"""主循环、工具分发、护栏判定、边界退出、交付校验。

见设计文档第 8 节与第 9 节。

决策权（设计文档 8.1）：
  LLM   —— 解读指令、声明能力边界、设定目标、每轮调参、调用工具、决定交付哪一轮
  harness —— 测量、算术、裁剪、护栏与回滚、预算与硬上限、校验输出合法性

本模块里不允许出现任何替 LLM 做优化决策的代码（设计文档 1.4 原则二）。
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from . import config
from .llm import LLMClient, LLMError
from .state import Round, TaskState
from .tools_aesthetic import aesthetic_score, measure as measure_aes, metric_info
from .tools_image import adjust
from .tools_stats import region_stats, stats
from .validate import clamp_params


# ================================================================ 小工具


def _sanitize(name: str, limit: int = 24) -> str:
    """把指令文本变成可做目录名的短标签。"""
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", "_", name, flags=re.UNICODE).strip("_")
    return (cleaned or "run")[:limit]


def _mean(values) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _error_of(stats_now: dict[str, float], target: dict[str, float]):
    """计算误差向量与平均绝对误差。只在 LLM 选定的维度上。"""
    ev = {k: stats_now[k] - v for k, v in target.items()}
    return ev, _mean(abs(v) for v in ev.values())


# ================================================================ 护栏


def guard_caps(original: dict[str, float] | None) -> tuple[float, float, float]:
    """护栏阈值。实现已移到 config（llm.py 组装上下文时也要用，见设计文档 3.4）。"""
    return config.guard_caps(original)


def check_image_guard(
    s: dict[str, float],
    original: dict[str, float] | None = None,
) -> tuple[bool, str]:
    """三条图像护栏。在轮次创建时判定（设计文档 9.1）。"""
    hi_cap, lo_cap, c_min = guard_caps(original)

    if s["clip_hi"] > hi_cap:
        return False, f"高光死白（clip_hi={s['clip_hi']:.4f} > {hi_cap:.4f}）"
    if s["clip_lo"] > lo_cap:
        return False, f"暗部死黑（clip_lo={s['clip_lo']:.4f} > {lo_cap:.4f}）"
    if s["contrast"] < c_min:
        return False, f"画面发灰（contrast={s['contrast']:.4f} < {c_min:.4f}）"
    return True, ""


def refresh_best_by_error(state: TaskState) -> None:
    """在通过护栏的轮次里取误差最小者（设计文档 6）。"""
    best: int | None = None
    for r in state.rounds:
        if not r.guard_ok or r.error is None:
            continue
        if best is None or r.error < state.rounds[best].error:
            best = r.index
    if best is not None:
        state.best_by_error = best


def apply_aesthetic_guard(state: TaskState) -> tuple[bool, str]:
    """审美硬红线的回溯判定（设计文档 9.1）。

    审美分是按需测量的，创建某一轮时通常还不知道它的 aes，因此必须在
    aes 被测出之后重新判定该轮是否合法。

    四步：写入结果 → 重判 guard_ok → 重算 best_by_error → 必要时回滚。

    Returns:
        ``(是否有轮次被改判为非法, 说明文本)``
    """
    flipped: Round | None = None
    for r in state.rounds:
        if r.aes is None or r.aes_delta is None or not r.guard_ok:
            continue
        if r.aes_delta <= -config.D_HARD:
            r.guard_ok = False
            r.guard_msg = (
                f"审美劣化超红线（{r.aes:.2f}，基线 {state.aes_baseline:.2f}，"
                f"差 {r.aes_delta:+.2f} ≤ -{config.D_HARD:g}）"
            )
            flipped = r

    if flipped is None:
        return False, ""

    refresh_best_by_error(state)

    # 若被改判的正是当前参数所在轮，立即回滚
    if flipped.index == state.current_index:
        state.params = state.rounds[state.best_by_error].params
        state.current_index = state.best_by_error
        return True, (
            f"第 {flipped.index} 轮触发审美硬红线（{flipped.guard_msg}），"
            f"已回滚到第 {state.best_by_error} 轮"
        )
    return True, f"第 {flipped.index} 轮被改判为非法（{flipped.guard_msg}）"


# ================================================================ 工具


def _aesthetic_level(delta: float) -> str:
    if delta <= -config.D_HARD:
        return "越过硬红线"
    if delta <= -config.D_SOFT:
        return "越过软警告阈值"
    return "正常范围"


def run_tool(reply: dict[str, Any], state: TaskState) -> tuple[str, dict[str, Any]]:
    """执行一次工具调用，返回 ``(给 LLM 看的文本, 结构化结果)``。

    工具调用不产生 Round，也不推进轮次号。
    """
    name = reply["tool"]

    if name == "aesthetic":
        tgt = reply.get("target", "current")
        n = len(state.rounds) - 1 if tgt == "current" else int(tgt)
        if not (0 <= n < len(state.rounds)):
            return f"错误：轮次 {n} 不存在", {"error": "bad_round", "round": n}

        r = state.rounds[n]
        score, elapsed = measure_aes(r.image_path)
        r.aes = score
        r.aes_delta = score - state.aes_baseline
        level = _aesthetic_level(r.aes_delta)

        data = {
            "round": n,
            "score": round(score, 4),
            "baseline": round(state.aes_baseline, 4),
            "delta": round(r.aes_delta, 4),
            "level": level,
            "seconds": round(elapsed, 3),
        }
        text = (
            f"第 {n} 轮审美分 {score:.2f}（基线 {state.aes_baseline:.2f}，"
            f"差 {r.aes_delta:+.2f}，{level}），耗时 {elapsed:.2f}s"
        )
        return text, data

    if name == "region":
        region = reply["region"]
        rs = region_stats(state.rounds[-1].image_path, region)
        text = f"{region} 区域：" + ", ".join(f"{k} {v:+.4f}" for k, v in rs.items())
        return text, {"region": region, **{k: round(v, 4) for k, v in rs.items()}}

    return f"错误：未知工具 {name}", {"error": "unknown_tool", "tool": name}


# ================================================================ 交付校验


def check_delivery(state: TaskState, n: Any) -> tuple[str, str]:
    """校验 LLM 指定的交付轮次（设计文档 9.4）。

    Returns:
        ``(kind, msg)``，kind ∈ ``{"ok", "challenge", "reject"}``

    - ``reject``    客观错误，必须改，计数
    - ``challenge`` 把权衡摆出来要求确认，同一轮只质疑一次；不计数
    - ``ok``        接受
    """
    last = len(state.rounds) - 1
    if not isinstance(n, int) or isinstance(n, bool) or not (0 <= n <= last):
        return "reject", f"deliver_round 必须是已存在的轮次号（0 ~ {last}）"

    r = state.rounds[n]
    if not r.guard_ok:
        return "reject", f"第 {n} 轮触发了护栏（{r.guard_msg}），不能交付，请改选"

    if state.aesthetic_mode == "tool" and r.aes is None:
        return "reject", f"请先对第 {n} 轮调用 aesthetic 确认，再 finish"

    best = state.best_by_error
    if (
        n != best
        and n not in state.challenged
        and r.error is not None
        and state.rounds[best].error is not None
        and r.error > state.rounds[best].error
    ):
        return "challenge", (
            f"第 {n} 轮平均误差 {r.error:.4f}，第 {best} 轮为 "
            f"{state.rounds[best].error:.4f}（更低）。"
            f"请确认你仍然选择第 {n} 轮，并在 reason 中说明理由"
        )

    return "ok", ""


# ================================================================ 目标修正


def apply_target_revision(state: TaskState, new_target: dict, reason: str) -> str:
    """LLM 修正目标（设计文档 5.2）。

    用途：当原目标在参数能力范围内确实不可达时（典型情况是撞护栏），
    LLM 需要一个出口来修正自己的判断，而不是无限撞墙。

    修正后：
    - 新目标合并进现有目标（只覆盖给出的维度）
    - 所有历史轮次的误差按新目标重算，并刷新 best_by_error
    - 记录到 target_revisions，报告里会列出来
    """
    old = dict(state.target)

    merged = dict(state.target)
    for k, v in new_target.items():
        lo, hi = config.STAT_RANGE[k]
        merged[k] = float(min(max(float(v), lo), hi))
    state.target = merged

    for r in state.rounds:
        r.error_vector, r.error = _error_of(r.stats, merged)

    refresh_best_by_error(state)

    state.target_revisions.append(
        {
            "round": len(state.rounds),
            "old_target": old,
            "new_target": dict(merged),
            "reason": reason,
        }
    )

    changed = ", ".join(
        f"{k}: {old.get(k, '—')} → {merged[k]}" for k in merged
    )
    return f"目标已修正（{changed}）。所有轮次的误差已按新目标重算。"


# ================================================================ 边界


def _recent_improvements(state: TaskState, n: int) -> list[float] | None:
    """最近 n 轮的改善量（error(上一轮) - error(本轮)）。不足则返回 None。

    只把护栏合法的轮次放进链条。触发护栏的轮次会被回滚，它的 error 是在一个
    并未真正停留过的参数位置上测得的，与合法轮次的 error 不可比。若把它们一并
    入链，会凭空产生一次"改善"（非法轮的误差恰好比上一轮小），紧随其后再产生
    一次"退步"（合法轮从回滚后的位置重新出发，误差比非法轮大）。两次假信号就
    顶满 PLATEAU_ROUNDS，让预算还剩大半的任务提前退出。
    """
    rounds = [r for r in state.rounds if r.guard_ok]
    if len(rounds) < n + 1:
        return None
    out = []
    for i in range(len(rounds) - n, len(rounds)):
        a, b = rounds[i - 1].error, rounds[i].error
        if a is None or b is None:
            return None
        out.append(a - b)
    return out


def _consecutive_violations(state: TaskState) -> int:
    """从最后一轮往回数，连续触发护栏的轮次数（跳过第 0 轮，原图不判护栏）。"""
    n = 0
    for r in reversed(state.rounds[1:]):
        if r.guard_ok:
            break
        n += 1
    return n


def check_boundaries(state: TaskState) -> str:
    """退出条件与优先级（设计文档 9.3）。

    返回 ``"continue"`` / ``"exit"`` / ``"rollback"``，并在退出时设置 ``exit_reason``。
    """
    rounds = state.rounds
    last = rounds[-1]

    # 1. 消耗的轮次预算达到上限（调整轮次 + 目标修正次数）
    if state.rounds_used >= config.ROUND_LIMIT:
        state.exit_reason = "round_limit"
        return "exit"

    # 2. 连续多轮护栏非法（阈值见 config.GUARDRAIL_STREAK_LIMIT 的说明）
    if _consecutive_violations(state) >= config.GUARDRAIL_STREAK_LIMIT:
        state.exit_reason = "guardrail_violation"
        return "exit"

    # 3. 全部参数裁剪后增量为 0（顶死在边界）
    if last.effective_delta and all(
        abs(v) < 1e-12 for v in last.effective_delta.values()
    ):
        state.exit_reason = "no_move_possible"
        return "exit"

    # 4/5. 连续若干轮不再改善。先看是否达标，再决定记为哪个原因。
    # 注意：这里的原因名不能与 LLM 主动 finish 时给的状态重名，
    # 否则报告里会出现"由安全网兜底"却写着"LLM 主动结束"的矛盾。
    imps = _recent_improvements(state, config.PLATEAU_ROUNDS)
    if imps is not None and all(x < config.IMPROVE_EPS for x in imps):
        state.exit_reason = "stalled_after_goal" if state.is_goal_met() else "plateau"
        return "exit"

    # 本轮触发护栏但还没到连续上限 → 回滚
    if not last.guard_ok:
        return "rollback"

    # 组装给下一轮的提示
    msgs: list[str] = []

    if state.is_goal_met():
        msgs.append("误差已达标，建议 finish")

    # 多轮未使用审美工具时给一次提醒。
    # 注意：这不是优化决策，只是提醒"这个工具存在且还没用过"。
    if (
        state.aesthetic_mode == "tool"
        and state.tools_used == 0
        and state.adjust_count >= config.AESTHETIC_REMINDER_AFTER
        and not state.aesthetic_reminded
    ):
        state.aesthetic_reminded = True
        msgs.append(
            f"你已经连续调整了 {state.adjust_count} 轮，但还没有确认过画面观感。"
            "统计量达标不等于画面好看，建议调用一次 aesthetic 看看"
            "（调用它不消耗轮次，只消耗工具次数）"
        )

    if msgs:
        state.pending_warning = "；".join(msgs)

    return "continue"


# ================================================================ 产物


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def finalize_outputs(state: TaskState, keep_intermediate: bool) -> str:
    """阶段 4：确定交付图、清理中间产物、落盘状态。

    见设计文档 11.1：只保留最终交付的那一张图片。中间图写在 tmp_dir，
    交付时复制交付轮次的那一张到 ``delivered.jpg``，然后删除 tmp_dir。

    删除时机必须在交付阶段，因为运行期间 ``aesthetic(target=<历史轮次>)``
    随时可能回看某一轮的图片。
    """
    run_dir = Path(state.run_dir)
    tmp_dir = Path(state.tmp_dir)

    n = state.delivered if state.delivered is not None else state.best_by_error
    state.delivered = n

    delivered_src = Path(state.rounds[n].image_path)
    delivered_dst = run_dir / "delivered.jpg"
    for src, dst in ((Path(state.original_path), run_dir / "original.jpg"),
                     (delivered_src, delivered_dst)):
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
    state.rounds[n].image_path = str(delivered_dst)

    if keep_intermediate:
        # 调试模式：把中间图从 tmp 搬到 run 目录
        for r in state.rounds:
            src = Path(r.image_path)
            if r.index == n or not src.exists():
                continue
            dst = run_dir / f"round_{r.index:02d}.jpg"
            shutil.copyfile(src, dst)
            r.image_path = str(dst)
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        for r in state.rounds:
            if r.index != n:
                r.image_path = ""  # 已删除，需要时用 --regenerate 重算

    _write_json(run_dir / "rounds.json", [r.to_dict() for r in state.rounds])
    _write_json(run_dir / "state.json", state.to_dict())

    with (run_dir / "tool_calls.jsonl").open("w", encoding="utf-8") as fh:
        for rec in state.tool_log:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return str(delivered_dst)


def regenerate(run_dir: str | Path, round_index: int) -> str:
    """用 rounds.json 里的参数向量重算某一轮的画面（设计文档 11.1）。

    因为候选图永远等于 ``adjust(原图, 累计参数)``，所以结果是确定性的、
    与当轮完全一致。中间图因此可以安全删除。
    """
    run_dir = Path(run_dir)
    rounds = json.loads((run_dir / "rounds.json").read_text(encoding="utf-8"))
    if not (0 <= round_index < len(rounds)):
        raise ValueError(f"轮次号超出范围：0 ~ {len(rounds) - 1}")

    out_path = run_dir / f"round_{round_index:02d}.jpg"
    adjust(run_dir / "original.jpg", rounds[round_index]["params"], out_path)
    return str(out_path)


# ================================================================ 主流程


def run(
    instruction: str,
    original_path: str | Path,
    run_root: str | Path = "runs",
    api_key: str | None = None,
    confirm: bool | None = None,
    keep_intermediate: bool | None = None,
    aesthetic_mode: str | None = None,
    run_name: str | None = None,
) -> TaskState:
    """跑完一次完整的 agent 循环，返回最终状态。

    流程见设计文档 8.2。

    Args:
        run_name: 产物目录名。默认用"时间戳 + 指令摘要"；批处理会传图片名，
            便于把每个任务的产物放在固定位置。
    """
    confirm = config.CONFIRM if confirm is None else confirm
    keep_intermediate = (
        config.KEEP_INTERMEDIATE_IMAGES if keep_intermediate is None else keep_intermediate
    )
    aesthetic_mode = aesthetic_mode or config.AESTHETIC_MODE

    stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    name = run_name or f"{stamp}_{_sanitize(instruction)}"
    run_dir = Path(run_root) / name
    tmp_dir = run_dir / ".work"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    state = TaskState(
        instruction=instruction,
        original_path=str(Path(original_path).resolve()),
        run_dir=str(run_dir),
        tmp_dir=str(tmp_dir),
        aesthetic_mode=aesthetic_mode,
    )

    # ---------------- 阶段 1：测量原图 ----------------
    state.original_stats = stats(original_path)
    state.aes_baseline = aesthetic_score(original_path)
    state.metric_info = metric_info()

    shutil.copyfile(original_path, tmp_dir / "round_00.jpg")
    r0 = Round(
        index=0,
        params=dict(config.ZERO_PARAMS),
        requested_delta={},
        effective_delta={},
        stats=state.original_stats,
        error=None,  # 目标确定后补算
        error_vector={},
        guard_ok=True,
        guard_msg="",
        aes=state.aes_baseline,
        aes_delta=0.0,
        llm_reason="（原图，无动作）",
        image_path=str(tmp_dir / "round_00.jpg"),
    )
    state.rounds.append(r0)

    client = LLMClient(api_key=api_key, log_path=run_dir / "llm_calls.jsonl")

    # ---------------- 阶段 2：解读指令 + 制定目标 ----------------
    try:
        reply = client.interpret(state)
    except LLMError as exc:
        # 调用一失败：交付原图，仍写出报告与日志，便于排查
        state.exit_reason = "llm_output_invalid"
        state.target_reason = f"调用一失败：{exc}"
        state.delivered = state.best_by_error
        state.delivered_by_safety_net = True
        finalize_outputs(state, keep_intermediate)
        from .report import write_report

        write_report(state)
        return state

    state.interpretation = reply["interpretation"]
    state.can_do = reply["can_do"]
    state.cannot_do = reply["cannot_do"]
    state.target = reply["target"]
    state.target_reason = reply["reason"]

    # 目标确定后才能算第 0 轮的误差
    r0.error_vector, r0.error = _error_of(state.original_stats, state.target)
    refresh_best_by_error(state)

    if confirm:
        from .report import print_interpretation

        print_interpretation(state)
        # 非交互式终端（管道、CI、cron）里 input() 会直接挂住，所以先探测
        if not sys.stdin.isatty():
            print("\n[提示] 当前不是交互式终端，跳过确认步骤，直接继续。")
        else:
            try:
                input("回车继续，Ctrl-C 中止 ... ")
            except EOFError:
                print("\n[提示] 输入流已结束，跳过确认，直接继续。")
            except KeyboardInterrupt:
                print("\n已中止。")
                raise

    # ---------------- 阶段 3：迭代 ----------------
    tool_log_path = run_dir / "tool_calls.jsonl"

    while True:
        warnings: list[str] = []

        try:
            # client.decide 内部先读取 pending_warning 组装上下文，随后立即清空
            reply = client.decide(state)
        except LLMError as exc:
            state.exit_reason = "llm_output_invalid"
            state.pending_warning = str(exc)
            break

        # ---- 3A 工具类：不消耗轮次 ----
        if "tool" in reply:
            if state.tools_used >= config.TOOL_BUDGET:
                state.tool_reject_count += 1
                state.pending_warning = "工具预算已用尽，请直接 adjust 或 finish"
                if state.tool_reject_count > config.TOOL_REJECT_LIMIT:
                    state.exit_reason = "tool_budget_exhausted"
                    break
                continue

            text, data = run_tool(reply, state)
            state.tools_used += 1
            record = {"call": reply, "result": text, "data": data}
            state.tool_log.append(record)
            with tool_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            flipped, msg = apply_aesthetic_guard(state)
            state.pending_warning = msg if flipped else None
            continue

        # ---- 3A2 修正目标（消耗轮次预算，次数受限）----
        if "revise_target" in reply:
            if len(state.target_revisions) >= config.TARGET_REVISION_LIMIT:
                state.pending_warning = (
                    f"目标修正次数已用尽（上限 {config.TARGET_REVISION_LIMIT} 次），"
                    "请基于当前目标继续调整，或 finish"
                )
                continue
            msg = apply_target_revision(state, reply["revise_target"], reply["reason"])
            state.pending_warning = msg

            action = check_boundaries(state)
            if action == "exit":
                break
            continue

        # ---- 3B 结束：校验交付轮次 ----
        if "finish" in reply:
            kind, msg = check_delivery(state, reply.get("deliver_round"))
            if kind == "challenge":
                state.challenged.add(reply["deliver_round"])
                state.pending_warning = msg
                continue
            if kind == "reject":
                state.reject_count += 1
                state.pending_warning = msg
                if state.reject_count > config.DELIVER_REJECT_LIMIT:
                    state.exit_reason = "finish_rejected"
                    state.delivered_by_safety_net = True
                    break
                continue
            state.exit_reason = reply["finish"]
            state.finish_status = reply["finish"]
            state.finish_reason = reply["reason"]
            state.delivered = reply["deliver_round"]
            break

        # ---- 3C 调参 ----
        new_params, effective = clamp_params(state.params, reply["adjust"])
        img_path = adjust(
            state.original_path,
            new_params,
            tmp_dir / f"round_{len(state.rounds):02d}.jpg",
        )
        s = stats(img_path)
        ev, err = _error_of(s, state.target)
        guard_ok, guard_msg = check_image_guard(s, state.original_stats)

        r = Round(
            index=len(state.rounds),
            params=new_params,
            requested_delta=dict(reply["adjust"]),
            effective_delta=effective,
            stats=s,
            error=err,
            error_vector=ev,
            guard_ok=guard_ok,
            guard_msg=guard_msg,
            llm_reason=reply["reason"],
            image_path=img_path,
        )
        state.rounds.append(r)
        state.current_index = r.index  # 当前参数对应的轮次

        # auto 模式：harness 每轮自动测审美分（设计文档 5.5）
        if state.aesthetic_mode == "auto":
            score, elapsed = measure_aes(img_path)
            r.aes = score
            r.aes_delta = score - state.aes_baseline
            flipped, msg = apply_aesthetic_guard(state)
            if flipped:
                warnings.append(msg)

        refresh_best_by_error(state)

        action = check_boundaries(state)
        if action == "exit":
            if state.delivered is None:
                state.delivered_by_safety_net = True
            break
        if action == "rollback":
            # 回滚：参数与 current_index 一起退回，否则上下文会把非法轮的参数
            # 当成"当前参数"告诉 LLM，让它在错误的位置上做决策。
            state.params = state.rounds[state.best_by_error].params
            state.current_index = state.best_by_error
            state.pending_warning = (
                f"第 {r.index} 轮触发护栏（{guard_msg}），已回滚到第 {state.best_by_error} 轮。"
                f"你现在的参数是第 {state.best_by_error} 轮的那一组，"
                f"请以此为基准重新计算增量。"
            )
            continue

        state.params = new_params
        if warnings:
            # 与 check_boundaries 可能已设置的"已达标"提示合并，不要互相覆盖
            merged = ([state.pending_warning] if state.pending_warning else []) + warnings
            state.pending_warning = "；".join(merged)

    # ---------------- 阶段 4：交付 ----------------
    if state.delivered is None:
        state.delivered = state.best_by_error
        state.delivered_by_safety_net = True

    finalize_outputs(state, keep_intermediate)

    from .report import write_report

    write_report(state)
    return state
