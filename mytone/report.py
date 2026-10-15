"""报告生成。

见设计文档 11.2：report.md 固定五部分。

    一、指令与解读
    二、参数变化表
    三、统计量与审美分轨迹
    四、交付决策
    五、逐轮决策依据与工具调用
"""

from __future__ import annotations

from pathlib import Path

from . import config
from .state import TaskState

EXIT_REASON_TEXT = {
    # --- LLM 主动 finish 时给的状态（这三个由 LLM 选择） ---
    "goal_met": "LLM 主动结束：声明目标已达成",
    "converged": "LLM 主动结束：声明已收敛",
    "best_effort": "LLM 主动结束：尽力而为",
    # --- harness 边界强制退出（这些不是 LLM 的选择） ---
    "round_limit": "轮次耗尽（硬上限），由 harness 结束",
    "guardrail_violation": f"连续 {config.GUARDRAIL_STREAK_LIMIT} 轮触发护栏，由 harness 结束",
    "no_move_possible": "全部参数已顶到边界，无法继续，由 harness 结束",
    "plateau": "未达标且陷入平台期，由 harness 结束",
    "stalled_after_goal": "已达标但连续多轮无改善，由 harness 结束",
    "tool_budget_exhausted": "工具预算耗尽后仍反复请求工具，由 harness 结束",
    "finish_rejected": "交付轮次多次不合法，由 harness 结束",
    "llm_output_invalid": "LLM 输出多次不合法或调用失败，由 harness 结束",
}


def exit_reason_text(reason: str) -> str:
    return EXIT_REASON_TEXT.get(reason, reason or "（未记录）")


def print_interpretation(state: TaskState) -> None:
    """--confirm 模式下打印解读与能力边界，等待用户确认。"""
    print("\n" + "=" * 68)
    print("指令解读")
    print("=" * 68)
    print(f"原始指令：{state.instruction}")
    print(f"解    读：{state.interpretation}")
    print()
    print("能做到：")
    for item in state.can_do:
        print(f"  - {item}")
    print("做不到：")
    for item in state.cannot_do or ["（无）"]:
        print(f"  - {item}")
    print()
    print("目标统计量：")
    for k, v in state.target.items():
        now = state.original_stats.get(k)
        print(f"  {k:<11} {now:.4f} → {v:.4f}")
    print(f"\n依据：{state.target_reason}")
    print("=" * 68)


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{v:.4f}"


def _signed(v: float | None) -> str:
    return "—" if v is None else f"{v:+.4f}"


def _params_row(params: dict[str, float]) -> str:
    return " | ".join(f"{params[k]:+.2f}" for k in config.PARAM_NAMES)


def _delta_text(delta: dict[str, float]) -> str:
    if not delta:
        return "原图"
    return ", ".join(f"{k} {v:+.3f}" for k, v in delta.items())


def build_report(state: TaskState) -> str:
    """生成 report.md 的完整文本。"""
    lines: list[str] = []
    a = lines.append

    a("# 调色报告")
    a("")
    a(f"- 原始指令：**{state.instruction}**")
    a(f"- 原图：`{state.original_path}`")
    a(f"- 运行目录：`{state.run_dir}`")
    a(f"- 审美工具：`{state.metric_info.get('name', config.AESTHETIC_METRIC)}`"
      f"（设备 {state.metric_info.get('device', config.DEVICE)}，"
      f"分数区间 {state.metric_info.get('score_range', '—')}）")
    a(f"- 审美分模式：`{state.aesthetic_mode}`")
    a(f"- LLM 调用次数：{state.llm_call_count}")
    a("")

    # ---------------------------------------------------------- 一、指令与解读
    a("## 一、指令与解读")
    a("")
    a(f"**解读**：{state.interpretation or '（无）'}")
    a("")
    a("| 能做到 | 做不到 |")
    a("|---|---|")
    rows = max(len(state.can_do), len(state.cannot_do)) or 1
    for i in range(rows):
        left = state.can_do[i] if i < len(state.can_do) else ""
        right = state.cannot_do[i] if i < len(state.cannot_do) else ""
        a(f"| {left} | {right} |")
    a("")
    a(f"**目标设定依据**：{state.target_reason or '（无）'}")
    a("")

    # ---------------------------------------------------------- 二、参数变化表
    a("## 二、参数变化表")
    a("")
    a("参数为**累计绝对值**。`请求→生效` 一列在发生裁剪时标注出来。")
    a("")
    header = " | ".join(config.PARAM_NAMES)
    a(f"| 轮次 | {header} | 请求 → 生效 | 备注 |")
    a("|---" * (len(config.PARAM_NAMES) + 3) + "|")
    for r in state.rounds:
        if r.index == 0:
            a(f"| 0 | {_params_row(r.params)} | 原图 | 起点 |")
            continue
        note = ""
        if r.clipped:
            note = "发生裁剪"
        if not r.guard_ok:
            note = (note + "；" if note else "") + f"护栏未通过：{r.guard_msg}"
        a(
            f"| {r.index} | {_params_row(r.params)} | "
            f"{_delta_text(r.requested_delta)} → {_delta_text(r.effective_delta)} | {note} |"
        )
    a("")

    # ------------------------------------------------- 三、统计量与审美分轨迹
    a("## 三、统计量与审美分轨迹")
    a("")
    a(
        "| 轮次 | luma | contrast | saturation | warmth | clip_hi | clip_lo "
        "| 平均误差 | 审美分 | 审美差 | 护栏 |"
    )
    a("|---|---|---|---|---|---|---|---|---|---|---|")

    tgt = state.target
    a(
        "| **目标** | "
        + " | ".join(f"**{tgt[k]:.4f}**" if k in tgt else "—" for k in config.STAT_NAMES)
        + " | — | — | — | — | — | — |"
    )
    for r in state.rounds:
        s = r.stats
        guard = "通过" if r.guard_ok else "非法"
        a(
            f"| {r.index} | {s['luma']:.4f} | {s['contrast']:.4f} | {s['saturation']:.4f} "
            f"| {s['warmth']:+.4f} | {s['clip_hi']:.4f} | {s['clip_lo']:.4f} "
            f"| {_pct(r.error)} | {'—' if r.aes is None else f'{r.aes:.2f}'} "
            f"| {_signed(r.aes_delta)} | {guard} |"
        )
    a("")
    a(f"审美分基线（第 0 轮）：**{state.aes_baseline:.2f}**")
    a("")
    if state.target_revisions:
        a(
            f"> 注意：本次运行中目标被修正过 {len(state.target_revisions)} 次（见第四部分）。"
            "上表所有轮次的误差都**已按最终目标**重算，因此与各轮当时的判断依据可能不一致。"
        )
        a("")
    a(f"收敛容差：平均绝对误差 < {config.ERROR_TOL}")
    a("")

    # ---------------------------------------------------------- 四、交付决策
    a("## 四、交付决策")
    a("")
    n = state.delivered if state.delivered is not None else state.best_by_error
    r = state.rounds[n]
    best = state.rounds[state.best_by_error]
    a("| 项 | 内容 |")
    a("|---|---|")
    a(f"| 交付轮次 | **第 {n} 轮** |")
    a(f"| 是否由 LLM 选择 | {'否，安全网兜底' if state.delivered_by_safety_net else '是'} |")
    a(f"| 误差最小轮次 | 第 {best.index} 轮（{_pct(best.error)}） |")
    a(f"| 交付轮次误差 | {_pct(r.error)} |")
    a(
        f"| 审美分 | "
        + ("未测量" if r.aes is None else f"{r.aes:.2f}（相对基线 {_signed(r.aes_delta)}）")
        + " |"
    )
    a(f"| 是否达标 | {'是' if (r.error is not None and r.error < config.ERROR_TOL) else '否'} |")
    a(f"| 目标修正次数 | {len(state.target_revisions)}（上限 {config.TARGET_REVISION_LIMIT}） |")
    a(f"| 退出原因 | {exit_reason_text(state.exit_reason)} |")
    a(f"| LLM 状态标记 | {state.finish_status or '（安全网，未由 LLM 给出）'} |")
    a("")

    if state.target_revisions:
        a("**目标修正记录**（由 LLM 发起，用于处理原目标不可达的情况）：")
        a("")
        a("| 发生在 | 原目标 | 新目标 | 理由 |")
        a("|---|---|---|---|")
        for rev in state.target_revisions:
            old = "，".join(f"{k}={v}" for k, v in rev["old_target"].items())
            new = "，".join(f"{k}={v}" for k, v in rev["new_target"].items())
            a(f"| 第 {rev['round']} 轮前 | {old} | {new} | {rev['reason']} |")
        a("")
    if state.finish_reason:
        a("**LLM 的交付决策理由（原文）**：")
        a("")
        a(f"> {state.finish_reason}")
    else:
        a("**交付由安全网兜底**：LLM 未给出合法的交付轮次，harness 交付了误差最小的合规轮次。"
          "这不是 LLM 的选择。")
    a("")

    # --------------------------------------------- 五、逐轮决策依据与工具调用
    a("## 五、逐轮决策依据与工具调用")
    a("")
    for r in state.rounds:
        if r.index == 0:
            continue
        a(f"### 第 {r.index} 轮")
        a("")
        a(f"- 动作：{_delta_text(r.requested_delta)}")
        if r.clipped:
            a(f"- **发生裁剪**：真实生效 {_delta_text(r.effective_delta)}")
        a(f"- 统计量：luma {r.stats['luma']:.4f}, contrast {r.stats['contrast']:.4f}, "
          f"saturation {r.stats['saturation']:.4f}, warmth {r.stats['warmth']:+.4f}")
        a(f"- 平均误差：{_pct(r.error)}")
        a(f"- 护栏：{'通过' if r.guard_ok else '未通过 — ' + r.guard_msg}")
        a(f"- 依据：{r.llm_reason}")
        a("")

    if state.tool_log:
        a("### 工具调用记录")
        a("")
        a("| # | 调用 | 理由 | 结果 |")
        a("|---|---|---|---|")
        for i, rec in enumerate(state.tool_log, 1):
            call = rec["call"]
            if call.get("tool") == "aesthetic":
                what = f"aesthetic(target={call.get('target', 'current')})"
            else:
                what = f"{call.get('tool')}({call.get('region', '')})"
            a(f"| {i} | `{what}` | {call.get('reason', '')} | {rec.get('result', '')} |")
        a("")
    else:
        a("### 工具调用记录")
        a("")
        a("（本次运行没有工具调用）")
        a("")

    return "\n".join(lines)


def write_report(state: TaskState) -> str:
    """把报告写到运行目录，返回路径。"""
    path = Path(state.run_dir) / "report.md"
    path.write_text(build_report(state), encoding="utf-8")
    state.report_path = str(path)
    return str(path)
