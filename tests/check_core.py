"""核心逻辑验证脚本（不需要 pytest，直接跑）。

    .venv/bin/python tests/check_core.py

覆盖：
    1. adjust 的确定性与"从原图重算"性质
    2. 五个参数的耦合方向是否与设计文档 4.4 的耦合表一致
    3. clamp_params 的两级裁剪语义
    4. validate_target / validate_action 的校验规则
    5. parse_json 的容错
    6. check_delivery 的三态语义，特别是"同一轮只质疑一次"
    7. check_boundaries 的退出优先级与 plateau/converged 的互斥，
       以及回滚轮不计入平台判定
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mytone import config  # noqa: E402
from mytone import validate as V  # noqa: E402
from mytone.llm import parse_json  # noqa: E402
from mytone.loop import check_boundaries, check_delivery, check_image_guard  # noqa: E402
from mytone.state import Round, TaskState  # noqa: E402
from mytone.tools_image import adjust  # noqa: E402
from mytone.tools_stats import stats  # noqa: E402

PASSED = 0
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED
    if cond:
        PASSED += 1
        print(f"  [OK]   {name}")
    else:
        FAILED.append(f"{name}  {detail}")
        print(f"  [FAIL] {name}  {detail}")


def make_image(path: Path, size: int = 256, base: float = 0.25) -> None:
    """造一张有亮度层次与色彩变化的测试图。

    ``base`` 控制整体亮度；调到 0.8 左右会得到一张有高光溢出的图，
    用于测 clip_hi 相关的耦合。
    """
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:size, 0:size] / size
    r = base + 0.5 * x
    g = base + 0.05 + 0.4 * y
    b = base + 0.20 - 0.2 * (x * y)
    arr = np.stack([r, g, b], -1) + rng.normal(0, 0.02, (size, size, 3))
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)


def make_state(**kw) -> TaskState:
    st = TaskState(instruction="测试", original_path="/x.jpg", run_dir="/r", tmp_dir="/t")
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def make_round(index: int, error: float, guard_ok: bool = True, aes: float | None = 5.0) -> Round:
    return Round(
        index=index,
        params=dict(config.ZERO_PARAMS),
        requested_delta={},
        effective_delta={},
        stats={k: 0.5 for k in config.STAT_NAMES + config.GUARD_STATS},
        error=error,
        error_vector={},
        guard_ok=guard_ok,
        guard_msg="" if guard_ok else "测试用非法",
        aes=aes,
        aes_delta=None if aes is None else 0.0,
    )


def set_rounds(st: TaskState, rounds: list[Round]) -> TaskState:
    """设置轮次并把 current_index 指到最后一轮。

    current_index 表示"当前参数实际对应的轮次"；正常前进时它就是最后一轮，
    触发护栏回滚后它会指回被回滚到的那一轮（见 state.TaskState.current_index）。
    手工构造 state 时必须一并设置，否则 is_goal_met() 会读错轮次。
    """
    st.rounds = rounds
    st.current_index = len(rounds) - 1
    return st


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="mytone_test_"))
    img = tmp / "base.jpg"
    make_image(img)

    # ================================================ 1. 确定性与可重算
    print("\n[1] adjust 的确定性与可重算性质")
    p = dict(config.ZERO_PARAMS)
    p.update(brightness=0.1, saturation=-0.05, color_temp=-100)
    a1 = adjust(img, p, tmp / "a1.jpg")
    a2 = adjust(img, p, tmp / "a2.jpg")
    s1, s2 = stats(a1), stats(a2)
    check("相同参数两次渲染，统计量完全一致", s1 == s2, f"{s1} vs {s2}")

    # 先做一半再补另一半，必须等于一次到位（因为永远从原图重算）
    half1 = dict(config.ZERO_PARAMS)
    half1["brightness"] = 0.05
    adjust(img, half1, tmp / "half.jpg")
    s_direct = stats(adjust(img, p, tmp / "direct.jpg"))
    check("从原图重算 == 一次到位（不受中间产物影响）", s_direct == s1)

    # ================================================ 2. 耦合方向
    print("\n[2] 五个参数的耦合方向（对照设计文档 4.4）")
    base = stats(img)
    expectations = {
        # 参数: (主要影响的统计量, 正向时该统计量应如何变化)
        "brightness": ("luma", +1),
        "contrast": ("contrast", +1),
        "saturation": ("saturation", +1),
        "color_temp": ("warmth", +1),
        "gamma": ("luma", -1),  # gamma>0 使中间调变暗
    }
    for name, (main_stat, sign) in expectations.items():
        for direction in (+1, -1):
            pp = dict(config.ZERO_PARAMS)
            pp[name] = direction * config.PARAMS[name]["step"]
            s = stats(adjust(img, pp, tmp / f"c_{name}_{direction}.jpg"))
            delta = s[main_stat] - base[main_stat]
            ok = (delta * sign * direction) > 0
            check(
                f"{name} {'+' if direction > 0 else '-'} → {main_stat} "
                f"{'增' if delta > 0 else '减'} ({delta:+.4f})",
                ok,
            )

    # 实测耦合矩阵：打印出来用于核对设计文档 4.4 的耦合表
    print("  -- 实测耦合矩阵（一步的影响量，含两个护栏量）--")
    all_stats = config.STAT_NAMES + config.GUARD_STATS
    header = f"  {'参数':<12}" + "".join(f"{s:>11}" for s in all_stats)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, (main_stat, _) in expectations.items():
        pp = dict(config.ZERO_PARAMS)
        pp[name] = config.PARAMS[name]["step"]
        s = stats(adjust(img, pp, tmp / f"m_{name}.jpg"))
        deltas = {k: s[k] - base[k] for k in all_stats}
        line = f"  {name:<12}" + "".join(f"{deltas[k]:>+11.4f}" for k in all_stats)
        print(line + "  <- 主要影响")

        # 断言：文档声明的主要影响确实存在且非平凡
        main_delta = deltas[main_stat]
        check(
            f"{name} 对 {main_stat} 的影响非平凡（|{main_delta:+.4f}| > 0.005）",
            abs(main_delta) > 0.005,
        )

    # brightness 应是推高 clip_hi 最猛的那个（结构性质，写进了提示词）。
    #
    # 度量用"全范围最大增幅"而不是中心差分：color_temp 的效应是 U 形的，
    # 中心差分会把加暖与加冷两侧抵消掉。
    #
    # 注意：本用例只能验证方向（谁最贵），不能验证倍数。
    # 合成渐变图的高光分布与真实照片差别很大（真实照片的过曝区集中在天空等高光区，
    # 主体仍在中间调），倍数关系由 calibrate.py 在真实照片上测。
    # 实测参考：真实照片上 brightness 的截断代价约是 gamma 的 28 倍。
    from mytone.calibrate import max_increase

    clip_costs = {}
    for name in config.PARAM_NAMES:
        rows = []
        for mult in (-3, -2, -1, 0, 1, 2, 3):
            lo, hi = config.PARAMS[name]["range"]
            value = max(lo, min(hi, mult * config.PARAMS[name]["step"]))
            pp = dict(config.ZERO_PARAMS)
            pp[name] = value
            rows.append(
                {"mult": mult, "stats": stats(adjust(img, pp, tmp / f"ch_{name}_{mult}.jpg"))}
            )
        clip_costs[name] = max_increase(rows, "clip_hi")

    worst_clip = max(clip_costs, key=clip_costs.get)
    print("  -- 截断的代价（全范围 clip_hi 最大增幅）--")
    for name in sorted(clip_costs, key=lambda n: -clip_costs[n]):
        print(f"     {name:<12} {clip_costs[name]:+.4f}")
    check(
        f"推高 clip_hi 最猛的是 brightness（实际 {clip_costs['brightness']:+.4f}）",
        worst_clip == "brightness",
    )
    check(
        f"brightness 的截断代价显著（{clip_costs['brightness']:+.4f} > 0.02）",
        clip_costs["brightness"] > 0.02,
    )

    # ================================================ 3. 裁剪语义
    print("\n[3] clamp_params 的两级裁剪")
    zero = dict(config.ZERO_PARAMS)

    new, eff = V.clamp_params(zero, {"brightness": 0.9})  # 步长上限 0.2
    check("超过步长上限被裁剪到 0.2", abs(new["brightness"] - 0.2) < 1e-9, str(new["brightness"]))
    check("生效增量反映真实值", abs(eff["brightness"] - 0.2) < 1e-9)

    at_max = dict(zero)
    at_max["brightness"] = config.PARAMS["brightness"]["range"][1]  # 已在上界
    new, eff = V.clamp_params(at_max, {"brightness": 0.2})
    check("已在上界时正向增量生效为 0", abs(eff["brightness"]) < 1e-12, str(eff))
    check("累计值不超过区间上界", new["brightness"] <= 0.6 + 1e-9)

    new, eff = V.clamp_params(zero, {"color_temp": -300})
    check("color_temp 步长 400 内不裁剪", abs(new["color_temp"] + 300) < 1e-9)

    # ================================================ 4. validate
    print("\n[4] validate_target / validate_action")
    ok, prob, clean = V.validate_target(
        {
            "interpretation": "理解为清冷",
            "can_do": ["降饱和"],
            "cannot_do": ["磨皮"],
            "target": {"luma": 0.52, "warmth": -0.04},
            "reason": "依据",
        }
    )
    check("合法目标通过", ok, prob)
    check("目标被规范化", clean == {"luma": 0.52, "warmth": -0.04})

    ok, prob, _ = V.validate_target(
        {
            "interpretation": "x",
            "can_do": [],
            "cannot_do": [],
            "target": {"sharpness": 0.5},  # 自造指标
            "reason": "x",
        }
    )
    check("自造指标名被打回", not ok, prob)

    ok, prob, clean = V.validate_target(
        {
            "interpretation": "x",
            "can_do": [],
            "cannot_do": [],
            "target": {"luma": 99.0},  # 越界
            "reason": "x",
        }
    )
    check("越界数值被裁剪而不是打回", ok and clean["luma"] == config.STAT_RANGE["luma"][1], str(clean))

    ok, prob, _ = V.validate_target({"interpretation": "x", "can_do": [], "cannot_do": []})
    check("缺 target 被打回", not ok, prob)

    tools = {"aesthetic", "region"}
    ok, prob, _ = V.validate_action(
        {"adjust": {"brightness": 0.1}, "reason": "r"}, tools
    )
    check("合法 adjust 通过", ok, prob)

    ok, prob, _ = V.validate_action(
        {"adjust": {"brightness": 0.1, "contrast": 0.1, "gamma": 0.1}, "reason": "r"}, tools
    )
    check("单轮超过 2 个参数被打回", not ok, prob)

    ok, prob, _ = V.validate_action({"adjust": {"sharpness": 0.1}, "reason": "r"}, tools)
    check("未知参数被打回", not ok, prob)

    ok, prob, _ = V.validate_action(
        {"finish": "goal_met", "deliver_round": 1, "reason": "r"}, tools
    )
    check("合法 finish 通过", ok, prob)

    ok, prob, _ = V.validate_action({"finish": "goal_met", "reason": "r"}, tools)
    check("finish 缺 deliver_round 被打回", not ok, prob)

    ok, prob, _ = V.validate_action({"tool": "aesthetic", "target": "current", "reason": "r"}, tools)
    check("合法工具调用通过", ok, prob)

    ok, prob, _ = V.validate_action({"tool": "aesthetic", "target": "current", "reason": "r"}, {"region"})
    check("auto 模式下 aesthetic 不在清单内被打回", not ok, prob)

    ok, prob, _ = V.validate_action({"tool": "region", "region": "sky", "reason": "r"}, tools)
    check("未知区域被打回", not ok, prob)

    # ================================================ 5. parse_json
    print("\n[5] parse_json 容错")
    check("裸 JSON", parse_json('{"a": 1}') == {"a": 1})
    check("带围栏", parse_json('```json\n{"a": 1}\n```') == {"a": 1})
    check("前后有废话", parse_json('好的，这是结果：{"a": 1} 希望有帮助') == {"a": 1})
    check("非法输入返回 None", parse_json("完全不是 JSON") is None)
    check("空输入返回 None", parse_json("") is None)

    # ================================================ 6. check_delivery
    print("\n[6] check_delivery 的三态语义（重点：同一轮只质疑一次）")
    st = make_state(aesthetic_mode="tool")
    set_rounds(st, [make_round(0, 0.065), make_round(1, 0.003), make_round(2, 0.018)])
    st.aes_baseline = 5.41
    st.best_by_error = 1  # 第 1 轮误差最小

    kind, msg = check_delivery(st, 99)
    check("不存在的轮次 -> reject", kind == "reject", f"{kind}: {msg}")

    kind, _ = check_delivery(st, 1)
    check("选择误差最小轮 -> ok", kind == "ok", kind)

    kind, msg = check_delivery(st, 2)
    check("选择误差更高轮 -> challenge", kind == "challenge", kind)
    st.challenged.add(2)
    kind, msg = check_delivery(st, 2)
    check("同一轮再次提交 -> 直接 ok（不重复质疑）", kind == "ok", kind)

    st2 = make_state(aesthetic_mode="tool")
    set_rounds(st2, [make_round(0, 0.065), make_round(1, 0.003, guard_ok=False), make_round(2, 0.018)])
    st2.best_by_error = 0
    kind, msg = check_delivery(st2, 1)
    check("未过护栏的轮次 -> reject（与质疑区分）", kind == "reject", kind)

    st3 = make_state(aesthetic_mode="tool")
    set_rounds(st3, [make_round(0, 0.065), make_round(1, 0.02, aes=None)])
    st3.best_by_error = 0
    kind, msg = check_delivery(st3, 1)
    check("未测审美分 -> reject", kind == "reject", kind)

    st4 = make_state(aesthetic_mode="auto")
    set_rounds(st4, [make_round(0, 0.065), make_round(1, 0.018, aes=None)])
    st4.best_by_error = 0
    kind, _ = check_delivery(st4, 1)
    check("auto 模式不因缺 aes 而 reject", kind != "reject", kind)

    # ================================================ 7. check_boundaries
    print("\n[7] check_boundaries 的优先级与互斥")

    st = make_state()
    st.target = {"luma": 0.5}
    set_rounds(st, [make_round(0, 0.065)] + [make_round(i, 0.01) for i in range(1, 11)])
    st.rounds[-1].effective_delta = {"brightness": 0.1}
    check("调整轮次达上限 -> exit", check_boundaries(st) == "exit")
    check("退出原因为 round_limit", st.exit_reason == "round_limit", st.exit_reason)

    # 连续多轮护栏非法 -> exit（阈值 = config.GUARDRAIL_STREAK_LIMIT）
    n = config.GUARDRAIL_STREAK_LIMIT
    st = make_state()
    set_rounds(st, [make_round(0, 0.065)]
               + [make_round(i, 0.05, guard_ok=False) for i in range(1, n + 1)])
    st.rounds[-1].effective_delta = {"brightness": 0.1}
    check(f"连续 {n} 轮护栏非法 -> exit", check_boundaries(st) == "exit")
    check("退出原因为 guardrail_violation", st.exit_reason == "guardrail_violation", st.exit_reason)

    # 只连续 n-1 轮、最新一轮合法时不应因该条退出（给 agent 一次收敛机会）
    st = make_state()
    set_rounds(st, [make_round(0, 0.065)]
               + [make_round(i, 0.05, guard_ok=False) for i in range(1, n)]
               + [make_round(n, 0.04)])
    st.rounds[-1].effective_delta = {"brightness": 0.1}
    check(f"只连续 {n - 1} 轮非法、最新一轮合法 -> 不因该条退出",
          st.exit_reason != "guardrail_violation" if check_boundaries(st) == "exit" else True,
          st.exit_reason)

    st = make_state()
    set_rounds(st, [make_round(0, 0.065), make_round(1, 0.05)])
    st.rounds[-1].effective_delta = {"brightness": 0.0}  # 顶死在边界
    check("增量为 0 -> exit", check_boundaries(st) == "exit")
    check("退出原因为 no_move_possible", st.exit_reason == "no_move_possible", st.exit_reason)

    # 未达标 + 最近两轮都无改善 -> plateau
    st = make_state()
    set_rounds(st, [make_round(0, 0.20), make_round(1, 0.100),
                    make_round(2, 0.0995), make_round(3, 0.0992)])
    st.rounds[-1].effective_delta = {"brightness": 0.1}
    check("未达标且陷入平台 -> exit", check_boundaries(st) == "exit")
    check("退出原因为 plateau", st.exit_reason == "plateau", st.exit_reason)

    # 已达标 + 最近两轮都无改善 -> stalled_after_goal（不得被误记为 plateau，
    # 也不得与 LLM 主动 finish 时给的状态 converged 重名）
    st = make_state()
    set_rounds(st, [make_round(0, 0.20), make_round(1, 0.019),
                    make_round(2, 0.0185), make_round(3, 0.0182)])
    st.rounds[-1].effective_delta = {"brightness": 0.1}
    check("已达标且无改善 -> exit", check_boundaries(st) == "exit")
    check("退出原因为 stalled_after_goal（不是 plateau，也不是 converged）",
          st.exit_reason == "stalled_after_goal", st.exit_reason)

    # 达标但仍在改善 -> continue 且带出提示
    st = make_state()
    set_rounds(st, [make_round(0, 0.20), make_round(1, 0.019)])
    st.rounds[-1].effective_delta = {"brightness": 0.1}
    check("达标后仍可继续 -> continue", check_boundaries(st) == "continue")
    check("给出达标提示", st.pending_warning is not None, str(st.pending_warning))

    # 本轮非法但未连续 -> rollback
    st = make_state()
    set_rounds(st, [make_round(0, 0.2, guard_ok=True), make_round(1, 0.05, guard_ok=False)])
    st.rounds[-1].effective_delta = {"brightness": 0.1}
    check("单轮非法 -> rollback", check_boundaries(st) == "rollback")

    # 被护栏拒绝并回滚的轮次不得进入平台判定的改善链条。
    # 数据取自首次批量运行中的 65.JPG：第 1 轮触发护栏被回滚，它的误差是在一个
    # 并未真正停留过的参数位置上测得的。旧实现把它一并入链，于是 r1->r2 被记成
    # 一次"退步"，r2->r3 再记一次，两次假信号顶满 PLATEAU_ROUNDS，任务只用掉
    # 3 轮（预算 10 轮）就以 plateau 退出，误差停在 0.0318。
    st = make_state()
    set_rounds(st, [make_round(0, 0.0364),
                    make_round(1, 0.0312, guard_ok=False),
                    make_round(2, 0.0318),
                    make_round(3, 0.0359)])
    st.rounds[-1].effective_delta = {"brightness": 0.1}
    action = check_boundaries(st)
    check("回滚轮不计入平台判定 -> 不提前退出", action != "exit", st.exit_reason or "")
    check("回滚轮不计入平台判定 -> 继续迭代", action == "continue", action)

    # ================================================ 8. 图像护栏
    print("\n[8] 图像护栏（绝对阈值 + 相对原图的容差，取较宽松者）")
    normal = {**{k: 0.5 for k in config.STAT_NAMES}, "clip_hi": 0.0, "clip_lo": 0.0,
              "contrast": 0.25}
    ok, msg = check_image_guard(dict(normal), normal)
    check("正常图片通过", ok, msg)
    ok, _ = check_image_guard({**normal, "clip_hi": 0.5}, normal)
    check("高光死白被拦", not ok)
    ok, _ = check_image_guard({**normal, "clip_lo": 0.5}, normal)
    check("暗部死黑被拦", not ok)
    ok, _ = check_image_guard({**normal, "contrast": 0.05}, normal)
    check("对比度大幅下降被拦", not ok)

    # 关键：原图本身就是低对比时，不能被绝对阈值误判
    lowc = {**{k: 0.5 for k in config.STAT_NAMES}, "clip_hi": 0.0, "clip_lo": 0.0,
            "contrast": 0.03}
    ok, msg = check_image_guard(dict(lowc), lowc)
    check("低对比原图不因'原图如此'被误判", ok, msg)
    ok, _ = check_image_guard({**lowc, "contrast": 0.005}, lowc)
    check("低对比原图继续掉对比仍被拦", not ok)

    # 原图本身有截断时同理
    clipped = {**{k: 0.5 for k in config.STAT_NAMES}, "clip_hi": 0.03, "clip_lo": 0.06,
               "contrast": 0.25}
    ok, msg = check_image_guard(dict(clipped), clipped)
    check("原图已有截断时不误判", ok, msg)
    ok, _ = check_image_guard({**clipped, "clip_hi": 0.10}, clipped)
    check("在原图截断基础上继续恶化仍被拦", not ok)

    # ================================================ 9. 目标修正
    print("\n[9] revise_target：不可达时的出口")
    ok, prob, _ = V.validate_action(
        {"revise_target": {"luma": 0.60}, "reason": "原目标撞护栏不可达"}, tools
    )
    check("合法 revise_target 通过", ok, prob)

    ok, prob, _ = V.validate_action({"revise_target": {"luma": 0.60}}, tools)
    check("revise_target 缺 reason 被打回", not ok, prob)

    ok, prob, _ = V.validate_action(
        {"revise_target": {"sharpness": 0.5}, "reason": "r"}, tools
    )
    check("revise_target 含未知统计量被打回", not ok, prob)

    from mytone.loop import apply_target_revision

    st = make_state()
    st.target = {"luma": 0.60, "saturation": 0.20}
    st.aes_baseline = 5.0
    r0 = make_round(0, 0.10)
    r0.stats = {**{k: 0.5 for k in config.STAT_NAMES + config.GUARD_STATS}}
    r0.stats.update(luma=0.50, saturation=0.30)
    r1 = make_round(1, 0.12)
    r1.stats = {**r0.stats}
    r1.stats.update(luma=0.58, saturation=0.22)
    set_rounds(st, [r0, r1])
    st.best_by_error = 0  # 旧目标下第 0 轮更优

    msg = apply_target_revision(st, {"luma": 0.58}, "原目标撞护栏")
    check("目标被合并（只覆盖给出的维度）",
          st.target == {"luma": 0.58, "saturation": 0.20}, str(st.target))
    # 新目标下第 0 轮误差 = (|0.50-0.58| + |0.30-0.20|) / 2 = 0.09
    check("历史轮次误差按新目标重算", abs(r0.error - 0.09) < 1e-9, str(r0.error))
    # 第 1 轮误差 = (|0.58-0.58| + |0.22-0.20|) / 2 = 0.01
    check("第 1 轮误差按新目标重算", abs(r1.error - 0.01) < 1e-9, str(r1.error))
    check("best_by_error 被刷新（新目标下第 1 轮更优）",
          st.best_by_error == 1, str(st.best_by_error))
    check("修正被记入历史", len(st.target_revisions) == 1, str(st.target_revisions))
    check("修正消耗轮次预算", st.rounds_used == 1 + 1, str(st.rounds_used))
    check("返回说明包含变化", "目标已修正" in msg, msg)

    # 超出合理区间的修正值被裁剪
    st2 = make_state()
    st2.target = {"luma": 0.60}
    set_rounds(st2, [make_round(0, 0.1)])
    apply_target_revision(st2, {"luma": 99.0}, "越界")
    check("修正值越界被裁剪到合理区间",
          st2.target["luma"] == config.STAT_RANGE["luma"][1], str(st2.target))

    # ================================================ 10. 回滚后的"当前轮次"
    print("\n[10] current_index：回滚后当前参数必须指向被回滚到的轮次")
    from mytone.llm import build_decide_context

    st = make_state()
    st.target = {"luma": 0.60}
    good = make_round(0, 0.05)
    bad = make_round(1, 0.30, guard_ok=False)
    bad.params = {**config.ZERO_PARAMS, "brightness": 0.4}  # 非法轮尝试过的参数
    set_rounds(st, [good, bad])
    st.best_by_error = 0
    st.current_index = 0  # 模拟回滚后的状态

    check("回滚后 current_round 指向被回滚到的轮次",
          st.current_round.index == 0, str(st.current_round.index))
    check("last_attempt 仍指向那次失败尝试",
          st.last_attempt.index == 1, str(st.last_attempt.index))
    check("is_goal_met 读的是当前轮而不是失败轮",
          st.current_round.error == 0.05, str(st.current_round.error))

    # 上下文里给出的"当前参数"必须是回滚后的那一组，
    # 否则 LLM 会在一个它并不在的位置上计算增量（首次实跑就踩了这个坑）
    ctx = build_decide_context(st)
    cur_line = next(ln for ln in ctx.splitlines() if ln.startswith("当前累计参数"))
    check("上下文中的当前参数是回滚后的（brightness +0.00，不是非法轮的 +0.40）",
          "brightness +0.00" in cur_line, cur_line)

    # ================================================ 汇总
    print("\n" + "=" * 60)
    print(f"通过 {PASSED} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for f in FAILED:
            print(f"  FAIL: {f}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
