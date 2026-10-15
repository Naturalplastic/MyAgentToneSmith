"""审美分灵敏度标定 + 参数耦合实测。

见设计文档 3.6 与 12.3 第 3 步。建议在正式使用前跑一次，
它的产出决定 config 里的阈值与提示词里注入的数字。

做法：
    对五个参数各做 7 档单参数变化（以"步长"为单位：-3 ~ +3 步），
    共 35 张变体，记录每张的六个统计量与审美分。

产出两件事：
    1. 参数耦合实测 —— 验证设计文档 4.4 的耦合表
       "一步"会让每个统计量动多少。
    2. 审美分灵敏度 —— 决定 D_SOFT / D_HARD
       NIMA 是确定性的（同图同分），所以这里测的不是重复性，
       而是"参数变化能不能让它动"。

建议阈值的取法：
    以"一个步长能造成的审美分变化"为基准单位。
    D_SOFT = 1.5 倍该值（比任何单个合法步长的影响都大 → 值得警告）
    D_HARD = 3 倍该值（约三个步长的累积影响 → 该回滚了）
    这只是启发式，输出里会同时给出全范围极差供参考。

用法：
    python -m mytone.calibrate -i photo.jpg
    python -m mytone.calibrate --reanalyze runs/_calibration/DSC07186/calibration.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from . import config
from .tools_aesthetic import aesthetic_score, metric_info
from .tools_image import adjust
from .tools_stats import stats

#: 以步长为单位的采样点
LEVELS = (-3, -2, -1, 0, 1, 2, 3)

#: 建议阈值相对"单步最大影响"的倍数
SOFT_FACTOR = 1.5
HARD_FACTOR = 3.0

#: 单步影响低于此值就认为审美分对本组参数不敏感
INSENSITIVE_STEP = 0.02


# ================================================================ 扫描


def sweep_param(image: Path, name: str, out_dir: Path) -> list[dict]:
    """对单个参数做多档扫描，返回每一档的测量结果。"""
    spec = config.PARAMS[name]
    lo, hi = spec["range"]
    step = spec["step"]

    rows = []
    for mult in LEVELS:
        value = max(lo, min(hi, mult * step))
        params = dict(config.ZERO_PARAMS)
        params[name] = value

        path = adjust(image, params, out_dir / f"{name}_{mult:+d}.jpg")
        rows.append(
            {
                "mult": mult,
                "value": value,
                "stats": stats(path),
                "aes": aesthetic_score(path),
            }
        )
    return rows


def central_gain(rows: list[dict], key: str) -> float:
    """中心差分估出的"每前进一个步长"的典型影响。

    ``(f(+1步) - f(-1步)) / 2``

    ``key`` 可以是统计量名，也可以是 ``"aes"``。

    注意：只适用于单调的效应。像 ``color_temp`` 对 ``clip_hi`` 的影响是 U 形的
    （加暖与加冷都推高截断），中心差分会把两侧抵消掉，给出误导性的小数字。
    这种量要用 :func:`max_increase`。
    """

    def value_of(row: dict) -> float:
        return row["aes"] if key == "aes" else row["stats"][key]

    by_mult = {r["mult"]: value_of(r) for r in rows}
    if 1 not in by_mult or -1 not in by_mult:
        return 0.0
    return (by_mult[1] - by_mult[-1]) / 2.0


def max_increase(rows: list[dict], key: str) -> float:
    """扫描范围内相对 0 档的最大增幅（可为负，表示全程都在降低）。

    护栏类指标（``clip_hi`` / ``clip_lo``）应该用这个而不是中心差分。
    """

    def value_of(row: dict) -> float:
        return row["stats"][key]

    zero = next((value_of(r) for r in rows if r["mult"] == 0), None)
    if zero is None:
        return 0.0
    return max(value_of(r) - zero for r in rows)


# ================================================================ 分析


def analyze(results: dict[str, list[dict]], baseline: dict) -> dict:
    """打印耦合矩阵与灵敏度，返回建议值。"""
    base_stats = baseline["stats"]
    base_aes = baseline["aes"]

    print(f"基准（原图）：审美分 {base_aes:.4f}")
    print(
        "              luma {luma:.4f}  contrast {contrast:.4f}  "
        "saturation {saturation:.4f}  warmth {warmth:+.4f}".format(**base_stats)
    )
    print()

    # ---------------------------------------------------------- 1. 耦合实测
    all_stats = config.STAT_NAMES + config.GUARD_STATS
    print("=" * 100)
    print("一、参数耦合实测：每前进一个步长，各统计量的变化量")
    print("=" * 100)
    header = f"{'参数':<12}" + "".join(f"{s:>11}" for s in all_stats) + f"{'审美分':>10}"
    print(header)
    print("-" * len(header))

    gains: dict[str, dict[str, float]] = {}
    for name in config.PARAM_NAMES:
        rows = results[name]
        g = {s: central_gain(rows, s) for s in all_stats}
        g["aes"] = central_gain(rows, "aes")
        gains[name] = g
        line = f"{name:<12}" + "".join(f"{g[s]:>+11.4f}" for s in all_stats)
        line += f"{g['aes']:>+10.4f}"
        print(line)

    print()
    print("高光截断的代价（全范围内 clip_hi 相对 0 档的最大增幅，越大越容易撞护栏）：")
    clip_cost = {n: max_increase(results[n], "clip_hi") for n in config.PARAM_NAMES}
    for name in sorted(config.PARAM_NAMES, key=lambda n: -clip_cost[n]):
        cost = clip_cost[name]
        verdict = (
            "代价很大" if cost > 0.1 else "代价中等" if cost > 0.03 else "代价很小"
        )
        print(f"  {name:<12} clip_hi 最大增幅 {cost:+.4f}    {verdict}")

    if base_stats["clip_hi"] > 0.01:
        worst = max(clip_cost, key=clip_cost.get)
        per = clip_cost[worst]
        print()
        print(
            f"  原图 clip_hi = {base_stats['clip_hi']:.4f}"
            f"（{base_stats['clip_hi']*100:.1f}% 的像素至少有一个通道打顶），"
            f"护栏只允许再增加 {config.CLIP_HI_DELTA:.4f}。"
        )
        if per > 0:
            print(
                f"  而 {worst} 在整个范围内的增幅可达 {per:+.4f}"
                f" → 有效余量很小（约 {config.CLIP_HI_DELTA / per:.2f} 个全范围步长）。"
            )
        print(
            f"  → 想提高 luma 应优先用 gamma 负值"
            f"（gamma 的 clip_hi 最大增幅 {clip_cost['gamma']:+.4f}），而不是 brightness。"
        )

    print()
    print("各参数实际影响最大的统计量（用于核对设计文档 4.4）：")
    for name in config.PARAM_NAMES:
        g = gains[name]
        main = max(config.STAT_NAMES, key=lambda s: abs(g[s]))
        declared = config.PRIMARY_EFFECT[name]
        mark = "符合" if main == declared else f"与声明不符（声明为 {declared}）"
        print(f"  {name:<12} 实际 -> {main:<11} ({g[main]:+.4f}/步)   {mark}")
    print()
    print(f"  contrast 对 saturation 的影响 {gains['contrast']['saturation']:+.4f}"
          f"  vs 对 contrast 本身 {gains['contrast']['contrast']:+.4f}"
          f"  -> {'降饱和更明显' if abs(gains['contrast']['saturation']) > abs(gains['contrast']['contrast']) else '以对比度为主'}")
    print(f"  color_temp 对 saturation 的影响 {gains['color_temp']['saturation']:+.4f}"
          "  （随题材差异较大，不要当成稳定规律）")

    # ------------------------------------------------------ 2. 审美分灵敏度
    print()
    print("=" * 84)
    print("二、审美分灵敏度")
    print("=" * 84)
    print(f"{'参数':<12}{'最小值':>10}{'最大值':>10}{'全范围极差':>12}"
          f"{'一步影响':>10}{'标准差':>10}")
    print("-" * 64)

    spans = {}
    step_gains = {}
    for name in config.PARAM_NAMES:
        aes_vals = [r["aes"] for r in results[name]]
        span = max(aes_vals) - min(aes_vals)
        spans[name] = span
        step_gains[name] = gains[name]["aes"]
        print(
            f"{name:<12}{min(aes_vals):>10.4f}{max(aes_vals):>10.4f}{span:>12.4f}"
            f"{step_gains[name]:>+10.4f}{statistics.pstdev(aes_vals):>10.4f}"
        )

    max_step = max(abs(v) for v in step_gains.values())
    max_span = max(spans.values())
    worst = max(step_gains, key=lambda k: abs(step_gains[k]))

    print()
    print(f"单参数一个步长能造成的审美分最大变化：{max_step:.4f}（来自 {worst}）")
    print(f"单参数走完全范围（±3 步）的最大极差：  {max_span:.4f}")

    d_soft = round(max_step * SOFT_FACTOR, 2)
    d_hard = round(max_step * HARD_FACTOR, 2)
    sensitive = max_step >= INSENSITIVE_STEP

    print()
    if not sensitive:
        print(f"[注意] 审美分对参数变化不敏感（单步影响 < {INSENSITIVE_STEP}）。")
        print("  结论：它只能当护栏，不能当参考信号。设计文档 3.3 需据此调整。")
    else:
        print("建议写进 config.py：")
        print(f"  D_SOFT        = {d_soft}    （{SOFT_FACTOR} 倍单步影响 → 比任何单个合法步长的影响都大，值得警告）")
        print(f"  D_HARD        = {d_hard}    （{HARD_FACTOR} 倍单步影响 → 约三个步长的累积影响，该回滚了）")
        print(f"  AES_NOISE_BAND = {d_soft}    （会注入提示词，告诉 LLM 多大波动没有意义）")
    print()
    print("说明：这三个数是基于单张照片单参数扫描的启发式建议。")
    print("      多参数叠加、不同题材会给出不同结论；建议多跑几张再定。")
    print()

    return {
        "max_step_gain": max_step,
        "worst_param": worst,
        "max_full_span": max_span,
        "suggested_D_SOFT": d_soft,
        "suggested_D_HARD": d_hard,
        "sensitive_enough": sensitive,
    }


# ================================================================ 入口


def calibrate(image: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"设备：{config.DEVICE}")
    print(f"审美工具：{config.AESTHETIC_METRIC}  {metric_info()}")
    print(f"测试图：{image}")
    print(f"输出目录：{out_dir}")
    print()

    base_stats = stats(image)
    base_aes = aesthetic_score(image)

    results: dict[str, list[dict]] = {}
    for name in config.PARAM_NAMES:
        print(f"  扫描 {name} ...", flush=True)
        results[name] = sweep_param(image, name, out_dir)
    print()

    baseline = {"stats": base_stats, "aes": base_aes}
    suggestion = analyze(results, baseline)

    payload = {
        "image": str(image),
        "metric": metric_info(),
        "baseline": baseline,
        "gains_per_step": {k: v for k, v in _gains(results).items()},
        "aes_sensitivity": suggestion,
        "sweeps": results,
    }
    out_file = out_dir / "calibration.json"
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完整数据已写入：{out_file}")
    print(f"（{len(config.PARAM_NAMES) * len(LEVELS)} 张标定变体留在 {out_dir}，确认无误后可自行删除）")

    return payload


def _gains(results: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    all_stats = config.STAT_NAMES + config.GUARD_STATS
    out = {}
    for name, rows in results.items():
        g = {s: central_gain(rows, s) for s in all_stats}
        g["aes"] = central_gain(rows, "aes")
        out[name] = g
    return out


def reanalyze(json_path: Path) -> dict:
    """只重跑分析，不重新扫描。改了建议算法或想换倍数时用。"""
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    print(f"从 {json_path} 读取，测试图：{payload['image']}")
    print(f"审美工具：{payload.get('metric')}")
    print()
    suggestion = analyze(payload["sweeps"], payload["baseline"])

    payload["gains_per_step"] = _gains(payload["sweeps"])
    payload["aes_sensitivity"] = suggestion
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已更新：{json_path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mytone.calibrate",
        description="审美分灵敏度标定 + 参数耦合实测",
    )
    parser.add_argument("-i", "--image", help="测试图路径（建议用自己的 coser 人像）")
    parser.add_argument("-o", "--out", help="输出目录，默认 runs/_calibration/<图名>")
    parser.add_argument("--reanalyze", help="只重新分析已有的 calibration.json，不重新扫描")
    args = parser.parse_args(argv)

    if args.reanalyze:
        path = Path(args.reanalyze)
        if not path.exists():
            print(f"[错误] 找不到 {path}")
            return 1
        reanalyze(path)
        return 0

    if not args.image:
        parser.error("需要 --image，或使用 --reanalyze")
    image = Path(args.image)
    if not image.exists():
        print(f"[错误] 找不到测试图：{image}")
        return 1

    out_dir = Path(args.out) if args.out else Path("runs") / "_calibration" / image.stem
    calibrate(image, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
