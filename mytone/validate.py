"""校验与裁剪：LLM 输出的合法性检查，以及参数的累加与边界裁剪。

见设计文档 7.1 / 7.3 / 9.4。

本模块只做校验与裁剪，不做任何优化决策。
校验的原则是：客观错误 → 打回重试；数值越界 → 直接裁剪（不重新请求）。
"""

from __future__ import annotations

from typing import Any

from . import config

FINISH_STATUSES = ("goal_met", "converged", "best_effort")


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# ---------------------------------------------------------------- 目标校验


def validate_target(reply: dict[str, Any]) -> tuple[bool, str, dict[str, float]]:
    """校验调用一的输出（设计文档 7.1）。

    客观错误（缺字段、自造指标名、空 target）→ 打回；
    数值越界 → 裁剪并记录，不重新请求。

    Returns:
        ``(是否通过, 问题描述, 规范化后的 target)``
    """
    for key in ("interpretation", "can_do", "cannot_do", "target", "reason"):
        if key not in reply:
            return False, f"缺少字段 {key!r}", {}

    if not isinstance(reply["interpretation"], str) or not reply["interpretation"].strip():
        return False, "interpretation 必须是非空字符串", {}
    for key in ("can_do", "cannot_do"):
        if not isinstance(reply[key], list) or not all(isinstance(x, str) for x in reply[key]):
            return False, f"{key} 必须是字符串数组", {}
    if not isinstance(reply["target"], dict) or not reply["target"]:
        return False, "target 必须是非空对象", {}

    clean: dict[str, float] = {}
    for name, value in reply["target"].items():
        if name not in config.STAT_RANGE:
            return False, (
                f"target 中出现未知统计量 {name!r}。"
                f"允许的统计量只有：{', '.join(config.STAT_NAMES)}"
            ), {}
        if not _is_number(value):
            return False, f"target[{name!r}] 必须是数字", {}
        lo, hi = config.STAT_RANGE[name]
        # 数值越界直接裁剪，不重新请求（设计文档 7.1）
        clean[name] = float(min(max(float(value), lo), hi))

    if len(clean) > 4:
        return False, f"target 维度过多（{len(clean)} 个），最多 4 个", {}

    return True, "", clean


# ---------------------------------------------------------------- 动作校验


def validate_action(
    reply: dict[str, Any],
    tool_names: set[str],
) -> tuple[bool, str, dict[str, Any]]:
    """校验调用二的输出（设计文档 7.3）。

    Returns:
        ``(是否通过, 问题描述, 原样动作)``
    """
    keys = [k for k in ("tool", "adjust", "revise_target", "finish") if k in reply]
    if len(keys) != 1:
        return False, (
            "顶层必须且只能包含 tool / adjust / revise_target / finish 四者之一，"
            f"实际收到：{sorted(reply.keys())}"
        ), {}
    kind = keys[0]

    if kind == "revise_target":
        new = reply["revise_target"]
        if not isinstance(new, dict) or not new:
            return False, "revise_target 必须是非空对象", {}
        if "reason" not in reply or not str(reply["reason"]).strip():
            return False, "revise_target 必须带 reason", {}
        for name, value in new.items():
            if name not in config.STAT_RANGE:
                return False, (
                    f"revise_target 中出现未知统计量 {name!r}。"
                    f"允许的统计量只有：{', '.join(config.STAT_NAMES)}"
                ), {}
            if not _is_number(value):
                return False, f"revise_target[{name!r}] 必须是数字", {}
        return True, "", reply

    if kind == "adjust":
        delta = reply["adjust"]
        if not isinstance(delta, dict) or not delta:
            return False, "adjust 必须是非空对象", {}
        if "reason" not in reply or not str(reply["reason"]).strip():
            return False, "adjust 必须带 reason", {}
        for name, value in delta.items():
            if name not in config.PARAMS:
                return False, (
                    f"adjust 中出现未知参数 {name!r}。"
                    f"允许的参数只有：{', '.join(config.PARAM_NAMES)}"
                ), {}
            if not _is_number(value):
                return False, f"adjust[{name!r}] 必须是数字", {}
        if len(delta) > config.MAX_PARAMS_PER_ROUND:
            return False, (
                f"单轮最多调整 {config.MAX_PARAMS_PER_ROUND} 个参数，"
                f"实际请求了 {len(delta)} 个"
            ), {}
        return True, "", reply

    if kind == "finish":
        status = reply["finish"]
        if status not in FINISH_STATUSES:
            return False, f"finish 必须是 {'/'.join(FINISH_STATUSES)} 之一", {}
        if "deliver_round" not in reply:
            return False, "finish 必须带 deliver_round", {}
        if not isinstance(reply["deliver_round"], int) or isinstance(reply["deliver_round"], bool):
            return False, "deliver_round 必须是整数轮次号", {}
        if "reason" not in reply or not str(reply["reason"]).strip():
            return False, "finish 必须带 reason", {}
        return True, "", reply

    # kind == "tool"
    name = reply["tool"]
    if name not in tool_names:
        return False, (
            f"未知工具 {name!r}。当前可用工具：{', '.join(sorted(tool_names)) or '（无）'}"
        ), {}
    if "reason" not in reply or not str(reply["reason"]).strip():
        return False, "tool 调用必须带 reason", {}

    if name == "aesthetic":
        target = reply.get("target", "current")
        ok_target = target == "current" or (isinstance(target, int) and not isinstance(target, bool))
        if not ok_target:
            return False, "aesthetic 的 target 必须是 \"current\" 或整数轮次号", {}
    elif name == "region":
        if reply.get("region") not in config.REGIONS:
            return False, f"region 必须是 {'/'.join(config.REGIONS)} 之一", {}

    return True, "", reply


# ---------------------------------------------------------------- 参数裁剪


def clamp_params(
    params: dict[str, float],
    delta: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """累加增量并裁剪到合法区间（设计文档 4.1 / 6）。

    两步裁剪，顺序不可颠倒：

    1. 单轮步长上限：``|delta| <= step``
    2. 累计取值区间：结果裁剪到 ``range``

    返回的 ``effective`` 是裁剪后真实生效的增量。它必须回喂给 LLM，
    否则 LLM 会以为自己请求的动作已完全生效，下一轮的判断就建立在错误前提上。

    Returns:
        ``(新的累计参数, 真实生效的增量)``
    """
    new_params = dict(params)
    effective: dict[str, float] = {}

    for name, raw in delta.items():
        spec = config.PARAMS[name]
        lo, hi = spec["range"]
        step = spec["step"]

        want = float(raw)
        limited = max(-step, min(step, want))

        before = new_params[name]
        after = max(lo, min(hi, before + limited))

        effective[name] = after - before
        new_params[name] = after

    return new_params, effective
