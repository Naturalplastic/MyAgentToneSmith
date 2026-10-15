"""状态定义。

见设计文档第 6 节。

关键设计点（设计文档 6）：
- ``rounds[i].index == i`` 恒成立，因此可以用轮次号直接索引；
- ``requested_delta`` 与 ``effective_delta`` 必须分开记录；
- 第 0 轮（原图）也计算 ``error``；
- ``best_by_error`` 与 ``delivered`` 是两个不同的东西，后者由 LLM 决策。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import config


@dataclass
class Round:
    """一轮的完整记录。"""

    index: int  # 轮次，从 0 开始，第 0 轮是原图
    params: dict[str, float]  # 该轮的累计绝对参数
    requested_delta: dict[str, float]  # LLM 请求的增量（原样记录）
    effective_delta: dict[str, float]  # 裁剪后真实生效的增量
    stats: dict[str, float]  # 六个统计量
    error: float | None  # 平均绝对误差，第 0 轮在目标确定后补算
    error_vector: dict[str, float]  # 每维的 当前值 - 目标值
    guard_ok: bool  # 是否通过全部护栏（审美硬红线可在之后被回溯改写）
    guard_msg: str  # 未通过的原因
    aes: float | None = None  # 审美分，只在被测量过的轮次有值
    aes_delta: float | None = None  # 与基线的差
    llm_reason: str = ""  # 该轮 LLM 给出的依据原文
    image_path: str = ""  # 候选图路径

    @property
    def clipped(self) -> bool:
        """请求的增量是否被边界裁剪过。"""
        return self.requested_delta != self.effective_delta

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "params": self.params,
            "requested_delta": self.requested_delta,
            "effective_delta": self.effective_delta,
            "clipped": self.clipped,
            "stats": self.stats,
            "error": self.error,
            "error_vector": self.error_vector,
            "guard_ok": self.guard_ok,
            "guard_msg": self.guard_msg,
            "aes": self.aes,
            "aes_delta": self.aes_delta,
            "llm_reason": self.llm_reason,
            "image_path": self.image_path,
        }


@dataclass
class TaskState:
    """整次运行的状态。harness 维护，LLM 不可见其内部结构。"""

    instruction: str  # 用户原始指令，一字不改
    original_path: str
    run_dir: str  # 本次运行的产物目录
    tmp_dir: str  # 临时工作目录（中间候选图写在这里）

    # --- 调用一的产出 ---
    interpretation: str = ""
    can_do: list[str] = field(default_factory=list)
    cannot_do: list[str] = field(default_factory=list)
    target: dict[str, float] = field(default_factory=dict)
    target_reason: str = ""

    # --- 测量结果 ---
    original_stats: dict[str, float] = field(default_factory=dict)
    aes_baseline: float = 0.0

    # --- 迭代过程 ---
    params: dict[str, float] = field(default_factory=lambda: dict(config.ZERO_PARAMS))
    rounds: list[Round] = field(default_factory=list)
    #: LLM 修正目标的历史记录（每条含 round / new_target / reason）
    target_revisions: list[dict[str, Any]] = field(default_factory=list)

    #: 误差最小的合规轮次号，由 harness 维护，只作为安全参考和校验基准
    best_by_error: int = 0
    #: 当前参数实际对应的轮次号。正常情况下等于最后一轮；
    #: 触发护栏回滚后，它会指回被回滚到的那一轮。
    #: 没有这个字段时，上下文会把"最后一轮"（非法轮次）的参数当成当前参数告诉 LLM，
    #: 于是 LLM 在一个它并不在的位置上做决策 —— 首次实跑就因此出现了"被罚后过度回调"。
    current_index: int = 0
    #: LLM 最终指定的交付轮次；None 表示尚未指定
    delivered: int | None = None

    # --- 工具 ---
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    tools_used: int = 0
    tool_reject_count: int = 0

    # --- 控制 ---
    pending_warning: str | None = None  # 给下一轮的提示，组装上下文后立即清空
    aesthetic_reminded: bool = False  # 是否已就"未使用审美工具"提醒过一次
    reject_count: int = 0  # deliver_round 被判定"不合法"的次数
    challenged: set[int] = field(default_factory=set)  # 已被质疑过的交付轮次号
    llm_retry_failed: bool = False
    exit_reason: str = ""
    delivered_by_safety_net: bool = False  # 交付是否由安全网兜底
    finish_status: str = ""  # LLM 主动 finish 时给的状态：goal_met / converged / best_effort
    finish_reason: str = ""  # LLM 主动 finish 时给出的理由原文

    # --- 运行期附加信息 ---
    aesthetic_mode: str = config.AESTHETIC_MODE
    metric_info: dict[str, Any] = field(default_factory=dict)
    llm_call_count: int = 0
    report_path: str = ""

    # ------------------------------------------------------------ 便捷访问
    @property
    def current_round(self) -> Round:
        """当前参数实际对应的轮次。

        触发护栏回滚后它不再等于最后一轮 —— 见 ``current_index`` 的说明。
        """
        return self.rounds[self.current_index]

    @property
    def last_attempt(self) -> Round:
        """最后一次尝试的轮次（可能就是被回滚掉的那个非法轮次）。"""
        return self.rounds[-1]

    def round_by_index(self, n: int) -> Round:
        return self.rounds[n]

    @property
    def adjust_count(self) -> int:
        """新增的调整轮次数量，不含第 0 轮。"""
        return len(self.rounds) - 1

    @property
    def rounds_used(self) -> int:
        """已消耗的轮次预算 = 调整轮次 + 目标修正次数（不含第 0 轮）。

        目标修正也消耗轮次预算，否则 agent 可以靠不断改目标来假装收敛。
        """
        return self.adjust_count + len(self.target_revisions)

    @property
    def last_error(self) -> float | None:
        return self.rounds[-1].error

    def improvement(self) -> float | None:
        """相对上一轮的误差改善量，正数表示变好。"""
        if len(self.rounds) < 2:
            return None
        prev, cur = self.rounds[-2].error, self.rounds[-1].error
        if prev is None or cur is None:
            return None
        return prev - cur

    def is_goal_met(self) -> bool:
        """当前参数实际对应的那一轮的误差是否已进容差。"""
        err = self.current_round.error
        return err is not None and err < config.ERROR_TOL

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "interpretation": self.interpretation,
            "can_do": self.can_do,
            "cannot_do": self.cannot_do,
            "target": self.target,
            "target_reason": self.target_reason,
            "original_path": self.original_path,
            "original_stats": self.original_stats,
            "aes_baseline": self.aes_baseline,
            "params": self.params,
            "rounds": [r.to_dict() for r in self.rounds],
            "best_by_error": self.best_by_error,
            "delivered": self.delivered,
            "target_revisions": self.target_revisions,
            "delivered_by_safety_net": self.delivered_by_safety_net,
            "tool_log": self.tool_log,
            "tools_used": self.tools_used,
            "tool_reject_count": self.tool_reject_count,
            "reject_count": self.reject_count,
            "challenged": sorted(self.challenged),
            "exit_reason": self.exit_reason,
            "finish_status": self.finish_status,
            "finish_reason": self.finish_reason,
            "aesthetic_mode": self.aesthetic_mode,
            "metric_info": self.metric_info,
            "llm_call_count": self.llm_call_count,
        }
