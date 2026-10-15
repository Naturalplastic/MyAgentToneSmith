"""LLM 调用契约：两个 system prompt、上下文组装、DeepSeek API 调用与重试。

见设计文档第 7 节。

两个调用点：
  调用一  ``interpret``  解读指令 + 声明能力边界 + 制定目标
  调用二  ``decide``     每轮决策（adjust / finish / tool）

设计原则（设计文档 1.4）：harness 负责一切算术，LLM 只负责判断。
因此喂给 LLM 的所有数值都由 harness 算好，LLM 不需要做任何计算。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import requests

from . import config
from . import validate as V
from .state import TaskState


class LLMError(RuntimeError):
    """LLM 调用或输出解析失败（重试耗尽）。"""


# ================================================================ prompt 模板

SYSTEM_INTERPRET = """\
你是照片调色助手的指令解读模块。你不看图片，只能读数字。

输入会给你：
1. 用户的原始指令（可能是简短风格词，也可能是一段话，可能引用某个圈层的惯例做法）
2. 原图的图像统计量
3. 原图的审美分
4. 可选统计量的清单（名称、含义、合理范围）
5. 你可用的五个调整参数，以及它们的能力范围

你的任务分三步：

第一步，解读指令。用一句话复述你如何理解这条指令想要的效果。

第二步，判断能力边界。五个参数都是全局参数，只能整体调整亮度、对比度、
饱和度、色温和中间调。凡是指令中需要局部处理、蒙版、人脸皮肤处理、
区域分离的操作，你都无法完成，必须如实列出。
不要为了让结果好看而声称自己能做到。

第三步，制定目标统计量。
目标必须在你可用的参数范围内确实可以达到。判断方法举例：
- brightness 取值 -0.6 ~ +0.6，即最多把亮度乘 1.6。若原图 luma 是 0.24，
  仅靠 brightness 最多到 0.38；把目标定成 0.60 是达不到的。
- gamma 取负值也能提高中间调亮度，可与 brightness 配合，但会连带推高对比度。
- color_temp 的单位是千分之二，有效调整幅度是 ±100 ~ ±400。
请在 reason 里简要说明你的目标为什么是可以达到的。

只输出一个 JSON：
{
  "interpretation": "一句话复述你的理解",
  "can_do": ["指令中你能做到的部分，逐条列出"],
  "cannot_do": ["指令中你做不到的部分，逐条列出；没有则给空数组"],
  "target": {"统计量名": 目标数值, ...},
  "reason": "为什么选这些维度、为什么定这些数值"
}

规则：
- 统计量名必须来自清单，不得自造
- 目标值必须在清单给出的合理范围内
- target 里只放与该指令真正相关的维度，数量 1 到 4 个
- cannot_do 必须诚实
- reason 必须引用具体数值
- 不要输出 JSON 以外的任何内容
"""

SYSTEM_DECIDE_TEMPLATE = """\
你是照片调色助手的调参决策模块。你不看图片，只能读数字。

每次会给你：
1. 用户的原始指令（原文）
2. 你对这条指令的解读，以及你已经声明的能力边界
3. 目标统计量及其设定依据
4. 当前五个参数的累计值
5. 当前图片的统计量
6. 误差向量（每维的 当前值减目标值）与平均绝对误差
7. 图像护栏状态
8. 审美分基线与已知的审美分
9. 历次轮次的参数、误差、审美分汇总
10. 最近三轮的详细轨迹
11. 最近的工具调用结果
12. 轮次进度、剩余轮次、剩余工具次数

你的任务：决定下一步。只输出一个 JSON，四选一：

动作类（消耗轮次）：
  {{"adjust": {{"参数名": 增量, ...}}, "reason": "依据"}}
  {{"finish": "goal_met" 或 "converged" 或 "best_effort",
   "deliver_round": 轮次号, "reason": "依据"}}
  {{"revise_target": {{"统计量名": 新目标值, ...}}, "reason": "依据"}}

  关于 revise_target：当你发现自己设定的目标在参数能力范围内确实达不到时
  （典型信号：反复撞同一面护栏、某个误差项怎么调都降不下去），
  用它修正目标，而不是一直撞墙。只覆盖你给出的维度，其余维度保持原值。
  修正后所有历史轮次的误差会按新目标重算。
  它会消耗一次轮次预算，且最多允许 {rev_limit} 次 ——
  不要用它把目标改成"已经达到的值"来假装收敛，必须在 reason 里
  说明原目标为什么不可达。

工具类（不消耗轮次，消耗工具次数）：
{tool_examples}

参数含义与影响：
{coupling_table}

必须注意的耦合：
{coupling_notes}

{tool_docs}
关于交付：
- finish 时必须给出 deliver_round，指定你希望交付哪一轮的结果
- 你可以选择任何通过护栏的轮次，不一定是最后一轮，也不一定是误差最小的那一轮
- 如果"误差最小"和"观感最好"不是同一轮，由你判断并做出选择，并在 reason 里说明理由
- 交付前必须已经对该轮调用过 aesthetic（若 aesthetic 在可用工具列表中），否则 finish 会被拒绝

规则：
- 每次只动 1 到 2 个参数，增量不得超过步长上限，超过会被裁剪
- 参数被裁剪时你请求的增量不会完全生效，给你的是裁剪后的真实生效值，以它为准则
- 你的调整必须始终服务于用户原始指令。如果统计量已达标但你认为结果偏离了指令本意，
  应当在 reason 里说明
- reason 必须引用具体数值
- 不要输出 JSON 以外的任何内容
"""

_AESTHETIC_DOC = """\
工具：aesthetic
- 返回 1 到 10 的分数，越大越好
- 来自 NIMA 模型，训练数据是 AVA（多人对真实摄影作品打分的公开数据集）
- 衡量"这张照片作为一张照片的整体观感"，主要由构图、主体、光线、清晰度决定，
  对纯色彩调整的敏感度较低
- 因此：正负 {noise_band:g} 以内的波动通常没有意义，不要据此改变判断
- 它只告诉你整体观感是否变差，不能告诉你该调哪个参数
- 基线是原图的分数；只有明显低于基线时这个信息才有价值
- 如果你已经连续调整 3 轮以上，建议至少调用一次它确认画面没有在变差
- 调用耗时约 {latency}，但会占用工具调用次数，请在有必要时调用
"""

_REGION_DOC = """\
工具：region
- 返回画面某个区域（上/中/下/中心）的 luma、contrast、saturation、warmth
- 用途：判断你的调整主要落在哪里，例如天空、背景还是人物
- 全局统计量看不出改动集中在哪个区域，这是补充信息
"""


def _build_tool_examples(tools: set[str]) -> str:
    lines = []
    if "aesthetic" in tools:
        lines.append('  {"tool": "aesthetic", "target": "current", "reason": "依据"}')
        lines.append('  {"tool": "aesthetic", "target": 轮次号, "reason": "依据"}')
    if "region" in tools:
        lines.append(
            '  {"tool": "region", "region": "top|middle|bottom|center", "reason": "依据"}'
        )
    return "\n".join(lines) if lines else "  （本次运行没有可用工具，只能输出动作类）"


def _build_tool_docs(tools: set[str]) -> str:
    parts = []
    if "aesthetic" in tools:
        parts.append(
            _AESTHETIC_DOC.format(
                noise_band=config.AES_NOISE_BAND,
                latency=config.AES_LATENCY,
            )
        )
    if "region" in tools:
        parts.append(_REGION_DOC)
    return "\n".join(parts)


def build_system_decide(aesthetic_mode: str | None = None) -> str:
    """按当前模式生成调用二的 system prompt（提示词里的数字从 config 注入）。"""
    tools = config.available_tools(aesthetic_mode)
    return SYSTEM_DECIDE_TEMPLATE.format(
        tool_examples=_build_tool_examples(tools),
        coupling_table=config.COUPLING_TABLE,
        coupling_notes=config.COUPLING_NOTES,
        tool_docs=_build_tool_docs(tools),
        rev_limit=config.TARGET_REVISION_LIMIT,
    )


# ================================================================ 上下文组装


def _fmt_stats(stats: dict[str, float]) -> str:
    return ", ".join(f"{k} {v:+.4f}" for k, v in stats.items())


def _fmt_params(params: dict[str, float]) -> str:
    return ", ".join(f"{k} {v:+.2f}" for k, v in params.items())


def _fmt_signed(v: float) -> str:
    return f"{v:+.4f}"


def build_interpret_input(state: TaskState) -> str:
    """调用一的 user 消息。

    必须把护栏余量也给它：调一决定的目标会贯穿整次运行，
    而实测中最主要的约束不是参数范围、而是护栏余量。
    不告诉它余量，它就会定出"参数上可达、但调整过程中必然撞护栏"的目标，
    然后整场都在贴着护栏线打转、误差降不下去（实测见过多次）。
    """
    s = state.original_stats
    hi_cap, lo_cap, c_min = config.guard_caps(s)

    return (
        f"用户指令：{state.instruction}\n\n"
        f"原图统计量：luma {s['luma']:.4f}, contrast {s['contrast']:.4f}, "
        f"saturation {s['saturation']:.4f}, warmth {s['warmth']:+.4f}\n"
        f"            clip_hi {s['clip_hi']:.4f}, clip_lo {s['clip_lo']:.4f}\n"
        f"原图审美分：{state.aes_baseline:.2f}\n\n"
        f"可选统计量：\n{config.stat_help()}\n\n"
        f"可用参数：\n{config.param_help()}\n"
        f"          （全部为全局参数，没有局部处理、蒙版或人脸处理能力）\n\n"
        f"【硬约束：图像护栏】每一轮的结果都必须满足下面三条，否则该轮直接作废并回滚：\n"
        f"  clip_hi   原图 {s['clip_hi']:.4f}，不能超过 {hi_cap:.4f}"
        f"（余量只有 {hi_cap - s['clip_hi']:+.4f}）\n"
        f"  clip_lo   原图 {s['clip_lo']:.4f}，不能超过 {lo_cap:.4f}"
        f"（余量只有 {lo_cap - s['clip_lo']:+.4f}）\n"
        f"  contrast  原图 {s['contrast']:.4f}，不能低于 {c_min:.4f}\n\n"
        f"【各参数对护栏的代价】\n{config.COUPLING_TABLE}\n\n"
        f"【制定目标时必须同时满足两件事】\n"
        f"  1. 参数上可达 —— 例如 brightness 最多乘 1.6；\n"
        f"  2. 调整过程中不会撞上上面的护栏余量。\n"
        f"特别注意：如果原图 clip_hi 已经接近上限，那么提亮与增艳都会很快撞墙，\n"
        f"此时 luma / saturation 的目标应当定得保守一些，\n"
        f"并在 reason 里说明你考虑了护栏余量。"
    )


def _round_rows(state: TaskState) -> str:
    """历次轮次汇总表。

    带上 ``clip_hi`` / ``clip_lo`` 是有意为之：agent 看不到参数到截断的映射，
    只有看到截断量逐轮怎么变，它才能校准自己的步长。
    实测中 agent 反复落在护栏上限边缘（差 0.0001 量级），正是因为缺少这张表。
    """
    head = "  轮次  平均误差   审美分   clip_hi   clip_lo  护栏"
    rows = [head]
    for r in state.rounds:
        err = "—" if r.error is None else f"{r.error:.4f}"
        aes = "—" if r.aes is None else f"{r.aes:.2f}"
        guard = "通过" if r.guard_ok else f"非法({r.guard_msg.split('（')[0]})"
        rows.append(
            f"  {r.index:<5} {err:<9} {aes:<8} "
            f"{r.stats['clip_hi']:>8.4f} {r.stats['clip_lo']:>9.4f}  {guard}"
        )
    return "\n".join(rows)


def _recent_traces(state: TaskState, n: int = 3) -> str:
    """最近 n 轮的详细轨迹。"""
    recent = state.rounds[1:][-n:]  # 跳过第 0 轮（原图，没有动作）
    if not recent:
        return "  （无，这是第 1 轮）"

    blocks = []
    for r in recent:
        req = ", ".join(f"{k} {v:+.3f}" for k, v in r.requested_delta.items()) or "（无）"
        eff = ", ".join(f"{k} {v:+.3f}" for k, v in r.effective_delta.items()) or "（无）"
        prev_err = state.rounds[r.index - 1].error
        prev = "—" if prev_err is None else f"{prev_err:.4f}"
        cur = "—" if r.error is None else f"{r.error:.4f}"
        blocks.append(
            f"  第 {r.index} 轮：请求 {req}\n"
            f"           生效 {eff}"
            + ("（发生裁剪，以生效值为准）" if r.clipped else "（无裁剪）")
            + f"\n           误差 {prev} → {cur}"
        )
        if not r.guard_ok:
            blocks[-1] += f"\n           护栏未通过：{r.guard_msg}"
    return "\n".join(blocks)


def _recent_tools(state: TaskState, n: int = 4) -> str:
    if not state.tool_log:
        return "  （无）"
    lines = []
    for rec in state.tool_log[-n:]:
        call = rec["call"]
        if call.get("tool") == "aesthetic":
            tgt = call.get("target", "current")
            lines.append(f"  aesthetic(target={tgt}) → {rec['result']}")
        else:
            lines.append(f"  {call.get('tool')}({call.get('region', '')}) → {rec['result']}")
    return "\n".join(lines)


def _aesthetic_line(state: TaskState) -> str:
    measured = [(r.index, r) for r in state.rounds if r.aes is not None]
    if not measured:
        return f"基线 {state.aes_baseline:.2f} ｜ 尚无其他测量"
    parts = []
    for idx, r in measured:
        flag = ""
        if r.aes_delta is not None:
            if r.aes_delta <= -config.D_HARD:
                flag = "越过硬红线"
            elif r.aes_delta <= -config.D_SOFT:
                flag = "越过软警告阈值"
            else:
                flag = "正常范围"
        parts.append(f"第 {idx} 轮 {r.aes:.2f}（{r.aes_delta:+.2f}，{flag}）")
    return f"基线 {state.aes_baseline:.2f} ｜ 已测：" + "；".join(parts)


def _guard_stats(state: TaskState) -> str:
    """护栏触发统计。纯聚合，不做任何判断。"""
    counts: dict[str, int] = {}
    for r in state.rounds:
        if r.guard_ok:
            continue
        key = r.guard_msg.split("（")[0]
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return ""
    parts = [f"{k} ×{v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])]
    return "护栏触发统计：" + "，".join(parts)


def _guard_headroom(state: TaskState) -> str:
    """护栏余量。纯测量，不做任何判断。

    给出余量数字很关键：agent 需要知道"还能往上走多少"，
    光说"你可能做不到"它是不会改主意的。
    """
    s = state.current_round.stats
    hi_cap, lo_cap, c_min = config.guard_caps(state.original_stats)
    return (
        f"护栏余量：clip_hi {s['clip_hi']:.4f} / 上限 {hi_cap:.4f}"
        f"（还剩 {hi_cap - s['clip_hi']:+.4f}）"
        f" ｜ clip_lo {s['clip_lo']:.4f} / 上限 {lo_cap:.4f}"
        f"（还剩 {lo_cap - s['clip_lo']:+.4f}）"
        f" ｜ contrast {s['contrast']:.4f} / 下限 {c_min:.4f}"
        f"（还剩 {s['contrast'] - c_min:+.4f}）"
    )


def build_decide_context(state: TaskState) -> str:
    """调用二的 user 消息。"""
    cur = state.current_round
    tools_left = config.TOOL_BUDGET - state.tools_used

    ev_lines = ", ".join(f"{k} {_fmt_signed(v)}" for k, v in cur.error_vector.items())
    err = "—" if cur.error is None else f"{cur.error:.4f}"

    guard = "正常"
    if not cur.guard_ok:
        guard = f"本轮触发护栏：{cur.guard_msg}"

    blocks = [
        f"用户原始指令：{state.instruction}",
        f"你的解读：{state.interpretation}",
        "你声明的能力边界：",
        "  能做到：" + " / ".join(state.can_do),
        "  做不到：" + " / ".join(state.cannot_do or ["（无）"]),
        "",
        f"目标（依据：{state.target_reason}）"
        + (f"【已修正 {len(state.target_revisions)} 次】" if state.target_revisions else "")
        + "：",
        "  " + " ｜ ".join(f"{k} {v:.4f}" for k, v in state.target.items()),
        "",
        f"当前累计参数：{_fmt_params(cur.params)}",
        f"当前统计量：  {_fmt_stats(cur.stats)}",
        f"误差向量（= 当前值 − 目标值）：{ev_lines}",
        "  【读法】负数表示当前值低于目标，需要把这个统计量往大调；",
        "          正数表示当前值高于目标，需要往小调。",
        "          不要把用户指令的方向当成永远的方向 —— 指令的方向只决定目标值，",
        "          具体往哪调要看这里的误差正负。比如指令说降低饱和度，",
        "          但若当前饱和度已经低于目标，就应该把它调高回来。",
        f"平均误差：    {err}",
        f"护栏状态：    {guard}",
        _guard_headroom(state),
    ]

    gs = _guard_stats(state)
    if gs:
        blocks.append(gs)
        blocks.append(
            "  （若同一面护栏反复触发、而余量已经很小，说明当前目标可能不可达。"
            "此时可以改用对护栏代价更小的参数（例如用 gamma 负值而非 brightness 提亮），"
            "或者用 revise_target 修正目标）"
        )

    blocks += [
        f"审美分：{_aesthetic_line(state)}",
        "",
        "历次轮次汇总：",
        _round_rows(state),
        "",
        "最近三轮详细轨迹：",
        _recent_traces(state),
        "",
        "最近工具调用：",
        _recent_tools(state),
        "",
        f"进度：当前第 {len(state.rounds) - 1} 轮"
        f" ｜ 轮次预算已用 {state.rounds_used}/{config.ROUND_LIMIT}"
        f"（含目标修正 {len(state.target_revisions)}/{config.TARGET_REVISION_LIMIT}）"
        f" ｜ 剩余工具次数：{tools_left}",
    ]

    if state.pending_warning:
        blocks.append(f"提示：{state.pending_warning}")

    return "\n".join(blocks)


# ================================================================ JSON 解析


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json(text: str) -> dict[str, Any] | None:
    """从模型输出中提取 JSON 对象。容忍 markdown 围栏与前后废话。"""
    if not text:
        return None
    candidates = _FENCE_RE.findall(text) or [text]
    for cand in candidates:
        cand = cand.strip()
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start, end = cand.find("{"), cand.rfind("}")
        if start != -1 and end > start:
            try:
                obj = json.loads(cand[start : end + 1])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
    return None


# ================================================================ LLM 客户端


class LLMClient:
    """DeepSeek API 客户端（OpenAI 兼容接口）。

    负责调用、解析、校验、重试，并把每一次请求与响应原文写成 jsonl 日志。
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        log_path: str | Path | None = None,
    ) -> None:
        self.api_key = api_key or config.DEEPSEEK_API_KEY
        self.base_url = (base_url or config.DEEPSEEK_BASE_URL).rstrip("/")
        self.model = model or config.DEEPSEEK_MODEL
        self.log_path = Path(log_path) if log_path else None
        self._seq = 0

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ 底层调用

    def _log(self, record: dict[str, Any]) -> None:
        if not self.log_path:
            return
        self._seq += 1
        record["seq"] = self._seq
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _chat(self, messages: list[dict[str, str]], tag: str) -> str:
        if not self.api_key:
            raise LLMError(
                "未设置 DEEPSEEK_API_KEY。请先执行 export DEEPSEEK_API_KEY=sk-xxxx"
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": config.LLM_TEMPERATURE,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=config.LLM_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise LLMError(f"网络请求失败：{exc}") from exc

        if resp.status_code != 200:
            raise LLMError(f"API 返回 {resp.status_code}：{resp.text[:400]}")

        data = resp.json()
        text = data["choices"][0]["message"]["content"]

        self._log(
            {
                "tag": tag,
                "model": self.model,
                "request_messages": messages,
                "response_text": text,
                "usage": data.get("usage"),
            }
        )
        return text

    # ------------------------------------------------------------ 通用重试

    def _call_with_validation(
        self,
        messages: list[dict[str, str]],
        tag: str,
        normalizer,
    ) -> dict[str, Any]:
        """调用 → 解析 → 校验，不通过则把问题附回去重试。

        ``normalizer(reply) -> (ok, problem, final_reply)``
        """
        msgs = list(messages)
        last_problem = "未知"

        for attempt in range(config.LLM_RETRY_LIMIT + 1):
            text = self._chat(msgs, f"{tag}#{attempt + 1}")

            reply = parse_json(text)
            if reply is None:
                last_problem = "输出不是合法 JSON"
            else:
                ok, problem, final_reply = normalizer(reply)
                if ok:
                    return final_reply
                last_problem = problem

            msgs = msgs + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": (
                        f"你上一次的输出未通过校验：{last_problem}\n"
                        "请重新输出一个合法的 JSON，不要输出任何其他内容。"
                    ),
                },
            ]

        raise LLMError(f"[{tag}] 重试 {config.LLM_RETRY_LIMIT} 次后仍不合法：{last_problem}")

    # ------------------------------------------------------------ 两个调用点

    @staticmethod
    def _normalize_target(reply: dict[str, Any]):
        """目标校验：数值越界在这里被裁剪，返回规范化后的完整 reply。"""
        ok, problem, clean = V.validate_target(reply)
        if not ok:
            return False, problem, {}
        return True, "", {**reply, "target": clean}

    @staticmethod
    def _normalize_action(state: TaskState):
        tools = config.available_tools(state.aesthetic_mode)

        def _norm(reply: dict[str, Any]):
            ok, problem, _ = V.validate_action(reply, tools)
            return ok, problem, reply

        return _norm

    def interpret(self, state: TaskState) -> dict[str, Any]:
        """调用一：解读指令 + 声明能力边界 + 制定目标。"""
        messages = [
            {"role": "system", "content": SYSTEM_INTERPRET},
            {"role": "user", "content": build_interpret_input(state)},
        ]
        state.llm_call_count += 1
        return self._call_with_validation(messages, "interpret", self._normalize_target)

    def decide(self, state: TaskState) -> dict[str, Any]:
        """调用二：每轮决策。"""
        # 先组装上下文（此时读取 pending_warning），随后立即清空，
        # 见设计文档 8.2 的注释与 6 节"关键设计点"。
        context = build_decide_context(state)
        state.pending_warning = None

        messages = [
            {"role": "system", "content": build_system_decide(state.aesthetic_mode)},
            {"role": "user", "content": context},
        ]
        state.llm_call_count += 1
        return self._call_with_validation(
            messages, "decide", self._normalize_action(state)
        )
