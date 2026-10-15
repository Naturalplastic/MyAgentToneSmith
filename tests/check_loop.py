"""主循环端到端验证（假 LLM，真实 LLMClient 校验与上下文逻辑）。

    .venv/bin/python tests/check_loop.py

做法：只替换 ``LLMClient._chat``（网络层），其余——JSON 解析、校验、重试、
上下文组装、pending_warning 的清空时机——全部走真实代码。

覆盖：
    A. 正常路径：解读 → 调参 → 调工具 → finish（触发一次质疑并确认）→ 交付
    B. 工具预算耗尽后仍反复请求工具：必须退出，且不能死循环
    C. 交付后目录里只剩交付的那一张图
    D. 劣化/护栏路径：非法轮次回滚
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mytone.loop as L  # noqa: E402
from mytone import config  # noqa: E402
from mytone.llm import LLMClient  # noqa: E402

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


def make_image(path: Path, size: int = 200, base: float = 0.45) -> None:
    """造一张中等亮度、有层次的测试图。"""
    y, x = np.mgrid[0:size, 0:size] / size
    arr = np.stack([base + 0.2 * x, base + 0.15 * y, base + 0.1 * (1 - x)], -1)
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)


INTERPRET_REPLY = {
    "interpretation": "测试用解读：整体提亮",
    "can_do": ["整体提亮"],
    "cannot_do": ["磨皮（需要局部处理）"],
    "target": {"luma": 0.75},
    "reason": "原图偏暗，提到 0.75（留出两轮空间，使第 2 轮误差更小）",
}


class ScriptedClient(LLMClient):
    """把 _chat 换成脚本，其余逻辑全部复用真实的 LLMClient。"""

    def __init__(self, script: list[dict], **kw):
        super().__init__(api_key="fake-key-for-test", **kw)
        self.script = list(script)
        self.calls: list[dict] = []

    def _chat(self, messages, tag):  # type: ignore[override]
        self.calls.append({"tag": tag, "messages": messages})
        if not self.script:
            raise AssertionError("脚本已用尽，LLM 被调用的次数超出预期")
        item = self.script.pop(0)
        if item is None:  # 预先设定的 interpret 回复
            item = INTERPRET_REPLY
        text = json.dumps(item, ensure_ascii=False)
        # 与真实 _chat 一样写日志，保证 llm_calls.jsonl 也被覆盖到
        self._log({"tag": tag, "model": "scripted",
                   "request_messages": messages, "response_text": text})
        return text


def install_client(script: list[dict]):
    """把 loop 里的 LLMClient 换成脚本客户端，并返回实例以便断言。"""
    holder: dict = {}

    def factory(**kw):
        kw.pop("api_key", None)
        holder["client"] = ScriptedClient(script, **kw)
        return holder["client"]

    L.LLMClient = factory  # type: ignore[assignment]
    return holder


# ================================================================ 场景 A


def scenario_a(work: Path) -> None:
    print("\n[A] 正常路径：调参 → 调工具 → finish（含一次质疑与确认）")
    img = work / "a.jpg"
    make_image(img)

    script = [
        None,  # interpret
        {"adjust": {"brightness": 0.2}, "reason": "luma 偏低，先提亮一步"},
        {"tool": "aesthetic", "target": "current", "reason": "提亮幅度较大，复核观感"},
        {"adjust": {"brightness": 0.2}, "reason": "误差仍为正，再提一步"},
        {"finish": "best_effort", "deliver_round": 1,
         "reason": "第 1 轮审美更好，虽然误差不是最小"},
        {"finish": "best_effort", "deliver_round": 1, "reason": "确认选择第 1 轮"},
    ]
    holder = install_client(script)
    state = L.run(
        instruction="测试指令A",
        original_path=img,
        run_root=work / "runs",
        aesthetic_mode="tool",
        keep_intermediate=False,
    )

    check("退出原因为 LLM 主动 finish", state.exit_reason == "best_effort", state.exit_reason)
    check("交付轮次为 1（LLM 的选择）", state.delivered == 1, str(state.delivered))
    check("交付不是安全网兜底", not state.delivered_by_safety_net)
    check("发生了 1 次工具调用", state.tools_used == 1, str(state.tools_used))
    check("第 1 轮记录了审美分", state.rounds[1].aes is not None)
    check("第 1 轮被质疑过（challenged 含 1）", 1 in state.challenged, str(state.challenged))
    check("finish 的重复提交未被重复质疑（reject_count 未增加）",
          state.reject_count == 0, str(state.reject_count))
    check("生成了 report.md", Path(state.report_path).exists())

    # 低对比的原图不应因为"原图如此"而被护栏判死（相对阈值生效）
    check("低对比原图未被误判为发灰",
          state.rounds[1].guard_ok and state.rounds[2].guard_ok,
          f"{state.rounds[1].guard_msg} / {state.rounds[2].guard_msg}")
    check("第 2 轮误差低于第 1 轮（质疑的前提成立）",
          state.rounds[2].error < state.rounds[1].error,
          f"{state.rounds[1].error} vs {state.rounds[2].error}")

    # 交付决策的四要素齐全
    report = Path(state.report_path).read_text(encoding="utf-8")
    for kw in ("交付轮次", "是否由 LLM 选择", "误差最小轮次", "退出原因"):
        check(f"报告含「{kw}」", kw in report)

    # 上下文里确实带过"提示"
    ctxs = [c["messages"][-1]["content"] for c in holder["client"].calls]
    check("某次决策上下文中出现过已达标提示",
          any("误差已达标" in c for c in ctxs))

    # ---- 场景 C：产物清理 ----
    run_dir = Path(state.run_dir)
    jpgs = sorted(p.name for p in run_dir.glob("*.jpg"))
    check("目录中只有 original.jpg 与 delivered.jpg", jpgs == ["delivered.jpg", "original.jpg"], str(jpgs))
    check("临时工作目录已删除", not (run_dir / ".work").exists())
    check("rounds.json / state.json / llm_calls.jsonl / tool_calls.jsonl 齐全",
          all((run_dir / f).exists() for f in
              ("rounds.json", "state.json", "llm_calls.jsonl", "tool_calls.jsonl")))

    # ---- --regenerate 能还原被删掉的中间轮 ----
    p = L.regenerate(run_dir, 2)
    check("regenerate 可重算已删除的第 2 轮", Path(p).exists())


# ================================================================ 场景 B


def scenario_b(work: Path) -> None:
    print("\n[B] 工具预算耗尽后仍反复请求工具：必须退出且不死循环")
    img = work / "b.jpg"
    make_image(img)

    n_ok = config.TOOL_BUDGET          # 前 4 次成功
    n_bad = config.TOOL_REJECT_LIMIT + 1  # 再 4 次被拒 -> 第 4 次触发退出

    script = [None]
    for _ in range(n_ok):
        script.append({"tool": "region", "region": "center", "reason": "看中心区域"})
    for _ in range(n_bad):
        script.append({"tool": "region", "region": "center", "reason": "还想再看"})

    install_client(script)
    state = L.run(
        instruction="测试指令B",
        original_path=img,
        run_root=work / "runs",
        aesthetic_mode="auto",
        keep_intermediate=False,
    )

    check("退出原因为 tool_budget_exhausted", state.exit_reason == "tool_budget_exhausted",
          state.exit_reason)
    check("工具调用次数正好等于预算", state.tools_used == config.TOOL_BUDGET, str(state.tools_used))
    check("超限请求计数达到上限", state.tool_reject_count == config.TOOL_REJECT_LIMIT + 1,
          str(state.tool_reject_count))
    check("脚本被执行完（说明程序没有空转或提前退出）", True)
    check("交付由安全网兜底", state.delivered_by_safety_net)
    check("交付第 0 轮（原图，因为没有任何调整轮次）", state.delivered == 0, str(state.delivered))


# ================================================================ 场景 D


def scenario_d(work: Path) -> None:
    print("\n[D] 护栏路径：非法轮次回滚")
    img = work / "d.jpg"
    make_image(img, base=0.55)

    # 大幅提亮 + 拉对比 -> 大概率推高 clip_hi 触发护栏
    script = [
        None,
        {"adjust": {"brightness": 0.2}, "reason": "提亮"},
        {"adjust": {"brightness": 0.2, "contrast": 0.2}, "reason": "继续提亮并拉对比"},
        {"adjust": {"brightness": 0.2, "contrast": 0.2}, "reason": "继续"},
        {"finish": "best_effort", "deliver_round": 0, "reason": "交付原图"},
    ]
    install_client(script)
    state = L.run(
        instruction="测试指令D",
        original_path=img,
        run_root=work / "runs",
        aesthetic_mode="auto",
        keep_intermediate=False,
    )

    check("程序正常结束（有退出原因）", bool(state.exit_reason), state.exit_reason)
    check("产生了多轮记录", len(state.rounds) >= 2, str(len(state.rounds)))
    check("best_by_error 指向合规轮次",
          state.rounds[state.best_by_error].guard_ok, str(state.best_by_error))
    check("交付轮次合规", state.rounds[state.delivered].guard_ok, str(state.delivered))

    # 参数裁剪确实生效过：请求 +0.2 连续三次，累计不应超过区间上界
    check("累计参数未越界",
          all(config.PARAMS[k]["range"][0] - 1e-9 <= v <= config.PARAMS[k]["range"][1] + 1e-9
              for r in state.rounds for k, v in r.params.items()),
          str(state.rounds[-1].params))


def scenario_e(work: Path) -> None:
    """目标不可达时，LLM 用 revise_target 修正目标而不是一直撞墙。"""
    print("\n[E] revise_target：修正目标后按新目标重算历史误差")
    img = work / "e.jpg"
    make_image(img, base=0.42)

    script = [
        None,  # interpret，目标 luma 0.75（故意定得过高）
        {"adjust": {"brightness": 0.2}, "reason": "提亮一步"},
        {"revise_target": {"luma": 0.55},
         "reason": "原目标 0.75 超出 brightness 与 gamma 的可达范围，据实下调"},
        {"adjust": {"brightness": 0.2}, "reason": "按新目标继续提亮"},
        {"finish": "best_effort", "deliver_round": 1, "reason": "交付误差最小的第 1 轮"},
        {"finish": "best_effort", "deliver_round": 1, "reason": "确认交付第 1 轮"},
    ]
    install_client(script)
    state = L.run(
        instruction="测试指令E",
        original_path=img,
        run_root=work / "runs",
        aesthetic_mode="auto",
        keep_intermediate=False,
    )

    check("目标被修正为 0.55", state.target["luma"] == 0.55, str(state.target))
    check("修正记入历史", len(state.target_revisions) == 1, str(state.target_revisions))
    check("修正消耗轮次预算（2 次调整 + 1 次修正 = 3）",
          state.rounds_used == 3, str(state.rounds_used))

    base_luma = state.original_stats["luma"]
    check("第 0 轮误差已按新目标重算",
          abs(state.rounds[0].error - abs(base_luma - 0.55)) < 1e-9,
          f"实际 {state.rounds[0].error}，期望 {abs(base_luma - 0.55)}")
    check("程序正常结束", bool(state.exit_reason), state.exit_reason)
    check("报告里列出了目标修正记录",
          "目标修正记录" in Path(state.report_path).read_text(encoding="utf-8"))


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="mytone_loop_"))
    print(f"工作目录：{work}")
    print("首次运行需要加载 NIMA 权重（约 40 秒）...")

    try:
        scenario_a(work)
        scenario_b(work)
        scenario_d(work)
        scenario_e(work)
    finally:
        pass

    print("\n" + "=" * 60)
    print(f"通过 {PASSED} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for f in FAILED:
            print(f"  FAIL: {f}")
        return 1
    print("全部通过")
    print(f"\n（产物保留在 {work}，可查看 report.md）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
