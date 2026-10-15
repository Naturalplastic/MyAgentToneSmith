"""批处理模块的验证（假 LLM 驱动真实 run()）。

    .venv/bin/python tests/check_batch.py

覆盖：
    1. read_instruction：注释剥离、多行、空内容
    2. scan_tasks：同名 txt 配对、默认指令兜底、扩展名过滤、排序
    3. 缺指令时的行为（报错并列出）
    4. --dry-run 不产生任何产物
    5. 端到端跑通：产物结构、汇总表内容、交付后只留一张图
    6. 单张失败不中断整批（错误隔离）+ 退出码
    7. 覆盖已有产物时给出告警
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mytone.loop as L  # noqa: E402
from mytone import batch  # noqa: E402
from mytone.llm import LLMClient  # noqa: E402

PASSED = 0
FAILED: list[str] = []

#: 保存原始 run()，避免重复安装假 LLM 时把包装器套成递归
_PRISTINE_RUN = L.run


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED
    if cond:
        PASSED += 1
        print(f"  [OK]   {name}")
    else:
        FAILED.append(f"{name}  {detail}")
        print(f"  [FAIL] {name}  {detail}")


def make_image(path: Path, size: int = 120, base: float = 0.45) -> None:
    y, x = np.mgrid[0:size, 0:size] / size
    arr = np.stack([base + 0.3 * x, base + 0.2 * y, base + 0.15 * (1 - x)], -1)
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)


# ---------------------------------------------------------------- 假 LLM

INTERPRET = {
    "interpretation": "测试解读",
    "can_do": ["整体提亮"],
    "cannot_do": ["磨皮"],
    "target": {"luma": 0.55},
    "reason": "测试",
}

DECIDE_SCRIPT = [
    {"adjust": {"gamma": -0.12}, "reason": "提亮中间调"},
    {"adjust": {"gamma": -0.08}, "reason": "再提一点"},
    {"finish": "best_effort", "deliver_round": 1, "reason": "交付第 1 轮"},
    {"finish": "best_effort", "deliver_round": 1, "reason": "确认"},
]

#: 当前正在跑的图片名，供 fake 按图注入失败
_CURRENT = {"stem": ""}


def install_fake_llm(fail_for: set[str] | None = None) -> None:
    """把 loop 里的 LLMClient 换成脚本客户端（只替换网络层）。"""

    class Fake(LLMClient):
        def __init__(self, **kw):
            kw.pop("api_key", None)
            super().__init__(api_key="fake", **kw)
            self.script = list(DECIDE_SCRIPT)

        def _chat(self, messages, tag):  # type: ignore[override]
            if (
                fail_for
                and _CURRENT["stem"] in fail_for
                and tag.startswith("interpret")
            ):
                raise RuntimeError(f"模拟失败：{_CURRENT['stem']}")
            payload = INTERPRET if tag.startswith("interpret") else self.script.pop(0)
            text = json.dumps(payload, ensure_ascii=False)
            self._log({"tag": tag, "model": "scripted",
                       "request_messages": messages, "response_text": text})
            return text

    def factory(**kw):
        kw.pop("api_key", None)
        return Fake(**kw)

    def tracking_run(instruction, original_path, **kw):
        _CURRENT["stem"] = Path(original_path).stem
        return _PRISTINE_RUN(instruction, original_path, **kw)

    L.LLMClient = factory  # type: ignore[assignment]
    L.run = tracking_run  # type: ignore[assignment]
    batch.run = tracking_run  # type: ignore[assignment]


def run_batch_quietly(*args, **kwargs) -> int:
    """跑一次批处理并吞掉它的 stdout（测试只关心返回值与产物）。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = batch.run_batch(*args, **kwargs)
    return rc


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="mytone_batch_"))
    print(f"工作目录：{work}\n")

    # ================================================ 1. read_instruction
    print("[1] read_instruction")
    p = work / "a.txt"
    p.write_text("# 注释行\n\n帮我进行coser界通用的基本照片调光\n第二行\n", encoding="utf-8")
    got = batch.read_instruction(p)
    check("剥离 # 注释与空行", "#" not in got, repr(got))
    check("保留多行内容", got.count("\n") == 1, repr(got))
    check("内容正确", got.startswith("帮我进行coser界通用的基本照片调光"), repr(got))

    p.write_text("# 只有注释\n\n", encoding="utf-8")
    check("纯注释 → 空字符串", batch.read_instruction(p) == "", repr(batch.read_instruction(p)))

    # ================================================ 2. scan_tasks
    print("\n[2] scan_tasks")
    in_dir = work / "in"
    in_dir.mkdir()
    make_image(in_dir / "paired.jpg")
    make_image(in_dir / "fallback.png")
    make_image(in_dir / "noinst.jpg")
    (in_dir / "paired.txt").write_text("# 注释\n配对的指令\n", encoding="utf-8")
    (in_dir / "notes.md").write_text("不是图片", encoding="utf-8")
    (in_dir / "orphan.txt").write_text("没有对应图片", encoding="utf-8")

    tasks = batch.scan_tasks(in_dir, "默认指令")
    names = [t["image"].name for t in tasks]
    check("只收图片、忽略 md 与孤立 txt",
          names == ["fallback.png", "noinst.jpg", "paired.jpg"], str(names))
    check("按文件名排序", names == sorted(names))

    by_name = {t["image"].name: t for t in tasks}
    check("有同名 txt → 用 txt", by_name["paired.jpg"]["instruction"] == "配对的指令",
          by_name["paired.jpg"]["instruction"])
    check("txt 里的注释没有进指令", "#" not in by_name["paired.jpg"]["instruction"])
    check("无同名 txt → 用默认指令", by_name["noinst.jpg"]["instruction"] == "默认指令",
          by_name["noinst.jpg"]["instruction"])
    check("指令来源记录了 txt 名", "paired.txt" in by_name["paired.jpg"]["source"],
          by_name["paired.jpg"]["source"])

    (in_dir / "fallback.txt").write_text("# 全是注释\n", encoding="utf-8")
    fb = next(t for t in batch.scan_tasks(in_dir, "默认指令")
              if t["image"].name == "fallback.png")
    check("空 txt → 回退默认指令", fb["instruction"] == "默认指令", fb["instruction"])
    check("来源说明了回退", "回退" in fb["source"], fb["source"])
    (in_dir / "fallback.txt").unlink()

    # ================================================ 3/4. 缺指令 & dry-run
    print("\n[3] 缺指令 / dry-run")
    check("有图但都没指令 → 退出码 1",
          batch.main(["-d", str(in_dir), "--dry-run"]) == 1)
    check("给了 -p → dry-run 退出码 0",
          batch.main(["-d", str(in_dir), "-p", "统一指令", "--dry-run"]) == 0)

    out_root = work / "runs"
    check("dry-run 没有创建产物目录", not out_root.exists(), str(out_root))

    # ================================================ 5. 端到端
    print("\n[5] 端到端（假 LLM）")
    install_fake_llm()
    rc = run_batch_quietly(in_dir=in_dir, default_instruction="统一指令", out_root=out_root)
    check("全部成功 → 退出码 0", rc == 0, str(rc))

    bds = sorted(out_root.glob("batch_*"))
    check("生成了一个 batch 目录", len(bds) == 1, str(bds))
    bd = bds[0]

    index = bd / "index.md"
    check("生成了 index.md", index.exists())
    text = index.read_text(encoding="utf-8")
    check("汇总表含表头", "| # | 图片 |" in text)
    check("汇总表列出全部 3 张", all(n in text for n in names))
    for col in ("退出原因", "交付轮次", "平均误差", "审美分", "耗时"):
        check(f"汇总表含「{col}」列", col in text)
    check("列出了各任务的报告链接", "report.md" in text)

    for n in names:
        d = bd / Path(n).stem
        need = ("delivered.jpg", "report.md", "rounds.json", "state.json", "llm_calls.jsonl")
        ok = d.is_dir() and all((d / f).exists() for f in need)
        check(f"{Path(n).stem}/ 产物齐全", ok,
              str(sorted(x.name for x in d.iterdir())) if d.is_dir() else "目录不存在")

    d0 = bd / Path(names[0]).stem
    jpgs = sorted(x.name for x in d0.glob("*.jpg"))
    check("交付后只留 original/delivered 两张图",
          jpgs == ["delivered.jpg", "original.jpg"], str(jpgs))

    st = json.loads((d0 / "state.json").read_text(encoding="utf-8"))
    check("state.json 里记录了退出原因", bool(st.get("exit_reason")), str(st.get("exit_reason")))

    # ================================================ 6. 错误隔离 + 退出码
    print("\n[6] 错误隔离与退出码")
    out2 = work / "runs2"
    install_fake_llm(fail_for={"noinst"})
    rc = run_batch_quietly(in_dir=in_dir, default_instruction="统一指令", out_root=out2)
    check("有任务失败 → 退出码 1", rc == 1, str(rc))

    bd2 = sorted(out2.glob("batch_*"))[0]
    text2 = (bd2 / "index.md").read_text(encoding="utf-8")
    check("失败的标为「失败」", "失败" in text2)
    check("失败详情里有错误信息", "模拟失败" in text2)
    made = [Path(n).stem for n in names if (bd2 / Path(n).stem / "delivered.jpg").exists()]
    check("失败的那张没有成片", "noinst" not in made, str(made))
    check("其余任务仍然跑完（错误隔离）", len(made) == 2, str(made))

    # ================================================ 7. 覆盖告警
    print("\n[7] 覆盖已有产物时告警")
    pinned = work / "pinned"
    install_fake_llm()
    run_batch_quietly(in_dir=in_dir, default_instruction="统一指令", out_dir=pinned)
    check("第一次跑：产物就绪", (pinned / Path(names[0]).stem / "delivered.jpg").exists())

    buf = io.StringIO()
    install_fake_llm()
    with contextlib.redirect_stdout(buf):
        batch.run_batch(in_dir=in_dir, default_instruction="统一指令", out_dir=pinned)
    out_text = buf.getvalue()
    check("第二次跑同一目录时打印了覆盖告警",
          "已存在，本次运行会覆盖它" in out_text,
          out_text[-300:])

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
