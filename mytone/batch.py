"""批处理入口：把一整个文件夹的照片逐张跑一遍完整 agent。

    python -m mytone.batch                      # 扫描 input/，用每条图自己的指令
    python -m mytone.batch -p "整体提亮"        # 没配 txt 的图用这条默认指令
    python -m mytone.batch -d 我的照片 -p "清冷"  # 换输入目录
    python -m mytone.batch --dry-run            # 只列出计划，不真跑
    python -m mytone.batch --skip-existing      # 跳过已经跑出成片的

输入约定
--------
在输入目录（默认 ``input/``）里放图片；对某张图想用不同指令时，
在旁边放一个同名 txt，内容就是这条图的指令：

    input/
      照片A.jpg
      照片A.txt        ← 内容是"帮我进行coser界通用的基本照片调光"
      照片B.jpg        ← 没有同名 txt，用命令行 -p 的默认指令

txt 里以 ``#`` 开头的行会被当作注释忽略，其余内容整体作为指令。

产物
----
    runs/batch_2026-10-15_2130/
      index.md        汇总表：退出原因 / 是否达标 / 误差 / 审美分变化 / 耗时
      照片A/          与单张运行完全相同的目录结构
      照片B/

单张失败不会中断整批；失败的会在汇总表里标出来并附上错误信息。

退出码
------
``0`` 全部任务都正常跑完（"未达标"也算正常跑完 —— agent 只是没达成目标，
报告里能看出来）；``1`` 有任务执行失败（抛异常），或无图可跑、有图缺指令。

注意与设计文档 §12.4 的区别
---------------------------
这里不是"批量处理"（那条扩展指的是把一个已收敛的参数配方固定套用到整个文件夹）。
本模块是对每张图各跑一次完整 agent 循环，属于使用便利性，不改动 agent 核心。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

from . import config
from .loop import run
from .report import exit_reason_text

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"}


# ================================================================ 输入扫描


def read_instruction(txt_path: Path) -> str:
    """读指令 txt。以 # 开头的行是注释，其余内容整体作为指令。"""
    lines = []
    for raw in txt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def scan_tasks(in_dir: Path, default_instruction: str | None) -> list[dict]:
    """扫描输入目录，返回任务清单。"""
    if not in_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{in_dir}")

    tasks = []
    for img in sorted(in_dir.iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue

        txt = img.with_suffix(".txt")
        if txt.is_file():
            instruction = read_instruction(txt)
            source = f"{txt.name}"
            if not instruction:
                instruction = (default_instruction or "").strip()
                source = f"{txt.name}（内容为空，回退到默认指令）"
        else:
            instruction = (default_instruction or "").strip()
            source = "默认指令"

        tasks.append(
            {
                "image": img,
                "instruction": instruction,
                "source": source,
                "dir_name": img.stem,
            }
        )
    return tasks


# ================================================================ 汇总


def _pct(v) -> str:
    return "—" if v is None else f"{v:.4f}"


def write_index(out_dir: Path, records: list[dict], in_dir: Path, default_instruction: str | None) -> Path:
    """写 batch 汇总表。"""
    lines: list[str] = []
    a = lines.append

    a("# 批处理汇总")
    a("")
    a(f"- 输入目录：`{in_dir}`")
    a(f"- 默认指令：{default_instruction or '（未提供）'}")
    a(f"- 任务数：{len(records)}")
    ok = [r for r in records if r.get("status") == "ok"]
    done = [r for r in ok if r.get("goal_met")]
    a(f"- 完成：{len(ok)} / {len(records)}　其中交付轮次达标：{len(done)}")
    a("")
    a("`交付结果` 指交付的那一轮的误差是否进入容差，与 agent 是否收敛（见 `退出原因`）是两件事。")
    a("")
    a("| # | 图片 | 指令来源 | 交付结果 | 退出原因 | 交付轮次 | 平均误差 | 审美分 | 耗时 |")
    a("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(records, 1):
        if r["status"] != "ok":
            a(
                f"| {i} | {r['image'].name} | {r['source']} | 失败 "
                f"| {r.get('error', '')} | — | — | — | {r['elapsed']:.0f}s |"
            )
            continue
        mark = "达标" if r["goal_met"] else "未达标"
        if not r["guard_ok"]:
            mark = "交付轮次非法"
        aes = "未测"
        if r["aes_base"] is not None and r["aes_final"] is not None:
            aes = f"{r['aes_base']:.2f} → {r['aes_final']:.2f}（{r['aes_final'] - r['aes_base']:+.2f}）"
        a(
            f"| {i} | {r['image'].name} | {r['source']} | {mark} "
            f"| {exit_reason_text(r['exit_reason'])} | {r['delivered']} "
            f"| {_pct(r['error'])} | {aes} | {r['elapsed']:.0f}s |"
        )
    a("")

    failed = [r for r in records if r["status"] != "ok"]
    if failed:
        a("## 失败详情")
        a("")
        for r in failed:
            a(f"### {r['image'].name}")
            a("")
            a("```")
            a(str(r.get("traceback", r.get("error", ""))).strip())
            a("```")
            a("")

    a("## 各任务的详细报告")
    a("")
    for r in records:
        if r["status"] != "ok":
            continue
        rel = Path(r["run_dir"]).name
        a(f"- [{r['image'].name}]({rel}/report.md)　成片：`{rel}/delivered.jpg`")
    a("")

    path = out_dir / "index.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ================================================================ 主流程


def run_batch(
    in_dir: Path,
    default_instruction: str | None,
    out_root: Path = Path("runs"),
    aesthetic_mode: str | None = None,
    keep_intermediate: bool = False,
    api_key: str | None = None,
    limit: int | None = None,
    skip_existing: bool = False,
    dry_run: bool = False,
    out_dir: Path | None = None,
) -> int:
    tasks = scan_tasks(in_dir, default_instruction)
    if limit:
        tasks = tasks[:limit]

    if not tasks:
        print(
            f"[错误] {in_dir} 里没有找到图片。\n"
            f"       请把照片放进 {in_dir}/"
            f"（支持 {'、'.join(sorted(IMAGE_EXTS))}）\n"
            f"       想给某张图单独指令，就在它旁边放一个同名 txt，内容写指令。\n"
            f"       详见 input/README.md"
        )
        return 1

    if out_dir is None:
        stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_dir = out_root / f"batch_{stamp}"

    print("=" * 78)
    print(f"输入目录：{in_dir}")
    print(f"输出目录：{out_dir}")
    print(f"任务数  ：{len(tasks)}")
    # 显示实际使用的设备：批处理耗时以小时计，用户需要确认是否用上了 GPU
    print(f"运行设备：{config.DEVICE}"
          + ("（未检测到可用 CUDA 设备，使用 CPU）" if config.DEVICE == "cpu" else ""))
    print("=" * 78)
    for i, t in enumerate(tasks, 1):
        inst = t["instruction"] or "（空，将报错）"
        print(f"  {i}. {t['image'].name}")
        print(f"     指令来源：{t['source']}")
        print(f"     指令内容：{inst[:70]}{'...' if len(inst) > 70 else ''}")

    # 缺指令的任务直接标失败，不浪费一次运行
    bad = [t for t in tasks if not t["instruction"]]
    if bad:
        print()
        print(f"[错误] 有 {len(bad)} 张图没有指令（既无同名 txt，也没有 -p 默认指令）：")
        for t in bad:
            print(f"       {t['image'].name}")
        print("       请补上同名 txt，或用 -p 给一个默认指令。")
        return 1

    if dry_run:
        print()
        print("--dry-run：只列计划，不执行。")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for i, t in enumerate(tasks, 1):
        img: Path = t["image"]
        run_dir = out_dir / t["dir_name"]

        if skip_existing and (run_dir / "delivered.jpg").exists():
            print(f"\n[{i}/{len(tasks)}] {img.name} —— 已有成片，跳过")
            records.append(
                {
                    **t,
                    "status": "ok",
                    "skipped": True,
                    "exit_reason": "（跳过）",
                    "delivered": "—",
                    "error": None,
                    "goal_met": True,
                    "guard_ok": True,
                    "aes_base": None,
                    "aes_final": None,
                    "run_dir": str(run_dir),
                    "elapsed": 0.0,
                }
            )
            continue

        print(f"\n[{i}/{len(tasks)}] {img.name}")
        print(f"    指令：{t['instruction'][:70]}")
        t0 = time.time()
        try:
            if run_dir.exists():
                # 覆盖前明确告警（不做静默销毁）
                print(f"    注意：{run_dir.name}/ 已存在，本次运行会覆盖它")
                shutil.rmtree(run_dir)
            state = run(
                instruction=t["instruction"],
                original_path=img,
                run_root=out_dir,
                run_name=t["dir_name"],
                api_key=api_key,
                confirm=False,
                keep_intermediate=keep_intermediate,
                aesthetic_mode=aesthetic_mode,
            )
            elapsed = time.time() - t0
            n = state.delivered if state.delivered is not None else 0
            r = state.rounds[n]
            records.append(
                {
                    **t,
                    "status": "ok",
                    "skipped": False,
                    "exit_reason": state.exit_reason,
                    "delivered": n,
                    "error": r.error,
                    "goal_met": r.error is not None and r.error < config.ERROR_TOL,
                    "guard_ok": r.guard_ok,
                    "aes_base": state.aes_baseline,
                    "aes_final": r.aes,
                    "run_dir": state.run_dir,
                    "report": state.report_path,
                    "elapsed": elapsed,
                }
            )
            print(
                f"    → {exit_reason_text(state.exit_reason)}　"
                f"交付第 {n} 轮　误差 {_pct(r.error)}　"
                f"{'达标' if records[-1]['goal_met'] else '未达标'}　{elapsed:.0f}s"
            )
        except Exception as exc:  # 单张失败不中断整批
            elapsed = time.time() - t0
            records.append(
                {
                    **t,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "elapsed": elapsed,
                }
            )
            print(f"    → 失败：{type(exc).__name__}: {exc}")

    index = write_index(out_dir, records, in_dir, default_instruction)

    ok = [r for r in records if r["status"] == "ok" and not r.get("skipped")]
    done = [r for r in ok if r.get("goal_met")]
    errors = [r for r in records if r["status"] != "ok"]

    print()
    print("=" * 78)
    print(f"完成 {len(ok)}/{len(records)}　达标 {len(done)}　失败 {len(errors)}")
    for r in records:
        if r["status"] != "ok":
            print(f"  [失败] {r['image'].name}：{r['error']}")
        elif not r.get("skipped") and not r.get("goal_met"):
            print(f"  [未达标] {r['image'].name}：误差 {_pct(r['error'])}")
    print(f"\n汇总表：{index}")
    if errors:
        print(f"退出码：1（有 {len(errors)} 个任务执行失败）")
    print("=" * 78)

    # 退出码约定：有任何任务执行失败（抛异常）→ 1；否则 0。
    # "未达标"不算失败 —— 那是 agent 正常跑完、只是没达成目标，报告里能看出来。
    return 1 if errors else 0


# ================================================================ CLI


def _line_buffer_stdout() -> None:
    """让 stdout 行缓冲。

    批处理运行时间以小时计，用户通常会把输出重定向到日志文件。
    此时 Python 默认使用块缓冲，进度要等很久才落盘，无法观察运行状态。
    改为行缓冲后，每打印一行就写出去。
    """
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # 非标准 stdout（如被替换过的流）
        pass


def main(argv: list[str] | None = None) -> int:
    _line_buffer_stdout()
    p = argparse.ArgumentParser(
        prog="mytone.batch",
        description="批量跑：把一整个文件夹的照片逐张跑一遍完整 agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("-d", "--dir", default="input", help="输入目录（默认 input/）")
    p.add_argument("-p", "--instruction", help="默认指令：没有同名 txt 的图用它")
    p.add_argument("-o", "--out", default="runs", help="产物根目录（默认 runs/）")
    p.add_argument("--aesthetic-mode", choices=["tool", "auto"], default=config.AESTHETIC_MODE)
    p.add_argument("--keep-intermediate", action="store_true", help="保留中间产物图片")
    p.add_argument("--limit", type=int, help="只跑前 N 张（调试用）")
    p.add_argument("--skip-existing", action="store_true", help="已有成片的跳过")
    p.add_argument("--dry-run", action="store_true", help="只列出计划，不执行")
    args = p.parse_args(argv)

    # --dry-run 只列计划，不需要 API key
    if not args.dry_run and not config.DEEPSEEK_API_KEY:
        print(
            "[错误] 未设置 DEEPSEEK_API_KEY。请二选一：\n"
            "       1) cp .env.example .env  然后填上密钥\n"
            "       2) export DEEPSEEK_API_KEY=sk-xxxxxxxx",
            file=sys.stderr,
        )
        return 1

    try:
        return run_batch(
            in_dir=Path(args.dir),
            default_instruction=args.instruction,
            out_root=Path(args.out),
            aesthetic_mode=args.aesthetic_mode,
            keep_intermediate=args.keep_intermediate,
            limit=args.limit,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
