"""命令行入口。

用法：

    # 跑一次调色
    python -m mytone -i photo.jpg -p "帮我进行coser界通用的基本照片调光"

    # 先看模型如何解读指令与声明能力边界，确认后再跑
    python -m mytone -i photo.jpg -p "清冷" --confirm

    # 保留中间产物图片（调试用）
    python -m mytone -i photo.jpg -p "清冷" --keep-intermediate

    # 用 rounds.json 里的参数向量重算某一轮的画面
    python -m mytone --regenerate runs/2026-10-15_143022_清冷 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config
from .loop import regenerate, run
from .report import exit_reason_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mytone",
        description="评价器在环的 AI 摄影后期 Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-i", "--image", help="原图路径")
    parser.add_argument("-p", "--instruction", help="自然语言调色指令")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="在解读指令后暂停，等确认再继续",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="保留中间产物图片（默认交付后删除，只留成片）",
    )
    parser.add_argument(
        "--aesthetic-mode",
        choices=["tool", "auto"],
        default=config.AESTHETIC_MODE,
        help="tool=由 LLM 决定何时测量审美分；auto=每轮自动测量（默认 tool）",
    )
    parser.add_argument("--run-root", default="runs", help="运行产物根目录")
    parser.add_argument(
        "--regenerate",
        nargs=2,
        metavar=("RUN_DIR", "ROUND"),
        help="重算指定运行中某一轮的画面，不跑 Agent",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # ---------- 重算模式 ----------
    if args.regenerate:
        run_dir, round_str = args.regenerate
        try:
            path = regenerate(run_dir, int(round_str))
        except (ValueError, FileNotFoundError) as exc:
            print(f"[错误] {exc}", file=sys.stderr)
            return 1
        print(f"已重算：{path}")
        return 0

    # ---------- 正常运行 ----------
    if not args.image or not args.instruction:
        parser.error("需要同时提供 --image 与 --instruction（或使用 --regenerate）")

    image = Path(args.image)
    if not image.exists():
        print(f"[错误] 找不到原图：{image}", file=sys.stderr)
        return 1

    if not config.DEEPSEEK_API_KEY:
        print(
            "[错误] 未设置 DEEPSEEK_API_KEY。\n"
            "       请先执行： export DEEPSEEK_API_KEY=sk-xxxxxxxx",
            file=sys.stderr,
        )
        return 1

    print(f"设备：{config.DEVICE}    审美工具：{config.AESTHETIC_METRIC}")
    print(f"原图：{image}")
    print(f"指令：{args.instruction}")
    print("正在运行 agent 循环 ...\n")

    state = run(
        instruction=args.instruction,
        original_path=image,
        run_root=args.run_root,
        confirm=args.confirm,
        keep_intermediate=args.keep_intermediate,
        aesthetic_mode=args.aesthetic_mode,
    )

    delivered = state.rounds[state.delivered if state.delivered is not None else 0]
    print("\n" + "=" * 68)
    print("完成")
    print("=" * 68)
    print(f"退出原因  ：{exit_reason_text(state.exit_reason)}")
    print(f"交付轮次  ：第 {delivered.index} 轮"
          + ("（安全网兜底）" if state.delivered_by_safety_net else "（由 LLM 选择）"))
    print(f"平均误差  ：{'—' if delivered.error is None else f'{delivered.error:.4f}'}"
          f"（容差 {config.ERROR_TOL}）")
    print(f"审美分    ：{'未测量' if delivered.aes is None else f'{delivered.aes:.2f}'}"
          f"（基线 {state.aes_baseline:.2f}）")
    print(f"LLM 调用  ：{state.llm_call_count} 次   工具调用：{state.tools_used} 次")
    print(f"运行目录  ：{state.run_dir}")
    print(f"成片      ：{Path(state.run_dir) / 'delivered.jpg'}")
    print(f"报告      ：{state.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
