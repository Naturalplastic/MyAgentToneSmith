"""MyToneSmith —— 评价器在环的 AI 摄影后期 Agent。

设计文档：DESIGN.md
"""

from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"


def _load_dotenv() -> None:
    """加载项目根目录的 ``.env``（如果存在）。

    用 python-dotenv，这是 Python 应用读取本地配置的通行做法。
    行为约定：

    - 已存在的环境变量优先，``.env`` 不会覆盖它们
      （所以 ``export DEEPSEEK_API_KEY=...`` 永远比 ``.env`` 优先）；
    - 找不到 ``.env`` 就静默跳过，环境变量照常可用；
    - 在包导入时执行一次，因此下面的 ``config`` 能读到 ``.env`` 里的值。

    ``.env`` 已列入 .gitignore；模板见项目根目录的 ``.env.example``。
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # 没装 python-dotenv 也能用，只是不读 .env
        return

    # 从本文件向上找到项目根（含 pyproject.toml 或 requirements.txt 的那一层）
    here = Path(__file__).resolve().parent
    for candidate in (here.parent, *here.parents):
        if (candidate / ".env").is_file():
            load_dotenv(candidate / ".env", override=False)
            return


_load_dotenv()

