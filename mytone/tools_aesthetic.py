"""审美评判层：NIMA 审美分的封装。

见设计文档 3.5。主选指标为 ``nima``（NIMA，Inception-ResNet-v2，AVA 训练）。

设计文档：
    Talebi, H., & Milanfar, P. (2018). NIMA: Neural Image Assessment.
    IEEE Transactions on Image Processing, 27(8), 3998-4011. arXiv:1709.05424

本模块只负责"测量"，不做任何判断。审美分绝不参与误差计算与 best 判定
（设计文档 3.3 / 7）。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from . import config

_metric: Any = None
_load_seconds: float | None = None


def _get_metric() -> Any:
    """惰性加载指标。首次调用会下载权重（约 200 MB），耗时数十秒。"""
    global _metric, _load_seconds
    if _metric is None:
        import pyiqa  # 延迟导入：import pyiqa 本身较慢

        t0 = time.time()
        _metric = pyiqa.create_metric(config.AESTHETIC_METRIC, device=config.DEVICE)
        _load_seconds = time.time() - t0
    return _metric


def warmup() -> float:
    """预加载指标并返回加载耗时（秒）。便于在循环开始前把权重下好。"""
    _get_metric()
    return float(_load_seconds or 0.0)


def aesthetic_score(img_path: str | Path) -> float:
    """返回 1~10 的审美分，越大越好。"""
    return float(_get_metric()(str(img_path)))


def measure(img_path: str | Path) -> tuple[float, float]:
    """测量并同时返回耗时（秒），用于把真实耗时写进日志。"""
    t0 = time.time()
    score = aesthetic_score(img_path)
    return score, time.time() - t0


def metric_info() -> dict[str, Any]:
    """返回指标元信息，用于写进报告与日志。"""
    m = _get_metric()
    return {
        "name": config.AESTHETIC_METRIC,
        "device": config.DEVICE,
        "lower_better": bool(getattr(m, "lower_better", False)),
        "score_range": getattr(m, "score_range", None),
        "load_seconds": _load_seconds,
    }
