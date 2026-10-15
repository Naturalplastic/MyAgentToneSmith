"""感知层：图像统计量的确定性测量。

见设计文档 3.2 与 5。

六个统计量：
  luma / contrast / saturation / warmth    —— 可作为优化目标
  clip_hi / clip_lo                        —— 仅用于护栏判定

关键性质：一个调整参数主要影响一个统计量，方向单调。这是盲 LLM 能推理的前提。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import config

_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _load_rgb_float(img_path: str | Path) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def _luma(arr: np.ndarray) -> np.ndarray:
    """亮度通道，0~1。"""
    return arr @ _LUMA_WEIGHTS


def _core_stats(arr: np.ndarray) -> dict[str, float]:
    """四个可优化目标量的测量。"""
    luma = _luma(arr)
    mx = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    # HSV 中的 S 通道
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return {
        "luma": float(luma.mean()),
        "contrast": float(luma.std()),
        "saturation": float(sat.mean()),
        "warmth": float(arr[..., 0].mean() - arr[..., 2].mean()),
    }


def stats(img_path: str | Path) -> dict[str, float]:
    """测量整图的六个统计量。

    Returns:
        ``{"luma", "contrast", "saturation", "warmth", "clip_hi", "clip_lo"}``

    Note:
        ``clip_hi`` / ``clip_lo`` 判的是任一通道触顶/触底，而不是加权亮度。

        这一点很关键：用加权 luma 判截断会出错 —— 一个纯白区域
        ``(1, 1, 1)`` 在 ``color_temp +400`` 之后变成 ``(1, 1, 0.8)``，
        加权 luma 降到 0.977、低于 0.98 阈值，于是**明明还在过曝的像素
        被判为"没有截断"**。实测中这会让 +400 与 −400 两个相反方向
        都把 ``clip_hi`` 变成 0，从而给出完全错误的耦合结论。

        数字图像里的"截断"就是指某个通道打到了编码上限（255），
        与感知亮度无关。
    """
    arr = _load_rgb_float(img_path)

    out = _core_stats(arr)
    out["clip_hi"] = float((arr.max(axis=-1) > 0.98).mean())
    out["clip_lo"] = float((arr.min(axis=-1) < 0.02).mean())
    return out


def _crop_region(arr: np.ndarray, region: str) -> np.ndarray:
    """按区域裁剪。未知区域名抛 ValueError。"""
    h, w = arr.shape[:2]
    if region == "top":
        return arr[: h // 3, :]
    if region == "middle":
        return arr[h // 3 : (2 * h) // 3, :]
    if region == "bottom":
        return arr[(2 * h) // 3 :, :]
    if region == "center":
        return arr[h // 4 : (3 * h) // 4, w // 4 : (3 * w) // 4]
    raise ValueError(f"未知区域: {region!r}，可选 {config.REGIONS}")


def region_stats(img_path: str | Path, region: str) -> dict[str, float]:
    """测量某个区域的四个统计量。

    补的是全局统计量最大的盲区：全局均值看不出改动集中在天空、背景还是人物。

    Args:
        region: ``"top"`` / ``"middle"`` / ``"bottom"`` / ``"center"``

    Returns:
        ``{"luma", "contrast", "saturation", "warmth"}``
    """
    arr = _load_rgb_float(img_path)
    return _core_stats(_crop_region(arr, region))
