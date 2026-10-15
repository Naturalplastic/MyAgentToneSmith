"""调整层：确定性的像素操作。

见设计文档第 4 节。

两条必须遵守的约束：
1. 永远从原图重新渲染（``adjust(原图, 累计参数)``），不在上一轮结果上叠加；
2. 五个操作的顺序固定为 config.APPLY_ORDER。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from . import config

#: 本模块实现的顺序必须与 config.APPLY_ORDER 一致。
#: 顺序会改变最终像素，属于设计的一部分（设计文档 4.2），因此在这里做一次断言保护。
_EXPECTED_ORDER = ["color_temp", "gamma", "brightness", "contrast", "saturation"]
if config.APPLY_ORDER != _EXPECTED_ORDER:
    raise RuntimeError(
        "config.APPLY_ORDER 与 tools_image.adjust 的实现顺序不一致："
        f"{config.APPLY_ORDER} != {_EXPECTED_ORDER}。"
        "改动顺序时必须同步修改两处。"
    )


def _load_rgb_float(src_path: str | Path) -> np.ndarray:
    """读入图片，转成 RGB float32 且归一化到 0~1。"""
    img = Image.open(src_path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def _apply_color_temp(arr: np.ndarray, t: float) -> np.ndarray:
    """R/B 通道增益。t 单位为千分之二：t=400 表示 R 通道 +20%、B 通道 -20%。"""
    gain = t / config.COLOR_TEMP_DIVISOR
    out = arr.copy()
    out[..., 0] *= 1.0 + gain
    out[..., 2] *= 1.0 - gain
    return out


def _apply_gamma(arr: np.ndarray, g: float) -> np.ndarray:
    """中间调指数偏移。g > 0 使中间调变暗。"""
    return np.power(np.clip(arr, 0.0, 1.0), 1.0 + g)


def adjust(src_path: str | Path, params: dict[str, float], out_path: str | Path) -> str:
    """对原图应用累计绝对参数，渲染候选图并返回其路径。

    Args:
        src_path: 原图路径，只读。
        params: 五个参数的累计绝对值，缺失的键按 0 处理。
        out_path: 候选图输出路径，父目录会自动创建。

    Returns:
        候选图路径（字符串）。

    Note:
        步骤 1、2 在 float 域完成以避免二次量化；步骤 3~5 交给 PIL 的
        ``ImageEnhance``，与设计文档 4.1 的实现列一致。中间有一次
        float→uint8 转换，这是刻意保留的（贴近真实后期管线的行为），
        且完全确定性。
    """
    arr = _load_rgb_float(src_path)

    # 1. color_temp
    arr = _apply_color_temp(arr, float(params.get("color_temp", 0.0)))
    # 2. gamma
    arr = _apply_gamma(arr, float(params.get("gamma", 0.0)))

    out = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8))

    # 3. brightness  4. contrast  5. saturation
    # ImageEnhance.Brightness: Image.blend(黑, 图, f) == 图 * f
    out = ImageEnhance.Brightness(out).enhance(1.0 + float(params.get("brightness", 0.0)))
    # ImageEnhance.Contrast: 与"亮度均值灰底"做 blend
    out = ImageEnhance.Contrast(out).enhance(1.0 + float(params.get("contrast", 0.0)))
    # ImageEnhance.Color: 与灰度图做 blend
    out = ImageEnhance.Color(out).enhance(1.0 + float(params.get("saturation", 0.0)))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    return str(out_path)
