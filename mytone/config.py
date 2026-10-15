"""全局常量。

以设计文档 DESIGN.md 第 12.1.1 节为准。
这里只放常量，不放逻辑。
"""

from __future__ import annotations

import os

import torch

# ---------------------------------------------------------------- 调整参数
# 见设计文档 4.1
PARAMS: dict[str, dict[str, float]] = {
    "brightness": {"range": (-0.6, 0.6), "step": 0.2},
    "contrast": {"range": (-0.6, 0.6), "step": 0.2},
    "saturation": {"range": (-0.6, 0.6), "step": 0.2},
    "color_temp": {"range": (-400.0, 400.0), "step": 400.0},
    "gamma": {"range": (-0.5, 0.5), "step": 0.15},
}

PARAM_NAMES: list[str] = list(PARAMS)

#: 单轮最多允许同时调整的参数个数（提示词中声明该规则，这里同步校验）
MAX_PARAMS_PER_ROUND = 2

#: 固定的应用顺序，顺序会改变最终像素，属设计的一部分。见设计文档 4.2
APPLY_ORDER: list[str] = ["color_temp", "gamma", "brightness", "contrast", "saturation"]

#: color_temp 的单位：千分之二通道增益。t=400 表示 R+20%、B-20%。见设计文档 4.1
COLOR_TEMP_DIVISOR = 2000.0

ZERO_PARAMS: dict[str, float] = {name: 0.0 for name in PARAM_NAMES}

#: 每个参数的单位说明。必须写清楚，否则 LLM 会用错量纲
#: （首次真实运行就踩了这个坑：模型给 color_temp 输出 -0.09，而它的单位是千分之二，
#:   -0.09 等于完全没有效果）。
PARAM_UNITS = {
    "brightness": "偏移量。0.2 = 亮度乘 1.2；0.6 = 乘 1.6（这是能到的最亮）",
    "contrast": "偏移量。0.2 = 对比度乘 1.2；-0.2 = 乘 0.8",
    "saturation": "偏移量。0.2 = 饱和度乘 1.2；-0.2 = 乘 0.8",
    "color_temp": (
        "千分之二通道增益。-400 = R 通道减 20%、B 通道加 20%（明显变冷）；"
        "+400 = 明显变暖。绝对值小于 50 基本看不出效果，有效幅度通常在 ±100 ~ ±400"
    ),
    "gamma": "中间调指数偏移。+0.15 = 指数 1.15（中间调变暗）；-0.15 = 指数 0.85（中间调变亮）",
}

#: 参数耦合说明，必须写进提示词。见设计文档 4.4
#: 结构性耦合由 tests/check_core.py 与 calibrate.py 实测得出，不是推测。
#: 幅度随题材变化，所以只声明方向与代价结构，不写死数字。
COUPLING_TABLE = """\
  brightness  整体亮度      提亮 luma 最直接，但推高（高光/暗部）截断也最猛
  gamma       中间调明暗    负值提亮中间调；对截断的影响远小于 brightness
                            —— 想提亮又不想爆高光时，优先用它
  contrast    对比度        主要作用于 saturation 与 contrast，同时明显推高亮部与暗部截断
  saturation  饱和度        基本只影响饱和度，对截断影响较小
  color_temp  冷暖          主要影响 warmth；但它会同时推高截断
                            （R/B 总有一个通道被增益，偏冷比偏暖更明显）"""

COUPLING_NOTES = """\
  - 【最容易出错】color_temp 的单位与其他四个完全不同，是"千分之二"。
    想明显变冷要输出 -200 ~ -400 这样的值；输出 -0.1 等于什么都没做。
  - 【截断的代价，最容易被忽略】brightness 每提亮一步都会大幅推高截断比例。
    如果原图本身已经有大面积高光溢出，brightness 的可用余量会很小、甚至接近于零。
    这种情况下想提高 luma，应当优先使用 gamma 负值，而不是继续加 brightness。
    请看上方"护栏触发统计"：若反复因高光死白被回滚，就是撞上了这个问题。
  - contrast 会同时推高亮部与暗部截断，用它时留意 clip_hi / clip_lo。
  - 截断的判定是"任一通道打顶"（RGB 中最大值 > 0.98），与加权亮度无关。
    所以加暖或加冷都可能推高截断，不要以为调色温是"免费的"。
  - contrast 是"向中性灰混合"，通常连带降低 saturation，降幅可能超过它对 contrast 本身的影响。
  - brightness / contrast / saturation 都会按各自倍数放大画面已有的 R-B 差值：
    原图偏暖或偏冷时这三个参数会把这个色偏一起放大；原图接近中性时影响很小。
    修正色偏要用 color_temp，不要用它们。
  - brightness 和 gamma 都改变 luma，方向相同，同时用会冲过头。"""

#: 每个参数"主要影响"的统计量，用于提示词与文档对照
PRIMARY_EFFECT = {
    "brightness": "luma",
    "contrast": "contrast",
    "saturation": "saturation",
    "color_temp": "warmth",
    "gamma": "luma",
}

# ---------------------------------------------------------------- 统计量
# 见设计文档 3.2 / 5
#: 可作为优化目标的统计量
STAT_NAMES: list[str] = ["luma", "contrast", "saturation", "warmth"]
#: 仅用于护栏判定的统计量
GUARD_STATS: list[str] = ["clip_hi", "clip_lo"]

#: 给 LLM 看的合理范围（同时用作目标值的合法区间）
STAT_RANGE: dict[str, tuple[float, float]] = {
    "luma": (0.30, 0.75),
    "contrast": (0.12, 0.32),
    "saturation": (0.05, 0.60),
    "warmth": (-0.15, 0.15),
}

STAT_DESC: dict[str, str] = {
    "luma": "平均亮度",
    "contrast": "亮度标准差",
    "saturation": "饱和度均值",
    "warmth": "R均值-B均值（正=暖）",
}

# ---------------------------------------------------------------- 护栏阈值
# 见设计文档 3.4 / 9.1
#
# 设计要点：不设绝对上限，而是一律以原图为基准算相对容差。
# 理由是实测出来的：一张雾景照片的 contrast 本来就低于 0.10，
# 用绝对阈值会在第 1 轮就把它判成"画面发灰"，agent 当场被掐死。
# 参见设计文档 3.4。

#: 对比度下限：取"绝对下限"与"原图的若干比例"中较宽松者
CONTRAST_MIN = 0.10
CONTRAST_RATIO = 0.6  # 对比度掉到原图的这个比例以下就算发灰

#: 允许比原图多出的截断比例。
#: 实际容差 = max(绝对容差, 原图自身截断量 × 相对比例)，两者取大：
#:   - 原图几乎没有过曝时，绝对容差生效（0 的 20% 还是 0）
#:   - 原图本身大面积过曝时，相对容差生效
#:     （原图已有 23% 过曝，再多 2 个百分点对它其实是微不足道的变化）
#:
#: 为什么是 0.03 而不是更严的 0.02：实测中 agent 反复落在 0.0238 ~ 0.0240，
#: 而上限是 0.0237 —— 差 0.0001 量级就被判非法。而"提亮 + 增艳"这个任务本身
#: 必然会推高截断，0.02 窄到把合法工作也挡掉了。
#: 3% 的像素打顶，对一张 2064×4608 的图约等于一个 535×535 的区域，仍是实质约束。
#:
#: 这是口味相关的参数，可以按自己的接受度调整。 调大更宽松。
CLIP_HI_DELTA = 0.03  # 绝对：允许多 3% 的像素打顶
CLIP_HI_RELATIVE = 0.20  # 相对：再允许多出"原图自身过曝量"的 20%
CLIP_LO_DELTA = 0.05  # 暗部同理
CLIP_LO_RELATIVE = 0.20

#: 连续多少轮触发护栏就强制退出。
#:
#: 为什么不是 2（实跑教训）：被护栏罚过一次之后，agent 通常会主动把幅度减半
#: 再试一次，正在往可用区间收敛。设成 2 时，第二次尝试往往仍在越线区，
#: 于是 agent 在收敛过程中被掐断。
#: 实测案例：某次运行里第 2 轮 clip_hi 0.0677（上限 0.0237）被罚，
#: 第 3 轮它把幅度减半后 clip_hi 0.0480 —— 方向完全正确，却被判为"连续 2 轮"而退出。
#: 设成 3 给它一次真正的收敛机会；每一轮越线都会消耗轮次预算，本身也有自我限制。
GUARDRAIL_STREAK_LIMIT = 3

#: 审美分软/硬阈值。
#: 这两个值不是拍脑袋填的，来自 calibrate.py 在真实照片上的实测（见设计文档 3.7）：
#: 单参数一个步长能造成的审美分最大变化是 0.1932（来自 contrast），
#: 故 D_SOFT = 1.5 × 0.1932 ≈ 0.29，D_HARD = 3 × 0.1932 ≈ 0.58。
#: 换照片或换题材后建议重跑 calibrate.py。
D_SOFT = 0.29
D_HARD = 0.58

# ---------------------------------------------------------------- 预算与上限
# 见设计文档 5.3 / 9.3
ROUND_LIMIT = 10  # 新增调整轮次上限，不含第 0 轮
#: 允许多少次目标修正。修正会消耗轮次预算，且次数受限 ——
#: 否则 agent 可以把目标改成"已经达到的值"来假装收敛。
TARGET_REVISION_LIMIT = 2
TOOL_BUDGET = 4
TOOL_REJECT_LIMIT = 3  # 预算耗尽后仍请求工具的次数上限
LLM_RETRY_LIMIT = 2  # 输出不合法时的重试次数
DELIVER_REJECT_LIMIT = 2  # 交付轮次非法被拒次数上限

# ---------------------------------------------------------------- 收敛判据
# 见设计文档 9.2
ERROR_TOL = 0.02
#: "几乎没动"的判定阈值。取值要足够小：设成 0.005 时，一次真实运行在
#: 误差 0.0286 → 0.0241 → 0.0200 稳步收敛的过程中被判为平台期提前掐断。
#: 真正卡住的表现是改善量接近 0 或为负，所以用 0.001。
IMPROVE_EPS = 0.001
PLATEAU_ROUNDS = 2

#: 连续调整这么多轮还没用过审美工具时，harness 给一次提醒（只提醒一次）。
#: 理由：工具存在但从未被使用，会失去"工具调用"这部分的学习价值。
#: 注意这只是"提醒有这个工具"，不涉及任何优化决策。
AESTHETIC_REMINDER_AFTER = 4

# ---------------------------------------------------------------- 开关
# 见设计文档 5.5 / 7.1 / 11.1
AESTHETIC_MODE = "tool"  # "tool" | "auto"
CONFIRM = False  # 是否在解读指令后暂停等用户确认
KEEP_INTERMEDIATE_IMAGES = False  # 交付时是否保留中间产物图片

# ---------------------------------------------------------------- 运行设备
# 不要硬编码 cuda，见设计文档 2.6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- 审美工具
# 见设计文档 3.5
AESTHETIC_METRIC = "nima"  # 备选：nima-vgg16-ava / musiq-ava / topiq_iaa / laion_aes

#: 注入提示词的占位数字，见设计文档 3.6 / 7.2
AES_NOISE_BAND = D_SOFT
AES_LATENCY = "约 0.1 秒"  # 实测 CPU 0.10s，见设计文档 2.6

# ---------------------------------------------------------------- LLM
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
LLM_TIMEOUT = 120  # 秒
#: LLM 采样温度。可用环境变量 MYTONE_TEMPERATURE 覆盖。
#:
#: 0.2 是默认值：留一点探索空间，让 agent 每轮不必死板。
#: 但实测同一张图同一条指令多次运行结果会不同（一次 0.0165 达标、一次 0.0268 未达标），
#: 想复现某次运行做排障时，设 MYTONE_TEMPERATURE=0。
LLM_TEMPERATURE = float(os.environ.get("MYTONE_TEMPERATURE", "0.2"))

# ---------------------------------------------------------------- 区域工具
# 见设计文档 5.4
REGIONS: list[str] = ["top", "middle", "bottom", "center"]


def guard_caps(original: dict[str, float] | None) -> tuple[float, float, float]:
    """算出本次运行的三条图像护栏阈值。

    一律以原图为基准，不设绝对上限：

    - 截断：``原图 + max(绝对容差, 原图 × 相对比例)``
    - 对比度：``min(绝对下限, 原图 × 比例)``

    这样原图本身就是低对比（雾景、柔光）或大面积过曝时不会被误判，
    但相对原图继续恶化仍然会被拦下。

    Returns:
        ``(clip_hi 上限, clip_lo 上限, contrast 下限)``
    """
    orig = original or {}
    o_hi = orig.get("clip_hi", 0.0)
    o_lo = orig.get("clip_lo", 0.0)
    o_ct = orig.get("contrast", CONTRAST_MIN)

    hi_cap = o_hi + max(CLIP_HI_DELTA, o_hi * CLIP_HI_RELATIVE)
    lo_cap = o_lo + max(CLIP_LO_DELTA, o_lo * CLIP_LO_RELATIVE)
    c_min = min(CONTRAST_MIN, o_ct * CONTRAST_RATIO)
    return hi_cap, lo_cap, c_min


def guard_headroom(original: dict[str, float] | None, stats_now: dict[str, float]) -> dict[str, float]:
    """各护栏还剩多少余量（负数表示已经越线）。纯测量，供上下文与报告使用。"""
    hi_cap, lo_cap, c_min = guard_caps(original)
    return {
        "clip_hi": hi_cap - stats_now.get("clip_hi", 0.0),
        "clip_lo": lo_cap - stats_now.get("clip_lo", 0.0),
        "contrast": stats_now.get("contrast", 0.0) - c_min,
    }


def available_tools(aesthetic_mode: str | None = None) -> set[str]:
    """当前模式下 LLM 可调用的工具集合（设计文档 5.5 / 5.6）。

    ``auto`` 模式下审美分由 harness 每轮自动测量，``aesthetic`` 工具不在清单内。
    """
    mode = aesthetic_mode or AESTHETIC_MODE
    tools = {"region"}
    if mode == "tool":
        tools.add("aesthetic")
    return tools


def param_help() -> str:
    """生成参数取值范围、步长上限与单位说明，用于拼接提示词。

    单位说明不能省：首次真实运行就因为 LLM 不知道 color_temp 的量纲
    而输出了 -0.09 这种等于零效果的值。
    """
    lines = []
    for name, spec in PARAMS.items():
        lo, hi = spec["range"]
        lines.append(
            f"  {name:<11} 取值 {lo:g} ~ {hi:g}    单轮步长上限 {spec['step']:g}"
        )
        lines.append(f"              单位：{PARAM_UNITS[name]}")
    return "\n".join(lines)


def stat_help() -> str:
    """生成可选统计量清单，用于拼接提示词。"""
    lines = []
    for name in STAT_NAMES:
        lo, hi = STAT_RANGE[name]
        lines.append(f"  {name:<11} {STAT_DESC[name]:<14} 合理范围 {lo:g} ~ {hi:g}")
    return "\n".join(lines)
