# AI 摄影后期 Agent 设计文档

> 日期：2026-10-15 ｜ 状态：待实施
> 定位：学习项目。目标是完整经历"评价器在环 agent"的编写、组织与运作。
> 调图效果的打磨不在本期范围。目标用户：自用。

---

## 0. 文档说明

本设计在成稿过程中做过以下调整。这些内容最初是作为与早期草稿的差异记录的，
草稿已不在本仓库中，此处保留调整内容与原因，作为设计依据。

| # | 变更 | 原因 |
|---|---|---|
| 1 | 输入从"一句话风格词"改为任意自然语言指令，并新增"能力边界声明"环节 | 实际输入可能是"帮我进行 coser 界通用的基本照片调光"这类引用圈层惯例的指令 |
| 2 | 审美分不再作为优化目标，改为参考信号 + 护栏 | 审美分不可归因，盲 agent 无法据此推理 |
| 3 | 优化目标改为可归因的图像统计量（亮度/对比度/饱和度/冷暖） | 统计量单调、可归因、有设定值，盲 agent 能推理 |
| 4 | 引入审美评判工具（`pyiqa` 的 `nima`），并做成 LLM 主动调用的工具 | 提供不可归因的参考信号，制造"数值达标但观感变差"的真实冲突 |
| 5 | 交付哪一轮由 LLM 决策，harness 只做校验与质疑 | 这是 agent 决策的一环，不是"指出冲突" |
| 6 | 新增工具调用协议（动作类 + 工具类，两套预算） | 明确"哪些测量自动做、哪些交给 LLM 决定" |
| 7 | 新增区域统计量工具（可选） | 全局统计量看不出改动落在画面哪个区域 |
| 8 | 移除 `ask_user`，改用 CLI 层 `--confirm` 暂停 | 降低复杂度，效果相同 |
| 9 | 记录运行环境与代理配置 | 网络可达性直接决定工具能否安装运行 |

---

## 1. 项目定位与范围

### 1.1 目标

一张原图 + 一条自然语言指令 → 自动迭代调参 → 输出成片 + 完整报告。

学习的重点是评价器在环（evaluator-in-the-loop）agent 的完整工程结构：工具调用、状态管理、反馈回喂、安全回滚、边界退出，以及决策权在 LLM 与 harness 之间的划分。

### 1.2 本期做

- 任意自然语言指令 → 解读 → 制定目标统计量
- 5 个全局调整参数：亮度、对比度、饱和度、色温、伽马
- 6 个图像统计量：亮度、对比度、饱和度、冷暖、高光截断、暗部截断
- 1 个审美评判工具（NIMA，1~10 分）
- 工具调用协议：动作类（消耗轮次）+ 工具类（消耗工具预算）
- 完整 agent 循环：解读 → 目标 → 迭代 → 工具调用 → 边界退出 → 汇报
- 安全回滚与三种交付保证
- 完整的运行日志，可逐轮回放

### 1.3 本期不做

见第 12.4 节。

### 1.4 三条硬原则

**原则一：harness 负责一切算术，LLM 只负责判断。**

LLM 做算术不可靠；"算误差、算平均、判阈值"属于测量工作，不属于决策工作。所有数值计算放在 harness，LLM 的输出才能保持简短可靠。

**原则二：harness 里不允许出现任何替 LLM 做决定的代码。**

代码中不允许出现 `solve()`、不允许"自动挑误差最大的那一项去调"、不允许"误差够小就自动结束"。只要有一处代码替 LLM 回答了"下一步做什么"，系统就退化成普通脚本，本项目失去学习价值。

**原则三：harness 的边界动作是安全规则，不是优化决策。**

参数裁剪、边界退出、护栏触发的回滚，属于安全规则，由 harness 执行。LLM 的决策范围是：目标设定、每轮调参、工具调用、最终交付哪一轮。

---

## 2. 运行环境与已知事实

### 2.1 硬件

| 项目 | 值 |
|---|---|
| GPU | NVIDIA RTX 4070 Super（12 GB 显存），在 WSL 中通过主机代理联网 |
| CPU | 13th Gen Intel i7-13700K，24 线程 |
| 内存 | 15 GB（这是比显存更紧的资源） |
| 磁盘 | 约 789 GB 可用 |
| Python | 3.10.12 |

说明：本文档早期版本基于"无 GPU"的假设做了若干取舍，已作废。GPU 可用带来的主要变化是审美评判工具的成本从"每次 1~3 秒"降到"毫秒级"，相关设计影响见第 5.5 节。

### 2.2 网络与配置（规范化后的约定）

**配置只有两种方式：真实环境变量、或项目根目录的 `.env`**（模板 `.env.example`）。
优先级：环境变量 > `.env` > 代码默认值。
项目不读取任何自定义格式的密钥文件。

```bash
cp .env.example .env      # 然后填 DEEPSEEK_API_KEY
```

代理不由项目设置，它属于使用者机器的环境。

早期版本的 `env.sh` 会自动探测主机 IP 与端口并 export 三个代理变量。
该行为已移除，原因是它属于特定机器（WSL + 主机 Clash）的环境细节，
不应成为仓库的一部分 —— 换一台机器就完全不对，而且会让"项目为什么跑不起来"变得难以判断。

现在若需要代理，由使用者自行设置（或写进 `~/.bashrc`）：

```bash
export http_proxy="http://<主机>:<端口>"
export https_proxy="http://<主机>:<端口>"
```

> 代理即使只支持 SOCKS，也必须写成 `http://`。
> `huggingface_hub` / `timm` 走 httpx，`socks5://` 需要额外的 `socksio` 包，
> 否则会在连本地缓存都没检查前就抛 `ImportError`（详见 2.5 第 6 条）。

`requests`、`transformers`、`huggingface_hub`、`torch.hub` 都会自动读取
`http_proxy` / `https_proxy` 环境变量，无需在代码里额外设置。

缓存目录：`env.sh` 会把 `UV_CACHE_DIR` / `PIP_CACHE_DIR` / `HF_HOME` / `TORCH_HOME`
指到项目内，便于整体清理、不污染家目录。该行为是可选的 —— 系统默认的 `~/.cache`
在任何机器上都能正常工作，不需要的话把那一段注释掉即可。

### 2.3 网络可达性实测记录

下表为实测结果，用于日后排障。

| 目标 | 不挂代理 | 挂代理 | 说明 |
|---|---|---|---|
| `huggingface.co` | 超时不通 | 200 | 挂代理后直连可用 |
| `hf-mirror.com` | 200 | 200 | 不挂代理时的备用通道 |
| `pypi.org` | 200 | 200 | |
| `github.com` / `raw.githubusercontent.com` | 200 | 200 | |
| `api.deepseek.com` | 401 | 401 | 401 = 可达，仅缺认证 |
| `export.arxiv.org` | 不通 | 可用（需跟随 301） | 用于核实论文引用 |
| `openaipublic.azureedge.net` 的 ViT-B-32 | 200 | 200 | 存在 |
| `openaipublic.azureedge.net` 的 ViT-L-14 | 404 | 404 | 该文件确实已不存在，不是网络问题 |

结论与推论：

- 挂了代理之后，`huggingface.co` 直连即可用，不再必须依赖镜像。若某次没挂代理，用 `export HF_ENDPOINT=https://hf-mirror.com` 兜底。
- 凡依赖 OpenAI CLIP 官方 CDN 下载 ViT-L/14 的工具，在本机都会失败。受影响的有：
  - `pip install aesthetic_predictor`（内部调 `open_clip(..., pretrained="openai")`）
  - pyiqa 的 `clipiqa+_vitL14_512`
  - 本设计选用的 `nima`（Inception-ResNet-v2）不受影响，它从 HuggingFace 下载。

### 2.4 依赖安装

已实际执行并验证过的完整步骤：

```bash
# 1. 创建虚拟环境并安装全部依赖
uv sync                   # torch 由 [tool.uv.sources] 路由到 PyTorch 官方索引

# 2. 配置密钥
cp .env.example .env      # 填上 DEEPSEEK_API_KEY
```

`pyiqa` 的许可是 PolyForm Noncommercial 1.0.0：自用学习合规，不可商用。

三条需要遵守的约束：

1. torch 必须来自 PyTorch 官方索引而非 PyPI，否则会装到 CPU 版。
   `pyproject.toml` 中已用 `[tool.uv.sources]` 声明，`uv sync` 会自动处理；
   不使用 uv 时需先手动安装 CUDA 版 torch。装完后确认：

   ```bash
   .venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```

2. 受限环境下需要把 `TORCH_HOME` 指到可写目录。`pyiqa` 通过 `torch.hub.get_dir()`
   决定权重存放位置，默认落在 `~/.cache/torch`。正常机器上这就是正确行为，无需配置；
   只有当 `~/.cache` 不可写（容器、沙箱、权限受限）时才会报 `PermissionError`。
   `env.sh` 默认把它指到项目内（可选，见 2.2）。

3. `uv` 同理：默认写 `~/.cache/uv`，受限环境下才需要 `UV_CACHE_DIR`。

依赖在 `pyproject.toml` 中声明，确定的版本由 `uv.lock` 锁定。

### 2.5 已确认的问题（记录以免重复排查）

1. OpenAI CLIP CDN 的 ViT-L-14 返回 404（见 2.3）。任何需要它的工具直接放弃。
2. pyiqa 的 `clipiqa` 不接受自定义文本 prompt。其 `forward(self, x)` 只接收图像，prompt 对是硬编码的（`Good image`/`bad image`、`Sharp image`/`blurry image` 等共 10 条），`default_model_configs.py` 中也没有 `text_prompt` 选项。因此不能用它把风格词变成可优化标量。早期设计曾据此假设该用法可行，属于事实性错误。
3. `piq` 库的 `clipiqa` 同样硬编码 anchor（`Good photo.`/`Bad photo.`），换库解决不了上一个问题。
4. `pyiqa` 下载权重会写 `~/.cache/torch`，需要 `TORCH_HOME`（见 2.4 第 2 条）。
5. `nvidia-cusolver-cu12` 从 `pypi.nvidia.com` 下载容易超时。实测首次安装时该包重试 3 次后失败；重新执行同一条命令（已下载的走缓存）并设置 `UV_HTTP_TIMEOUT=600` 后成功。
6. `all_proxy` 不能设成 `socks5://`。`huggingface_hub` 与 `timm` 走 httpx，
   若 `all_proxy` 是 socks5 且未装 `socksio`，会在构造 HTTP 客户端时
   直接抛 `ImportError: Using SOCKS proxy, but the 'socksio' package is not installed`。
   该失败发生在检查本地缓存之前，所以即使权重已下载也会报错。
   对策：自行设置代理时用 `http://` 形式（Clash 的混合端口两种都支持），
   或者装上 `httpx[socks]`。项目本身不再设置代理（见 2.2）。
7. `timm` 会额外下载 Inception-ResNet-v2 主干。首次使用 `nima` 时，
   除了 pyiqa 自己的权重，`timm` 还会通过 `hf_hub_download` 拉一次主干权重，
   因此第 6 条那个 httpx 问题会在第一次运行时暴露。

### 2.6 环境配置验证记录

以下为实际执行结果，可作为日后排障的基线。

| 项目 | 结果 |
|---|---|
| Python | 3.10.12（`.venv`，由 `uv venv` 创建） |
| torch | `2.6.0+cu124`（编译针对 CUDA 12.4） |
| torchvision | `0.21.0+cu124` |
| pyiqa | `0.1.16` |
| 安装包总数 | 148（见 `uv.lock`） |
| `.venv` 体积 | 6.2 GB |
| NIMA 权重下载 | 成功。`NIMA_InceptionV2_ava-b0c77c00.pth`，208 MB，43 秒，来自 HuggingFace `chaofengc/IQA-PyTorch-Weights` |
| `nima` 的 `score_range` | `0, 10`（与 3.5 节描述一致） |
| `nima` 的 `lower_better` | `False`（即越大越好，与 3.5 节一致） |
| NIMA 单次推理耗时 | 0.10 秒（CPU，24 线程） |
| 冒烟测试 | 生成一张 384×384 渐变测试图，打分 5.1120，链路完整 |

关于 GPU：`torch.cuda.is_available()` 在当前沙箱内返回 `False`，原因是沙箱屏蔽了 GPU（`nvidia-smi` 报 `GPU access blocked by the operating system`），不属于安装问题。该确认需在非沙箱的终端中执行一次：

```bash
source env.sh --venv
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

关于 5.5 节的实测数据：NIMA 在 CPU 上仅需 0.10 秒，在 4070S 上会更快，这确认了 5.5 节的判断 —— "因为贵所以做成工具"的成本论据不成立，把审美分做成 LLM 主动调用的工具，理由只有"决策可见"与"学习目标"两条。因此 5.5 节的 `AESTHETIC_MODE` 对照开关有实测价值。

代码要求：设备选择必须写成与硬件无关的形式，不能硬编码 `cuda`：

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

否则在无 GPU 环境下（例如 CI 或沙箱）会直接崩溃。`create_metric` 与后续所有张量操作都要传这个 `DEVICE`。

---

## 3. 评价体系

### 3.1 为什么分三层

本节是评价体系的核心。区分标准是信号能否被"归因"：它的变化能否对应到某个具体的调整参数上。

| 信号 | 数值可读 | 动作可归因 | 有设定值（知道还差多远） | 噪声 | 角色 |
|---|---|---|---|---|---|
| 图像统计量 | 是 | 是，一个参数主要影响一个统计量 | 是，目标值是具体数字 | 零（确定性计算） | 优化目标 |
| 审美分 | 是 | 否，受构图/主体/光线共同影响 | 否，只知道"越高越好" | 有 | 参考信号 + 护栏 |

"可归因"是盲 agent 唯一的立足点。它看不见图，只能靠"我动了 X，数变了 Y"反推规律。信号不可归因时，它只能编造因果关系。

同时，审美分必须保留：它不可归因，但它和可归因的统计量可能互相冲突（例如统计量达标了但画面变灰）。该冲突没有公式解，这正是 LLM 需要做判断的地方。把审美分降级为"参考 + 护栏"的目的就是保留这个冲突。

### 3.2 第一层：图像统计量（优化目标）

六个统计量，全部由 numpy 直接计算，毫秒级，确定性。

| 名字 | 含义 | 典型范围 | 角色 |
|---|---|---|---|
| `luma` | 平均亮度 | 0 ~ 1 | 可优化目标 |
| `contrast` | 亮度标准差 | 0.10 ~ 0.35 | 可优化目标 |
| `saturation` | HSV 中 S 通道均值 | 0 ~ 1 | 可优化目标 |
| `warmth` | R 通道均值 − B 通道均值 | −0.3 ~ +0.3 | 可优化目标 |
| `clip_hi` | 亮度 > 0.98 的像素占比 | 0 ~ 1 | 护栏 |
| `clip_lo` | 亮度 < 0.02 的像素占比 | 0 ~ 1 | 护栏 |

前四个由 LLM 决定选哪几个、目标值多少；后两个不设目标，只用于判定本轮是否合法。

**误差的定义**：

```python
error_vector = {维度: 当前统计量 - 目标值 for 维度 in LLM选定的维度}
error        = mean(abs(v) for v in error_vector.values())    # 平均绝对误差
```

`error` 越小越好。它同时提供方向（哪一项是正还是负）和距离（差多少），因此收敛判据是明确的：`error < 0.02`。

第 0 轮（原图）也计算 `error`（用原图统计量对目标求误差），因此原图完整参与比较。

### 3.3 第二层：审美分（参考信号 + 护栏）

明确不参与的事：不进入 `error` 的计算，不作为优化方向。

参与的事：

1. 参考信号：第 0 轮自动测基线；后续轮次由 LLM 通过工具调用获取。写进上下文供 LLM 判断。
2. 护栏（硬）：`aes < 基线 − D_hard` → 本轮判为非法，回滚。
3. 护栏（软）：`aes < 基线 − D_soft` → 写进下一轮上下文，由 LLM 判断是否调整策略。

阈值 `D_soft` / `D_hard` 必须先标定（见 3.6），初始建议 `0.5` / `1.5`（1~10 尺度）。

### 3.4 第三层：图像护栏（硬规则）

| 条件 | 判定 |
|---|---|
| `clip_hi` 超过上限 | 本轮非法：高光死白 |
| `clip_lo` 超过上限 | 本轮非法：暗部死黑 |
| `contrast` 低于下限 | 本轮非法：画面发灰 |
| `aes < 基线 − D_hard` | 本轮非法：审美劣化超红线 |

**阈值一律以原图为基准**（实现时发现的修正，后又做了一次调整）：

```
clip_hi 上限  = 原图.clip_hi + max(0.03, 原图.clip_hi × 0.20)
clip_lo 上限  = 原图.clip_lo + max(0.05, 原图.clip_lo × 0.20)
contrast 下限 = min(0.10,  原图.contrast × 0.60)
```

阈值必须是相对的：只用绝对阈值时，一张本身就是低对比的照片（雾景、柔光人像）会在第 1 轮就被判"画面发灰"，
连续几轮即强制退出，即原图本身的特性使 agent 提前退出。改成相对阈值后：

- 相对原图恶化会被拦下；
- 原图本身就是低对比或有截断时不会被误判；
- 原图大面积过曝时，相对项生效（原图已有 23% 过曝，再多 2 个百分点是微不足道的变化）。

`CLIP_HI_DELTA` 定为 0.03 而非 0.02 的原因：实测中 agent 反复落在
`clip_hi = 0.0238 ~ 0.0240`，而上限是 0.0237，差 0.0001 量级就被判非法。
而"提亮 + 增艳"这个任务本身必然推高截断，0.02 窄到把合法工作也挡掉了。
3% 的像素打顶，对一张 2064×4608 的图约等于一个 535×535 的区域，仍是实质约束。

该参数与口味相关，可以在 `config.py` 里按自己的接受度调整。

非法轮次：不进候选交付集，harness 回滚到安全参数并告知 LLM。
这三条与审美硬红线的判定时机不同，见 9.1。

### 3.5 审美评判工具的选型与参考文献

#### 选型结论

选用 `pyiqa` 的 `nima` 指标。

```python
import pyiqa
metric = pyiqa.create_metric('nima', device='cuda')
score  = float(metric(img_path))      # 1 ~ 10，越大越好
```

- 实际模型：NIMA，Inception-ResNet-v2 主干，在 AVA 数据集上训练
- 权重文件：`NIMA_InceptionV2_ava-b0c77c00.pth`（来自 HuggingFace 仓库 `chaofengc/IQA-PyTorch-Weights`）
- 输出：1~10 的平均主观分（MOS）

#### 参考文献

```
[1] Talebi, H., & Milanfar, P. (2018). NIMA: Neural Image Assessment.
    IEEE Transactions on Image Processing, 27(8), 3998-4011.
    arXiv:1709.05424

[2] Ke, J., Wang, Q., Wang, Y., Milanfar, P., & Yang, F. (2021).
    MUSIQ: Multi-scale Image Quality Transformer. ICCV 2021.
    arXiv:2108.05997

[3] Chen, C., Mo, J., Hou, J., et al. (2024). TOPIQ: A Top-down Approach
    from Semantics to Distortions for Image Quality Assessment.
    IEEE Transactions on Image Processing, 33.
    DOI:10.1109/TIP.2024.3378466  ｜ arXiv:2308.03060

[4] Schuhmann, C., Beaumont, R., Vencu, R., et al. (2022).
    LAION-5B: An open large-scale dataset for training next generation
    image-text models. NeurIPS 2022, Datasets and Benchmarks Track.
    arXiv:2210.08402

[5] Wang, J., Chan, K. C. K., & Loy, C. C. (2023). Exploring CLIP for
    Assessing the Look and Feel of Images. AAAI 2023. arXiv:2207.12396
```

以上 arXiv 编号均已通过 arXiv API 核实标题一致。

[1] 是主选指标的依据。[2][3][4] 是同一个 `pyiqa` 包内可直接替换的备选，接口完全一致，切换只改一个字符串：

| 指标名 | 出处 |
|---|---|
| `nima` | [1]，AVA 训练，Inception-ResNet-v2 |
| `nima-vgg16-ava` | [1]，VGG16 主干 |
| `musiq-ava` | [2] |
| `topiq_iaa` | [3] |
| `laion_aes` | [4] |

[5] 记录我们为什么放弃"用 CLIPIQA 承载风格 prompt"这条路（见 2.5 第 2 条）。

> 待办：安装后核对 `pyiqa` 自带 `docs/ModelCard.md` 中各指标与论文的对应关系，并把结果补记到本节。

#### 为什么不用其他方案

| 方案 | 不采用的原因 |
|---|---|
| `pip install aesthetic_predictor`（LAION 审美预测器） | 内部走 OpenAI CDN 下载 ViT-L/14，该文件实测 404，本机必然失败 |
| 自己组装 LAION 预测器 | 可行，但只有社区模型、无独立论文，且训练数据含大量生成图，偏离"摄影审美" |
| `clipiqa` / `clipiqa+` | 无法接收自定义 prompt（见 2.5），且 `clipiqa+_vitL14_512` 依赖 404 的权重 |
| `qalign` / `qalign_4bit`（Q-Align，ICML 2024） | 是能看图的多模态大模型，与"无视觉"约束冲突；7B 级需 4bit 量化。留作 v2 的"昂贵复核工具" |
| 自己实现 CLIP 成对 prompt 打分 | 需要自己写度量，与"尽量调度现成工具"的定位冲突 |

### 3.6 审美分灵敏度标定（实施前必做）

NIMA 是确定性的（同图同分），不需要测重复性。需要测的是灵敏度：NIMA 主要由构图、主体、光线决定，
对纯色彩调整的敏感度可能远低于直觉。

做法：取一张测试照片（最好是 coser 人像），对五个参数各做 7 档单参数变化，共约 35 张，记录审美分的取值范围与标准差。GPU 上几十秒可完成。

产出：

| 产出 | 用途 |
|---|---|
| 典型变化幅度 | 决定"降多少算异常" |
| 噪声带 | 决定 `D_soft` / `D_hard` |

若标定发现"参数怎么变分数都不动"，则审美分只能当护栏、不能当参考信号。该结论本身要记进文档，因为它直接决定第 3.3 节是否成立。

标定结果出来后要做两件事：

1. 把 `D_soft` / `D_hard` 填进 `config.py`（本设计第 3.3 节的建议值只是起点）
2. 回填提示词里的数字。第 7.2 节提示词中"审美分工具"那段里的"正负 0.5 以内通常没有意义"是占位数值，实现时必须从 config 注入真实的噪声带，否则模型会拿一个编造的数字做判断。建议在 `llm.py` 里用模板变量（如 `{AES_NOISE_BAND}`）拼接，而不是写死在提示词字符串里。
   （已实现：`config.AES_NOISE_BAND` 与 `config.AES_LATENCY` 会注入 `SYSTEM_DECIDE_TEMPLATE`。）

### 3.7 标定实跑记录

`calibrate.py` 已在真实照片上跑过（`DSC07186(1).JPG`，6000×4000）。

原图基线：审美分 5.3824，luma 0.5885、contrast 0.3129、saturation 0.2411、warmth −0.0091，
clip_hi 0.1886（18.9% 的像素亮度超过 0.98，大面积高光溢出）。

实测耦合矩阵（每前进一个步长的影响量）：

| 参数 | luma | contrast | saturation | warmth | 审美分 |
|---|---|---|---|---|---|
| `brightness` | **+0.0939** | +0.0350 | −0.0089 | +0.0031 | +0.0583 |
| `contrast` | −0.0074 | +0.0500 | **+0.0741** | −0.0007 | +0.1932 |
| `saturation` | −0.0001 | −0.0001 | **+0.0385** | −0.0016 | −0.0266 |
| `color_temp` | +0.0178 | +0.0063 | −0.0005 | **+0.1886** | +0.1713 |
| `gamma` | **−0.0323** | +0.0154 | +0.0282 | −0.0025 | −0.0467 |

审美分灵敏度：

| 参数 | 最小 | 最大 | 全范围极差 | 一步影响 |
|---|---|---|---|---|
| `brightness` | 5.1392 | 5.5288 | 0.3896 | +0.0583 |
| `contrast` | 4.7491 | 5.6021 | 0.8530 | +0.1932 |
| `saturation` | 5.3553 | 5.5273 | 0.1719 | −0.0266 |
| `color_temp` | 5.3696 | 5.7537 | 0.3841 | +0.1713 |
| `gamma` | 5.2305 | 5.4106 | 0.1800 | −0.0467 |

据此确定的阈值：单步最大影响 0.1932，取
`D_SOFT = 0.29`（1.5 倍）、`D_HARD = 0.58`（3 倍）、`AES_NOISE_BAND = 0.29`。

两个结论：

1. NIMA 在真实照片上是敏感的（单步最大影响 0.19，全范围极差 0.85），
   远好于合成图（极差仅 0.33）。它作为参考信号是可用的。
   之前基于合成图的悲观判断已被推翻。
   这也说明标定必须使用真实照片，合成图的数字会误导。
2. `clip_hi = 0.1886`，这张照片大面积高光溢出，远超绝对阈值 0.02。
   若还用 3.4 节的绝对阈值，这张照片从第 1 轮起就永远非法、agent 立即退出。
   相对阈值（`max(0.02, 原图 + 0.02) = 0.2086`）使这次运行得以继续，
   这是 3.4 节那处修正的实证价值。

---

## 4. 调整层（工具）

### 4.1 五个参数定义

| 参数 | 含义 | 取值范围 | 单轮步长上限 | 实现 |
|---|---|---|---|---|
| `brightness` | 整体亮度偏移 | −0.6 ~ +0.6 | 0.2 | `ImageEnhance.Brightness(1 + b)` |
| `contrast` | 对比度偏移 | −0.6 ~ +0.6 | 0.2 | `ImageEnhance.Contrast(1 + c)` |
| `saturation` | 饱和度偏移 | −0.6 ~ +0.6 | 0.2 | `ImageEnhance.Color(1 + s)` |
| `color_temp` | R/B 通道增益，单位千分之二 | −400 ~ +400 | 400 | `R × (1 + t/2000)`，`B × (1 − t/2000)` |
| `gamma` | 中间调指数偏移 | −0.5 ~ +0.5 | 0.15 | `LUT: out = in ** (1 + g)` |

- `color_temp` 的单位定义为"千分之二"：`t = 400` 表示 R 通道 +20%、B 通道 −20%。
- `gamma > 0` 使中间调变暗，`gamma < 0` 使中间调变亮。
- 参数值是累计绝对值，不是增量。LLM 输出的是增量，harness 负责累加与裁剪。

### 4.2 固定应用顺序

```
color_temp  →  gamma  →  brightness  →  contrast  →  saturation
```

顺序会改变最终像素结果，因此它属于设计的一部分，不可随意调整。理由：先做白平衡（线性通道增益），再做影调曲线，最后做亮度/对比度/饱和度这类感知层调整。

### 4.3 adjust 的实现要点

```python
def adjust(src_path, params) -> out_path:
    img = 读入原图, 转 float32, 归一化到 0-1
    按 4.2 的顺序依次应用
    裁剪到 0-1, 转回 8bit, 写入 out_path
```

两条必须遵守：

1. **永远从原图重新渲染**：`候选图 = adjust(原图, 累计参数)`，不在上一轮的结果上叠加。这样任意一轮都能用它的参数向量完整重现，也不会累积量化误差。
2. 原图只读，输出写临时目录。

### 4.4 参数耦合（必须写进提示词）

本节最初的耦合表是推测的，已由 `tests/check_core.py` 与 `calibrate.py` 实测修正。
实测方法：对每个参数施加恰好一个步长，测量六个统计量的变化量。

实测矩阵（左：真实照片 `DSC07186(1).JPG`；括号内为合成测试图的对照值）：

| 动这个 | luma | contrast | saturation | warmth |
|---|---|---|---|---|
| `brightness` (+0.2) | **+0.0939** (+0.0704) | +0.0350 (+0.0210) | −0.0089 (−0.0004) | +0.0031 (+0.0004) |
| `contrast` (+0.2) | −0.0074 (−0.0002) | +0.0500 (+0.0210) | **+0.0741** (+0.0620) | −0.0007 (+0.0005) |
| `saturation` (+0.2) | −0.0001 (−0.0001) | −0.0001 (+0.0001) | **+0.0385** (+0.0530) | −0.0016 (+0.0002) |
| `color_temp` (+400) | +0.0178 (+0.0132) | +0.0063 (+0.0043) | −0.0005 (+0.0035) | **+0.1886** (+0.1413) |
| `gamma` (+0.15) | **−0.0323** (−0.0524) | +0.0154 (−0.0015) | +0.0282 (+0.0375) | −0.0025 (+0.0011) |

结构性耦合与题材相关性的区分（两张图的对照暴露了哪些结论不稳固）：

| # | 结论 | 跨题材是否稳定 | 说明 |
|---|---|---|---|
| 1 | `contrast` 对 `saturation` 的影响大于它对 `contrast` 本身的影响 | 稳定 | 两张图都成立（0.0741 > 0.0500；0.0620 > 0.0210）。原因是 PIL 的 `ImageEnhance.Contrast` 是"向亮度均值的灰底混合"，混向中性灰必然去饱和 |
| 2 | `brightness`/`contrast`/`saturation` 都会放大 `warmth` | 有条件 | 放大的是画面已有的 R−B 差值，因此原图偏暖/偏冷时明显，接近中性时可忽略。合成图 warmth 基线 +0.0985 → 三行增量都是 +0.0197；真实照片基线 −0.0091 → 增量只有 +0.0031。原设计把它写成无条件规律，是错的 |
| 3 | `color_temp` 会明显改变 `saturation` | 不稳定 | 合成图 +0.0983，真实照片仅 −0.0005。原设计把它当成稳定规律，是错的，已在提示词中改成"单位是千分"这样的确定事实，不再声明这条 |
| 4 | `brightness` 与 `gamma` 都改变 `luma`；`contrast` 与 `gamma` 都改变 `contrast` | 稳定 | 两张图都成立 |

结论：参数耦合的方向是结构性的，但幅度强烈依赖题材。
单张图上的实测矩阵只能用来核对量级，不能当成普适规律写进提示词。

### 4.5 截断的代价（实跑后发现的关键约束）

截断代价需要单独列出，因为它是"目标不可达"最常见的真实原因，而最初的耦合表完全漏掉了它。

测量方法：扫描该参数的全范围（±3 步），取 `clip_hi` 相对 0 档的最大增幅。
这里不能用中心差分 `(f(+1)−f(−1))/2`：`color_temp` 对截断的影响是 U 形的
（加暖与加冷都推高截断），中心差分会把两侧抵消掉，给出 −0.0334 这种误导性的小数字。
护栏类指标必须用"最大增幅"来度量。

| 参数 | `clip_hi` 最大增幅 | 含义 |
|---|---|---|
| `brightness` | **+0.3423** | 提亮最直接，但推高截断最猛 |
| `contrast` | +0.1781 | 同时推高亮部与暗部截断（`clip_lo` +0.0485/步） |
| `color_temp` | +0.1368 | 两个方向都会推高（R/B 总有一个通道被增益），偏冷更明显 |
| `saturation` | +0.0632 | 代价中等 |
| `gamma` | **+0.0122** | 几乎不影响截断 —— 想提亮又不想爆高光时首选它 |

该约束的重要性（以 `DSC07186(1).JPG` 为例）：

- 原图 `clip_hi = 0.2356`（23.6% 的像素至少有一个通道打顶，大面积高光溢出）
- 护栏只允许再增加 `CLIP_HI_DELTA = 0.02`
- 而 `brightness` 全范围的增幅可达 0.3423 → 有效余量只有约 0.06 个全范围步长

也就是说，在这张照片上，几乎任何幅度的 `brightness` 提亮都会立刻触发高光护栏。
agent 想把 `luma` 从 0.5885 提到 0.65，需要付出远超 0.02 的 `clip_hi`；
该目标在护栏约束下不可达，但只看参数范围会误判为"完全可达"。

正确做法是改用 `gamma` 负值提亮中间调：`clip_hi` 增幅仅 +0.0122，代价极低。

### 4.6 一处 metric 定义 bug，以及它如何污染了 agent 的推理

该问题的特点是：错误的指标不会引发异常，只会使 LLM 在错误的前提上做出正确形式的推理。

现象：实测发现 `color_temp` 的 +400 与 −400 两个相反方向都把 `clip_hi` 变成 0.0000。
两个相反的操作产生完全相同的极端结果，物理上不可能。

根因：原来的 `clip_hi` 定义为"加权亮度 > 0.98 的像素占比"。
一个纯白区域 `(1, 1, 1)` 在 `color_temp +400` 后变成 `(1, 1, 0.8)`，
加权亮度 = 0.299 + 0.587 + 0.114×0.8 = 0.977 < 0.98，
此时仍有一个通道打顶，却因为加权亮度掉到阈值以下而被判为"没有截断"。

后果链条：

1. 依据这个错误数据，提示词里写下了"偏暖会显著降低高光截断"这条错误规律；
2. agent 读到这条规律，在第 3 轮主动选择 `color_temp +150` 来"降低截断"，
   理由是"偏暖会降低高光截断（当前 clip_hi 0.2027 已有一段余量）"；
3. 结果是当场触发高光死白被回滚（clip_hi 0.2192 > 0.2086），
   此后第 5、7、9 轮又反复触发同一护栏。

修正：`clip_hi` / `clip_lo` 改为判任一通道触顶/触底：

```python
clip_hi = (arr.max(axis=-1) > 0.98).mean()   # 任一通道打顶
clip_lo = (arr.min(axis=-1) < 0.02).mean()   # 任一通道触底
```

数字图像里的"截断"就是指某个通道打到编码上限（255），与感知亮度无关。

修正后的数据变得合理：原图 `clip_hi` 从 0.1886 变为 0.2356；
`color_temp +400` 从 −0.1886 变为 +0.0697（升高，符合物理）。

结论：

- 指标定义错误时，agent 的推理会以正确的形式建立在错误的前提上 ——
  它本身没有出错，问题来自输入给它的数值。这类问题不会抛异常，表现为 agent 反复触发同一护栏。
- 排查手法：怀疑指标时先做"对称性检查" ——
  两个物理上相反的操作若给出相同结果，几乎一定是指标或实现有问题。
- 这也说明 agent 的失败模式需要逐轮留档：
  如果没有 `llm_calls.jsonl` 里 agent 那句"偏暖会降低高光截断"，
  很难把问题追溯回指标的实现。

因此提示词里增加了两条（见 `config.COUPLING_TABLE` / `COUPLING_NOTES`）：

1. `brightness` 是推高截断最猛的参数；原图已有高光溢出时它的可用余量极小甚至为零，
   此时应改用 `gamma` 负值。
2. 截断判的是"任一通道打顶"，加暖或加冷都可能推高截断，色温同样有代价。

`calibrate.py` 的耦合矩阵也已补上 `clip_hi` / `clip_lo` 两列，
并直接算出"还剩几个步长的余量" —— 这是判断目标可达性的关键数字。

核对方式：`tests/check_core.py` 第 2 节会打印合成图矩阵；
`python -m mytone.calibrate -i 你的照片.jpg` 会在真实照片上测同一张表，
并逐参数标出"实际影响最大的统计量"是否与本节声明一致。

---

## 5. 工具调用协议

### 5.1 分界原则

> **没有决策的地方，由 harness 自动执行；有决策的地方，做成工具交给 LLM。**

| 行为 | 有没有决策 | 归属 |
|---|---|---|
| 每轮渲染候选图 | 无：给定参数必然要渲染 | harness 自动 |
| 每轮测六个统计量 | 无：每轮必然需要 | harness 自动 |
| 第 0 轮测审美基线 | 无：基线必然要建 | harness 自动 |
| 后续轮次测审美分 | 有 | 工具，LLM 决定 |
| 测区域统计量 | 有 | 工具，LLM 决定 |

设计结论：不要把能力全塞给 LLM 当工具，那会浪费决策预算；也不要把事全写死在流程里，那就没有 agent 了。

### 5.2 两类输出

**动作类（消耗轮次预算）**

```json
{"adjust": {"color_temp": -400, "saturation": -0.10}, "reason": "依据"}
{"revise_target": {"luma": 0.58}, "reason": "依据"}
{"finish": "goal_met", "deliver_round": 2, "reason": "依据"}
```

**`revise_target` 的用途与约束**（见 15.7）：

当 LLM 发现自己设定的目标在参数能力范围内确实不可达时（典型信号：同一护栏反复触发），
它需要一个出口来修正自己的判断，而不是持续触发同一护栏。约束有三条：

1. 消耗轮次预算 —— 与 `adjust` 一样计一次，因此 `rounds_used = 调整次数 + 修正次数`。
   否则 agent 可以靠不断改目标来假装收敛。
2. 次数上限 `TARGET_REVISION_LIMIT = 2`。
3. 修正后所有历史轮次的误差按新目标重算，并刷新 `best_by_error`；
   修正记录写入 `target_revisions`，报告里单独列出。

prompt 里明确要求不得把目标改成"已经达到的值"来假装收敛，必须说明原目标为什么不可达。

**工具类（消耗工具预算，不消耗轮次）**

```json
{"tool": "aesthetic", "target": "current", "reason": "依据"}
{"tool": "aesthetic", "target": 2, "reason": "依据"}
{"tool": "region", "region": "center", "reason": "依据"}
```

工具类输出后，harness 执行工具、把结果写入"最近工具结果"缓冲区、轮次计数不变，然后重新请求 LLM 决策。

### 5.3 预算

| 预算 | 常量 | 上限 | 作用 |
|---|---|---|---|
| 轮次 | `ROUND_LIMIT` | 10 | 限制调参次数（不含第 0 轮，见 9.3） |
| 工具调用 | `TOOL_BUDGET` | 4 | 限制"只观测不行动" |

两套预算独立计算。工具调用不消耗轮次，因此必须另设 `TOOL_REJECT_LIMIT`（建议 3）来兜住"预算耗尽后仍反复请求工具"的情况 —— 否则该分支会绕开全部边界检查而成死循环（见 9.3 第 7 条）。

### 5.4 工具清单

| 工具 | 参数 | 返回 | 消耗 | 必需 |
|---|---|---|---|---|
| `aesthetic` | `target`：`"current"` 或轮次号 | 审美分、与基线差值、是否越过软/硬阈值 | 1 | 是 |
| `region` | `region`：`"top"`/`"middle"`/`"bottom"`/`"center"` | 该区域 luma / contrast / saturation / warmth | 1 | 否 |

`aesthetic` 的 `target` 参数语义

- `"current"` 指最近完成的那一轮的候选图（即当前参数对应的图）。
- 整数轮次号指指定的历史轮次，用于回看早期结果。
- 测量结果一律回溯写入被测量轮次的 `aes` / `aes_delta`，而不是写入"正在决策的那一轮"。否则 9.4 的交付校验会找错对象。

`region` 补的是全局统计量最大的盲区：全局均值看不出改动集中在天空、背景还是人物。做 coser 人像时价值较高。约 15 行代码，先不做也不影响其他部分。

### 5.5 关于"GPU 让审美分变便宜后，还要不要做成工具"

原来的成本理由不成立：4070S 上 NIMA 是毫秒级，完全可以每轮自动测。

保留工具化的三个理由：

1. 上下文经济：每轮把所有权值都塞进上下文，token 与噪声都会累积。
2. 决策可见：做成工具后，"什么时候需要复核画面"这一判断在日志里是显式的；自动测的话这个判断根本不存在。
3. 学习目标：本项目的重点之一是工具调用。

因此保留工具化，同时提供配置开关：

```python
AESTHETIC_MODE = "tool"    # "tool" | "auto"
```

`auto` 模式下 harness 每轮自动测量并写入上下文，`aesthetic` 工具不再可调用。

该开关对应一次对照实验：同一张图跑两种模式，可以直接观察"LLM 自主决定观测"与"数据自动喂到位"对决策质量的影响。

### 5.6 强制工具使用规则

> **调用 `finish` 时，必须已经对所选交付轮次成功调用过一次 `aesthetic`。**

未确认就 finish，harness 拒绝该 finish，附提示"请先对第 N 轮调用 aesthetic"，重新请求（不消耗轮次）。被拒超过 2 次则退出，原因记为 `finish_rejected`。

这条规则保证每次运行至少有一次真实的工具调用，使工具调用成为必经环节。

`AESTHETIC_MODE="auto"` 时本条自动满足：harness 每轮都已测过审美分，所以每次 `finish` 都能通过这一校验。此时 `aesthetic` 工具不出现在工具清单里，LLM 若请求它，按"工具名不在清单内"处理（打回重试）。

---

## 6. 状态设计

```python
@dataclass
class Round:
    index: int                    # 轮次，从 0 开始，第 0 轮是原图
    params: dict                  # 该轮的累计绝对参数
    requested_delta: dict         # LLM 请求的增量（原样记录）
    effective_delta: dict         # 裁剪后真实生效的增量
    stats: dict                   # 六个统计量
    error: float | None           # 平均绝对误差（第 0 轮在目标确定后补算）
    error_vector: dict            # 每维的 当前值 - 目标值
    guard_ok: bool                # 是否通过全部护栏（审美硬红线可在之后被回溯改写）
    guard_msg: str                # 未通过的原因
    aes: float | None             # 审美分（只在被测量过的轮次有值）
    aes_delta: float | None       # 与基线的差
    llm_reason: str               # 该轮 LLM 给出的依据原文
    image_path: str               # 候选图路径

@dataclass
class TaskState:
    instruction: str              # 用户原始指令，一字不改
    interpretation: str           # LLM 对指令的解读
    can_do: list[str]             # LLM 认为五个参数能做到的部分
    cannot_do: list[str]          # LLM 认为做不到的部分
    original_path: str
    original_stats: dict
    aes_baseline: float           # 第 0 轮自动测得的审美基线
    target: dict                  # 目标统计量
    target_reason: str
    params: dict                  # 当前累计参数
    rounds: list[Round]
    best_by_error: int            # 误差最小的合规轮次号（harness 维护，安全参考）
    delivered: int | None         # LLM 最终指定的交付轮次
    tool_log: list[dict]          # 每次工具调用的入参与结果
    tools_used: int
    pending_warning: str | None   # 给下一轮的提示，组装上下文后立即清空
    reject_count: int             # deliver_round 被判定"不合法"的次数（见 9.4）
    challenged: set[int]          # 已被质疑过的交付轮次号；同一轮只质疑一次
    tool_reject_count: int        # 工具预算耗尽后仍继续请求工具的次数
    exit_reason: str
    report_path: str
```

### 关键设计点

0. 轮次编号与列表下标一致。
   `rounds[i].index == i` 恒成立（轮次只追加、不回退），因此下文用轮次号直接索引 `state.rounds[n]`。第 0 轮是原图。

1. `requested_delta` 与 `effective_delta` 必须分开记录。
   如果请求的增量被边界裁掉了，LLM 必须知道自己的动作没有完全生效，否则下一轮的判断会建立在错误的前提上。

2. 第 0 轮也计算 `error`。
   用原图统计量对目标求误差。这样原图完整参与比较，并支撑"交付不会比原图更差"的检查。

3. `best_by_error` 与 `delivered` 的区别。

   - `best_by_error`：误差最小的合规轮次，由 harness 维护，只作为安全参考和校验基准，不是最终交付。
   - `delivered`：由 LLM 决策指定交付哪一轮。

   这是 v2.0 的关键变更（见 9.4）。

4. 轮次与工具调用是两套独立计数。
   工具调用不产生 `Round`，因此轮次号只随 `adjust` 递增。同一次决策里 LLM 可能先调用若干工具、再给出 `adjust`，这些都不影响轮次号。日志与上下文中必须把两者分开标注，否则会误判进度。

---

## 7. LLM 调用契约

整个系统只发起两类 LLM 请求。

### 7.1 调用一：解读指令 + 声明能力边界 + 制定目标

#### system prompt

```
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
```

#### 输入示例

```
用户指令：帮我进行coser界通用的基本照片调光

原图统计量：luma 0.48, contrast 0.26, saturation 0.36, warmth +0.08
            clip_hi 0.002, clip_lo 0.030
原图审美分：5.41

可选统计量：
  luma        平均亮度        合理范围 0.30 ~ 0.75
  contrast    亮度标准差      合理范围 0.12 ~ 0.32
  saturation  饱和度均值      合理范围 0.05 ~ 0.60
  warmth      R均值-B均值     合理范围 -0.15 ~ +0.15

可用参数：brightness, contrast, saturation, color_temp, gamma
          （全部为全局参数，无局部处理能力）
```

#### 输出示例

```json
{
  "interpretation": "理解为 cosplay 人像的通用基础调光：整体提亮、去灰提升通透感、让肤色显得白净、色调统一干净",
  "can_do": [
    "整体提亮",
    "提升对比去掉灰蒙感",
    "通过整体降低饱和度和略微偏冷让肤色显白净",
    "统一整体色调"
  ],
  "cannot_do": [
    "磨皮与皮肤平滑（需要局部处理）",
    "单独压暗背景（需要蒙版）",
    "单独提亮面部（需要人脸识别或蒙版）",
    "眼睛与妆容强化（需要局部处理）"
  ],
  "target": {"luma": 0.58, "contrast": 0.30, "saturation": 0.30, "warmth": 0.02},
  "reason": "原图偏暗（luma 0.48）且偏暖（warmth +0.08）。coser 通用调光需要提亮到 0.58；对比度从 0.26 提到 0.30 去灰；饱和度从 0.36 降到 0.30、暖度从 +0.08 降到 +0.02 让肤色白净。四项都在合理范围内。"
}
```

#### harness 的校验

- 三个字段 `interpretation` / `can_do` / `cannot_do` 必须存在且类型正确（字符串、字符串数组、字符串数组）
- `target` 的维度名必须在清单内；出现清单外的维度名 → 打回重试（因为这代表模型在自造指标）
- `target` 的数值越界 → 裁剪到合法区间并记日志，不重新请求
- `target` 为空对象 → 打回重试
- 任何一项重试 2 次仍不通过 → 退出，原因 `llm_output_invalid`

#### 输出要打印并写入报告

`interpretation` / `can_do` / `cannot_do` 是指令含糊时唯一能让用户发现理解偏差的地方。CLI 提供 `--confirm` 开关，开启后在这一步暂停等待用户确认。

### 7.2 调用二：每轮决策

#### system prompt

```
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
  {"adjust": {"参数名": 增量, ...}, "reason": "依据"}
  {"finish": "goal_met" 或 "converged" 或 "best_effort",
   "deliver_round": 轮次号, "reason": "依据"}

工具类（不消耗轮次，消耗工具次数）：
  {"tool": "aesthetic", "target": "current", "reason": "依据"}
  {"tool": "aesthetic", "target": 轮次号, "reason": "依据"}
  {"tool": "region", "region": "top|middle|bottom|center", "reason": "依据"}

参数含义与影响：
  brightness  整体亮度      主要影响 luma
  gamma       中间调明暗    同时影响 luma 和 contrast（大于 0 变暗）
  contrast    对比度        主要影响 contrast，过大会推高 clip_hi
  saturation  饱和度        主要影响 saturation
  color_temp  冷暖          主要影响 warmth，正值变暖

必须注意的耦合：
  brightness 和 gamma 都改变 luma，方向相同，同时用会冲过头
  contrast 和 gamma 都改变 contrast

工具：aesthetic
- 返回 1 到 10 的分数，越大越好
- 来自 NIMA 模型，训练数据是 AVA（多人对真实摄影作品打分的公开数据集）
- 衡量"这张照片作为一张照片的整体观感"，主要由构图、主体、光线、清晰度决定，
  对纯色彩调整的敏感度较低
- 因此：正负 0.5 以内的波动通常没有意义，不要据此改变判断
- 它只告诉你整体观感是否变差，不能告诉你该调哪个参数
- 基线是原图的分数；只有明显低于基线时这个信息才有价值
- 调用有耗时开销，并且占用工具调用预算，请在有必要时调用

工具：region
- 返回画面某个区域（上/中/下/中心）的 luma、contrast、saturation、warmth
- 用途：判断你的调整主要落在哪里，例如天空、背景还是人物
- 全局统计量看不出改动集中在哪个区域，这是补充信息

关于交付：
- finish 时必须给出 deliver_round，指定你希望交付哪一轮的结果
- 你可以选择任何通过护栏的轮次，不一定是最后一轮，也不一定是误差最小的那一轮
- 如果"误差最小"和"观感最好"不是同一轮，由你判断并做出选择，并在 reason 里说明理由
- 交付前必须已经对该轮调用过 aesthetic，否则 finish 会被拒绝

规则：
- 每次只动 1 到 2 个参数，增量不得超过步长上限，超过会被裁剪
- 参数被裁剪时你请求的增量不会完全生效，给你的是裁剪后的真实生效值，以它为准则
- 你的调整必须始终服务于用户原始指令。如果统计量已达标但你认为结果偏离了指令本意，
  应当在 reason 里说明
- reason 必须引用具体数值
- 不要输出 JSON 以外的任何内容
```

#### 第 9、11 条的作用

防止"只盯着数字优化、忘了用户原本想要什么"。

#### 提示词里的数字必须从 config 注入

上面"审美分工具"一节中的"正负 0.5 以内"和"调用有耗时开销"都是占位内容，实现时必须替换为：

| 占位 | 来源 |
|---|---|
| 噪声带（正负 0.5） | 3.6 节标定得到的噪声带，或 `D_soft` |
| 调用耗时 | 3.6 节实测的 GPU 单次耗时 |

建议在 `llm.py` 中用模板变量拼接（如 `{AES_NOISE_BAND}` / `{AES_LATENCY}`），不要写死在字符串里，否则标定结果无法反映到提示词，模型会拿编造的数字做判断。

#### 每轮输入的实际组装格式

```
用户原始指令：帮我进行coser界通用的基本照片调光
你的解读：理解为 cosplay 人像的通用基础调光：整体提亮、去灰提升通透感、
          让肤色显得白净、色调统一干净
你声明的能力边界：
  能做：整体提亮 / 提升对比去灰 / 整体降饱和略偏冷 / 统一色调
  做不到：磨皮 / 单独压暗背景 / 单独提亮面部 / 眼睛与妆容强化

目标（依据：原图偏暗且偏暖，coser 通用调光需提亮去灰、肤色白净）：
  luma 0.58 ｜ contrast 0.30 ｜ saturation 0.30 ｜ warmth 0.02

当前累计参数：brightness 0.18, contrast 0.05, saturation -0.06, color_temp -120, gamma 0.00
当前统计量：  luma 0.57, contrast 0.30, saturation 0.30, warmth 0.02
              clip_hi 0.009, clip_lo 0.018
误差向量：    luma -0.01, contrast 0.00, saturation 0.00, warmth 0.00
平均误差：    0.003
护栏状态：    正常

审美分：基线 5.41 ｜ 已测：第 1 轮 4.68（-0.73，越过软警告阈值 0.5）

历次轮次汇总：
  轮次  平均误差  审美分  护栏
  0     0.092    5.41    通过
  1     0.003    4.68    通过

最近三轮详细轨迹：
  第 1 轮：请求 brightness +0.18, contrast +0.05, saturation -0.06, color_temp -120
           生效 同请求（无裁剪）
           误差 0.092 → 0.003
  第 2 轮：（无）

最近工具调用：
  aesthetic(current) → 4.68（基线 5.41，差 -0.73）

轮次：第 2 轮 / 共 10 轮 ｜ 剩余工具次数：3
提示：误差已达标，建议 finish
```

### 7.3 校验与重试

| 检查项 | 不通过处理 |
|---|---|
| JSON 可解析 | 附错误信息重试，最多 2 次 |
| 顶层键是 `tool` / `adjust` / `finish` 之一 | 同上 |
| `adjust` 参数名合法、增量为数字 | 同上 |
| `tool` 名字在清单内、参数合法 | 同上 |
| 工具预算充足 | 附提示重新请求（不消耗预算） |
| `finish` 含合法 `deliver_round` | 附提示重新请求 |
| `deliver_round` 指向的轮次通过护栏 | 附提示重新请求 |
| `deliver_round` 对应的轮次已测审美分 | 附提示重新请求 |
| 重试 2 次仍失败 | 退出，原因 `llm_output_invalid` |

---

## 8. 主循环与决策权划分

### 8.1 决策权清单

这张表是本设计的核心。写代码时逐条对照。

| # | 事项 | 决策者 | 说明 |
|---|---|---|---|
| 1 | 如何理解用户指令 | LLM | 输出 `interpretation` |
| 2 | 哪些要求做不到 | LLM | 输出 `cannot_do` |
| 3 | 优化哪些统计量、目标值多少 | LLM | 输出 `target` |
| 4 | 每轮动哪个参数、动多少 | LLM | 输出 `adjust` |
| 5 | 什么时候需要额外观测 | LLM | 输出 `tool` |
| 6 | 什么时候结束 | LLM（或边界强制） | 输出 `finish` |
| 7 | 交付哪一轮 | LLM | 输出 `deliver_round` |
| 8 | 参数累加与裁剪 | harness | 安全规则 |
| 9 | 计算统计量、误差、所有数值 | harness | 测量工作 |
| 10 | 护栏判定与回滚 | harness | 安全规则 |
| 11 | 轮次/工具预算与硬上限 | harness | 安全规则 |
| 12 | 校验 LLM 输出合法性 | harness | 校验工作 |

第 7 条是 v2.0 的关键变更：交付轮次由 LLM 决策，而不是 harness 按误差自动选。

### 8.2 主循环

```python
def run(instruction, original_path):
    state = TaskState(instruction=instruction, original_path=original_path)

    # ---------- 阶段 1：测量原图 ----------
    state.original_stats = stats(original_path)
    state.aes_baseline   = aesthetic_score(original_path)     # 自动，无需决策
    r0 = Round(index=0, params=全零, stats=state.original_stats,
               error=None, guard_ok=True,
               aes=state.aes_baseline, aes_delta=0.0)
    state.rounds.append(r0)

    # ---------- 阶段 2：解读指令 + 制定目标（调用一） ----------
    reply = validate_target(call_llm_interpret(state))
    state.interpretation = reply["interpretation"]
    state.can_do         = reply["can_do"]
    state.cannot_do      = reply["cannot_do"]
    state.target         = reply["target"]
    state.target_reason  = reply["reason"]
    log(reply)

    # 第 0 轮误差必须在目标确定后才能算
    state.rounds[0].error_vector = {k: state.original_stats[k] - v
                                    for k, v in state.target.items()}
    state.rounds[0].error = mean(abs(v) for v in state.rounds[0].error_vector.values())
    state.best_by_error   = 0

    if CONFIRM:
        print_interpretation(state)
        input("回车继续，Ctrl-C 中止")

    # ---------- 阶段 3：迭代 ----------
    while True:
        ctx = build_context(state)         # 组装时要读取 pending_warning，随后清空
        state.pending_warning = None
        reply = validate_action(call_llm_decide(ctx))      # 最多重试 2 次

        # 3A 工具类：执行工具，不消耗轮次
        #     这条分支不推进轮次，因此必须自带次数上限，
        #     否则 LLM 反复请求工具就会绕开全部边界检查而死循环。
        if "tool" in reply:
            if state.tools_used >= TOOL_BUDGET:
                state.tool_reject_count += 1
                state.pending_warning = "工具预算已用尽，请直接 adjust 或 finish"
                if state.tool_reject_count > TOOL_REJECT_LIMIT:      # 建议 3
                    state.exit_reason = "tool_budget_exhausted"
                    state.delivered   = state.best_by_error          # 安全网
                    break
                continue
            result = run_tool(reply, state)     # 内部把结果回溯写入被测量轮次
            state.tool_log.append({"call": reply, "result": result})
            state.tools_used += 1
            apply_aesthetic_guard(state)        # 见 9.1：审美硬红线回溯判定
            continue

        # 3B 结束：校验交付轮次
        if "finish" in reply:
            n = reply.get("deliver_round")
            kind, msg = check_delivery(state, n)     # 见 9.4：返回 "ok" / "challenge" / "reject"
            if kind == "challenge":
                state.challenged.add(n)              # 同一轮只质疑一次
                state.pending_warning = msg
                continue
            if kind == "reject":
                state.reject_count += 1
                state.pending_warning = msg
                if state.reject_count > 2:
                    state.exit_reason = "finish_rejected"
                    state.delivered   = state.best_by_error      # 安全网
                    break
                continue
            state.exit_reason = reply["finish"]
            state.delivered   = n
            break

        # 3C 调参
        new_params, effective = clamp_params(state.params, reply["adjust"])
        img_path = adjust(state.original_path, new_params)
        s        = stats(img_path)
        ev       = {k: s[k] - state.target[k] for k in state.target}
        err      = mean(abs(v) for v in ev.values())
        # 此刻只判三条图像护栏；审美硬红线要等 aes 被测出后才判（9.1）
        guard_ok, guard_msg = check_guard(s, state)

        r = Round(index=len(state.rounds), params=new_params,
                  requested_delta=reply["adjust"], effective_delta=effective,
                  stats=s, error=err, error_vector=ev,
                  guard_ok=guard_ok, guard_msg=guard_msg,
                  llm_reason=reply["reason"], image_path=img_path)
        state.rounds.append(r)

        if state.aesthetic_mode == "auto":          # auto 模式每轮自动测
            measure_aesthetic(state, r.index)
            apply_aesthetic_guard(state)

        refresh_best_by_error(state)                # 只在 guard_ok 的轮次里取误差最小者

        d = check_boundaries(state)                 # 内部负责设置 exit_reason
        if d == "exit":
            if state.delivered is None:
                state.delivered = state.best_by_error      # 安全网
            break
        if d == "rollback":
            state.params = state.rounds[state.best_by_error].params
            state.pending_warning = f"第 {r.index} 轮触发护栏（{guard_msg}），已回滚"
            continue
        state.params = new_params

    # ---------- 阶段 4：交付 ----------
    finalize_outputs(state)   # 复制交付图为 delivered.jpg，删除中间图与临时目录（见 11.1）
    write_report(state)
    return run_dir / "delivered.jpg", state.report_path
```

### 8.3 每轮时序（`adjust` 类输出）

1. harness 组装上下文（当前参数、统计量、误差、护栏、审美、历次汇总、最近三轮、工具结果、预算、提示）
2. harness 调用 LLM
3. harness 解析校验（不合法则附错误重试，最多 2 次）
4. 分支：
   - `tool` → 检查预算 → 执行 → 记录 →（若预算已耗尽则计入 `tool_reject_count`，超限即退出）→ 回到第 1 步，轮次不增
   - `finish` → 校验交付轮次（见 9.4）→ `ok` 则退出；`challenge` 则回问确认；`reject` 则计数
   - `adjust` → 继续第 5 步
5. 叠加增量并裁剪到合法区间，同时记录请求值与生效值
6. 从原图重新渲染候选图
7. 测六个统计量
8. 算误差向量与平均绝对误差
9. 判护栏。此刻只判三条图像护栏；审美硬红线要等该轮的 aes 被测出后才回溯判定（见 9.1）
10. 记录本轮，落盘候选图
11. 更新 `best_by_error`（仅当通过护栏且误差更小）
12. 判边界：退出／回滚／给提示
13. 继续则回到第 1 步

第 3、5、7、8、9、11、12 步全部是 harness 在做事，没有一步涉及"选哪个参数、调多少、要不要调工具、交付哪一轮"。

---

## 9. 边界、护栏与退出

### 9.1 护栏

三条图像护栏在轮次创建时判定；审美硬红线在该轮 aes 被测出时回溯判定。

| 条件 | 级别 | 判定时机 | 动作 |
|---|---|---|---|
| `clip_hi` 超过上限 | 硬 | 轮次创建时 | 本轮非法：高光死白 |
| `clip_lo` 超过上限 | 硬 | 轮次创建时 | 本轮非法：暗部死黑 |
| `contrast` 低于下限 | 硬 | 轮次创建时 | 本轮非法：画面发灰 |
| `aes < 基线 − D_hard` | 硬 | aes 被测出时 | 本轮非法：审美劣化超红线 |
| `aes < 基线 − D_soft` | 软 | aes 被测出时 | 写进下一轮上下文，由 LLM 判断 |

前三条的阈值按 3.4 节的相对规则计算（取绝对阈值与相对原图容差中较宽松者）。

审美硬红线必须回溯判定的原因：审美分是按需测量的（见 5.4），创建某一轮时通常还不知道它的 aes。因此每次测出 aes 后执行 `apply_aesthetic_guard(state)`：

1. 把结果写入被测量轮次的 `aes` / `aes_delta`
2. 重新判定该轮的 `guard_ok`；若越过硬红线，将该轮标为非法并写入 `guard_msg`
3. 调用 `refresh_best_by_error(state)` 重算（该轮可能不再是候选）
4. 若被标为非法的正是当前参数所在轮，立即回滚到 `best_by_error` 的参数，并把回滚原因写入 `pending_warning`

硬护栏触发：该轮不进候选交付集；harness 回滚到 `best_by_error` 的参数，并告知 LLM 回滚原因。连续 2 轮触发则强制退出。这里"连续"按轮次判定，且回溯改写的 `guard_ok` 也计入。

### 9.2 收敛与平台期判据

```python
error       = mean(|当前统计量 - 目标值|)     # 只在 LLM 选定的维度上
improvement = error(上一轮) - error(本轮)     # 正数表示变好
```

| 情形 | 条件 | 动作 |
|---|---|---|
| 达标 | `error < 0.02` | 下一轮上下文加提示"已达标，建议 finish" |
| 达标后仍继续且不再改善 | 已达标，且连续 2 轮 `improvement < 0.001` | 强制退出，原因 `stalled_after_goal` |
| 未达标且陷入平台 | 未达标，且连续 2 轮 `improvement < 0.001` | 强制退出，原因 `plateau` |

两条互斥：先看是否达标，再决定记为哪个原因，避免已达标却被误记为平台期。

`IMPROVE_EPS` 的取值（经实跑修正）：最初取 `0.005`，一次真实运行在
`0.0286 → 0.0241 → 0.0200` 稳步收敛的过程中被误判为平台期、提前 2 轮掐断。
真正"卡住"的表现是改善量接近 0 或为负，所以改成 `0.001`。
改后同一条指令的另一次运行正常收敛到 `0.0131`。

原因命名的修正：`stalled_after_goal` 原来是 `converged`，
但 `converged` 同时是 LLM 主动 finish 时可给的三种状态之一，
导致报告里同时出现"由安全网兜底"和"LLM 主动结束"两句矛盾的话。
harness 强制退出的原因名不得与 LLM 状态名重名。

### 9.3 退出条件与优先级

按顺序检查，命中即退出。`exit_reason` 由 `check_boundaries` 内部设置。

| 优先级 | 条件 | `exit_reason` |
|---|---|---|
| 1 | 新增调整轮次达到上限（见下） | `round_limit` |
| 2 | 连续 `GUARDRAIL_STREAK_LIMIT`（=3）轮护栏非法 | `guardrail_violation` |
| 3 | 全部参数裁剪后增量为 0（顶死在边界） | `no_move_possible` |
| 4 | 未达标且连续 2 轮 `improvement < 0.001` | `plateau` |
| 5 | 已达标且连续 2 轮 `improvement < 0.001` | `stalled_after_goal` |
| 6 | LLM 主动 finish 且交付轮次通过校验 | `goal_met` / `converged` / `best_effort` |
| 7 | 工具预算耗尽后仍请求工具，超过 `TOOL_REJECT_LIMIT`（建议 3） | `tool_budget_exhausted` |
| 8 | 交付轮次累计非法被拒超过 2 次 | `finish_rejected` |
| 9 | LLM 输出重试 2 次仍不合法 | `llm_output_invalid` |

第 1 条的精确含义：上限指新增的调整轮次，即 `len(rounds) - 1 <= ROUND_LIMIT`（第 0 轮是原图，不占额度）。工具调用不消耗轮次，所以必须由第 7 条单独兜住。

第 1 条必须最先检查，否则其他逻辑出错会导致死循环。

### 9.4 交付轮次的选择（v2.0 核心变更）

**由 LLM 决策，harness 只做校验与质疑。**

`finish` 时必须带 `deliver_round`。harness 执行 `check_delivery(state, n)`，返回三态：

| 返回 | 触发条件 | 处理 |
|---|---|---|
| `("reject", msg)` | `n` 不是整数，或不在已记录轮次范围内 | 计数并回问 |
| `("reject", msg)` | 第 `n` 轮 `guard_ok == False` | 计数并回问 |
| `("reject", msg)` | 第 `n` 轮 `aes is None`（`auto` 模式不会出现） | 计数并回问 |
| `("challenge", msg)` | `n not in challenged`，且 `n != best_by_error`，且第 `n` 轮误差高于 `best_by_error` | 加入 `challenged`，回问确认 |
| `("ok", None)` | 以上都不触发 | 接受交付 |

质疑与拒绝的区别

- 质疑：只发生一次。harness 把权衡摆出来 ——"第 N 轮平均误差 X，第 M 轮为 Y（更低）。请确认你仍然选择第 N 轮，并在 reason 中说明理由"。LLM 再次提交同一个 `n` 时，因为 `n` 已在 `challenged` 集合里，直接接受。
- 拒绝：轮次不存在、未过护栏、未测审美分，这些是客观错误，必须改。累计超过 2 次即走安全网。

这样既保证"harness 不否决 LLM 的选择"，又不会因为同一条质疑反复触发而把合法选择顶掉。

LLM 可以选择任何通过护栏的轮次，不必是最后一轮，不必是误差最小的那一轮。当"误差最小"和"观感最好"不是同一轮时，由 LLM 判断，并在 `reason` 里说明理由。这是 agent 决策的一环。

### 9.5 安全网

在以下情况下，harness 用 `best_by_error` 兜底交付，并在报告中明确标注这是安全网交付、非 LLM 选择：

- `finish_rejected`（交付轮次累计非法被拒超过 2 次）
- `tool_budget_exhausted`
- 其他边界强制退出，且 LLM 从未给出合法的 `deliver_round`

安全网不受"必须测过审美分"这条规则约束（兜底轮次可能从未被测量过）。安全网属于防止流程崩溃的兜底机制，不属于优化决策。

### 9.6 交付保证

1. 交付的轮次必定 `guard_ok == True`（由 9.4 校验保证）
2. 若从未有轮次通过护栏，交付第 0 轮（原图）
3. 报告必须写明：是否真正达标（`error < 0.02`）、`best_by_error` 是哪一轮及其误差、交付轮次的误差、审美分相对基线的升降、以及交付是否为安全网兜底

---

## 10. 完整走查

输入指令："帮我进行 coser 界通用的基本照片调光"
配置：`AESTHETIC_MODE="tool"`，`TOOL_BUDGET=4`，软阈值 0.5，硬阈值 1.5

### 第 0 轮（自动，无需决策）

```
params：全零
stats： luma 0.48, contrast 0.26, saturation 0.36, warmth +0.08
        clip_hi 0.002, clip_lo 0.030
审美分基线：5.41
```

### 调用一：解读指令

输出即 7.1 节示例：`interpretation` + 四条 `can_do` + 四条 `cannot_do` + 四项目标。

目标确定后补算第 0 轮误差：
```
误差向量：luma -0.10, contrast -0.04, saturation +0.06, warmth +0.06
平均误差：0.065
best_by_error = 0
```

### 第 1 轮

```json
{"adjust": {"brightness": 0.18, "contrast": 0.05, "saturation": -0.06, "color_temp": -120},
 "reason": "luma 差 -0.10 最大，先提亮；contrast 差 -0.04，提通透；saturation 差 +0.06、warmth 差 +0.06，一并下调。"}
```

harness 执行：
```
生效：全部在步长内，无需裁剪
stats： luma 0.57, contrast 0.30, saturation 0.30, warmth 0.02
        clip_hi 0.009, clip_lo 0.018
误差向量：luma -0.01, contrast 0.00, saturation 0.00, warmth 0.00
平均误差：0.003
护栏：通过
best_by_error ← 1
边界：0.003 < 0.02，达标 → 下一轮上下文加"已达标，建议 finish"
```

### 决策第 2 轮：LLM 选择调用工具

工具调用不消耗轮次，因此这一节仍然属于"第 2 轮的决策"，`rounds` 长度没有变化。

```json
{"tool": "aesthetic", "target": "current",
 "reason": "brightness 一次加了 0.18，幅度较大，需要确认画面没有变灰或发白。"}
```

harness 返回：
```
审美分 4.68，基线 5.41，差 -0.73 → 越过软警告阈值 0.5
回溯写入：第 1 轮 aes = 4.68，aes_delta = -0.73
工具预算：4 → 3（轮次不变，rounds 长度仍为 2）
```

### 第 2 轮（继续决策）：出现冲突

```json
{"adjust": {"brightness": -0.08, "gamma": -0.03},
 "reason": "统计量误差已达 0.003，但审美分从 5.41 降到 4.68，降幅 0.73 超过标定噪声带 0.5，说明提亮 0.18 过头、画面偏灰。回调 brightness 0.08，并改用 gamma -0.03 把中间调提回来，而不是继续压亮度，避免再次推高截断。"}
```

harness 执行：
```
生效：无需裁剪
stats： luma 0.52, contrast 0.29, saturation 0.30, warmth 0.02
        clip_hi 0.006, clip_lo 0.021
误差向量：luma -0.06, contrast -0.01, saturation 0.00, warmth 0.00
平均误差：0.018
护栏：通过
best_by_error：仍是 1（0.003 < 0.018）
```

### 决策第 3 轮：再次调用工具复核

```json
{"tool": "aesthetic", "target": "current",
 "reason": "需要确认回调后的审美分是否恢复，才能决定交付第 1 轮还是第 2 轮。"}
```

harness 返回：
```
审美分 5.28，基线 5.41，差 -0.13 → 回到噪声带内
回溯写入：第 2 轮 aes = 5.28，aes_delta = -0.13
工具预算：3 → 2（轮次不变，rounds 长度仍为 3）
```

### 决策第 3 轮（继续决策）：LLM 做出交付决策

```json
{"finish": "best_effort", "deliver_round": 2,
 "reason": "我选择交付第 2 轮，而不是误差更小的第 1 轮。理由：第 1 轮虽然平均误差 0.003 更低，但审美分 4.68，比基线低 0.73，超出标定噪声带，画面偏灰；第 2 轮误差 0.018 仍在容差 0.02 之内，审美分 5.28 已回到噪声带内。用户要的是'coser 通用调光'，画面观感比把四个统计量精确推到目标更重要，所以我优先保住观感。另外，指令里的磨皮和局部提亮面部我做不到，已在交付说明里保留。"}
```

harness 执行 `check_delivery(state, 2)`：
```
第 2 轮存在（rounds 长度 3，合法）
第 2 轮 guard_ok == True
第 2 轮 aes = 5.28，已测量
第 2 轮误差 0.018 高于 best_by_error（第 1 轮，0.003）
2 不在 challenged 集合中
→ ("challenge", "第 2 轮平均误差 0.018，第 1 轮为 0.003（更低）。请确认仍选择第 2 轮")
→ challenged = {2}，reject_count 不变
```

LLM 确认（再次 finish）：
```json
{"finish": "best_effort", "deliver_round": 2,
 "reason": "确认选择第 2 轮。第 1 轮误差更低但审美分低 0.73，代价是画面偏灰。第 2 轮误差在容差内且观感恢复。"}
```

harness 再次执行 `check_delivery(state, 2)`：
```
2 已在 challenged 集合中 → 直接返回 ("ok", None)
```

交付：第 2 轮。

### 这个走查演示了什么

| 环节 | 展示的能力 |
|---|---|
| 调用一 | 诚实声明四条做不到的操作，而不是假装能做完 |
| 第 1 轮 | 依据误差向量同时调整多个参数，并说明理由 |
| 决策第 2 轮 | 主动花费工具预算做复核（不消耗轮次） |
| 第 2 轮决策 | 识别"误差达标但观感变差"，并改用 gamma 而非继续压亮度 |
| 决策第 3 轮 | 自主做出"交付哪一轮"的决策，并用审美分与误差的权衡说明理由 |
| harness | 不否决 LLM 的选择，只质疑一次要求确认 |

整个走查只用了 3 个调整轮次（0/1/2）和 2 次工具调用，剩下的轮次与工具预算都没有用上，这体现的是"有明确设定值就能几步到位"。

---

## 11. 产物、日志与报告

### 11.1 目录结构

```
runs/2026-10-15_143022_coser通用调光/
    original.jpg                        原图副本（只读）
    delivered.jpg                       最终交付 —— 只保留这一张图片
    state.json                          完整 TaskState
    rounds.json                         每轮完整 Round 记录（参数、统计量、误差、审美分）
    tool_calls.jsonl                    每次工具调用的入参与结果
    llm_calls.jsonl                     每次 LLM 请求与响应原文
    report.md                           人类可读报告
```

目录中只保留最终交付的那一张图片。中间产物图（`round_00.jpg` … `round_NN.jpg`）在运行期间写在临时工作目录，交付时只把交付轮次的那一张复制成 `delivered.jpg`，然后删除全部中间图与临时目录。

可以安全删除的原因：第 4.3 节强制"候选图永远等于 `adjust(原图, 累计参数)`"，因此任意一轮都能用 `rounds.json` 里的参数向量确定性重算出来。中间图属于可再生的派生数据，保留只占磁盘。

删除时机必须在阶段 4 交付，不能在循环里删：因为 `aesthetic` 工具的 `target` 可以指历史轮次号（见 5.4），运行期间随时可能回看某一轮的图片。

调试开关：`KEEP_INTERMEDIATE_IMAGES = False`（见 12.1.1）。置为 `True` 时保留全部中间图与临时目录，便于逐轮对比画面。

需要回看某一轮时用 CLI 子命令临时重算，不需要保留图片：

```bash
python -m mytone.cli --regenerate <run_dir> <round_index>
```

它读取 `rounds.json` 里第 N 轮的参数向量，调用同一个 `adjust()` 重新渲染。约 10 行代码，复用的是主流程的同一个函数，因此结果与当轮完全一致。

`llm_calls.jsonl` + `rounds.json` + `tool_calls.jsonl` 三者合起来能完整回放整次运行的决策过程（每一步的输入、输出、参数、误差、审美分、退出原因），配合上面的重算命令即可还原任意一轮的画面。

### 11.2 report.md 的固定结构

**第一部分：指令与解读**

| 项 | 内容 |
|---|---|
| 原始指令 | 帮我进行 coser 界通用的基本照片调光 |
| 解读 | 理解为 cosplay 人像的通用基础调光…… |
| 能做到 | 整体提亮 / 提升对比去灰 / 整体降饱和略偏冷 / 统一色调 |
| 做不到 | 磨皮 / 单独压暗背景 / 单独提亮面部 / 眼睛与妆容强化 |

**第二部分：参数变化表**

| 轮次 | brightness | contrast | saturation | color_temp | gamma | 请求→生效 |
|---|---|---|---|---|---|---|
| 0 | 0.00 | 0.00 | 0.00 | 0 | 0.00 | 原图 |
| 1 | +0.18 | +0.05 | −0.06 | −120 | 0.00 | 全部生效 |
| 2 | +0.10 | +0.05 | −0.06 | −120 | −0.03 | 全部生效 |

**第三部分：统计量与审美分轨迹**

| 轮次 | luma | contrast | saturation | warmth | clip_hi | clip_lo | 平均误差 | 审美分 | 护栏 |
|---|---|---|---|---|---|---|---|---|---|
| 目标 | 0.58 | 0.30 | 0.30 | 0.02 | — | — | — | — | — |
| 0 | 0.48 | 0.26 | 0.36 | +0.08 | 0.002 | 0.030 | 0.065 | 5.41 | 通过 |
| 1 | 0.57 | 0.30 | 0.30 | +0.02 | 0.009 | 0.018 | 0.003 | 4.68 | 通过 |
| 2 | 0.52 | 0.29 | 0.30 | +0.02 | 0.006 | 0.021 | 0.018 | 5.28 | 通过 |

**第四部分：交付决策**

| 项 | 内容 |
|---|---|
| 交付轮次 | 第 2 轮 |
| 是否 LLM 选择 | 是 |
| 误差最小轮次 | 第 1 轮（0.003） |
| 交付轮次误差 | 0.018（在容差 0.02 内） |
| 审美分 | 5.28（相对基线 −0.13，噪声带内） |
| 是否达标 | 是 |
| LLM 的决策理由 | （原文） |

**第五部分：逐轮决策依据与工具调用**

每轮附 `reason` 原文、参数裁剪提示、工具调用的理由与结果、以及 harness 提出的质疑与 LLM 的确认。

---

## 12. 代码结构与实施

### 12.1 文件划分

```
mytone/
  __init__.py           包标记                                      约 6 行
  __main__.py           python -m mytone 入口                      约 6 行
  config.py             常量：参数表、阈值、预算、开关、提示词片段    约 170 行
  state.py              Round / TaskState 数据类                    约 175 行
  tools_image.py        adjust()                                    约 95 行
  tools_stats.py        stats()、region_stats()                      约 90 行
  tools_aesthetic.py    pyiqa nima 封装                              约 70 行
  calibrate.py          审美分灵敏度标定 + 耦合实测                   约 210 行
  validate.py           目标校验、动作校验、参数裁剪                  约 195 行
  llm.py                prompt 模板、上下文组装、API 调用、重试        约 500 行
  loop.py               主循环、工具分发、护栏、边界、交付校验         约 430 行
  report.py             报告生成                                     约 245 行
  cli.py                命令行入口、--regenerate 子命令               约 150 行
tests/
  check_core.py         纯逻辑验证（68 项）                           约 300 行
  check_loop.py         主循环端到端验证，假 LLM 驱动（30 项）         约 300 行
```

合计约 2300 行（含注释与文档字符串）。比原估的 860 行多，主要多在：
提示词模板与上下文组装、校验的细节、报告的五部分、以及两个验证脚本。
`state.py` 是相对 12.1 原列表新增的（把数据类从 loop 里拆出来）。

### 12.1.1 `config.py` 需要定义的常量

文档中引用的所有常量集中在这里，实现时以此为准。

```python
# --- 调整参数 ---
PARAMS = {                       # 见 4.1
    "brightness": {"range": (-0.6, 0.6), "step": 0.2},
    "contrast":   {"range": (-0.6, 0.6), "step": 0.2},
    "saturation": {"range": (-0.6, 0.6), "step": 0.2},
    "color_temp": {"range": (-400, 400), "step": 400},
    "gamma":      {"range": (-0.5, 0.5), "step": 0.15},
}
APPLY_ORDER = ["color_temp", "gamma", "brightness", "contrast", "saturation"]  # 见 4.2

# --- 统计量 ---
STAT_NAMES  = ["luma", "contrast", "saturation", "warmth"]   # 可优化目标
GUARD_STATS = ["clip_hi", "clip_lo"]                         # 仅用于护栏
STAT_RANGE  = {                                              # 提示词里给 LLM 的合理范围
    "luma": (0.30, 0.75), "contrast": (0.12, 0.32),
    "saturation": (0.05, 0.60), "warmth": (-0.15, 0.15),
}

# --- 护栏阈值（标定后回填，见 3.6） ---
CLIP_HI_MAX   = 0.02
CLIP_LO_MAX   = 0.05
CONTRAST_MIN  = 0.10
D_SOFT        = 0.5      # 占位，以标定为准
D_HARD        = 1.5      # 占位，以标定为准
AES_NOISE_BAND = D_SOFT  # 注入提示词，见 7.2
AES_LATENCY    = "约 0.1 秒"   # 实测值，见 2.6；GPU 上更快，标定后回填

# --- 预算与上限（见 5.3 / 9.3） ---
ROUND_LIMIT       = 10   # 新增调整轮次上限，不含第 0 轮
TOOL_BUDGET       = 4
TOOL_REJECT_LIMIT = 3    # 预算耗尽后仍请求工具的次数上限
LLM_RETRY_LIMIT   = 2    # 输出不合法时的重试次数
DELIVER_REJECT_LIMIT = 2 # 交付轮次非法被拒次数上限

# --- 收敛判据（见 9.2） ---
ERROR_TOL      = 0.02
IMPROVE_EPS    = 0.005
PLATEAU_ROUNDS = 2

# --- 开关 ---
AESTHETIC_MODE = "tool"  # "tool" | "auto"，见 5.5
CONFIRM        = False   # --confirm，见 7.1
KEEP_INTERMEDIATE_IMAGES = False   # 交付时是否保留中间产物图片，见 11.1

# --- 运行设备（不要硬编码 cuda，见 2.6） ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 审美工具 ---
AESTHETIC_METRIC = "nima"   # 备选：nima-vgg16-ava / musiq-ava / topiq_iaa / laion_aes
```

### 12.2 依赖

```bash
uv sync                       # 安装全部依赖并锁定版本（见 uv.lock）
cp .env.example .env          # 填 DEEPSEEK_API_KEY
```

同时提供三个控制台命令，与 `python -m` 形式等价：
`mytone`、`mytone-batch`、`mytone-calibrate`。

配置只有两种方式：环境变量、或 `.env`（见 2.2）。

### 12.3 建议实施顺序

虽然计划一口气写完，但按依赖顺序推进可以每步都有可验证的产出：

| 步 | 内容 | 验证方式 |
|---|---|---|
| 1 | `config.py` + `tools_image.py` | 命令行手动传单参数，肉眼确认五个参数各自的效果与方向正确 |
| 2 | `tools_stats.py` | 对同一张图算统计量，确认数值合理；确认六个统计量对五个参数的反应方向符合第 4.4 节的耦合表 |
| 3 | `calibrate.py` + `tools_aesthetic.py` | 跑出审美分灵敏度标定结果，据此确定 `D_soft` / `D_hard` 并回填到 config 与提示词 |
| 4 | `validate.py` | 单测：各类非法输入被正确拒绝或裁剪 |
| 5 | `llm.py` | 用一张图跑调用一，确认 JSON 合法、`cannot_do` 诚实、目标值合理 |
| 6 | `loop.py`（先用假 LLM） | 手写 JSON 序列，走完完整循环，九种退出原因各触发一次 |
| 7 | `loop.py`（接真 LLM） | 一句真实指令端到端出片 |
| 8 | `report.py` + `cli.py` + README | 报告五部分齐全可读 |

第 3 步的标定结果会反过来影响提示词（工具说明里的耗时与噪声带数字）和 config（阈值），所以它必须早于第 5 步。

### 12.4 明确不做

| 不做 | 原因 |
|---|---|
| 让 LLM 发问（`ask_user`） | 用 `--confirm` 在 CLI 层暂停替代，效果相同且简单得多 |
| 参考图模式 | 扩展方向 |
| 坐标下降、搜索算法、任何优化器 | 与"决策权归 LLM"直接冲突 |
| 让 LLM 看图 / 使用 VLM | 与"无视觉"约束冲突。Q-Align 这类工具留作 v2 的昂贵复核工具 |
| 批量处理、preset 表、会话内学习、多分支探索 | 扩展方向 |
| 原生 function calling（`tools=` 参数） | v1 用 JSON 动作协议便于日志与回放；两者可 1:1 迁移 |
| 局部 / 蒙版处理（磨皮等） | 五个全局参数做不到，本设计明确不承诺，并由 LLM 在 `cannot_do` 中声明 |
| Web UI | 命令行足够 |

---

## 13. 验收标准

- [ ] 任意自然语言指令 + 原图 → 成片，端到端可复现
- [ ] 调用一能正确输出 `interpretation` / `can_do` / `cannot_do`，且 `cannot_do` 如实包含局部处理类操作
- [ ] 审美工具跑通：基线自动测得，LLM 至少主动调用过一次
- [ ] 未对交付轮次调用 `aesthetic` 就 finish 时，被正确拒绝并要求补调用
- [ ] 当 LLM 选择的交付轮次误差高于 `best_by_error` 时，harness 正确提出质疑而非直接接受或否决
- [ ] 同一交付轮次被质疑后再次提交时被直接接受，不会重复质疑
- [ ] 交付轮次不存在／未过护栏／未测审美分时被正确拒绝（与质疑区分开），累计超过 2 次走安全网
- [ ] 工具预算用尽后 LLM 仍请求工具，超过 `TOOL_REJECT_LIMIT` 时正确退出为 `tool_budget_exhausted`，不出现死循环
- [ ] `pending_warning` 在每轮组装上下文后正确清空，不会把上一轮的提示永久留在上下文里
- [ ] 审美硬红线回溯判定生效：某轮在成图之后才测出审美暴跌时，该轮被标为非法、`best_by_error` 被重算、必要时回滚
- [ ] `AESTHETIC_MODE="auto"` 时 `aesthetic` 工具不在清单内，且 finish 校验自动通过
- [ ] LLM 确认后交付其选择的轮次，报告中记录选择理由
- [ ] `--confirm` 开启时能显示解读与能力边界并等待确认
- [ ] 九种退出原因各构造一次并验证行为正确（清单见 9.3）
- [ ] 护栏生效：故意构造死白／死黑／发灰／审美暴跌，都能判非法并回滚
- [ ] 劣化回滚生效：故意构造劣化方向，能回到历史最优
- [ ] 工具预算用尽后无法再调用工具
- [ ] `AESTHETIC_MODE` 两种模式都能跑通，并留下对照结论
- [ ] 审美分灵敏度标定完成，阈值有实测依据，结论记入本文档第 3.6 节
- [ ] 每轮 LLM 输出、工具调用、统计量、误差、审美分全部落盘，可完整回放
- [ ] 交付后运行目录中只剩交付的那一张图片，中间产物图与临时目录已删除
- [ ] `KEEP_INTERMEDIATE_IMAGES=True` 时中间图被保留，可用于逐轮对比
- [ ] `--regenerate <run_dir> <n>` 重算出的图片与当轮画面一致（验证"从原图重算"的确定性）
- [ ] 运行期间 `aesthetic(target=<历史轮次>)` 能正常读到该轮图片（证明删除只发生在交付阶段）
- [ ] 干净 venv 一键跑通（README 含代理配置、CUDA 版 torch 安装命令）

---

## 14. 参考文献

以下全部条目的 arXiv 编号均已通过 arXiv API 核实标题一致。

```
[1] Talebi, H., & Milanfar, P. (2018). NIMA: Neural Image Assessment.
    IEEE Transactions on Image Processing, 27(8), 3998-4011.
    arXiv:1709.05424

[2] Ke, J., Wang, Q., Wang, Y., Milanfar, P., & Yang, F. (2021).
    MUSIQ: Multi-scale Image Quality Transformer. ICCV 2021.
    arXiv:2108.05997

[3] Chen, C., Mo, J., Hou, J., et al. (2024). TOPIQ: A Top-down Approach
    from Semantics to Distortions for Image Quality Assessment.
    IEEE Transactions on Image Processing, 33.
    DOI:10.1109/TIP.2024.3378466  ｜ arXiv:2308.03060

[4] Schuhmann, C., Beaumont, R., Vencu, R., et al. (2022).
    LAION-5B: An open large-scale dataset for training next generation
    image-text models. NeurIPS 2022, Datasets and Benchmarks Track.
    arXiv:2210.08402

[5] Wang, J., Chan, K. C. K., & Loy, C. C. (2023). Exploring CLIP for
    Assessing the Look and Feel of Images. AAAI 2023.
    arXiv:2207.12396

[6] Xu, J., Liu, X., Wu, Y., et al. (2023). ImageReward: Learning and
    Evaluating Human Preferences for Text-to-Image Generation.
    NeurIPS 2023.  arXiv:2304.05977

[7] Wu, H., Zhang, Z., Zhang, E., et al. (2024). Q-Align: Teaching LMMs
    for Visual Scoring via Discrete Text-Defined Levels. ICML 2024.
    arXiv:2312.17090

[8] Wu, Q., Shi, J., Jenni, S., et al. (2026). RetouchIQ: MLLM Agents for
    Instruction-Based Image Retouching with Generalist Reward.
    CVPR 2026.  arXiv:2602.17558

[9] Zhang, Z., et al. (2025). JarvisArt: Liberating Human Artistic
    Creativity via an Intelligent Photo Retouching Agent.
    NeurIPS 2025.  arXiv:2506.17612
```

各条与本项目的关系：

- [1] 主选审美指标（`nima`）的依据
- [2][3][4] 同一 `pyiqa` 包内可替换的备选审美指标
- [5] 记录为何放弃"用 CLIPIQA 承载风格 prompt"这条路（见 2.5 第 2 条）
- [6] 文本条件化的人类偏好奖励模型，供 v2 参考
- [7] v2 可选的重型复核工具方向（LMM 打分，7B 级需 4bit 量化）
- [8] 与本项目最接近的工业级工作。其评价器按每条指令动态生成评价维度再逐项打分；
  消融实验显示多维奖励显著优于单一奖励（奖励信号质量改进带来 6.89 → 7.51）
- [9] 同样是最接近的同类工作，且给出了本项目设计所需的关键消融证据：
  单类奖励组合（格式+参数准确度，或格式+感知质量）效果均次优，
  多维奖励组合最好，论文明确指出参数导向与感知导向两类奖励是互补的。
  其感知质量奖励定义为"色彩分布匹配 + 像素保真"，与本项目"统计量设定值"的思路同源，
  但它是经过 SFT + GRPO-R 训练的 7B 多模态模型，与本项目的
  "无视觉 + 冻结模型 + 推理时循环"设定不同。

---

## 15. 首次真实运行记录

实现完成后用真实 DeepSeek API 跑了 4 次，同一张测试图、同一条指令
（"整体提亮一些，降低一点饱和度，色调偏冷"）。每次运行都暴露一个真实问题，
这一节记录过程与据此做的修正。

| 次序 | 退出原因 | 最终平均误差 | 工具调用 | 暴露的问题 |
|---|---|---|---|---|
| 1 | `plateau` | 0.0661 | 0 次 | LLM 用错 `color_temp` 量纲；从未使用 `gamma` |
| 2 | `plateau` | 0.0390 | 0 次 | 量纲已修正，但 `saturation` 调整方向反了 |
| 3 | `plateau` | 0.0200 | 0 次 | 方向已修正，但收敛中被平台期规则提前掐断 |
| 4 | `stalled_after_goal` | 0.0131（达标） | 1 次 | 暴露退出原因命名冲突 |

### 15.1 量纲错误（第 1 次运行）

现象：`color_temp` 列出现了 `-0.09`、`-0.06`、`-0.33` 这样的值，
而它的单位是千分之二（有效幅度 ±100~±400）。这些值的效果等于零，
所以 `warmth` 从 +0.0516 一路涨到 +0.0782（目标 −0.04），完全没往目标走。
同时六轮里一次都没用 `gamma`，而目标 luma 0.42 靠 brightness 上限
（0.2386 × 1.6 = 0.38）无法达到。

根因：提示词只写了 `color_temp` 的数值范围，没写单位，
LLM 就按其他四个参数的量纲（0.1 量级）来输出了。

修正：
- `config.PARAM_UNITS` 逐参数写明单位，`param_help()` 会输出给 LLM
- 耦合说明里把"最容易出错"的一条放在第一条
- 调用一的提示词增加可达性判断指导，并要求在 reason 里说明目标为什么可达

### 15.2 调整方向反转（第 2 次运行）

现象：LLM 的依据写着"饱和度当前 0.0944，比目标 0.18 低 0.0856，需回补"，
动作却是 `saturation -0.090`；下一轮更写成"当前 0.0783 远高于目标 0.1800"
（实际 0.0783 < 0.18），越调越远。

根因：上下文里只给了误差向量，没有说明正负的含义。
LLM 被用户指令里的"降低饱和度"锚住，把指令方向当成了永远的方向；
而它前几轮已经把饱和度降过头（`gamma -0.15` 的连带降饱和效应），
此时正确的动作应该是调高。

修正：上下文里给误差向量加上读法说明 ——
"负数表示当前值低于目标，需要往大调；正数表示高于目标，需要往小调。
不要把用户指令的方向当成永远的方向"。修正后第 3 次运行里，
LLM 明确写出"需要调高而非继续降低"。

### 15.3 平台期规则误判（第 3 次运行）

现象：误差轨迹 `0.0286 → 0.0241 → 0.0200`，稳步收敛，
但连续两轮改善量（0.0045、0.0041）都小于 `IMPROVE_EPS = 0.005`，
被判为平台期、提前掐断。容差是 0.02，误差正好卡在 0.0200 上。

根因：`IMPROVE_EPS = 0.005` 太大，把稳步收敛误判成停滞。

修正：`IMPROVE_EPS` 改为 `0.001`（真正卡住的表现是改善量接近 0 或为负）。
改后误差收敛到 0.0131，达标。

### 15.4 退出原因命名冲突（第 4 次运行）

现象：报告里同时出现两句话 ——
"是否由 LLM 选择：否，安全网兜底"与"退出原因：LLM 主动结束：已收敛"。

根因：`converged` 这一个字符串被用在两个地方：
harness 因"达标后停滞"强制退出，以及 LLM 主动 finish 时声明"已收敛"。
这在设计文档 9.2 / 9.3 里就埋下了。

修正：harness 强制退出的那一个改名为 `stalled_after_goal`。
规则：harness 强制退出的原因名不得与 LLM 状态名重名。

### 15.5 工具调用提醒（第 4 次运行生效）

前三次运行都是 0 次工具调用 —— LLM 始终没觉得有必要测量审美分。
这会使"工具调用"这部分失去学习价值，因此加了
`AESTHETIC_REMINDER_AFTER = 4`：连续调整 4 轮仍未用过审美工具时，
harness 给一次提醒（只提醒一次，且只说明"这个工具存在且还没用过"，不涉及任何优化决策）。

第 4 次运行中 LLM 收到提醒后主动调用，理由原文：

> 已连续调整4轮，统计量误差降至0.0279，但从未确认画面观感是否变差，
> 需先确认再决定是否收尾

结果：第 4 轮审美分 2.91（基线 3.12，差 −0.21，正常范围），耗时 0.11s。

### 15.6 从这四次运行得到的结论

1. 提示词里的"数值契约"必须写死到量纲一级。只给取值范围不够，
   不同参数量纲相差三个数量级时，模型一定会用错。
2. 误差向量的正负含义必须解释。只给数字，模型会退回按指令方向办事。
3. 失败模式具体且可复现：都是可定位、可用一次修改解决的具体问题，
   这说明"评价器在环"结构有效 —— 每一轮的数字都有留痕，
   问题能定位到具体哪一轮、哪个参数。
4. 实测到的参数耦合（4.4 节）确实会成为 agent 的陷阱：
   第 2 次运行里 `gamma` 的连带降饱和直接导致了过冲，
   而 agent 在第 3 次运行里自行从过冲中纠正回来，
   这是"决策权归 LLM"的价值所在。

### 15.7 真实照片上的运行：三条新发现

在真实照片（`DSC07186(1).JPG`，6000×4000）上运行时又暴露了三个问题，
其中第三个最严重。这三个问题都属于只有真实数据才会暴露的类型。

（1）目标的可达性不能只看参数范围。

这张照片原图 `clip_hi = 0.2356` —— 23.6% 的像素至少有一个通道打顶。
护栏只允许再增加 0.02，而 `brightness` 全范围的截断增幅可达 0.3423。
余量只有约 0.06 个全范围步长，也就是几乎不能提亮。

但 agent 在依据里写"目标 0.68 完全可达"，因为它只算了参数范围
（brightness 最多 ×1.6 → luma 可达 0.94），没有把护栏算进去。

修正：把 `clip_hi` / `clip_lo` 纳入 `calibrate.py` 的耦合矩阵，
并在提示词里声明"`brightness` 是推高截断最猛的参数，原图已有高光溢出时余量极小"。
修正后 agent 在第 1 轮就主动选择了 `gamma` 负值而不是 `brightness`：

> "原图 clip_hi 已达 0.2356 偏高，用 brightness 提亮会明显推高高光截断，
> 故改用 gamma 负值提亮中间调（对高光截断影响小）"

（2）指标定义错误不会引发异常，只会使 agent 在错误的前提下推理。详见 4.6 节。
`clip_hi` 原本用加权亮度判定，导致"加暖"与"加冷"两个相反操作
都把截断算成 0；提示词据此写下了错误规律，agent 照着执行后当场触发护栏。
LLM 的推理过程本身没有错误，问题来自输入给它的数值。

（3）回滚之后，上下文把"当前参数"指错了轮次。

`state.params` 在回滚时被改回 `best_by_error` 的参数，
但 `state.rounds[-1]` 仍然是那次非法的尝试 —— 而上下文里的"当前累计参数"
取自 `rounds[-1]`。于是 LLM 被告知它处于一个它并不在的位置。

后果：agent 看到"我在 brightness +0.12"，为了"改回 0"而请求 −0.12，
实际是在 0 的基础上又减了 0.12，于是越修越偏。
这解释了前面几次运行里被护栏触发后一路往反方向跑的现象。

修正：引入 `TaskState.current_index` —— 当前参数实际对应的轮次号。
正常前进时等于最后一轮；回滚后指回被回滚到的那一轮。
`current_round` 用它，另外新增 `last_attempt` 表示"最后一次尝试（可能就是非法轮）"。
回滚时同时设置两者，并在提示里明确告诉 LLM：

> 第 N 轮触发护栏，已回滚到第 M 轮。你现在的参数是第 M 轮的那一组，
> 请以此为基准重新计算增量。

这三条的共同点：问题都不在于 LLM 的推理能力，
而在于它基于错误的信息做出了形式正确的推理。指标、护栏、状态三者任何一处输入错误，
agent 的行为都会表现为错误。

### 15.8 批处理时代的两个新发现

（1）"连续 2 轮护栏非法就退出"过于激进，已改为 3 轮。

一次批处理运行里：第 2 轮 `contrast +0.15, color_temp +120` 导致 `clip_hi = 0.0677`
（上限 0.0237）被罚；第 3 轮 agent 主动把幅度减半（`+0.10 / +60`），
`clip_hi` 降到 0.0480 —— 方向完全正确，正在往可用区间收敛，
却因为"连续 2 轮"被当场掐断，只跑了 3 轮就结束。

护栏触发一次之后，agent 通常会做一次更温和的尝试。规则本意是防止反复触发同一护栏，
但 2 轮太紧，把恢复过程也一起掐掉了。改为 `GUARDRAIL_STREAK_LIMIT = 3`。
每一轮越线都会消耗轮次预算，本身已有自我限制。

（2）`clip_hi` 的容差是最主要的约束，agent 长期停留在其边缘。

改完规则后重跑，护栏在 1/4/5/7/8/9 轮触发，7、8、9 连续 3 轮，正确退出。
逐轮数据如下：

| 轮次 | `clip_hi` | 上限 |
|---|---|---|
| 4 | 0.0240 | 0.0237 |
| 7 | 0.0238 | 0.0237 |
| 8 | 0.0239 | 0.0237 |

差值为 0.0001 ~ 0.0003。agent 精确地停留在可行域边缘，说明它在贴着约束走，
但缺少一个关键信息：它看不到 `clip_hi` 逐轮的变化，因此无法校准
"多大的参数增量会把截断推高多少"。

修正：在历次轮次汇总表里加上 `clip_hi` / `clip_lo` 两列。
这张表让 agent 能直接读出"上一轮 contrast +0.10 把 clip_hi 从 0.0199 推到 0.0240"，
从而自行推算步长该收多少。这是纯测量，不涉及任何判断。

（3）由此得到一个需要用户决策的结论。

`CLIP_HI_DELTA = 0.02`（相对原图允许再多 2% 的像素打顶）在两张真实照片上
都是限制 agent 的主要因素，而不是参数范围：

| 照片 | 原图 clip_hi | 上限 | agent 的处境 |
|---|---|---|---|
| DSC07186 | 0.2356 | 0.2556 | `brightness` 几乎完全不能用 |
| IMG_20261005 | 0.0037 | 0.0237 | 反复以 0.0001 之差越线 |

这属于护栏正常履行职责（阻止画面增加过曝），不是 bug。
如果实测下来认为"再多 2% 过曝像素"可以接受，
直接调大 `config.CLIP_HI_DELTA` 即可。这属于摄影口味的决定，应当由使用者决定，
而不是通过调参得出。

### 15.9 调用一补充护栏余量后的效果

该修正在各次修改中对结果的影响最大。

问题：调用一（制定目标）拿到的信息里只有参数范围（"brightness 最多乘 1.6"），
没有护栏余量。因此它会定出"参数上可达、但调整过程中必然触发护栏"的目标，
整场都在贴着护栏线运行、误差降不下去。

实测对比（同一张图、同一条指令）：

| | 调用一看到的信息 | 它定的 luma 目标 | 结果 |
|---|---|---|---|
| 修正前 | 只有参数范围 | 0.48 | 误差 0.0408，平台期退出，护栏触发 1 次但改善量只有 0.0005 |
| 修正后 | 参数范围 + 护栏余量 + 各参数对护栏的代价 | 0.42 | 误差 0.0093，达标，LLM 主动收尾 |

修正后的目标设定依据原文：

> 原图 luma 0.3657，目标 0.42 仅需乘约 1.15，brightness +0.15 或 gamma -0.1 即可达到，
> 远低于 brightness 上限 1.6；…… 原图 clip_hi 0.0037 余量仅 +0.0300，
> clip_lo 余量 +0.0500，因此提亮与增艳都保持保守，避免触发护栏。

这一处的重要性在于：目标在调用一只定一次、贯穿整次运行。
前面几处修正（护栏余量进决策上下文、逐轮 clip_hi 可见）都是让 agent 在过程中调整，
而这一处是让它在开始时就避开不可行的目标。定错目标的代价远大于走错一步。

引出的通用结论：凡是 LLM 要做判断的地方，都要把它做这个判断所需的最小信息集给全。
调用一需要的是"可达性"，提示词给了参数范围却漏了护栏余量，而实测里护栏才是真正的边界。
该错误和 4.6 节的指标 bug 属于同一类：信息不全时，
agent 会做出形式正确但建立在错误前提上的决定。

### 15.10 由此得到的最好一次运行

修正后同一张图跑出的结果：

| 轮次 | 误差 | luma | clip_hi | 上限 0.0337 |
|---|---|---|---|---|
| 0 | 0.0349 | 0.3657 | 0.0037 | |
| 1 | 0.0230 | 0.3953 | 0.0040 | |
| 2 | 0.0280 | 0.3931 | 0.0359 | 非法（唯一的越线） |
| 3 | 0.0176 | 0.3935 | 0.0141 | |
| 4 | 0.0238 | 0.3933 | 0.0317 | |
| 5 | 0.0143 | 0.3922 | 0.0255 | |
| 6 | 0.0093 | 0.4021 | 0.0270 | ← 交付 |
| 7 | 0.0082 | 0.4094 | 0.0289 | |
| 8 | 0.0083 | 0.4145 | 0.0294 | |

审美分 5.08 → 5.14（有提升）。退出原因：LLM 主动声明已收敛。

交付决策的理由原文，它没有简单地选误差最小的那一轮：

> 第 6 轮平均误差 0.0093、审美分 5.14（基线 5.08，+0.06），clip_hi 0.0270 / clip_lo 0.0344 均在护栏内；
> 第 7、8 轮误差仅从 0.0093 微降到 0.0082/0.0083，属噪声级改善，且第 8 轮误差反而回升 0.0001，
> 说明已收敛。第 6 轮审美分为已测轮次中最高，且护栏余量比第 7、8 轮更宽（clip_hi 0.0270 vs 0.0294），
> 故交付第 6 轮。

`best_by_error` 是第 7 轮，它选了第 6 轮 —— 于是 harness 按 9.4 节提出质疑，
LLM 确认后交付。"质疑而非否决"这条设计在这里被完整执行了一次。

它做的判断是"误差相差 0.001 属于噪声级，但护栏余量差 0.0024 是实打实的安全边际"，
这正是把决策权交给 LLM 而不是让代码取 argmin 的价值。

### 15.11 首次真实批量运行：33 张

用户在自己拍摄的 33 张漫展人像上运行。图片统一复制到 `input/` 并按文件名后两位
重命名（`DSC07131(1).JPG` → `31.JPG`），每张配一份内容相同的指令 txt：

> 帮我进行coser界通用的基本照片调光

| 指标 | 结果 |
|---|---|
| 任务数 | 33 |
| 完成 | 33（0 失败） |
| 误差达标（< 0.02） | 25 |
| 耗时 | 31 分 15 秒（约 57 秒/张） |
| 运行设备 | cpu（本环境 `torch.cuda.is_available()` 为 False，NVML 无法初始化） |
| LLM 调用 | 363 次，平均 11.0 次/张 |
| 工具调用 | 34 次，平均 1.0 次/张 |

运行设备为 CPU 并不构成瓶颈：单张耗时由 LLM 往返延迟主导（11 次调用），
唯一的重计算——审美指标——是按需测量的，33 张里只测了 3 张。

退出原因分布：

| 退出原因 | 张数 | 误差范围 | 平均误差 |
|---|---|---|---|
| `round_limit` | 15 | 0.0068 ~ 0.0213 | 0.0146 |
| `stalled_after_goal` | 6 | 0.0079 ~ 0.0183 | 0.0137 |
| `finish_rejected` | 4 | 0.0051 ~ 0.0112 | 0.0073 |
| `guardrail_violation` | 4 | 0.0138 ~ 0.0300 | 0.0232 |
| `plateau` | 3 | 0.0308 ~ 0.0356 | 0.0327 |
| `llm_output_invalid` | 1 | 0.0085 | 0.0085 |

达标组平均误差 0.0125，未达标组 0.0273。三条观察：

**一、误差是连续分布的，容差线切在一个连续谱上。** 最大的达标误差是 0.0193，
最小的未达标误差是 0.0202。两者相差 0.0009，远小于同一张图因 LLM 温度而
产生的运行间波动。也就是说，这 8 张"未达标"里有多少是真的没调好、有多少
只是抽签落在了容差线另一侧，这个数据本身答不了。它同时说明 `ERROR_TOL = 0.02`
这个值对当前的目标设定并非天然分界，只是一个可调的作业口径。

**二、护栏是主要约束，不是例外路径。** 33 张里有 28 张至少触发过一次护栏，
累计 66 个轮次被回滚。最大残差落在 `saturation`（12 张）与 `luma`（11 张），
而这两个量都直接耦合到 `clip_hi`（见 4.4 的耦合表）——想压 `saturation`
残差或抬 `luma`，都要消耗本就不宽的高光余量。护栏触发得如此频繁，
说明"目标统计量"与"护栏边界"在这批照片上贴得很近。

**三、统计量达标与观感达标是两件事。** 只有 3 张调用过审美工具，这 3 张的
审美分全部下降：5.332 → 5.095，单张降幅 0.16 ~ 0.34。三张都落在硬红线
`D_HARD = 0.58` 以内，因此都没有被回滚。样本只有 3 张，不足以支持任何结论，
但它出现的形态正是 5.1 节把审美分设为"参考信号 + 护栏"时所预期的冲突：
**统计量在收敛，观感在下滑。** 是否要把 `D_SOFT` 与 `D_HARD` 收得更紧，
需要更多测量数据才能判断。

### 15.12 回滚轮污染平台判定

**起因。** 15.11 的 8 张未达标里，有 3 张（65 / 71 / 73）的退出原因是 `plateau`，
而且分别只用了 3、3、5 轮（预算是 10 轮），护栏余量也还有富余。这既不是"预算耗尽"，
也不是"撞死在护栏上"，于是去查平台判定本身。

**问题。** `_recent_improvements` 把 `state.rounds` 里的轮次串成误差链条来计算
"最近 n 轮有没有改善"。但触发护栏的轮次会被回滚（见 9.3），而它的 `error` 是在
**一个并未真正停留过的参数位置**上测得的——这个数记录了"如果采用会怎样"，
不是"实际处在哪"。把它和合法轮次放进同一条链，等于拿两个不同位置的误差相减。

以 71.JPG 为例：

| 轮次 | 误差 | 护栏 | 说明 |
|---|---|---|---|
| 0 | 0.0462 | 合法 | 原图 |
| 1 | 0.0286 | **非法** | 触发护栏已回滚；0.0286 是这次未生效的尝试 |
| 2 | 0.0356 | 合法 | 从第 0 轮的位置重新出发 |
| 3 | 0.0376 | 合法 | |

旧实现把第 1 轮也入链，于是 0→1 被记成一次"改善"（0.0462 → 0.0286），紧接着
1→2 被记成一次"退步"（0.0286 → 0.0356）——可第 2 轮根本不是从第 1 轮出发的。
两次假信号正好顶满 `PLATEAU_ROUNDS = 2`，任务在第 3 轮就以 `plateau` 退出。

这不是个例。33 张里累计 66 个轮次触发护栏，其中 **50 个（76%）** 的误差比上一轮
更小——也就是说，多数护栏触发都会制造一次假的"改善"，并让紧随其后的合法轮
背上一次假的"退步"。

**修正。** 改善链条只保留护栏合法的轮次。这里要强调它的性质：这是**算术正确性**
问题，不是优化决策。harness 的职责是把量算对（原则一），"要不要继续调、调多少"
仍由 LLM 判断。删掉不可比的数据点不改变任何取舍规则，只是不再用错位的数字去
触发退出。

修正后同样三张、同样指令：

| 图 | 修正前 | 修正后 | 结果 |
|---|---|---|---|
| `65.JPG` | 0.0318（3 轮退出） | 0.0201 | 大幅改善，仍未达标 |
| `71.JPG` | 0.0356（3 轮退出） | **0.0165** | **达标** |
| `73.JPG` | 0.0308（5 轮退出） | 0.0212 | 大幅改善，仍未达标 |

三张全部改善，其中一张跨过容差线。

**一个需要分辨的测试失败。** 修正后 `check_loop.py` 的场景 D 报
"脚本已用尽，LLM 被调用的次数超出预期"。原因是夹具只准备了 5 条回复，而修正后
主循环会多走一轮，走到那条 `finish`；该 `finish` 声明交付第 0 轮，因
`best_by_error` 不是第 0 轮而被质疑一次（9.4 节），于是还需要第 6 条回复。
**这里的失败指的是夹具假设了旧行为，不是程序出错。** 这个区分很重要：
夹具是按旧的退出时机写的，行为修正后它会以"测试挂了"的形式报警，
若不加分辨就回退修正，等于把 bug 保住。

同时 `check_core.py` 增加了一条用 65.JPG 真实数据构造的回归用例，并已双向验证：
在旧实现下它失败（`plateau`），在新实现下通过（`continue`）。

