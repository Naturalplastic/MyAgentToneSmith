# MyToneSmith

评价器在环（evaluator-in-the-loop）的 AI 摄影后期 Agent。

输入一条自然语言调色指令与一张原图，输出成片、参数变化记录与完整决策日志。

本项目的定位是学习项目：目的是完整实现并运行一个"评价器在环"的 agent，
而不是打磨调图效果本身。目标用户为自用。

## 文档

| 文件 | 内容 |
|---|---|
| `README.md` | 安装、使用、输出解读、调参、排障 |
| [`DESIGN.md`](DESIGN.md) | 设计文档：架构决策、实测数据、已知问题 |
| [`LICENSE`](LICENSE) | MIT 许可，以及第三方组件的许可说明 |

许可方面需注意：本仓库采用 MIT，但运行时会使用 **pyiqa**，
该组件采用 PolyForm Noncommercial 1.0.0，不可用于商业用途。详见 `LICENSE`。

隐私方面：图像统计与评分全部在本地完成，不向大模型传输图片，模型只接收数值。
仓库的 `.gitignore` 对图片扩展名采取默认拒绝策略。

---

## 工作方式

```
用户指令（自然语言）
   │
   ▼
调用一：LLM 解读指令、声明能力边界、制定目标统计量
   │
   ▼
迭代循环（上限 10 轮）
   │  LLM 决策 ──► adjust() 渲染 ──► stats() 测量
   │      ▲                              │
   │      └── 误差向量回喂 ◄── 护栏判定 ◄── 边界判定
   │
   │  LLM 可主动调用工具：aesthetic（审美分）、region（区域统计）
   │  工具调用不消耗轮次，单独计预算（上限 4 次）
   ▼
交付 LLM 指定的轮次 + 报告
```

## 设计要点

1. **LLM 无视觉能力**。模型只接收数值，不接收图像。
   该约束是本项目存在 agent 结构的前提：若模型能看图，一次即可给出参数，
   循环、评价器、回滚与边界判定都不再需要。

2. **三层评价体系**。图像统计量作为优化目标，具有可归因、有设定值、无噪声三个性质；
   审美分作为参考信号与护栏，其作用是在"统计量已达标但画面变差"时产生冲突；
   截断与发灰作为硬约束。需要判断的地方只存在于无法用公式解决的冲突中。

3. **决策权归 LLM**。目标设定、每轮调参、工具调用时机、最终交付轮次均由 LLM 决定。
   harness 只负责测量、算术、参数裁剪、护栏与边界判定。
   代码中不存在替代 LLM 做优化决策的逻辑。

---

## 环境准备

### 前置要求

| 项 | 要求 | 说明 |
|---|---|---|
| 操作系统 | Linux / macOS / WSL | 未在原生 Windows 上验证 |
| Python | 3.10 | 无需自行安装，`uv` 会自动获取 |
| [`uv`](https://docs.astral.sh/uv/) | 包管理器 | 未安装时执行 `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| DeepSeek API Key | 必需 | 在 <https://platform.deepseek.com> 申请 |
| 磁盘空间 | 约 8 GB | 虚拟环境约 6.2 GB，模型权重约 0.4 GB |
| NVIDIA GPU | 可选 | 纯 CPU 可正常运行。本项目的耗时由 LLM 往返延迟主导，GPU 只加速按需测量的审美指标，对总时长的改善有限（实测见 `DESIGN.md` 15.11） |

### 安装

依赖在 `pyproject.toml` 中声明。CUDA 版 torch 通过 `[tool.uv.sources]` 指向
PyTorch 官方索引，因此使用 uv 时无需手动区分安装顺序。

```bash
uv sync                     # 创建 .venv 并安装全部依赖（首次下载约 2.5 GB）
cp .env.example .env        # 编辑 .env，填入 DEEPSEEK_API_KEY
```

`uv sync` 会依据 `uv.lock` 复现确定的依赖版本，其中 torch 为 `2.6.0+cu124`。

不使用 uv 时，注意必须先装 CUDA 版 torch 再装本项目，否则会装入 CPU 版
（`pyiqa` 声明依赖 torch，顺序颠倒会从 PyPI 解析到 CPU 版）：

```bash
python -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -e .
```

无 NVIDIA GPU 时，把 `pyproject.toml` 中 `[tool.uv.index]` 的 url 改为
`https://download.pytorch.org/whl/cpu` 后执行 `uv sync`。程序自动选择运行设备。

安装后提供三个命令，与 `python -m` 形式等价：

```bash
mytone --help                 # 单张运行
mytone-batch --help           # 批处理
mytone-calibrate --help       # 标定
```

### 配置

配置来源只有两种：进程环境变量，或项目根目录的 `.env` 文件。
优先级为：环境变量 > `.env` > 代码默认值。

| 变量 | 用途 | 必填 |
|---|---|---|
| `DEEPSEEK_API_KEY` | LLM 调用 | 是 |
| `DEEPSEEK_BASE_URL` | 接口地址，默认 `https://api.deepseek.com` | 否 |
| `DEEPSEEK_MODEL` | 模型名，默认 `deepseek-chat` | 否 |
| `MYTONE_TEMPERATURE` | 采样温度，默认 0.2，设为 0 可复现单次运行 | 否 |
| `HF_ENDPOINT` | HuggingFace 镜像地址，仅在无法直连时需要 | 否 |

`.env` 已列入 `.gitignore`。模板见 `.env.example`。程序不读取任何自定义格式的密钥文件。

### 网络

代理配置属于使用者机器的环境，不由本项目设置。若网络需要代理，自行在 shell 中
设置或写入 `~/.bashrc`：

```bash
export http_proxy="http://<主机>:<端口>"
export https_proxy="http://<主机>:<端口>"
```

代理即使仅支持 SOCKS，也应写成 `http://` 形式。`huggingface_hub` 与 `timm` 使用 httpx，
`socks5://` 需要额外的 `socksio` 包，否则会在检查本地缓存之前抛出 `ImportError`。

本项目从 HuggingFace 下载模型权重。无法直连 `huggingface.co` 时增加：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

### 自检

```bash
source env.sh --venv
```

该脚本载入 `.env`、激活虚拟环境、可选地将各类缓存目录指向项目内，
并输出以下状态：代理是否已设置、HF 镜像、API Key 是否载入、外网连通性、CUDA 是否可用。

---

## 使用

### 验证安装（不需要 API Key）

```bash
.venv/bin/python tests/check_core.py     # 86 项，完全离线
.venv/bin/python tests/check_loop.py     # 36 项，需模型权重
.venv/bin/python tests/check_batch.py    # 38 项，需模型权重
```

三份测试共 160 项，均不需要 API Key，用于确认依赖安装正确、核心逻辑正常。

`check_core.py` 不接触模型，断网也能跑；`check_loop.py` 与 `check_batch.py`
需要真实的审美评分来取基线，因此首次运行会下载模型权重，之后走本地缓存。
若只想知道依赖是否装对，先跑 `check_core.py` 即可。

### 首次运行会下载模型权重

审美评分使用 NIMA，其权重来自 HuggingFace，共约 400 MB：

| 内容 | 大小 | 来源 |
|---|---|---|
| NIMA 权重 `NIMA_InceptionV2_ava-b0c77c00.pth` | 208 MB | `chaofengc/IQA-PyTorch-Weights` |
| Inception-ResNet-v2 主干（由 `timm` 拉取） | 约 200 MB | `timm/inception_resnet_v2.tf_in1k` |

首次调用审美评分时会一并下载，实测在正常网络下约一到两分钟；
之后从本地缓存读取，不再联网。缓存位置由 `env.sh` 设为项目内的 `.torch-cache/`。

若下载卡住，参考「网络」一节的代理与 `HF_ENDPOINT` 设置。

### 批处理

将照片放入 `input/`。若某张照片需要单独指令，在它旁边放置同名 `.txt` 文件，
文件内容即为该照片的指令。

```
input/
  照片A.jpg
  照片A.txt        内容为"帮我进行coser界通用的基本照片调光"
  照片B.jpg        无同名 txt，使用命令行 -p 指定的默认指令
```

`.txt` 中以 `#` 开头的行视为注释。文件存在但内容为空时，同样回退到默认指令。

```bash
source env.sh --venv

python -m mytone.batch --dry-run -p "默认指令"    # 仅列出执行计划
python -m mytone.batch -p "默认指令"              # 实际执行
python -m mytone.batch -p "默认指令" --skip-existing   # 跳过已有成片的任务
python -m mytone.batch -d 其他目录 -p "默认指令"        # 指定其他输入目录
python -m mytone.batch -p "默认指令" --limit 2         # 只处理前两张
```

单张任务失败不中断整批。存在执行失败的任务时，退出码为 1；否则为 0。
"未达标"不计为失败。

### 单张

```bash
# 基本用法
python -m mytone -i photo.jpg -p "帮我进行coser界通用的基本照片调光"

# 先查看指令解读与能力边界，确认后继续
python -m mytone -i photo.jpg -p "清冷" --confirm

# 由 harness 每轮自动测量审美分，而非由 LLM 决定测量时机
python -m mytone -i photo.jpg -p "清冷" --aesthetic-mode auto

# 保留中间产物图片（默认在交付时删除）
python -m mytone -i photo.jpg -p "清冷" --keep-intermediate

# 重算某次运行中指定轮次的画面
python -m mytone --regenerate runs/<运行目录> 2
```

`--confirm` 在非交互式终端（管道、CI）下会跳过确认步骤并输出提示。

### 标定

对五个调整参数各施加 7 档单参数变化，实测参数耦合矩阵与审美分灵敏度。
输出结果包含可直接写入 `config.py` 的建议阈值。

```bash
python -m mytone.calibrate -i input/你的照片.jpg

# 仅重新分析已有结果，不重新扫描
python -m mytone.calibrate --reanalyze runs/_calibration/<图名>/calibration.json
```

标定必须使用真实照片。合成图像的高光分布与真实照片差异显著，
同一组参数在两类图像上的截断代价实测可相差一个数量级。

---

## 输出

### 目录结构

每次批处理生成 `runs/batch_<时间戳>/`；单张运行生成 `runs/<时间戳>_<指令摘要>/`。

```
runs/batch_2026-10-15_221228/
  index.md                 汇总表
  <照片名>/
    original.jpg           原图副本
    delivered.jpg          最终交付，目录中仅保留这一张图片
    rounds.json            每轮的参数、统计量、误差、审美分、LLM 依据
    state.json             完整状态
    tool_calls.jsonl       每次工具调用的入参与结果
    llm_calls.jsonl        每次 LLM 请求与响应原文
    report.md              报告
```

中间产物图片在交付阶段删除。该操作不损失信息：候选图恒等于
`adjust(原图, 累计参数)`，任意轮次均可由 `rounds.json` 中的参数向量确定性重算。

### 汇总表

`index.md` 每行一个任务，包含：结果、退出原因、交付轮次、平均误差、审美分变化、耗时。
执行失败的任务会附带完整 traceback。

### 报告

`report.md` 固定为五个部分。

| 部分 | 内容 |
|---|---|
| 一、指令与解读 | LLM 对指令的解读；`能做到` 与 `做不到` 两列为其声明的能力边界 |
| 二、参数变化表 | 每轮的参数累计值、请求增量与生效增量；发生裁剪时标注 |
| 三、统计量与审美分轨迹 | 目标值、逐轮实测值、平均误差、截断比例 |
| 四、交付决策 | 交付轮次；由 LLM 选择还是由安全网兜底；LLM 的决策理由原文 |
| 五、逐轮决策依据与工具调用 | 每轮 LLM 给出的依据原文；工具调用的理由与结果 |

第五部分记录了 LLM 每轮的实际推理过程。例如某次运行中它写道：

> 原图 clip_hi 已达 0.2356 偏高，用 brightness 提亮会明显推高高光截断，
> 故改用 gamma 负值提亮中间调。

此类输出只有在 LLM 获得可归因的数值时才可能出现。

---

## 调参

全部参数位于 `mytone/config.py`，修改后立即生效。

### 护栏阈值

```
clip_hi 上限  = 原图.clip_hi + max(CLIP_HI_DELTA, 原图.clip_hi × CLIP_HI_RELATIVE)
clip_lo 上限  = 原图.clip_lo + max(CLIP_LO_DELTA, 原图.clip_lo × CLIP_LO_RELATIVE)
contrast 下限 = min(CONTRAST_MIN, 原图.contrast × CONTRAST_RATIO)
```

阈值以原图为基准，不设绝对上限。原图本身低对比或大面积过曝时不会被误判，
但相对原图继续恶化仍会被拦截。

`CLIP_HI_DELTA` 默认 0.03，允许比原图多 3% 的像素触顶。
实测中该项是最主要的约束。在两张测试照片上，参数取值范围均未成为限制，
限制来自此容差。若认为可接受更多过曝，调大该值。

### 审美阈值

先运行 `calibrate.py`，再将输出的建议值写入 `D_SOFT`、`D_HARD`、`AES_NOISE_BAND`。
当前值 0.29 / 0.58 由一张真实照片标定得出，更换题材后建议重新标定。

### 预算与上限

| 配置项 | 默认值 | 含义 |
|---|---|---|
| `ROUND_LIMIT` | 10 | 调整轮次上限，不含第 0 轮 |
| `TOOL_BUDGET` | 4 | 工具调用次数上限，不消耗轮次 |
| `GUARDRAIL_STREAK_LIMIT` | 3 | 连续触发护栏多少轮后强制退出 |
| `TARGET_REVISION_LIMIT` | 2 | 允许的目标修正次数 |
| `ERROR_TOL` | 0.02 | 收敛容差 |

### 更换审美指标

```python
AESTHETIC_METRIC = "nima"    # 可选：nima-vgg16-ava、musiq-ava、topiq_iaa、laion_aes
```

---

## 排障

| 现象 | 原因与处理 |
|---|---|
| `未设置 DEEPSEEK_API_KEY` | 未创建 `.env` 且未设置环境变量。执行 `cp .env.example .env` 并填写 |
| `huggingface.co 不可达` | 先确认已 `source env.sh`。权重缓存在项目内 `.hf-cache` 与 `.torch-cache`，而这两个位置只由 `env.sh` 通过 `HF_HOME`/`TORCH_HOME` 指定；不设置时程序会去默认目录找，找不到才联网，于是表现为"权重明明下载过却仍要联网"。确认已 source 后仍不通，再设代理或 `HF_ENDPOINT=https://hf-mirror.com` |
| `PermissionError` 指向 `~/.cache/torch` | 缓存目录不可写。设置 `export TORCH_HOME=$PWD/.torch-cache` |
| `ImportError: Using SOCKS proxy` | 代理地址写成了 `socks5://`。改为 `http://` |
| 同一张照片多次运行结果不同 | LLM 采样具有随机性（温度 0.2），且路径与护栏强相关。排障时设置 `MYTONE_TEMPERATURE=0` |
| 反复"未达标" | 检查 `report.md` 第三部分中 `clip_hi` 是否持续接近上限。若是，说明任务需要更多过曝，考虑调大 `CLIP_HI_DELTA` |
| 单张任务失败 | 不影响整批。失败原因见 `index.md` 的失败详情，含完整 traceback。此时退出码为 1 |
| `--confirm` 未等待确认 | 当前非交互式终端。此为该参数的设计行为 |
| 运行设备显示 `cpu`，但机器确有 NVIDIA 显卡 | 批处理启动时会打印实际设备。WSL 下用 `/usr/lib/wsl/lib/nvidia-smi` 验证；若报 `GPU access blocked by the operating system`，说明 WSL 宿主未把 GPU 交出来，需在 Windows 侧更新 NVIDIA 驱动后执行 `wsl --update` 与 `wsl --shutdown`。注意不要在 WSL 内安装 Linux 版 NVIDIA 驱动 |

---

## 验证

```bash
source env.sh                            # 必须：指定项目内的权重缓存，否则 check_loop/check_batch 会去联网找模型
.venv/bin/python tests/check_core.py     # 88 项：耦合方向、裁剪、校验、护栏、边界、目标修正、回滚轮不计入平台判定
.venv/bin/python tests/check_loop.py     # 36 项：主循环端到端
.venv/bin/python tests/check_batch.py    # 38 项：批处理扫描、配对、错误隔离、汇总表、退出码
```

`check_core.py` 完全离线，不需要权重、不需要 API Key。`check_loop.py` 与
`check_batch.py` 仅替换网络层，需要从本地缓存加载 NIMA 权重（约 400 MB，
缓存缺失时才会下载）。三者的 JSON 解析、输出校验、重试、上下文组装、
`pending_warning` 的清理时机、产物清理均使用真实代码。

覆盖场景包括：正常交付（含质疑与确认流程）、工具预算耗尽、护栏回滚、
`revise_target` 修正目标后重算历史误差、批处理单任务失败不中断整批。

---

## 已知限制

- **审美分对参数变化的敏感度低**。同题材下单个步长造成的最大变化约 0.19，
  因此它适合作为参考信号与护栏，不适合作为优化目标。
- **目标的可达性受护栏约束**。参数取值范围内可达不等于实际可达。
  在大面积高光溢出的照片上，提亮会被截断护栏拦截。
  LLM 可通过 `revise_target` 修正目标，但实测中它并不总会使用该动作。
- **只能进行全局调整**。五个参数均为全局参数，无法实现磨皮、局部提亮面部、
  单独压暗背景等操作。LLM 会在指令解读阶段声明这些限制（写入 `cannot_do`）。
- **同一张照片多次运行结果不同**。要复现某次运行用于排障，设置 `MYTONE_TEMPERATURE=0`。
- **pyiqa 采用 PolyForm Noncommercial 许可**，自用学习合规，不可商用。
  本仓库自身采用 MIT，见 `LICENSE`。

---

## 项目结构

```
mytone/
  config.py             全部常量与开关：阈值、预算、提示词片段
  state.py              Round 与 TaskState 定义
  tools_image.py        adjust()：五个参数的确定性渲染
  tools_stats.py        stats()、region_stats()：六个统计量
  tools_aesthetic.py    NIMA 审美分封装
  validate.py           输出校验与参数裁剪
  llm.py                两个 prompt、上下文组装、API 调用与重试
  loop.py               主循环、工具分发、护栏、边界判定、交付校验
  report.py             report.md 生成
  cli.py                单张运行入口
  batch.py              批处理入口
  calibrate.py          标定脚本
tests/                  160 项验证，全部可离线运行
input/                  批处理输入目录
runs/                   运行产物，已列入 .gitignore

pyproject.toml          包元数据、依赖声明、控制台命令
uv.lock                 依赖版本锁定
env.sh                  环境准备脚本（载入 .env、可选激活虚拟环境）
.env.example            配置模板
DESIGN.md               设计文档
LICENSE                 MIT 许可 + 第三方许可说明
```

## 参考文献

完整清单见 `DESIGN.md` 的参考文献一节。主选审美指标的出处：

> Talebi, H., & Milanfar, P. (2018). NIMA: Neural Image Assessment.
> *IEEE Transactions on Image Processing*, 27(8), 3998-4011. arXiv:1709.05424
