#!/usr/bin/env bash
# MyToneSmith 环境准备
#
# 用法：
#   source env.sh            # 载入 .env、设置项目内缓存目录
#   source env.sh --venv     # 额外激活虚拟环境
#
# ---------------------------------------------------------------------------
# 【重要】本脚本不配置代理。
#
# 代理属于你机器的环境，不是这个项目的一部分 —— 所以它不在这里替你做决定。
# 如果你所在网络需要代理，请自己在 shell 里设置（或写进 ~/.bashrc）：
#
#     export http_proxy="http://<主机>:<端口>"
#     export https_proxy="http://<主机>:<端口>"
#
# 注意：如果你的代理只支持 SOCKS，请写成 http:// 而不是 socks5://
# （huggingface_hub / timm 走 httpx，socks5 需要额外的 socksio 包，
#   否则会在连缓存都没检查前就抛 ImportError）。
#
# 本项目会从 HuggingFace 下载模型权重（NIMA + Inception-ResNet-v2 主干）。
# 若无法直连 huggingface.co，再加一条：
#
#     export HF_ENDPOINT="https://hf-mirror.com"
#
# 详见 README.md 的「环境准备」一节与「排障」一节。
# ---------------------------------------------------------------------------

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT

# ---------- 载入 .env ----------
# .env 是配置的唯一文件来源（模板见 .env.example）。已存在的环境变量优先。
if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "${PROJECT_ROOT}/.env"
    set +a
    echo "已载入 .env"
fi

# ---------- 项目内缓存目录（可选） ----------
# 默认情况下 uv / pip / torch / huggingface 会把缓存写到 ~/.cache。
# 这里把与项目相关的几个重定向到项目内，好处是 `rm -rf .*-cache` 就能整体清掉，
# 也不污染家目录。**不需要的话把这段注释掉即可** —— 系统默认位置在任何机器上都可用。
export UV_CACHE_DIR="${PROJECT_ROOT}/.uv-cache"
export PIP_CACHE_DIR="${PROJECT_ROOT}/.pip-cache"
export HF_HOME="${PROJECT_ROOT}/.hf-cache"
export TORCH_HOME="${PROJECT_ROOT}/.torch-cache"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME"

# ---------- 可选激活虚拟环境 ----------
if [ "$1" = "--venv" ]; then
    if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
        # shellcheck disable=SC1091
        . "${PROJECT_ROOT}/.venv/bin/activate"
        echo "已激活虚拟环境：$(python -V 2>&1)"
    else
        echo "[错误] 未找到 .venv。请先执行："
        echo "       uv venv .venv --python 3.10"
    fi
fi

# ---------- 状态自检（只报告，不修改） ----------
echo ""
echo "项目根目录 : ${PROJECT_ROOT}"

if [ -n "${https_proxy}${HTTPS_PROXY}${http_proxy}${HTTP_PROXY}" ]; then
    echo "代理       : ${https_proxy:-${HTTPS_PROXY:-${http_proxy:-${HTTP_PROXY}}}}"
else
    echo "代理       : 未设置（如需代理请自行 export，见本文件顶部说明）"
fi
echo "HF 镜像    : ${HF_ENDPOINT:-未设置（直连 huggingface.co）}"

if [ -z "${DEEPSEEK_API_KEY}" ]; then
    echo ""
    echo "[提示] DEEPSEEK_API_KEY 未设置。请二选一："
    echo "       1) cp .env.example .env  然后填上密钥"
    echo "       2) export DEEPSEEK_API_KEY=sk-xxxxxxxx"
else
    echo "API Key    : 已载入（${DEEPSEEK_API_KEY:0:3}...${DEEPSEEK_API_KEY: -4}）"
fi

printf "外网连通   : "
if timeout 12 curl -s -o /dev/null -w "%{http_code}" https://huggingface.co 2>/dev/null | grep -q 200; then
    echo "huggingface.co 可达"
elif [ -n "${HF_ENDPOINT}" ]; then
    echo "huggingface.co 不可达，但已设置 HF_ENDPOINT 镜像"
else
    echo "huggingface.co 不可达"
    echo "             -> 如果你在网络受限环境，请设置代理，或 export HF_ENDPOINT=https://hf-mirror.com"
    echo "             -> 模型权重已下载过的话不影响使用"
fi

if [ -n "${VIRTUAL_ENV}" ]; then
    python - <<'PY' 2>/dev/null
try:
    import torch
    print(f"PyTorch    : {torch.__version__}   CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU        : {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch    : 未安装")
PY
fi
