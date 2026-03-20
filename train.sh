#!/usr/bin/env bash
# Stage 1 训练脚本：C-RADIOv4-SO400M encoder + ViT-XL decoder
#
# 用法：
#   bash train.sh              # 单卡训练
#   bash train.sh --ddp        # 多卡 DDP（自动检测 GPU 数量）
#   bash train.sh --gpus 4     # 指定 GPU 数量
#   bash train.sh --test-data  # 仅验证 dataloader（快速）
#   bash train.sh --test-run   # mini 训练（验证全流程通）

# # 1. 快速验证 dataloader（几秒）
# bash train.sh --test-data
# # 2. mini 全流程验证（跑10步）
# bash train.sh --test-run  --ddp --gpus 2
# # 3. 正式训练
# bash train.sh
# bash train.sh --ddp --gpus 2
set -e

# ─── 配置 ──────────────────────────────────────────────────────────────────────
RAE_ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="/root/paddlejob/workspace/env_run/penghaotian/datas/imagenet-1k"
ENV_ACTIVATE="/root/paddlejob/workspace/env_run/penghaotian/envs/rae/bin/activate"
CONFIG="configs/stage1/training/CRadiov4-SO400M_decXL_bs64.yaml"
RESULTS_DIR="ckpts/CRadiov4-SO400M_decXL_bs64"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-CRadiov4-SO400M_decXL_bs64}"

# ─── 颜色 / 日志 ───────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log_info() { echo -e "${CYAN}[INFO]${NC}  $*"; }
log_ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_err()  { echo -e "${RED}[ERROR]${NC} $*"; }

# ─── 参数解析 ──────────────────────────────────────────────────────────────────
USE_DDP=false; NUM_GPUS=""; MODE=train
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ddp)       USE_DDP=true; shift ;;
        --gpus)      USE_DDP=true; NUM_GPUS="$2"; shift 2 ;;
        --test-data) MODE=test_data; shift ;;
        --test-run)  MODE=test_run; shift ;;
        *) log_err "未知参数: $1"; exit 1 ;;
    esac
done
[ -z "$NUM_GPUS" ] && NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)

# ─── 环境准备 ──────────────────────────────────────────────────────────────────
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}   RAE Stage1 训练 - C-RADIOv4-SO400M  ${NC}"
echo -e "${GREEN}========================================${NC}\n"

cd "$RAE_ROOT"
source "$ENV_ACTIVATE"
export PYTHONPATH="$RAE_ROOT/src:$PYTHONPATH"

# models 软链接
if   [ ! -e "$RAE_ROOT/models" ]; then ln -s "/root/paddlejob/workspace/env_run/penghaotian/models/RAE" "$RAE_ROOT/models" && log_ok "models 软链接已创建"
elif [ -L "$RAE_ROOT/models"   ]; then log_ok "models 软链接已存在"
else log_warn "models 目录已存在（非软链接），跳过"; fi

# 必要文件检查
log_info "检查必要文件..."
ALL_OK=true
for f in "$CONFIG" \
         "models/discs/dino_vit_small_patch8_224.pth" \
         "/root/paddlejob/workspace/env_run/penghaotian/models/C-RADIOv4/C-RADIOv4-SO400M/model.safetensors"; do
    target="$f"; [[ "$f" != /* ]] && target="$RAE_ROOT/$f"
    [ -f "$target" ] && log_ok "  $f" || { log_err "  缺失: $f"; ALL_OK=false; }
done
[ "$ALL_OK" = false ] && { log_err "部分文件缺失，请检查路径后重试"; exit 1; }

# 数据集检查
[ ! -d "$DATA_DIR/train" ] && { log_err "ImageNet 训练集未找到: $DATA_DIR/train"; exit 1; }
log_ok "数据集: $DATA_DIR"

# GPU 信息
echo ""; log_info "GPU 信息:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null \
    | awk '{print "        " $0}' || echo "        (未检测到 GPU)"

# ─── 测试模式 ──────────────────────────────────────────────────────────────────
if [ "$MODE" = test_data ]; then
    log_info "验证 dataloader..."
    python - <<EOF
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder
data_path = Path("$DATA_DIR")
train_dir = data_path / "train"
ds = ImageFolder(str(train_dir if train_dir.is_dir() else data_path), transform=transforms.ToTensor())
print(f"  OK: {len(ds):,} 张图，{len(ds.classes)} 个类，前5类: {ds.classes[:5]}")
EOF
    log_ok "dataloader 验证通过"
    exit 0
fi

TRAIN_ARGS="--config $CONFIG --data-path $DATA_DIR --results-dir $RESULTS_DIR --precision bf16"
if [ "$MODE" = test_run ]; then
    log_info "启动 mini 训练（验证全流程）..."
    python src/train_stage1.py $TRAIN_ARGS --image-size 256 --max-steps 10 --no-resume
    log_ok "mini 训练通过"; exit 0
fi

# ─── 正式训练 ──────────────────────────────────────────────────────────────────
echo -e "\n  配置:  ${CYAN}$CONFIG${NC}\n  数据:  ${CYAN}$DATA_DIR${NC}\n  输出:  ${CYAN}$RESULTS_DIR${NC}"
[ "$USE_DDP" = true ] && echo -e "  模式:  ${CYAN}DDP (${NUM_GPUS} GPU)${NC}\n" || echo -e "  模式:  ${CYAN}单卡${NC}\n"
mkdir -p "$RESULTS_DIR"

if [ "$USE_DDP" = true ]; then
    MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
    log_info "启动 DDP 训练（${NUM_GPUS} 卡），port=${MASTER_PORT}..."
    torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
        src/train_stage1.py $TRAIN_ARGS --image-size 256
else
    log_info "启动单卡训练..."
    python src/train_stage1.py $TRAIN_ARGS --image-size 256
fi

echo ""; log_ok "训练完成！检查点保存于: $RESULTS_DIR"