#!/usr/bin/env bash
# Stage 2 训练脚本：C-RADIOv4-SO400M encoder + ViT-XL decoder + DiT-XL diffusion
#
# 用法：
# # Step 1：提取 decoder（几分钟）
# bash train_stage2.sh --extract-decoder
#
# # Step 2：计算 latent 统计（跑完 ImageNet train 一遍，约 30-60 分钟）
# bash train_stage2.sh --calc-stat --gpus 2
#
# # Step 3：正式训练
# bash train_stage2.sh --ddp --gpus 2
#
# # 或者三步一次性跑完
# bash train_stage2.sh --all --ddp --gpus 2

set -e

# ─── 配置 ──────────────────────────────────────────────────────────────────────
RAE_ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="/root/paddlejob/workspace/env_run/penghaotian/datas/imagenet-1k"
ENV_ACTIVATE="/root/paddlejob/workspace/env_run/penghaotian/envs/rae/bin/activate"

STAGE1_CONFIG="configs/stage1/training/CRadiov4-SO400M_decXL_bs128.yaml"
STAGE1_CKPT="ckpts/CRadiov4-SO400M_decXL_bs128/CRadiov4-SO400M_decXL_bs128/checkpoints/ep-last.pt"

STAGE2_CONFIG="configs/stage2/training/ImageNet256/DiTDH-XL_CRadiov4-SO400M.yaml"
RESULTS_DIR="ckpts/stage2"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-DiTDH-XL_CRadiov4-SO400M}"

DECODER_OUT="models/decoders/cradio/SO400M/ViTXL/model.pt"
STAT_OUT_DIR="models/stats/cradio/SO400M"
STAT_FOLDER="imagenet1k"   # SAVE_FOLDER env → normalization_stats.pt 保存在 ${STAT_OUT_DIR}/${STAT_FOLDER}/

# ─── 颜色 / 日志 ───────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log_info() { echo -e "${CYAN}[INFO]${NC}  $*"; }
log_ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_err()  { echo -e "${RED}[ERROR]${NC} $*"; }

# ─── 参数解析 ──────────────────────────────────────────────────────────────────
DO_EXTRACT=false; DO_STAT=false; DO_TRAIN=false
USE_DDP=false; NUM_GPUS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --extract-decoder) DO_EXTRACT=true; shift ;;
        --calc-stat)       DO_STAT=true;    shift ;;
        --all)             DO_EXTRACT=true; DO_STAT=true; DO_TRAIN=true; shift ;;
        --ddp)             USE_DDP=true;    shift ;;
        --gpus)            USE_DDP=true; NUM_GPUS="$2"; shift 2 ;;
        *)  # --ddp/--gpus 放到 --train 后也能生效，视为训练触发
            log_err "未知参数: $1"; exit 1 ;;
    esac
done
# 如果没有指定 extract/stat，默认执行训练
if ! $DO_EXTRACT && ! $DO_STAT; then DO_TRAIN=true; fi
[ -z "$NUM_GPUS" ] && NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)

# ─── 环境准备 ──────────────────────────────────────────────────────────────────
cd "$RAE_ROOT"
source "$ENV_ACTIVATE"
export PYTHONPATH="$RAE_ROOT/src:$PYTHONPATH"

# models 软链接
if   [ ! -e "$RAE_ROOT/models" ]; then ln -s "/root/paddlejob/workspace/env_run/penghaotian/models/RAE" "$RAE_ROOT/models" && log_ok "models 软链接已创建"
elif [ -L "$RAE_ROOT/models"   ]; then log_ok "models 软链接已存在"
else log_warn "models 目录已存在（非软链接），跳过"; fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  RAE Stage2 训练 - C-RADIOv4-SO400M   ${NC}"
echo -e "${GREEN}========================================${NC}\n"

# ─── Step 1：提取 EMA decoder ─────────────────────────────────────────────────
if $DO_EXTRACT; then
    log_info "Step 1: 从 stage1 checkpoint 提取 EMA decoder..."
    [ ! -f "$STAGE1_CKPT" ] && { log_err "Stage1 checkpoint 不存在: $STAGE1_CKPT"; exit 1; }

    python src/extract_decoder.py \
        --config "$STAGE1_CONFIG" \
        --ckpt   "$STAGE1_CKPT" \
        --use-ema \
        --out    "$DECODER_OUT"

    log_ok "Decoder 已保存至: $DECODER_OUT"
fi

# ─── Step 2：计算 latent 归一化统计 ──────────────────────────────────────────
if $DO_STAT; then
    log_info "Step 2: 计算 latent 归一化统计（需要 DDP，约跑完 ImageNet train 一遍）..."
    [ ! -d "$DATA_DIR/train" ] && { log_err "ImageNet train 目录不存在: $DATA_DIR/train"; exit 1; }

    MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
    SAVE_FOLDER="$STAT_FOLDER" torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port="$MASTER_PORT" \
        src/calculate_stat.py \
        --config            "$STAGE1_CONFIG" \
        --data-path         "$DATA_DIR/train" \
        --image-size        256 \
        --per-proc-batch-size 64 \
        --precision         bf16 \
        --sample-dir        "$STAT_OUT_DIR"

    STAT_PATH="$STAT_OUT_DIR/$STAT_FOLDER/normalization_stats.pt"
    [ ! -f "$STAT_PATH" ] && { log_err "stats 文件未生成: $STAT_PATH"; exit 1; }
    log_ok "Stats 已保存至: $STAT_PATH"
fi

# ─── Step 3：Stage2 训练 ──────────────────────────────────────────────────────
if $DO_TRAIN; then
    # 前置检查
    log_info "检查必要文件..."
    ALL_OK=true
    for f in "$STAGE2_CONFIG" \
             "$DECODER_OUT" \
             "${STAT_OUT_DIR}/${STAT_FOLDER}/normalization_stats.pt"; do
        target="$f"; [[ "$f" != /* ]] && target="$RAE_ROOT/$f"
        [ -f "$target" ] && log_ok "  $f" || { log_err "  缺失: $f  （请先运行 --extract-decoder / --calc-stat）"; ALL_OK=false; }
    done
    [ "$ALL_OK" = false ] && exit 1

    [ ! -d "$DATA_DIR/train" ] && { log_err "ImageNet train 目录不存在: $DATA_DIR/train"; exit 1; }

    mkdir -p "$RESULTS_DIR"
    TRAIN_ARGS="--config $STAGE2_CONFIG --data-path $DATA_DIR --results-dir $RESULTS_DIR --precision bf16 --compile"

    echo -e "\n  配置:  ${CYAN}$STAGE2_CONFIG${NC}"
    echo -e "  数据:  ${CYAN}$DATA_DIR${NC}"
    echo -e "  输出:  ${CYAN}$RESULTS_DIR/$EXPERIMENT_NAME${NC}"

    if [ "$USE_DDP" = true ]; then
        MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
        echo -e "  模式:  ${CYAN}DDP (${NUM_GPUS} GPU)${NC}\n"
        log_info "启动 DDP Stage2 训练，port=${MASTER_PORT}..."
        torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
            src/train.py $TRAIN_ARGS
    else
        echo -e "  模式:  ${CYAN}单卡${NC}\n"
        log_info "启动单卡 Stage2 训练..."
        python src/train.py $TRAIN_ARGS
    fi

    echo ""; log_ok "Stage2 训练完成！检查点保存于: $RESULTS_DIR"
fi
