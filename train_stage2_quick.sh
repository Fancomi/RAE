#!/usr/bin/env bash
# Stage 2 对比实验训练脚本（quick版）
# 使用 DiT-S decoder，1/10 训练数据，20 epochs，5000张eval，方便快速对比 DINOv2 vs CRadio
#
# 用法：
#   bash train_stage2_quick.sh --dino          # 只跑 DINOv2
#   bash train_stage2_quick.sh --cradio        # 只跑 CRadio
#   bash train_stage2_quick.sh --all           # 顺序跑两个实验（半天以内）
#   bash train_stage2_quick.sh --all --gpus 2  # 指定GPU数

set -e

RAE_ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="/root/paddlejob/workspace/env_run/penghaotian/datas/imagenet-1k"
ENV_ACTIVATE="/root/paddlejob/workspace/env_run/penghaotian/envs/rae/bin/activate"
RESULTS_DIR="ckpts/stage2"

CONFIG_DINO="configs/stage2/training/ImageNet256/DiTDH-S_DINOv2-B_quick.yaml"
CONFIG_CRADIO="configs/stage2/training/ImageNet256/DiTDH-S_CRadiov4-SO400M_quick.yaml"
NAME_DINO="DiTDH-S_DINOv2-B_quick"
NAME_CRADIO="DiTDH-S_CRadiov4-SO400M_quick"

GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
log_info() { echo -e "${CYAN}[INFO]${NC}  $*"; }
log_ok()   { echo -e "${GREEN}[OK]${NC}    $*"; }
log_err()  { echo -e "${RED}[ERROR]${NC} $*"; }

RUN_DINO=false; RUN_CRADIO=false; NUM_GPUS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dino)    RUN_DINO=true;   shift ;;
        --cradio)  RUN_CRADIO=true; shift ;;
        --all)     RUN_DINO=true; RUN_CRADIO=true; shift ;;
        --gpus)    NUM_GPUS="$2"; shift 2 ;;
        *) log_err "未知参数: $1"; exit 1 ;;
    esac
done
if ! $RUN_DINO && ! $RUN_CRADIO; then
    log_err "请指定 --dino / --cradio / --all"
    exit 1
fi
[ -z "$NUM_GPUS" ] && NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo 1)

cd "$RAE_ROOT"
source "$ENV_ACTIVATE"
export PYTHONPATH="$RAE_ROOT/src:$PYTHONPATH"

if [ ! -e "$RAE_ROOT/models" ]; then
    ln -s "/root/paddlejob/workspace/env_run/penghaotian/models/RAE" "$RAE_ROOT/models"
fi

run_experiment() {
    local config="$1"
    local name="$2"
    log_info "启动实验: ${name}  (config=${config}, ${NUM_GPUS} GPU)"
    MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
    EXPERIMENT_NAME="$name" torchrun \
        --nproc_per_node="$NUM_GPUS" \
        --master_port="$MASTER_PORT" \
        src/train.py \
        --config "$config" \
        --data-path "$DATA_DIR" \
        --results-dir "$RESULTS_DIR" \
        --precision bf16 \
        --compile
    log_ok "实验完成: ${name}  →  ${RESULTS_DIR}/${name}"
}

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  RAE Stage2 Quick对比实验              ${NC}"
echo -e "${GREEN}  数据: 1/10  Epochs: 20  EvalN: 5000  ${NC}"
echo -e "${GREEN}========================================${NC}\n"

$RUN_DINO   && run_experiment "$CONFIG_DINO"   "$NAME_DINO"
$RUN_CRADIO && run_experiment "$CONFIG_CRADIO" "$NAME_CRADIO"

echo ""
log_ok "全部实验完成！指标文件："
$RUN_DINO   && echo "  ${RESULTS_DIR}/${NAME_DINO}/metrics.csv"
$RUN_CRADIO && echo "  ${RESULTS_DIR}/${NAME_CRADIO}/metrics.csv"
