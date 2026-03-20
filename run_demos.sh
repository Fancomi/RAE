#!/usr/bin/env bash
# RAE Demo Runner
# 逐步运行各个 demo，每步完成后提示继续
# Demo 1	Stage1 重建，DINOv2-B 编码器	256 分辨率
# Demo 2	Stage1 重建，DINOv2-B 编码器	512 分辨率
# Demo 3	Stage1 重建，MAE 编码器	256 分辨率
# Demo 4	Stage1 重建，SigLIP2 编码器	256 分辨率
# Demo 5	Stage2 生成，CFG 模式	scale=1.0
# Demo 6	Stage2 生成，Autoguidance 模式	scale=1.42，效果更好
# Demo 7	Stage2 生成，512x512 高分辨率	显存需求较大
set -e

RAE_ROOT="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="/root/paddlejob/workspace/env_run/penghaotian/models/RAE"
ENV_ACTIVATE="/root/paddlejob/workspace/env_run/penghaotian/envs/rae/bin/activate"
OUTPUT_DIR="$RAE_ROOT/demo_outputs"

# 颜色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

log_info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_err()   { echo -e "${RED}[ERROR]${NC} $*"; }

pause() {
    echo ""
    read -r -p "$(echo -e "${YELLOW}>>> 按 Enter 继续下一步，Ctrl+C 退出...${NC}")"
    echo ""
}

# ─── 准备工作 ────────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}         RAE Demo Runner                ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

cd "$RAE_ROOT"

# 激活环境
log_info "激活 Python 环境..."
source "$ENV_ACTIVATE"
log_ok "环境已激活: $(which python)"

# 设置 PYTHONPATH
export PYTHONPATH="$RAE_ROOT/src:$PYTHONPATH"
log_ok "PYTHONPATH 已设置"

# 创建 models 软链接
if [ ! -e "$RAE_ROOT/models" ]; then
    log_info "创建 models 软链接..."
    ln -s "$MODELS_DIR" "$RAE_ROOT/models"
    log_ok "软链接已创建: $RAE_ROOT/models -> $MODELS_DIR"
elif [ -L "$RAE_ROOT/models" ]; then
    log_ok "models 软链接已存在"
else
    log_warn "models 目录已存在（非软链接），跳过创建"
fi

# 检查关键权重文件
log_info "检查关键权重文件..."
REQUIRED_FILES=(
    "models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt"
    "models/stats/dinov2/wReg_base/imagenet1k/stat.pt"
    "models/DiTs/Dinov2/wReg_base/ImageNet256/DiTDH-XL/stage2_model.pt"
)
ALL_OK=true
for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$RAE_ROOT/$f" ]; then
        log_ok "  $f"
    else
        log_err "  缺失: $f"
        ALL_OK=false
    fi
done
if [ "$ALL_OK" = false ]; then
    log_err "部分权重文件缺失，请检查 $MODELS_DIR 目录"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo ""
log_info "输出目录: $OUTPUT_DIR"
log_info "GPU 信息:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null \
    | awk '{print "        " $0}' || echo "        (未检测到 GPU，将使用 CPU)"

echo ""
echo -e "${GREEN}准备完毕，开始运行 Demo${NC}"

# ─── Demo 1：Stage1 重建 - 内置猫咪图片 ─────────────────────────────────────

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} Demo 1: Stage1 图片重建（内置测试图）${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "  编码器: DINOv2-B"
echo -e "  输入:   assets/pixabay_cat.png"
echo -e "  输出:   $OUTPUT_DIR/demo1_recon_cat.png"
echo ""
pause

log_info "运行 Stage1 重建..."
python src/stage1_sample.py \
    --config configs/stage1/pretrained/DINOv2-B.yaml \
    --image  assets/pixabay_cat.png \
    --output "$OUTPUT_DIR/demo1_recon_cat.png"

log_ok "Demo 1 完成！重建图片已保存至 $OUTPUT_DIR/demo1_recon_cat.png"

# ─── Demo 2：Stage1 重建 - 512 分辨率 ────────────────────────────────────────

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} Demo 2: Stage1 图片重建（512 分辨率）${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "  编码器: DINOv2-B (512 版本)"
echo -e "  输入:   assets/pixabay_cat.png"
echo -e "  输出:   $OUTPUT_DIR/demo2_recon_cat_512.png"
echo ""
pause

log_info "运行 Stage1 重建（512分辨率）..."
python src/stage1_sample.py \
    --config configs/stage1/pretrained/DINOv2-B_512.yaml \
    --image  assets/pixabay_cat.png \
    --output "$OUTPUT_DIR/demo2_recon_cat_512.png"

log_ok "Demo 2 完成！输出: $OUTPUT_DIR/demo2_recon_cat_512.png"

# ─── Demo 3：Stage1 重建 - MAE 编码器 ────────────────────────────────────────

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} Demo 3: Stage1 图片重建（MAE 编码器）${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "  编码器: MAE-B/16"
echo -e "  输出:   $OUTPUT_DIR/demo3_recon_cat_mae.png"
echo ""
pause

log_info "运行 Stage1 重建（MAE 编码器）..."
python src/stage1_sample.py \
    --config configs/stage1/pretrained/MAE.yaml \
    --image  assets/pixabay_cat.png \
    --output "$OUTPUT_DIR/demo3_recon_cat_mae.png"

log_ok "Demo 3 完成！输出: $OUTPUT_DIR/demo3_recon_cat_mae.png"

# ─── Demo 4：Stage1 重建 - SigLIP2 编码器 ────────────────────────────────────

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} Demo 4: Stage1 图片重建（SigLIP2 编码器）${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "  编码器: SigLIP2-B"
echo -e "  输出:   $OUTPUT_DIR/demo4_recon_cat_siglip2.png"
echo ""
pause

log_info "运行 Stage1 重建（SigLIP2 编码器）..."
python src/stage1_sample.py \
    --config configs/stage1/pretrained/SigLIP2.yaml \
    --image  assets/pixabay_cat.png \
    --output "$OUTPUT_DIR/demo4_recon_cat_siglip2.png"

log_ok "Demo 4 完成！输出: $OUTPUT_DIR/demo4_recon_cat_siglip2.png"

# ─── Demo 5：Stage2 图像生成 - CFG 模式 ──────────────────────────────────────

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} Demo 5: Stage2 图像生成（CFG 模式）  ${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "  模型:   DiTDH-XL + DINOv2-B，ImageNet 256x256"
echo -e "  类别:   207 (金毛犬), 360 (貂)"
echo -e "  引导:   CFG scale=1.0"
echo -e "  输出:   $OUTPUT_DIR/demo5_gen_cfg.png"
echo ""
pause

log_info "运行 Stage2 生成（CFG）..."
python src/sample.py \
    --config configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B.yaml \
    --seed 42

mv sample.png "$OUTPUT_DIR/demo5_gen_cfg.png" 2>/dev/null || true
log_ok "Demo 5 完成！输出: $OUTPUT_DIR/demo5_gen_cfg.png"

# ─── Demo 6：Stage2 图像生成 - Autoguidance 模式 ─────────────────────────────

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} Demo 6: Stage2 图像生成（Autoguidance）${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "  模型:   DiTDH-XL + DINOv2-B，ImageNet 256x256"
echo -e "  引导:   Autoguidance scale=1.42（效果更好）"
echo -e "  输出:   $OUTPUT_DIR/demo6_gen_autoguidance.png"
echo ""
pause

log_info "运行 Stage2 生成（Autoguidance）..."
python src/sample.py \
    --config configs/stage2/sampling/ImageNet256/DiTDHXL-DINOv2-B_AG.yaml \
    --seed 42

mv sample.png "$OUTPUT_DIR/demo6_gen_autoguidance.png" 2>/dev/null || true
log_ok "Demo 6 完成！输出: $OUTPUT_DIR/demo6_gen_autoguidance.png"

# ─── Demo 7：Stage2 图像生成 - 512 分辨率 ────────────────────────────────────

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} Demo 7: Stage2 图像生成（512x512）   ${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "  模型:   DiTDH-XL + DINOv2-B，ImageNet 512x512"
echo -e "  引导:   Autoguidance scale=1.5"
echo -e "  输出:   $OUTPUT_DIR/demo7_gen_512.png"
echo -e "  注意:   512 分辨率显存需求较大，建议 GPU >= 24GB"
echo ""
pause

log_info "运行 Stage2 生成（512x512）..."
python src/sample.py \
    --config configs/stage2/sampling/ImageNet512/DiTDH-XL_DINOv2-B_decXL_AG.yaml \
    --seed 42

mv sample.png "$OUTPUT_DIR/demo7_gen_512.png" 2>/dev/null || true
log_ok "Demo 7 完成！输出: $OUTPUT_DIR/demo7_gen_512.png"

# ─── 汇总 ────────────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}         所有 Demo 运行完毕！            ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "输出文件汇总："
ls -lh "$OUTPUT_DIR/"
echo ""
log_ok "全部完成！输出目录: $OUTPUT_DIR"