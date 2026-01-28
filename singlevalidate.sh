#!/bin/bash
#===============================================================================
# S2V 单机单卡推理启动脚本
# 直接使用 python 启动，不使用 mpirun
#===============================================================================

set -e  # 出错即退出

# ======================== 1. 环境及显卡设置 ========================
# 指定使用的显卡，0 表示第一块显卡
export CUDA_VISIBLE_DEVICES=0

export http_proxy=http://10.66.16.238:11080 
export https_proxy=http://10.66.16.238:11080
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
export PATH="/m2v_intern/mengzijie/env/wan2.2/bin:$PATH"
export PATH=/opt/xray/deps:$PATH
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore::FutureWarning"

# 分布式辅助变量（单卡通常设为 1 和 0）
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29502

# 生成统一的时间戳
export OUTPUT_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "Output timestamp: $OUTPUT_TIMESTAMP"

# ======================== 2. 推理参数配置 ========================
# 数据路径
IMAGE_LIST_PATH="/m2v_intern/mengzijie/DiffSynth-Studio/data/all_id_test_shuf2.txt" 
AUDIO_DIR="/m2v_intern/mengzijie/DiffSynth-Studio/data/audio" 
OUTPUT_BASE_DIR="output"

# 模型参数
CKPT_PATH="/m2v_intern/mengzijie/DiffSynth-Studio/models/train/s2v_v4_32gpu/step-10000.safetensors"
MODEL_ID="Wan-AI/Wan2.2-S2V-14B"

# 推理参数
NUM_FRAMES=25
HEIGHT=720
WIDTH=640
NUM_INFERENCE_STEPS=1
SEED=0
FPS=16
QUALITY=5
# 音频参数
NUM_AUDIOS_PER_IMAGE=1
AUDIO_SAMPLE_RATE=16000

# ======================== 3. 执行推理 ========================
cd /m2v_intern/mengzijie/DiffSynth-Studio/
PYTHON_EXE="/m2v_intern/mengzijie/env/wan2.2/bin/python"

# 确保日志目录存在
mkdir -p logs

echo "=============================================="
echo "Starting Single GPU Inference..."
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "Checkpoint: $CKPT_PATH"
echo "=============================================="

# 直接运行 Python
$PYTHON_EXE -u examples/wanvideo/model_training/validate_full/s2vinfer.py \
    --image_list_path "$IMAGE_LIST_PATH" \
    --audio_dir "$AUDIO_DIR" \
    --output_base_dir "$OUTPUT_BASE_DIR" \
    --output_timestamp "$OUTPUT_TIMESTAMP" \
    --ckpt_path "$CKPT_PATH" \
    --model_id "$MODEL_ID" \
    --num_frames $NUM_FRAMES \
    --height $HEIGHT \
    --width $WIDTH \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --seed $SEED \
    --fps $FPS \
    --quality $QUALITY \
    --num_audios_per_image $NUM_AUDIOS_PER_IMAGE \
    --audio_sample_rate $AUDIO_SAMPLE_RATE \
2>&1 | tee logs/s2v_inference_single_${OUTPUT_TIMESTAMP}.log

echo "=============================================="
echo "Inference finished!"
echo "Log file: logs/s2v_inference_single_${OUTPUT_TIMESTAMP}.log"
echo "=============================================="