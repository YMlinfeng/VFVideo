#!/bin/bash
#===============================================================================
# S2V 多机多卡分布式推理启动脚本
# 使用 mpirun 拉起多进程，每个进程处理不同的图片
#===============================================================================

set -e  # 出错即退出

# ======================== 1. 基础信息获取 ========================
hostfile=/etc/mpi/hostfile
Port=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2)

# 总进程数（总GPU数）
np=$(cat $hostfile | cut -d'=' -f2 | awk '{sum += $0} END {print sum}')
echo "=============================================="
echo "Total GPUs (processes): $np"
echo "=============================================="

# 主节点地址（用于分布式握手）
master_addr=$(head -n 1 $hostfile | awk '{print $1}')
echo "Master address: $master_addr"

# 显示 hostfile 内容
echo "Hostfile content:"
cat $hostfile
echo "=============================================="

# ======================== 2. 环境变量设置 ========================
export http_proxy=http://10.66.16.238:11080 
export https_proxy=http://10.66.16.238:11080
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
export PATH="/m2v_intern/mengzijie/env/wan2.2/bin:$PATH"
export PYTHONUNBUFFERED=1

# 生成统一的时间戳，确保所有进程使用相同的输出目录
export OUTPUT_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "Output timestamp: $OUTPUT_TIMESTAMP"

# ======================== 3. 推理参数配置 ========================
# 数据路径
IMAGE_LIST_PATH="/m2v_intern/mengzijie/DiffSynth-Studio/data/all_id_test_shuf2.txt"
AUDIO_DIR="/m2v_intern/mengzijie/DiffSynth-Studio/data/audio"
OUTPUT_BASE_DIR="output"

# 模型参数
CKPT_PATH="/m2v_intern/mengzijie/DiffSynth-Studio/models/train/Wan2.2-S2V-14B_32gpu/step-3200.safetensors"
MODEL_ID="Wan-AI/Wan2.2-S2V-14B"

# 推理参数
NUM_FRAMES=57
HEIGHT=640
WIDTH=560
NUM_INFERENCE_STEPS=40
SEED=0
FPS=16
QUALITY=5

# 音频参数
NUM_AUDIOS_PER_IMAGE=1
AUDIO_SAMPLE_RATE=16000

# Prompt 参数
# PROMPT="a person is singing or talking"
# NEGATIVE_PROMPT="画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

# ======================== 4. 准备工作 ========================
cd /m2v_intern/mengzijie/DiffSynth-Studio/
PYTHON_EXE="/m2v_intern/mengzijie/env/wan2.2/bin/python"

# 创建日志目录
# mkdir -p logs

# 显示将要使用的配置
echo "=============================================="
echo "Inference Configuration:"
echo "  Image list: $IMAGE_LIST_PATH"
echo "  Audio dir: $AUDIO_DIR"
echo "  Output dir: ${OUTPUT_BASE_DIR}/output_${OUTPUT_TIMESTAMP}"
echo "  Checkpoint: $CKPT_PATH"
echo "  Resolution: ${WIDTH}x${HEIGHT}"
echo "  Frames: $NUM_FRAMES"
echo "  Audios per image: $NUM_AUDIOS_PER_IMAGE"
echo "=============================================="

# ======================== 5. 执行 mpirun ========================
mpirun --allow-run-as-root -np $np \
    -mca plm_rsh_args "-p ${Port}" \
    -hostfile $hostfile \
    -bind-to none -map-by slot \
    --mca btl tcp,self \
    -x HOROVOD_MPI_THREADS_DISABLE=1 \
    -x MPI_THREAD_SINGLE=1 \
    -x NCCL_IB_DISABLE=0 \
    -x NCCL_IB_GID_INDEX=3 \
    -x NCCL_MIN_NCHANNELS=16 \
    -x NCCL_IB_HCA=mlx5 \
    -x NCCL_IB_QPS_PER_CONNECTION=4 \
    -x NCCL_IB_TIMEOUT=32 \
    -x NCCL_DEBUG=WARN \
    -x PATH \
    -x LD_LIBRARY_PATH \
    -x http_proxy \
    -x https_proxy \
    -x no_proxy \
    -x MASTER_ADDR=$master_addr \
    -x MASTER_PORT=29502 \
    -x WORLD_SIZE=$np \
    -x OUTPUT_TIMESTAMP \
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
    2>&1 | tee logs/s2v_inference_${OUTPUT_TIMESTAMP}.log

echo "=============================================="
echo "Inference finished!"
echo "Output directory: ${OUTPUT_BASE_DIR}/output_${OUTPUT_TIMESTAMP}"
echo "Log file: logs/s2v_inference_${OUTPUT_TIMESTAMP}.log"
echo "=============================================="