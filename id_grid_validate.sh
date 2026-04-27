#!/bin/bash
#===============================================================================
# ID Grid I2V 多机多卡分布式推理启动脚本
#===============================================================================

set -e  # 出错即退出

# ======================== 1. 基础信息获取 ========================
# BDY机器注意：
# if [ "${X_ROLE}" == "launcher" ] || [ "${ROLE_NAME}" == "master" ]; then wget https://halo.corp.kuaishou.com/api/cloud-storage/v1/public-objects/user-cloud-storage/xray/install_xray.sh -O install_xray.sh && bash install_xray.sh "all"; fi && if [[ "$PATH" != "/opt/xray/deps"* ]]; then export PATH=/opt/xray/deps:$PATH; fi;
export http_proxy=http://10.66.16.238:11080 
export https_proxy=http://10.66.16.238:11080
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
hostfile=/etc/mpi/hostfile
# Port=$(grep -v '^#' /etc/ssh/ssh_config | grep 'Port' | awk '{print $2}' | head -n 1)

# 总进程数（总GPU数）
np=$(cat $hostfile | cut -d'=' -f2 | awk '{sum += $0} END {print sum}')
echo "=============================================="
echo "Total GPUs (processes): $np"
echo "=============================================="

# 主节点地址（用于分布式握手）
master_addr=$(head -n 1 $hostfile | awk '{print $1}')
echo "Master address: $master_addr"

# ======================== 2. 环境变量设置 ========================
export PATH="/m2v_intern/mengzijie/env/wan2.2/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore::FutureWarning"

# 将 get_smpl_motion 所在的父目录加入 Python 搜索路径
export PYTHONPATH=$PYTHONPATH:/m2v_intern/mengzijie/DiffSynth-Studio/get_smpl_motion

export OUTPUT_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "Output timestamp: $OUTPUT_TIMESTAMP"

# ======================== 3. 推理参数配置 ========================
# 数据路径 (更新为 CSV 路径)
# DATASET_METADATA_PATH="dataset/all_id_test_shuf2.csv"
DATASET_METADATA_PATH="/m2v_intern/mengzijie/DiffSynth-Studio/dataset/testdataset/v4.0/testdata.csv"
OUTPUT_BASE_DIR="output_id_grid"

# 模型参数
# 修改为训练好的 Checkpoint 路径
# CKPT_PATH="/ytech_m2v4_hdd/mengzijie/DiffSynth-Studio/models/train/i2v_v1.0/step-1000.safetensors" 
CKPT_PATH="/ytech_m2v4_hdd/mengzijie/DiffSynth-Studio/models/train/i2v_v4.0/step-4200.safetensors" 
# ENABLE_ID_GRID=false
ENABLE_ID_GRID=true

MODEL_ID="Wan-AI/Wan2.1-I2V-14B-720P"

# 推理参数 (等效面积优先)
NUM_FRAMES=49
MAX_PIXELS=268800  # 480x560
NUM_INFERENCE_STEPS=40
SEED=42
FPS=15
QUALITY=5

# ID Grid 参数
ID_GRID_MAX_PIXELS=268800
ID_GRID_NUM_FRAMES=1

# ======================== 4. 准备工作 ========================
cd /m2v_intern/mengzijie/DiffSynth-Studio/
PYTHON_EXE="/m2v_intern/mengzijie/env/wan2.2/bin/python"

# # ======================== 5. 执行 mpirun BDY========================
# export PATH=/opt/xray/deps:$PATH
# export NCCL_TOPO_FILE="/share/huzhiwen/baidu/topo_a800_hpc_bcc.xml"
# mpirun --allow-run-as-root -np $np \
#     -mca plm_rsh_args "-p ${Port}" \
#     -hostfile $hostfile \
#     -bind-to none -map-by slot \
#     --mca btl tcp,self \
#     -x HOROVOD_MPI_THREADS_DISABLE=1 \
#     -x MPI_THREAD_SINGLE=1 \
#     -x NCCL_IB_DISABLE=0 \
#     -x NCCL_IB_GID_INDEX=3 \
#     -x NCCL_MIN_NCHANNELS=16 \
#     -x NCCL_IB_HCA=mlx5 \
#     -x NCCL_IB_QPS_PER_CONNECTION=4 \
#     -x NCCL_IB_TIMEOUT=32 \
#     -x NCCL_DEBUG=WARN \
#     -x PATH \
#     -x LD_LIBRARY_PATH \
#     -x http_proxy \
#     -x https_proxy \
#     -x no_proxy \
#     -x MASTER_ADDR=$master_addr \
#     -x MASTER_PORT=29509 \
#     -x WORLD_SIZE=$np \
#     -x OUTPUT_TIMESTAMP \
#     -x NCCL_TOPO_FILE \
#     $PYTHON_EXE -u examples/wanvideo/model_training/validate_full/id_grid_infer.py \
#         --dataset_metadata_path "$DATASET_METADATA_PATH" \
#         --output_base_dir "$OUTPUT_BASE_DIR" \
#         --output_timestamp "$OUTPUT_TIMESTAMP" \
#         --ckpt_path "$CKPT_PATH" \
#         --model_id "$MODEL_ID" \
#         --num_frames $NUM_FRAMES \
#         --max_pixels $MAX_PIXELS \
#         --num_inference_steps $NUM_INFERENCE_STEPS \
#         --seed $SEED \
#         --fps $FPS \
#         --quality $QUALITY \
#         --enable_id_grid \
#         --id_grid_max_pixels $ID_GRID_MAX_PIXELS \
#         --id_grid_num_frames $ID_GRID_NUM_FRAMES \
#     2>&1 | tee logs/id_grid_inference_${OUTPUT_TIMESTAMP}.log

# echo "=============================================="
# echo "Inference finished!"
# echo "Output directory: ${OUTPUT_BASE_DIR}/output_${OUTPUT_TIMESTAMP}"
# echo "=============================================="



# ======================== 5. 执行 mpirun A800 ========================
mpirun --allow-run-as-root -np $np \
    -mca plm_rsh_args "-F /etc/ssh/ssh_config" \
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
    -x NCCL_DEBUG=INFO \
    -x NCCL_SOCKET_IFNAME=eth0 \
    -x PATH \
    -x PYTHONPATH \
    -x LD_LIBRARY_PATH \
    -x http_proxy \
    -x https_proxy \
    -x no_proxy \
    -x MASTER_ADDR=$master_addr \
    -x MASTER_PORT=29509 \
    -x WORLD_SIZE=$np \
    -x OUTPUT_TIMESTAMP \
    $PYTHON_EXE -u examples/wanvideo/model_training/validate_full/id_grid_infer.py \
        --dataset_metadata_path "$DATASET_METADATA_PATH" \
        --output_base_dir "$OUTPUT_BASE_DIR" \
        --output_timestamp "$OUTPUT_TIMESTAMP" \
        --ckpt_path "$CKPT_PATH" \
        --model_id "$MODEL_ID" \
        --num_frames $NUM_FRAMES \
        --max_pixels $MAX_PIXELS \
        --num_inference_steps $NUM_INFERENCE_STEPS \
        --seed $SEED \
        --fps $FPS \
        --quality $QUALITY \
        --enable_id_grid \
        --id_grid_max_pixels $ID_GRID_MAX_PIXELS \
        --id_grid_num_frames $ID_GRID_NUM_FRAMES \
    2>&1 | tee logs/id_grid_inference_${OUTPUT_TIMESTAMP}.log

echo "=============================================="
echo "Inference finished!!"
echo "Output directory: ${OUTPUT_BASE_DIR}/output_${OUTPUT_TIMESTAMP}"
echo "=============================================="



# cd /m2v_intern/mengzijie/DiffSynth-Studio/

# export MASTER_ADDR=127.0.0.1
# export MASTER_PORT=29509
# export WORLD_SIZE=8
# export OUTPUT_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# export PYTHONPATH=$PYTHONPATH:/m2v_intern/mengzijie/DiffSynth-Studio/get_smpl_motion

# mpirun --allow-run-as-root -np 8 \
#     --mca btl tcp,self \
#     -x PATH="/m2v_intern/mengzijie/env/wan2.2/bin:$PATH" \
#     -x PYTHONPATH \
#     -x LD_LIBRARY_PATH \
#     -x PYTHONUNBUFFERED=1 \
#     -x MASTER_ADDR \
#     -x MASTER_PORT \
#     -x WORLD_SIZE \
#     -x OUTPUT_TIMESTAMP \
#     -x http_proxy=http://10.66.16.238:11080 \
#     -x https_proxy=http://10.66.16.238:11080 \
#     -x no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com \
#     /m2v_intern/mengzijie/env/wan2.2/bin/python -u /m2v_intern/mengzijie/DiffSynth-Studio/examples/wanvideo/model_training/validate_full/id_grid_infer.py \
#         --dataset_metadata_path "dataset/all_id_test_shuf2.csv" \
#         --output_base_dir "output_id_grid" \
#         --output_timestamp "$OUTPUT_TIMESTAMP" \
#         --ckpt_path "/ytech_m2v4_hdd/mengzijie/DiffSynth-Studio/models/train/i2v_v4.0/step-4200.safetensors" \
#         --model_id "Wan-AI/Wan2.1-I2V-14B-720P" \
#         --num_frames 49 \
#         --max_pixels 268800 \
#         --num_inference_steps 40 \
#         --seed 42 \
#         --fps 15 \
#         --quality 5 \
#         --enable_id_grid \
#         --id_grid_max_pixels 268800 \
#         --id_grid_num_frames 1