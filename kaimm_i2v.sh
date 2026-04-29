#!/bin/bash

# 1. 基础信息获取
hostfile=/etc/mpi/hostfile
Port=$(cat /etc/ssh/ssh_config | grep 'Port' | cut -d'"' -f2)
# 总进程数（总GPU数）
np=$(cat $hostfile | cut -d'=' -f2 | awk '{sum += $0} END {print sum}')
echo "Total GPUs: $np"
# 主节点地址（用于分布式握手）
master_addr=$(head -n 1 $hostfile | awk '{print $1}')

# 2. 环境变量设置
export http_proxy=http://10.66.16.238:11080 
export https_proxy=http://10.66.16.238:11080
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
export PATH="/m2v_intern/mengzijie/env/wan2.2/bin:$PATH"
export PATH=/opt/xray/deps:$PATH
export ACCELERATE_CONFIG_FILE="/m2v_intern/mengzijie/DiffSynth-Studio/examples/wanvideo/model_training/full/accelerate_config_14B_multigpu.yaml"
export DEEPSPEED_FORCE_MULTI_NODE=1
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore::FutureWarning"
export MODELSCOPE_HTTP_TIMEOUT=300
export MODELSCOPE_MAX_RETRIES=10
# export NCCL_TOPO_FILE="/share/huzhiwen/baidu/topo_a800_hpc_bcc.xml"
# cat $ACCELERATE_CONFIG_FILE

# 3. 准备 Python 启动指令
cd /m2v_intern/mengzijie/DiffSynth-Studio/
PYTHON_EXE="/m2v_intern/mengzijie/env/wan2.2/bin/python"

# 4. 执行 mpirun
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
    -x PATH \
    -x LD_LIBRARY_PATH \
    -x http_proxy \
    -x https_proxy \
    -x no_proxy \
    -x MASTER_ADDR=$master_addr \
    -x MASTER_PORT=29511 \
    -x ACCELERATE_CONFIG_FILE \
    -x DEEPSPEED_FORCE_MULTI_NODE \
    -x WORLD_SIZE=$np \
    $PYTHON_EXE -u examples/wanvideo/model_training/train.py \
      --dataset_base_path "" \
      --dataset_metadata_path "/m2v_intern/mengzijie/DiffSynth-Studio/dataset/traindataset/v4.0/fulltrainingdataset.txt" \
      --data_file_keys "video_path" \
      --dataset_num_workers 4 \
      --save_steps 200 \
      --max_pixels 268800 \
      --num_frames 49 \
      --dataset_repeat 1 \
      --model_paths '[["/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors", "/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors", "/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors", "/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors","/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors","/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors","/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"], "/m2v_intern/mengzijie/DiffSynth-Studio/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors", "/m2v_intern/mengzijie/DiffSynth-Studio/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors", "/m2v_intern/mengzijie/DiffSynth-Studio/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"]' \
      --learning_rate 1e-5 \
      --num_epochs 10 \
      --trainable_models "dit" \
      --remove_prefix_in_ckpt "pipe.dit." \
      --output_path "/ytech_m2v3_hdd/mengzijie/DiffSynth-Studio/models/train/i2v_v5.0" \
      --extra_inputs "input_image" \
      --offload_optimizer_device "cpu" \
      --gradient_accumulation_steps 1 \
      --enable_id_grid \
      --id_grid_max_pixels 1048576 \
      --id_grid_num_frames 1 \
      --id_drop_rate 0.1 \
      --use_swanlab \
      --swanlab_mode "local" \
      --swanlab_project "wan_video_id_grid_training_i2v_v5.0" \
      --swanlab_run_name "i2v_v5.0" \
    2>&1 | tee logs/wan_train_v5.0.log

# --trainable_models "dit.audio_injector" \

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
#     -x NCCL_DEBUG=INFO \
#     -x PATH \
#     -x LD_LIBRARY_PATH \
#     -x http_proxy \
#     -x https_proxy \
#     -x no_proxy \
#     -x MASTER_ADDR=$master_addr \
#     -x MASTER_PORT=29503 \
#     -x ACCELERATE_CONFIG_FILE \
#     -x DEEPSPEED_FORCE_MULTI_NODE \
#     -x WORLD_SIZE=$np \
#     -x NCCL_TOPO_FILE \
#     $PYTHON_EXE -u /ytech_milm/chenming09/codes/check_gpu_big.py \
#     2>&1 | tee logs/wan_train_$(date +%Y.%m.%d_%H:%M:%S).log


# sleep 1000d;
      # --dataset_metadata_path "/m2v_intern/mengzijie/DiffSynth-Studio/emo_ge81f_verified.csv" \
