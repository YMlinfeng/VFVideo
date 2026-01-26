
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
S2V (Speech-to-Video) 多机多卡分布式推理脚本
功能：
  - 支持多机多卡并行推理
  - 每张图片可配置多段音频
  - 自动生成带时间戳的输出目录
  - 视频命名规则：图片路径最后三个关键词 + 音频文件名
"""

import torch
import os
import argparse
import random
import datetime
import glob
from pathlib import Path
from PIL import Image, ImageOps
import librosa
from diffsynth.utils.data import save_video_with_audio
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
os.environ['http_proxy'] = 'http://oversea-squid1.jp.txyun:11080'
os.environ['https_proxy'] = 'http://oversea-squid1.jp.txyun:11080'
os.environ['no_proxy'] = 'localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com'
# 支持 MPI / torchrun / 单卡 三种模式
if "OMPI_COMM_WORLD_RANK" in os.environ:
    # MPI 模式
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
    os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
elif "RANK" not in os.environ:
    # 单卡模式（无分布式环境变量时设置默认值）
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")



def parse_args():
    parser = argparse.ArgumentParser(description="S2V Distributed Inference Script")
    
    # ================== 数据路径参数 ==================
    parser.add_argument(
        "--image_list_path", 
        type=str, 
        default="/m2v_intern/mengzijie/DiffSynth-Studio/data/all_id_test_shuf2.txt",
        help="Path to the text file containing image paths (one per line)"
    )
    parser.add_argument(
        "--audio_dir", 
        type=str,
        default="/m2v_intern/mengzijie/DiffSynth-Studio/data/audio",
        help="Directory containing audio files (mp3)"
    )
    parser.add_argument(
        "--output_base_dir", 
        type=str, 
        default="output",
        help="Base output directory"
    )
    parser.add_argument(
        "--output_timestamp",
        type=str,
        default=None,
        help="Output directory timestamp (for multi-process sync). If not provided, will generate one."
    )
    
    # ================== 模型参数 ==================
    parser.add_argument(
        "--ckpt_path", 
        type=str,
        default="/m2v_intern/mengzijie/DiffSynth-Studio/models/train/单卡/step-800.safetensors",
        help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="Wan-AI/Wan2.2-S2V-14B",
        help="Model ID for the pipeline"
    )
    
    # ================== 推理参数 ==================
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames (4n+1)")
    parser.add_argument("--height", type=int, default=832, help="Video height")
    parser.add_argument("--width", type=int, default=448, help="Video width")
    parser.add_argument("--num_inference_steps", type=int, default=40, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS")
    parser.add_argument("--quality", type=int, default=5, help="Output video quality")
    parser.add_argument(
        "--littletestdataset",
        action="store_true",
        help="If enabled, load audio in order (same as images) instead of random sampling"
    )
    
    # ================== 音频参数 ==================
    parser.add_argument(
        "--num_audios_per_image", 
        type=int, 
        default=1,
        help="Number of audio samples to use per image"
    )
    parser.add_argument(
        "--audio_sample_rate", 
        type=int, 
        default=16000, 
        help="Audio sample rate for processing"
    )
    
    # ================== Prompt 参数 ==================
    # parser.add_argument(
    #     "--prompt", 
    #     type=str, 
    #     default="a person is singing",
    #     help="Positive prompt for video generation"
    # )
    # parser.add_argument(
    #     "--negative_prompt", 
    #     type=str,
    #     default="画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    #     help="Negative prompt for video generation"
    # )
    
    # ================== 分布式参数 ==================
    parser.add_argument("--rank", type=int, default=0, help="Current process rank (overridden by MPI env)")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of processes (overridden by MPI env)")
    parser.add_argument("--local_rank", type=int, default=0, help="Local GPU rank (overridden by MPI env)")
    
    return parser.parse_args()


def get_distributed_info(args):
    """
    获取分布式环境信息，支持多种启动方式：
    - MPI (OMPI_COMM_WORLD_*)
    - torchrun (RANK, LOCAL_RANK, WORLD_SIZE)
    - 单卡直接运行 (使用命令行参数或默认值)
    """
    # 优先级: MPI环境变量 > 通用环境变量 > 命令行参数
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 
               os.environ.get("RANK", args.rank)))
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 
                     os.environ.get("WORLD_SIZE", args.world_size)))
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 
                     os.environ.get("LOCAL_RANK", args.local_rank)))
    
    # 确保 world_size 至少为 1
    world_size = max(1, world_size)
    
    return rank, world_size, local_rank


def get_all_audio_files(audio_dir):
    """
    递归获取目录下所有的mp3文件
    """
    audio_files = []
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.lower().endswith('.mp3'):
                audio_files.append(os.path.join(root, file))
    return sorted(audio_files)  # 排序以保证顺序一致性


def load_prompts(image_path, fps, rank):
    """
    Load positive and negative prompts from text files.
    
    Args:
        image_path: Path to the image file
        fps: FPS value for prompt replacement
        rank: Current process rank for logging
    
    Returns:
        tuple: (positive_prompt, negative_prompt)
    """
    # Construct prompt file paths
    # Handle both .png and other extensions
    base_path = image_path
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        if image_path.endswith(ext):
            base_path = image_path[:-len(ext)]
            break
    
    positive_prompt_path = base_path + '_2.txt'
    negative_prompt_path = base_path + '_negative_2.txt'
    
    # Load positive prompt
    positive_prompt = None
    if os.path.exists(positive_prompt_path):
        try:
            with open(positive_prompt_path, 'r', encoding='utf-8') as f:
                positive_prompt = f.read().strip()
            # Replace FPS if exists (safe replace, won't error if not found)
            if 'FPS-30' in positive_prompt:
                positive_prompt = positive_prompt.replace('FPS-30', f'FPS-{fps}')
        except Exception as e:
            print(f"[RANK {rank}] [WARN] Failed to read positive prompt file {positive_prompt_path}: {e}")
            positive_prompt = None
    else:
        print(f"[RANK {rank}] [WARN] Positive prompt file not found: {positive_prompt_path}, using default prompt")
    
    # Fallback to default positive prompt if not loaded
    if positive_prompt is None or positive_prompt == "":
        positive_prompt = "a person is singing"
        print(f"[RANK {rank}] [INFO] Using default positive prompt: {positive_prompt}")
    
    # Load negative prompt
    negative_prompt = None
    if os.path.exists(negative_prompt_path):
        try:
            with open(negative_prompt_path, 'r', encoding='utf-8') as f:
                negative_prompt = f.read().strip()
            # Replace FPS if exists (safe replace, won't error if not found)
            if 'FPS-30' in negative_prompt:
                negative_prompt = negative_prompt.replace('FPS-30', f'FPS-{fps}')
        except Exception as e:
            print(f"[RANK {rank}] [WARN] Failed to read negative prompt file {negative_prompt_path}: {e}")
            negative_prompt = None
    else:
        print(f"[RANK {rank}] [WARN] Negative prompt file not found: {negative_prompt_path}, using default negative prompt")
    
    # Fallback to default negative prompt if not loaded
    if negative_prompt is None or negative_prompt == "":
        negative_prompt = (
            f"FPS-{fps} The video plays in distorted slow motion with unstable speed and jittering frames. "
            "The camera captures the scene in slow motion. An abstract, computer-generated, unrealistic, "
            "animation, cartoon, scene with distorted and blurry visuals, with high saturation and high contrast. "
            "A deformed, disfigured figure without specific features, depicted as an illustration, with scene transition. "
            "The background is a collage of grainy textures and striped patterns, lacking clear visual content. "
            "The figure moves minimally with weak dynamics and a stuttering effect, displaying distorted and erratic motions. "
            "The style incorporates extremely high contrast and extremely high sharpness, combined with low-quality imagery, "
            "grainy effects, and includes logos and text elements. It is an unrealistic 3D animation. "
            "The camera employs disjointed and stuttering movements, inconsistent framing, and unstructured composition."
        )
        print(f"[RANK {rank}] [INFO] Using default negative prompt")
    
    return positive_prompt, negative_prompt


def generate_video_name(image_path, audio_path):
    """
    生成视频名称：图片路径最后三个关键词 + 音频文件名（不含后缀）
    
    示例：
    - 图片路径: /path/to/复杂情况/何炅/2.png
    - 音频路径: /path/to/英文_口播_正常节奏_老年男声_3.mp3
    - 输出: 复杂情况_何炅_2_英文_口播_正常节奏_老年男声_3.mp4
    """
    # 解析图片路径
    image_parts = image_path.rstrip('/').split('/')
    image_name = os.path.splitext(image_parts[-1])[0]  # 文件名（不含扩展名）
    image_folder1 = image_parts[-2] if len(image_parts) > 1 else ""  # 上一级目录
    image_folder2 = image_parts[-3] if len(image_parts) > 2 else ""  # 上两级目录
    
    # 解析音频文件名（不含扩展名）
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # 组合成最终的视频名称
    video_name = f"{image_folder2}_{image_folder1}_{image_name}_{audio_name}.mp4"
    
    # 清理文件名中的非法字符
    video_name = video_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    
    return video_name


def load_pipeline(args, device): 
    """
    加载模型管道
    """
    # print(f"[INFO] Loading pipeline on device: {device}")
    
    # pipe = WanVideoPipeline.from_pretrained(
    #     torch_dtype=torch.bfloat16,
    #     device="cuda",
    #     model_configs=[
    #         # 第一个ModelConfig：应该是4个safetensors文件作为一个列表
    #         ModelConfig(
    #             path=[
    #                 "/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/diffusion_pytorch_model-00001-of-00004.safetensors",
    #                 "/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/diffusion_pytorch_model-00002-of-00004.safetensors",
    #                 "/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/diffusion_pytorch_model-00003-of-00004.safetensors",
    #                 "/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/diffusion_pytorch_model-00004-of-00004.safetensors"
    #             ]
    #         ),
    #         # 第二个ModelConfig：wav2vec2模型
    #         ModelConfig(
    #             path="/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/model.safetensors"
    #         ),
    #         # 第三个ModelConfig：T5模型
    #         ModelConfig(
    #             path="/m2v_intern/mengzijie/DiffSynth-Studio/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"
    #         ),
    #         # 第四个ModelConfig：VAE模型
    #         ModelConfig(
    #             path="/m2v_intern/mengzijie/DiffSynth-Studio/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors"
    #         ),
    #     ],
    #     audio_processor_config=ModelConfig(
    #         path="/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english"
    #     )
    # )
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(
                model_id=args.model_id, 
                origin_file_pattern="diffusion_pytorch_model*.safetensors"
            ),
            ModelConfig(
                model_id=args.model_id, 
                origin_file_pattern="wav2vec2-large-xlsr-53-english/model.safetensors"
            ),
            ModelConfig(
                model_id=args.model_id, 
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"
            ),
            ModelConfig(
                model_id=args.model_id, 
                origin_file_pattern="Wan2.1_VAE.pth"
            ),
        ],
        audio_processor_config=ModelConfig(
            model_id=args.model_id, 
            origin_file_pattern="wav2vec2-large-xlsr-53-english/"
        ),
    )

    # 加载 checkpoint
    if args.ckpt_path and os.path.exists(args.ckpt_path):
        print(f"[INFO] Loading checkpoint from: {args.ckpt_path}")
        state_dict = load_state_dict(args.ckpt_path)
        missing, unexpected = pipe.dit.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARNING] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[WARNING] Unexpected keys: {len(unexpected)}")
        print(f"[INFO] Checkpoint loaded successfully")
    else:
        print(f"[INFO] No checkpoint specified or file not found, using pretrained weights")
    
    return pipe


def run_inference(pipe, image_path, audio_path, args, prompt, negative_prompt):
    """
    Execute single inference.
    
    Args:
        pipe: The pipeline object
        image_path: Path to input image
        audio_path: Path to input audio
        args: Command line arguments
        prompt: Positive prompt text
        negative_prompt: Negative prompt text
    """
    # Load and process image
    input_image = Image.open(image_path).convert("RGB")
    input_image = ImageOps.fit(input_image, (args.width, args.height), Image.LANCZOS)
    
    # Load audio
    input_audio, sample_rate = librosa.load(audio_path, sr=args.audio_sample_rate)
    
    # Execute inference
    video = pipe(
        prompt=prompt,
        input_image=input_image,
        negative_prompt=negative_prompt,
        seed=args.seed,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        audio_sample_rate=sample_rate,
        input_audio=input_audio,
        num_inference_steps=args.num_inference_steps
    )
    
    return video


def save_inference_config(output_dir, args, rank):
    """
    保存推理配置到文件（仅rank 0执行）
    """
    if rank != 0:
        return
    
    config_path = os.path.join(output_dir, "inference_config.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(f"=" * 60 + "\n")
        f.write(f"S2V Inference Configuration\n")
        f.write(f"Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"=" * 60 + "\n\n")
        
        f.write(f"[Data Paths]\n")
        f.write(f"  image_list_path: {args.image_list_path}\n")
        f.write(f"  audio_dir: {args.audio_dir}\n")
        f.write(f"  output_base_dir: {args.output_base_dir}\n\n")
        
        f.write(f"[Model]\n")
        f.write(f"  ckpt_path: {args.ckpt_path}\n")
        f.write(f"  model_id: {args.model_id}\n\n")
        
        f.write(f"[Inference Parameters]\n")
        f.write(f"  num_frames: {args.num_frames}\n")
        f.write(f"  height: {args.height}\n")
        f.write(f"  width: {args.width}\n")
        f.write(f"  num_inference_steps: {args.num_inference_steps}\n")
        f.write(f"  seed: {args.seed}\n")
        f.write(f"  fps: {args.fps}\n")
        f.write(f"  quality: {args.quality}\n\n")
        
        f.write(f"[Audio]\n")
        f.write(f"  num_audios_per_image: {args.num_audios_per_image}\n")
        f.write(f"  audio_sample_rate: {args.audio_sample_rate}\n\n")
        
    print(f"[INFO] Saved config to: {config_path}")

def log_inference_detail(output_dir, video_name, image_path, audio_path, prompt, negative_prompt, rank):
    """
    记录每次推理的详细信息到日志文件
    """
    log_path = os.path.join(output_dir, f"inference_details_rank{rank}.jsonl")
    
    import json
    log_entry = {
        "video_name": video_name,
        "image_path": image_path,
        "audio_path": audio_path,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

def main():
    args = parse_args()
    rank, world_size, local_rank = get_distributed_info(args)
    
    # 检查 CUDA 可用性
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    num_gpus = torch.cuda.device_count()
    # 修正 local_rank 防止越界
    if local_rank >= num_gpus:
        print(f"[WARN] local_rank ({local_rank}) >= available GPUs ({num_gpus}), resetting to 0")
        local_rank = 0
    
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    
    print(f"[RANK {rank}/{world_size}] Local Rank: {local_rank}, Device: {device}")
    
    # ================== 创建输出目录 ==================
    # 使用环境变量传递的时间戳确保所有进程使用相同的输出目录
    if args.output_timestamp:
        timestamp = args.output_timestamp
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = os.path.join(args.output_base_dir, f"output_{timestamp}")
    
    # 所有rank都创建目录（确保目录存在）
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[RANK {rank}] Output directory: {output_dir}")
    
    # ================== 读取图片列表 ==================
    with open(args.image_list_path, 'r', encoding='utf-8') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    
    total_images = len(image_paths)
    print(f"[RANK {rank}] Total images in list: {total_images}")
    
    # ================== 获取所有音频文件 ==================
    if args.littletestdataset:
        # littletestdataset模式：从txt文件读取音频路径列表
        with open(args.audio_dir, 'r', encoding='utf-8') as f:
            all_audio_files = [line.strip() for line in f if line.strip()]
        if not all_audio_files:
            raise ValueError(f"No audio paths found in {args.audio_dir}")
        if len(all_audio_files) != len(image_paths):
            print(f"[RANK {rank}] [WARN] Audio list length ({len(all_audio_files)}) != Image list length ({len(image_paths)})")
        print(f"[RANK {rank}] Loaded {len(all_audio_files)} audio paths from txt file")
    else:
        # 原有逻辑：从目录递归扫描音频文件
        all_audio_files = get_all_audio_files(args.audio_dir)
        if not all_audio_files:
            raise ValueError(f"No MP3 files found in {args.audio_dir}")
        print(f"[RANK {rank}] Found {len(all_audio_files)} audio files")
    
    # ================== 加载模型 ==================
    pipe = load_pipeline(args, device)
    
    # ================== 计算当前rank需要处理的图片 ==================
    # 分配策略：rank 0 处理第 0, world_size, 2*world_size, ... 行
    #          rank 1 处理第 1, world_size+1, 2*world_size+1, ... 行
    #          以此类推
    my_image_indices = list(range(rank, total_images, world_size))
    
    print(f"[RANK {rank}] Will process {len(my_image_indices)} images: indices {my_image_indices[:5]}{'...' if len(my_image_indices) > 5 else ''}")
    
    # 保存配置文件（仅rank 0）
    save_inference_config(output_dir, args, rank)
    
    # 设置随机种子（每个rank不同，确保选择不同的音频组合）
    random.seed(args.seed + rank)
    
    # ================== 开始推理 ==================
    success_count = 0
    fail_count = 0
    
    for idx, img_idx in enumerate(my_image_indices):
        image_path = image_paths[img_idx]
        
        if not os.path.exists(image_path):
            print(f"[RANK {rank}] [WARN] Image not found: {image_path}, skipping...")
            fail_count += 1
            continue
        
        print(f"[RANK {rank}] [{idx + 1}/{len(my_image_indices)}] Processing image index {img_idx}: {image_path}")
        
        prompt, negative_prompt = load_prompts(image_path, args.fps, rank)

        # 选择音频
        if args.littletestdataset:
            # littletestdataset模式：按行号一一对应
            if img_idx < len(all_audio_files):
                selected_audios = [all_audio_files[img_idx]]
            else:
                print(f"[RANK {rank}] [WARN] No corresponding audio for image index {img_idx}, skipping...")
                fail_count += 1
                continue
            num_audios = len(selected_audios)  # ← 添加这行
        else:
            # 原有逻辑：随机选择音频
            num_audios = min(args.num_audios_per_image, len(all_audio_files))
            selected_audios = random.sample(all_audio_files, num_audios)
        
        for audio_idx, audio_path in enumerate(selected_audios):
            if not os.path.exists(audio_path):
                print(f"[RANK {rank}] [WARN] Audio not found: {audio_path}, skipping...")
                fail_count += 1
                continue
            
            try:
                print(f"[RANK {rank}]   Audio [{audio_idx + 1}/{num_audios}]: {os.path.basename(audio_path)}")
                
                # 执行推理
                video = run_inference(pipe, image_path, audio_path, args, prompt, negative_prompt)
                
                # 生成视频名称
                video_name = generate_video_name(image_path, audio_path)
                video_save_path = os.path.join(output_dir, video_name)
                
                # 检查文件是否已存在，避免覆盖
                if os.path.exists(video_save_path):
                    base_name = os.path.splitext(video_name)[0]
                    video_name = f"{base_name}_rank{rank}.mp4"  # ← 同时更新 video_name
                    video_save_path = os.path.join(output_dir, video_name)
                
                # 保存视频
                save_video_with_audio(
                    video[1:],  # 跳过第一帧
                    video_save_path, 
                    audio_path, 
                    fps=args.fps, 
                    quality=args.quality
                )
                
                log_inference_detail(output_dir, video_name, image_path, audio_path, prompt, negative_prompt, rank)

                print(f"[RANK {rank}]   Saved: {video_name}")
                success_count += 1
                
            except Exception as e:
                print(f"[RANK {rank}] [ERROR] Failed to process {image_path} with {audio_path}: {e}")
                import traceback
                traceback.print_exc()
                fail_count += 1
                continue
    
    # ================== 输出统计信息 ==================
    print(f"\n" + "=" * 60)
    print(f"[RANK {rank}] Inference completed!")
    print(f"[RANK {rank}] Success: {success_count}, Failed: {fail_count}")
    print(f"[RANK {rank}] Output directory: {output_dir}")
    print(f"=" * 60 + "\n")


if __name__ == "__main__":
    if os.environ.get("LOCAL_RANK", "0") == "0":
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))
        print("=" * 50)
        print("Waiting for debugger to attach on port 5678...")
        print("=" * 50)
        debugpy.wait_for_client()  
        print("Debugger attached! Continuing...")
    main()
    # all_img_list = open("/ytech_m2v2_hdd/liujiwen/audio_v3/Qwen3-VL/明星照/all_id_test_shuf2.txt").read().strip().split('\n')
    # img_path1 = all_img_list[i]
    # caption_txt_path = img_path1.replace('.png', '_2.txt')
    # try:
    #     negative_prompt = open(caption_txt_path.replace('_2.txt', '_negative_2.txt')).read().strip().replace('FPS-24', 'FPS-30').replace('\n', ' ')
    # except:
    #     negative_prompt = "FPS-30. The video plays in distorted slow motion with unstable speed and jittering frames. The camera captures the scene in slow motion. An abstract, computer-generated, unrealistic, animation, cartoon, scene with distorted and blurry visuals, with high saturation and high contrast. A deformed, disfigured figure without specific features, depicted as an illustration, with scene transition. The background is a collage of grainy textures and striped patterns, lacking clear visual content. The figure moves minimally with weak dynamics and a stuttering effect, displaying distorted and erratic motions. The style incorporates extremely high contrast and extremely high sharpness, combined with low-quality imagery, grainy effects, and includes logos and text elements. It is an unrealistic 3D animation. The camera employs disjointed and stuttering movements, inconsistent framing, and unstructured composition." 
    # caption_txt_path = img_path1.replace('.png', '_2.txt')

    # caption_txt_path = img_path1.replace('.png', '_2.txt')
    # all_caption_list = open('/ytech_m2v2_hdd/liujiwen/audio_v3/m2v-diffusers/test2/all_caption.txt').read().strip().split('\n')
    # try:
    #     caption_str = open(caption_txt_path).read().strip().replace('FPS-24', 'FPS-30').replace('\n', ' ')
    # except:
    #     pass
