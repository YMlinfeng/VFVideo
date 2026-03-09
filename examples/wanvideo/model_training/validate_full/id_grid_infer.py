#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tifffile import TiledSequence
import torch
import os
import argparse
import random
import datetime
import glob
from pathlib import Path
from PIL import Image, ImageOps
import librosa
import numpy as np
from diffsynth.utils.data import save_video_with_audio
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
import warnings
warnings.filterwarnings("ignore")

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
        default="/m2v_intern/mengzijie/DiffSynth-Studio/dataset/all_id_test_shuf2.txt",
        help="Path to the text file containing image paths (one per line)"
    )
    parser.add_argument(
        "--audio_dir", 
        type=str,
        default="/m2v_intern/mengzijie/DiffSynth-Studio/dataset/audio",
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
        default=None,
        help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="Wan-AI/Wan2.1-I2V-14B-720P",
        help="Model ID for the pipeline"
    )
    
    # ================== 推理参数 ==================
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames (4n+1)")
    parser.add_argument("--height", type=int, default=None, help="Video height")
    parser.add_argument("--width", type=int, default=None, help="Video width")
    parser.add_argument("--max_pixels", type=int, default=268800, help="Equivalent area max pixels")
    parser.add_argument("--num_inference_steps", type=int, default=40, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--fps", type=int, default=16, help="Output video FPS")
    parser.add_argument("--quality", type=int, default=5, help="Output video quality")
    parser.add_argument(
        "--littletestdataset",
        action="store_true",
        help="If enabled, load audio in order (same as images) instead of random sampling"
    )

    # ================== ID Grid 参数 (New) ==================
    parser.add_argument("--enable_id_grid", action="store_true")
    parser.add_argument("--id_grid_max_pixels", type=int, default=268800)
    parser.add_argument("--id_grid_num_frames", type=int, default=1)
    
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
        positive_prompt = "high quality, video, The character turns their head left and right slowly while speaking."
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
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id=args.model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="Wan2.1_VAE.pth"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        ],
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


def generate_id_grid(image, args):
    """
    Generate a 3x3 ID Grid tensor from a single input image.
    Uses equivalent area logic and center cropping (simulating FaceGrid without pose).
    """
    if not args.enable_id_grid:
        return None
    
    w, h = image.size
    # 估算人脸/图像自然长宽比
    face_ar = w / h
    
    # 计算 Grid 尺寸 (等效面积)
    area = args.id_grid_max_pixels
    grid_w_raw = (area * face_ar) ** 0.5
    grid_h_raw = (area / face_ar) ** 0.5
    
    # 对齐到 48 (16*3)
    final_w = int(round(grid_w_raw / 48) * 48)
    final_h = int(round(grid_h_raw / 48) * 48)
    if final_w == 0: final_w = 48
    if final_h == 0: final_h = 48
    
    cell_w = final_w // 3
    cell_h = final_h // 3
    
    # Center Crop & Resize to Cell
    target_ar = cell_w / cell_h
    img_ar = w / h
    
    if img_ar > target_ar:
        new_w = int(h * target_ar)
        offset = (w - new_w) // 2
        crop = image.crop((offset, 0, offset + new_w, h))
    else:
        new_h = int(w / target_ar)
        offset = (h - new_h) // 2
        crop = image.crop((0, offset, w, offset + new_h))
        
    cell_img = crop.resize((cell_w, cell_h), Image.LANCZOS)
    
    # Create 3x3 Grid (Tiling)
    grid_img = Image.new('RGB', (final_w, final_h))
    for i in range(3):
        for j in range(3):
            grid_img.paste(cell_img, (i * cell_w, j * cell_h))
            
    # Convert to Tensor (C, T, H, W)
    grid_np = np.array(grid_img)
    grid_tensor = torch.from_numpy(grid_np).float() # (H, W, 3)
    grid_tensor = grid_tensor.permute(2, 0, 1) # (3, H, W)
    
    # Stack T frames
    grid_tensor = grid_tensor.unsqueeze(1).repeat(1, args.id_grid_num_frames, 1, 1) # (3, T, H, W)
    
    # Normalize [-1, 1]
    grid_tensor = grid_tensor / 127.5 - 1.0
    
    return grid_tensor


def run_inference(pipe, image_path, audio_path, args, prompt, negative_prompt, id_grid=None, target_h=None, target_w=None):
    """
    Execute single inference.
    """
    # Load and process image
    input_image = Image.open(image_path).convert("RGB")
    
    # Resize to target (Equivalent Area) if specified
    if target_h is not None and target_w is not None:
        input_image = ImageOps.fit(input_image, (target_w, target_h), Image.LANCZOS)
    else:
        # Fallback to legacy
        if args.width and args.height:
             input_image = ImageOps.fit(input_image, (args.width, args.height), Image.LANCZOS)
    
    # Load audio (kept for filename compatibility)
    input_audio, sample_rate = librosa.load(audio_path, sr=args.audio_sample_rate)
    
    # Execute inference
    video = pipe(
        prompt=prompt,
        input_image=input_image,
        negative_prompt=negative_prompt,
        seed=args.seed,
        num_frames=args.num_frames,
        height=target_h if target_h else args.height,
        width=target_w if target_w else args.width,
        tiled=True,
        num_inference_steps=args.num_inference_steps,
        id_grid=id_grid # Pass ID Grid
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
        f.write(f"ID Grid I2V Inference Configuration\n")
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
        f.write(f"  max_pixels: {args.max_pixels}\n")
        f.write(f"  num_inference_steps: {args.num_inference_steps}\n")
        f.write(f"  seed: {args.seed}\n")
        f.write(f"  fps: {args.fps}\n")
        f.write(f"  quality: {args.quality}\n\n")
        
        f.write(f"[ID Grid]\n")
        f.write(f"  enable_id_grid: {args.enable_id_grid}\n")
        f.write(f"  id_grid_max_pixels: {args.id_grid_max_pixels}\n")
        f.write(f"  id_grid_num_frames: {args.id_grid_num_frames}\n\n")
        
    print(f"[INFO] Saved config to: {config_path}")

def log_inference_detail(output_dir, video_name, image_path, audio_path, prompt, negative_prompt, rank):
    return 
    """
    记录每次推理的详细信息到日志文件
    """
    # ... (Original commented out)

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
    with open(args.image_list_path, 'r', encoding='utf-8') as f: #todo
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
        
        # Load Image for Pre-processing (Generate Grid & Calc Size)
        try:
            pil_image = Image.open(image_path).convert("RGB")
            
            # Generate ID Grid
            id_grid = generate_id_grid(pil_image, args)
            if id_grid is not None:
                id_grid = id_grid.to(device)

            # Calculate Target H/W (Equivalent Area)
            target_h, target_w = args.height, args.width
            if target_h is None or target_w is None:
                w, h = pil_image.size
                if w * h > args.max_pixels:
                    scale = (w * h / args.max_pixels) ** 0.5
                    target_h = int(h / scale)
                    target_w = int(w / scale)
                else:
                    target_h, target_w = h, w
                # Align to 16
                target_h = target_h // 16 * 16
                target_w = target_w // 16 * 16
        except Exception as e:
            print(f"[RANK {rank}] Error loading image {image_path}: {e}")
            fail_count += 1
            continue

        for audio_idx, audio_path in enumerate(selected_audios):
            if not os.path.exists(audio_path):
                print(f"[RANK {rank}] [WARN] Audio not found: {audio_path}, skipping...")
                fail_count += 1
                continue
            
            try:
                print(f"[RANK {rank}]   Audio [{audio_idx + 1}/{num_audios}]: {os.path.basename(audio_path)}")
                
                # 执行推理
                video = run_inference(pipe, image_path, audio_path, args, prompt, negative_prompt, 
                                      id_grid=id_grid, target_h=target_h, target_w=target_w)
                
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
                    video, 
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
        print(f"RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}")
        print(f"OMPI_COMM_WORLD_RANK={os.environ.get('OMPI_COMM_WORLD_RANK')}")
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))
        print("=" * 50)
        print("Waiting for debugger to attach on port 5678...")
        print("=" * 50)
        debugpy.wait_for_client()  
        print("Debugger attached! Continuing...")
    main()



































# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# import torch
# import os
# import argparse
# import random
# import datetime
# import numpy as np
# from PIL import Image, ImageOps
# import librosa
# from diffsynth.utils.data import save_video_with_audio
# from diffsynth.core import load_state_dict
# from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
# import warnings
# warnings.filterwarnings("ignore")

# os.environ['http_proxy'] = 'http://oversea-squid1.jp.txyun:11080'
# os.environ['https_proxy'] = 'http://oversea-squid1.jp.txyun:11080'
# os.environ['no_proxy'] = 'localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com'

# # 分布式环境支持
# if "OMPI_COMM_WORLD_RANK" in os.environ:
#     os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
#     os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
#     os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
# elif "RANK" not in os.environ:
#     os.environ.setdefault("RANK", "0")
#     os.environ.setdefault("WORLD_SIZE", "1")
#     os.environ.setdefault("LOCAL_RANK", "0")

# def parse_args():
#     parser = argparse.ArgumentParser(description="ID Grid I2V Inference Script")
    
#     # 基础参数
#     parser.add_argument("--image_list_path", type=str, required=True)
#     parser.add_argument("--audio_dir", type=str, default=None)
#     parser.add_argument("--output_base_dir", type=str, default="output")
#     parser.add_argument("--output_timestamp", type=str, default=None)
    
#     # 模型参数
#     parser.add_argument("--ckpt_path", type=str, default=None)
#     parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-I2V-14B-720P")
    
#     # 视频参数 (等效面积优先)
#     parser.add_argument("--num_frames", type=int, default=81)
#     parser.add_argument("--height", type=int, default=None)
#     parser.add_argument("--width", type=int, default=None)
#     parser.add_argument("--max_pixels", type=int, default=268800)
    
#     # 推理参数
#     parser.add_argument("--num_inference_steps", type=int, default=40)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--fps", type=int, default=16)
#     parser.add_argument("--quality", type=int, default=5)
    
#     # ID Grid 参数
#     parser.add_argument("--enable_id_grid", action="store_true")
#     parser.add_argument("--id_grid_max_pixels", type=int, default=268800)
#     parser.add_argument("--id_grid_num_frames", type=int, default=1)
    
#     # Audio
#     parser.add_argument("--num_audios_per_image", type=int, default=1)
#     parser.add_argument("--audio_sample_rate", type=int, default=16000)
    
#     # 分布式
#     parser.add_argument("--rank", type=int, default=0)
#     parser.add_argument("--world_size", type=int, default=1)
#     parser.add_argument("--local_rank", type=int, default=0)

#     return parser.parse_args()

# def get_distributed_info(args):
#     rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", args.rank)))
#     world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("WORLD_SIZE", args.world_size)))
#     local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("LOCAL_RANK", args.local_rank)))
#     return rank, world_size, local_rank

# def load_pipeline(args, device):
#     print(f"[INFO] Loading model: {args.model_id}")
#     pipe = WanVideoPipeline.from_pretrained(
#         torch_dtype=torch.bfloat16,
#         device=device,
#         model_configs=[
#             ModelConfig(model_id=args.model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
#             ModelConfig(model_id=args.model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
#             ModelConfig(model_id=args.model_id, origin_file_pattern="Wan2.1_VAE.pth"),
#             ModelConfig(model_id=args.model_id, origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
#         ],
#     )
    
#     if args.ckpt_path and os.path.exists(args.ckpt_path):
#         print(f"[INFO] Loading checkpoint from: {args.ckpt_path}")
#         state_dict = load_state_dict(args.ckpt_path)
#         missing, unexpected = pipe.dit.load_state_dict(state_dict, strict=False)
#         if missing: print(f"[WARNING] Missing keys: {len(missing)}")
#         if unexpected: print(f"[WARNING] Unexpected keys: {len(unexpected)}")
    
#     return pipe

# def generate_id_grid(image, args):
#     """
#     Generate a 3x3 ID Grid tensor from a single input image.
#     Uses equivalent area logic and center cropping (simulating FaceGrid without pose).
#     """
#     if not args.enable_id_grid:
#         return None
    
#     w, h = image.size
#     # 估算人脸/图像自然长宽比
#     face_ar = w / h
    
#     # 计算 Grid 尺寸 (等效面积)
#     area = args.id_grid_max_pixels
#     grid_w_raw = (area * face_ar) ** 0.5
#     grid_h_raw = (area / face_ar) ** 0.5
    
#     # 对齐到 48 (16*3)
#     final_w = int(round(grid_w_raw / 48) * 48)
#     final_h = int(round(grid_h_raw / 48) * 48)
#     if final_w == 0: final_w = 48
#     if final_h == 0: final_h = 48
    
#     cell_w = final_w // 3
#     cell_h = final_h // 3
    
#     # Center Crop & Resize to Cell
#     target_ar = cell_w / cell_h
#     img_ar = w / h
    
#     if img_ar > target_ar:
#         new_w = int(h * target_ar)
#         offset = (w - new_w) // 2
#         crop = image.crop((offset, 0, offset + new_w, h))
#     else:
#         new_h = int(w / target_ar)
#         offset = (h - new_h) // 2
#         crop = image.crop((0, offset, w, offset + new_h))
        
#     cell_img = crop.resize((cell_w, cell_h), Image.LANCZOS)
    
#     # Create 3x3 Grid (Tiling)
#     grid_img = Image.new('RGB', (final_w, final_h))
#     for i in range(3):
#         for j in range(3):
#             grid_img.paste(cell_img, (i * cell_w, j * cell_h))
            
#     # Convert to Tensor (C, T, H, W)
#     # T = args.id_grid_num_frames
#     # We duplicate the grid image T times
#     grid_np = np.array(grid_img)
#     grid_tensor = torch.from_numpy(grid_np).float() # (H, W, 3)
#     grid_tensor = grid_tensor.permute(2, 0, 1) # (3, H, W)
    
#     # Stack T frames
#     grid_tensor = grid_tensor.unsqueeze(1).repeat(1, args.id_grid_num_frames, 1, 1) # (3, T, H, W)
    
#     # Normalize [-1, 1]
#     grid_tensor = grid_tensor / 127.5 - 1.0
    
#     # Add Batch Dimension for Pipeline (B, C, T, H, W)
#     # Actually pipeline expects unbatched usually? 
#     # Pipeline.__call__ handles tensors.
#     # Usually inputs are (C, T, H, W) and pipeline unsqueezes.
#     # Let's verify WanVideoUnit_IDGridEmbedder.
#     # It does `if id_grid.ndim == 4: unsqueeze`.
#     # So we can return (3, T, H, W).
    
#     return grid_tensor


# def load_prompts(image_path, fps, rank):
#     """
#     Load positive and negative prompts from text files.
#     """
#     base_path = image_path
#     for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
#         if image_path.endswith(ext):
#             base_path = image_path[:-len(ext)]
#             break
    
#     positive_prompt_path = base_path + '_2.txt'
#     negative_prompt_path = base_path + '_negative_2.txt'
    
#     positive_prompt = None
#     if os.path.exists(positive_prompt_path):
#         try:
#             with open(positive_prompt_path, 'r', encoding='utf-8') as f:
#                 positive_prompt = f.read().strip()
#             if 'FPS-30' in positive_prompt:
#                 positive_prompt = positive_prompt.replace('FPS-30', f'FPS-{fps}')
#         except Exception as e:
#             print(f"[RANK {rank}] [WARN] Failed to read positive prompt: {e}")

#     if positive_prompt is None or positive_prompt == "":
#         print(f"[Bug] positive_prompt is None!!!")
#         positive_prompt = "high quality, video"
#         print(f"[RANK {rank}] [INFO] Using default positive prompt: {positive_prompt}")
    
#     negative_prompt = None
#     if os.path.exists(negative_prompt_path):
#         try:
#             with open(negative_prompt_path, 'r', encoding='utf-8') as f:
#                 negative_prompt = f.read().strip()
#             if 'FPS-30' in negative_prompt:
#                 negative_prompt = negative_prompt.replace('FPS-30', f'FPS-{fps}')
#         except Exception as e:
#             print(f"[RANK {rank}] [WARN] Failed to read negative prompt: {e}")

#     # Fallback to default negative prompt if not loaded
#     if negative_prompt is None or negative_prompt == "":
#         negative_prompt = (
#             f"FPS-{fps} The video plays in distorted slow motion with unstable speed and jittering frames. "
#             "The camera captures the scene in slow motion. An abstract, computer-generated, unrealistic, "
#             "animation, cartoon, scene with distorted and blurry visuals, with high saturation and high contrast. "
#             "A deformed, disfigured figure without specific features, depicted as an illustration, with scene transition. "
#             "The background is a collage of grainy textures and striped patterns, lacking clear visual content. "
#             "The figure moves minimally with weak dynamics and a stuttering effect, displaying distorted and erratic motions. "
#             "The style incorporates extremely high contrast and extremely high sharpness, combined with low-quality imagery, "
#             "grainy effects, and includes logos and text elements. It is an unrealistic 3D animation. "
#             "The camera employs disjointed and stuttering movements, inconsistent framing, and unstructured composition."
#         )
#         print(f"[RANK {rank}] [INFO] Using default negative prompt")
    
#     return positive_prompt, negative_prompt

# def main():
#     args = parse_args()
#     rank, world_size, local_rank = get_distributed_info(args)
    
#     if not torch.cuda.is_available():
#         raise RuntimeError("CUDA is not available")
    
#     torch.cuda.set_device(local_rank)
#     device = f"cuda:{local_rank}"
    
#     # Create output dir
#     timestamp = args.output_timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_dir = os.path.join(args.output_base_dir, f"output_{timestamp}")
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Load Model
#     pipe = load_pipeline(args, device)
    
#     # Load Images
#     with open(args.image_list_path, 'r', encoding='utf-8') as f:
#         image_paths = [line.strip() for line in f if line.strip()]
        
#     my_indices = list(range(rank, len(image_paths), world_size))
#     print(f"[RANK {rank}] Processing {len(my_indices)} images")
    
#     for idx in my_indices:
#         image_path = image_paths[idx]
#         if not os.path.exists(image_path): continue
        
#         print(f"[RANK {rank}] Processing {image_path}")
        
#         try:
#             # Load Prompts
#             prompt, negative_prompt = load_prompts(image_path, args.fps, rank)
            
#             # Load Image
#             input_image = Image.open(image_path).convert("RGB") #! 无需调整大小
            
#             # Generate ID Grid
#             id_grid = generate_id_grid(input_image, args)
#             if id_grid is not None:
#                 id_grid = id_grid.to(device)
            
#             # Determine Main Video Size (Equivalent Area)
#             target_h = args.height
#             target_w = args.width
            
#             if target_h is None or target_w is None:
#                 w, h = input_image.size
#                 if w * h > args.max_pixels:
#                     scale = (w * h / args.max_pixels) ** 0.5
#                     target_h = int(h / scale)
#                     target_w = int(w / scale)
#                 else:
#                     target_h, target_w = h, w
#                 # Align to 16
#                 target_h = target_h // 16 * 16
#                 target_w = target_w // 16 * 16

#             video = pipe(
#                 prompt=prompt,
#                 negative_prompt=negative_prompt,
#                 input_image=input_image,
#                 num_frames=args.num_frames,
#                 height=target_h,
#                 width=target_w,
#                 num_inference_steps=args.num_inference_steps,
#                 seed=args.seed,
#                 id_grid=id_grid
#             )
            
#             save_name = os.path.join(output_dir, os.path.basename(image_path).split('.')[0] + ".mp4")
#             save_video_with_audio(video, save_name, None, fps=args.fps, quality=args.quality)
#             print(f"[RANK {rank}] Saved {save_name}")
            
#         except Exception as e:
#             print(f"[RANK {rank}] Error processing {image_path}: {e}")
#             import traceback
#             traceback.print_exc()

# if __name__ == "__main__":
#     # if os.environ.get("LOCAL_RANK", "0") == "0":
#     #         print(f"RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}")
#     #         print(f"OMPI_COMM_WORLD_RANK={os.environ.get('OMPI_COMM_WORLD_RANK')}")
#     #         import debugpy
#     #         debugpy.listen(("0.0.0.0", 5678))
#     #         print("=" * 50)
#     #         print("Waiting for debugger to attach on port 5678...")
#     #         print("=" * 50)
#     #         debugpy.wait_for_client()  
#     #         print("Debugger attached! Continuing...")
#     main()
