#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import os
import sys
import argparse
import random
import datetime
import numpy as np
import pandas as pd
import ast
import cv2
from PIL import Image, ImageOps
import librosa
from diffsynth.utils.data import save_video_with_audio
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
import warnings
warnings.filterwarnings("ignore")

# 引入项目根目录，以便导入 get_smpl_motion
sys.path.append(os.getcwd())
try:
    from get_smpl_motion.GVHMR.smpl_Infer_service_ljw import SmplInfer # 导入人脸裁剪和九宫格构建模块
except ImportError:
    print("[WARN] Failed to import SmplInfer from get_smpl_motion. Make sure it exists in the root path.")
    SmplInfer = None

os.environ['http_proxy'] = 'http://oversea-squid1.jp.txyun:11080'
os.environ['https_proxy'] = 'http://oversea-squid1.jp.txyun:11080'
os.environ['no_proxy'] = 'localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com'

if "OMPI_COMM_WORLD_RANK" in os.environ:
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
    os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
elif "RANK" not in os.environ:
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

def parse_args():
    parser = argparse.ArgumentParser(description="ID Grid I2V Inference Script")
    
    # 按照要求，将所有参数独立在一行，且默认值和 id_grid_validate.sh 完全一致
    parser.add_argument("--dataset_metadata_path", type=str, default="dataset/all_id_test_shuf2.csv", help="Path to the CSV dataset") # 使用 CSV 文件
    parser.add_argument("--audio_dir", type=str, default="/m2v_intern/mengzijie/DiffSynth-Studio/data/audio", help="Audio directory") # 音频目录
    parser.add_argument("--output_base_dir", type=str, default="output_id_grid", help="Output base directory") # 输出根目录
    parser.add_argument("--output_timestamp", type=str, default=None, help="Output timestamp string") # 运行时间戳
    parser.add_argument("--ckpt_path", type=str, default="/ytech_m2v4_hdd/mengzijie/DiffSynth-Studio/models/train/i2v_v1.0/step-1000.safetensors", help="Model checkpoint path") # 模型路径
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-I2V-14B-720P", help="Base model ID") # 基础模型ID
    parser.add_argument("--num_frames", type=int, default=41, help="Main video frame count") # 主视频帧数
    parser.add_argument("--height", type=int, default=None, help="Fixed height, None for equivalent area") # 固定高
    parser.add_argument("--width", type=int, default=None, help="Fixed width, None for equivalent area") # 固定宽
    parser.add_argument("--max_pixels", type=int, default=268800, help="Equivalent area max pixels") # 等效面积
    parser.add_argument("--num_inference_steps", type=int, default=40, help="Denoising steps") # 推理步数
    parser.add_argument("--seed", type=int, default=42, help="Random seed") # 随机种子
    parser.add_argument("--fps", type=int, default=15, help="Video FPS") # 帧率
    parser.add_argument("--quality", type=int, default=5, help="Video quality") # 质量
    parser.add_argument("--littletestdataset", action="store_true", help="Audio load mode") # 音频对齐模式
    parser.add_argument("--enable_id_grid", action="store_true", help="Enable ID Grid generation") # 开启九宫格
    parser.add_argument("--id_grid_max_pixels", type=int, default=268800, help="ID Grid equivalent area") # 九宫格等效面积
    parser.add_argument("--id_grid_num_frames", type=int, default=1, help="Number of frames for ID Grid") # 九宫格帧数 (控制长度)
    parser.add_argument("--id_video_path", type=str, default=None, help="Optional reference ID video") # 支持传入视频扣ID
    parser.add_argument("--num_audios_per_image", type=int, default=1, help="Audios per inference") # 每个图片音频数
    parser.add_argument("--audio_sample_rate", type=int, default=16000, help="Audio SR") # 音频采样率
    parser.add_argument("--rank", type=int, default=0, help="Global rank") # Rank
    parser.add_argument("--world_size", type=int, default=1, help="World size") # 总卡数
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank") # 单机卡数
    
    return parser.parse_args()

def get_distributed_info(args):
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("RANK", args.rank)))
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", os.environ.get("WORLD_SIZE", args.world_size)))
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("LOCAL_RANK", args.local_rank)))
    world_size = max(1, world_size)
    return rank, world_size, local_rank

def get_all_audio_files(audio_dir):
    audio_files = []
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.lower().endswith('.mp3'):
                audio_files.append(os.path.join(root, file))
    return sorted(audio_files)

def load_prompts(image_path, fps, rank):
    base_path = image_path
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        if image_path.endswith(ext):
            base_path = image_path[:-len(ext)]
            break
            
    positive_prompt_path = base_path + '_2.txt'
    negative_prompt_path = base_path + '_negative_2.txt'
    
    use_default_pos = False # 记录是否使用了默认正向提示词
    use_default_neg = False # 记录是否使用了默认负向提示词
    
    positive_prompt = None
    if os.path.exists(positive_prompt_path):
        try:
            with open(positive_prompt_path, 'r', encoding='utf-8') as f:
                positive_prompt = f.read().strip()
            if 'FPS-30' in positive_prompt:
                positive_prompt = positive_prompt.replace('FPS-30', f'FPS-{fps}')
        except Exception as e:
            positive_prompt = None
    
    if positive_prompt is None or positive_prompt == "":
        positive_prompt = "high quality, video, The character turns their head left and right slowly while speaking."
        use_default_pos = True
        
    negative_prompt = None
    if os.path.exists(negative_prompt_path):
        try:
            with open(negative_prompt_path, 'r', encoding='utf-8') as f:
                negative_prompt = f.read().strip()
            if 'FPS-30' in negative_prompt:
                negative_prompt = negative_prompt.replace('FPS-30', f'FPS-{fps}')
        except Exception as e:
            negative_prompt = None
            
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
        use_default_neg = True
        
    return positive_prompt, negative_prompt, use_default_pos, use_default_neg

def generate_video_name(image_path, audio_path):
    image_parts = image_path.rstrip('/').split('/')
    image_name = os.path.splitext(image_parts[-1])[0]
    image_folder1 = image_parts[-2] if len(image_parts) > 1 else ""
    image_folder2 = image_parts[-3] if len(image_parts) > 2 else ""
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    video_name = f"{image_folder2}_{image_folder1}_{image_name}_{audio_name}.mp4"
    video_name = video_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return video_name

def load_pipeline(args, device): 
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
    if args.ckpt_path and os.path.exists(args.ckpt_path):
        print(f"[INFO] Loading checkpoint from: {args.ckpt_path}")
        state_dict = load_state_dict(args.ckpt_path)
        missing, unexpected = pipe.dit.load_state_dict(state_dict, strict=False)
        if missing: print(f"[WARNING] Missing keys: {len(missing)}")
        if unexpected: print(f"[WARNING] Unexpected keys: {len(unexpected)}")
        print(f"[INFO] Checkpoint loaded successfully")
    return pipe

def load_and_expand_id_images(id_image_paths):
    """
    加载并扩充 ID 图片到 9 张。
    使用左右翻转或直接复制。
    """
    images_bytes = []
    for p in id_image_paths:
        try:
            with open(p, "rb") as f:
                images_bytes.append(f.read()) # 读取二进制
        except Exception as e:
            print(f"[WARN] Failed to load ID image {p}: {e}")
            
    if not images_bytes:
        return []
        
    expanded_binaries = list(images_bytes)
    while len(expanded_binaries) < 9: # 扩充到 9 张
        src_binary = random.choice(images_bytes)
        if random.random() < 0.5:
            try:
                np_img = cv2.imdecode(np.frombuffer(src_binary, np.uint8), cv2.IMREAD_COLOR) # 解码为图像阵列
                flipped = cv2.flip(np_img, 1) # 水平翻转增强多样性
                _, encoded = cv2.imencode('.png', flipped) # 重新编码
                expanded_binaries.append(encoded.tobytes()) # 加入扩充列表
            except:
                expanded_binaries.append(src_binary) # 失败则原样复制
        else:
            expanded_binaries.append(src_binary) # 直接原样复制
            
    return expanded_binaries[:9] # 确保只返回 9 张

def generate_id_grid_with_smpl(image_path, id_image_paths, args, smpl_infer):
    """
    调用 SmplInfer 扣人脸并生成九宫格。
    """
    if not args.enable_id_grid or smpl_infer is None:
        return None
        
    with open(image_path, "rb") as f:
        image_data = f.read() # 首帧图片数据
        
    id_video_data = None
    input_id_image_list_binary = None

    if args.id_video_path and os.path.exists(args.id_video_path):
        with open(args.id_video_path, "rb") as f:
            id_video_data = f.read() # 模式A: 用户传入参考视频提取人脸
    elif id_image_paths:
        input_id_image_list_binary = load_and_expand_id_images(id_image_paths) # 模式B: 使用扩充后的 ID 图像列表
    else:
        input_id_image_list_binary = load_and_expand_id_images([image_path]) # 兜底：只用首帧

    # 确定等效面积尺寸
    try:
        img = Image.open(image_path)
        w, h = img.size
    except:
        w, h = 560, 480
    face_ar = w / h
    area = args.id_grid_max_pixels
    grid_w_raw = (area * face_ar) ** 0.5
    grid_h_raw = (area / face_ar) ** 0.5
    final_w = int(round(grid_w_raw / 48) * 48) # 对齐到 48 的倍数
    final_h = int(round(grid_h_raw / 48) * 48)
    if final_w == 0: final_w = 48
    if final_h == 0: final_h = 48

    debug_dir = os.path.join(args.output_base_dir, 'debug_face_grid')
    os.makedirs(debug_dir, exist_ok=True)
    
    # 核心调用：使用提供的 get_face_grid 获取多帧九宫格数组
    res = smpl_infer.get_face_grid(
        image_data, 
        id_video_data=id_video_data, 
        input_id_image_list_binary=input_id_image_list_binary, 
        output_dir=debug_dir, 
        target_size=[final_h, final_w], 
        save_path_dir=debug_dir
    )
    
    if not res:
        return None
        
    frames = []
    for frame_np in res:
        frame_tensor = torch.from_numpy(frame_np.copy()).float() # 转换为浮点 Tensor
        frame_tensor = frame_tensor.permute(2, 0, 1) # (C, H, W)
        frames.append(frame_tensor)
        
    video_tensor = torch.stack(frames, dim=1) # (C, T_raw, H, W)
    
    # 适配输出序列长度 (id_grid_num_frames)
    N = video_tensor.shape[1]
    T = args.id_grid_num_frames
    if N != T:
        if N == 1:
            video_tensor = video_tensor.repeat(1, T, 1, 1) # 退化为静态图
        else:
            indices = np.linspace(0, N - 1, T, dtype=int) # 动态采样多帧
            video_tensor = video_tensor[:, indices, :, :]
            
    video_tensor = video_tensor / 127.5 - 1.0 # 像素值归一化至 [-1, 1]
    return video_tensor

def run_inference(pipe, image_path, audio_path, args, prompt, negative_prompt, id_grid, target_h, target_w):
    input_image = Image.open(image_path).convert("RGB")
    if target_h is not None and target_w is not None:
        input_image = ImageOps.fit(input_image, (target_w, target_h), Image.LANCZOS) # 按计算好的等效面积目标宽高缩放
    elif args.height and args.width:
        input_image = ImageOps.fit(input_image, (args.width, args.height), Image.LANCZOS) # 兜底逻辑
        
    input_audio, sample_rate = librosa.load(audio_path, sr=args.audio_sample_rate) # 保持原样读取
    
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
        id_grid=id_grid # 传入动态生成的 ID Grid
    )
    return video

def save_inference_config(output_dir, args, rank):
    if rank != 0: return
    config_path = os.path.join(output_dir, "inference_config.txt")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(f"ID Grid I2V Inference Configuration\n")
        f.write(f"Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Data Paths:\n  image_list_path: {args.dataset_metadata_path}\n  audio_dir: {args.audio_dir}\n")
        f.write(f"Model:\n  ckpt_path: {args.ckpt_path}\n  model_id: {args.model_id}\n")
        f.write(f"Inference:\n  num_frames: {args.num_frames}\n  height: {args.height}\n  width: {args.width}\n  max_pixels: {args.max_pixels}\n")
        f.write(f"ID Grid:\n  enable_id_grid: {args.enable_id_grid}\n  id_grid_max_pixels: {args.id_grid_max_pixels}\n  id_grid_num_frames: {args.id_grid_num_frames}\n")

def main():
    args = parse_args()
    rank, world_size, local_rank = get_distributed_info(args)
    
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is not available")
    num_gpus = torch.cuda.device_count()
    if local_rank >= num_gpus: local_rank = 0
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    
    # Init output and config
    timestamp = args.output_timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_base_dir, f"output_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    save_inference_config(output_dir, args, rank)
    
    # Load pipeline and external tools
    pipe = load_pipeline(args, device)
    if SmplInfer is not None:
        # 初始化外部的扣人脸/九宫格模块，设置 is_image=True 以适配推理单图/少数图场景
        smpl_infer = SmplInfer(smpl_checkpoints_path='/ytech_milm/liujiwen/kling_motion_service/smpl_all_checkpoints', is_image=True)
    else:
        smpl_infer = None

    # Parse dataset (CSV parsing instead of txt)
    try:
        df = pd.read_csv(args.dataset_metadata_path)
        if 'image' not in df.columns:
            raise ValueError(f"Column 'image' not found in CSV {args.dataset_metadata_path}")
        total_images = len(df)
        my_indices = list(range(rank, total_images, world_size))
        print(f"[RANK {rank}] Loaded CSV with {total_images} rows. Processing {len(my_indices)} rows.")
    except Exception as e:
        print(f"[ERROR] Failed to load CSV: {e}")
        return

    # Audio loading
    if args.littletestdataset:
        with open(args.audio_dir, 'r', encoding='utf-8') as f:
            all_audio_files = [line.strip() for line in f if line.strip()]
    else:
        all_audio_files = get_all_audio_files(args.audio_dir)

    random.seed(args.seed + rank)
    
    success_count = 0
    fail_count = 0
    default_prompt_count = 0 # 记录缺失 prompt 的数量

    for row_idx in my_indices:
        row = df.iloc[row_idx]
        image_path = str(row['image']).strip()
        
        # 提取 ID 字段（列表）
        id_image_paths = []
        if 'ID' in df.columns:
            id_field = row['ID']
            try:
                if isinstance(id_field, str):
                    id_image_paths = ast.literal_eval(id_field) # 将字符串的 list 转换成真的 list
                elif isinstance(id_field, list):
                    id_image_paths = id_field
            except:
                id_image_paths = []

        if not os.path.exists(image_path):
            print(f"[RANK {rank}] [WARN] Image not found: {image_path}, skipping...")
            fail_count += 1
            continue
            
        print(f"[RANK {rank}] Processing row index {row_idx}: {image_path}")
        
        # Prompt 加载与统计
        prompt, negative_prompt, use_def_pos, use_def_neg = load_prompts(image_path, args.fps, rank)
        if use_def_pos or use_def_neg:
            default_prompt_count += 1 # 如果任意一个使用了默认值，则计数加1
            
        # Audio 选择
        if args.littletestdataset:
            selected_audios = [all_audio_files[row_idx]] if row_idx < len(all_audio_files) else []
            num_audios = len(selected_audios)
        else:
            num_audios = min(args.num_audios_per_image, len(all_audio_files))
            selected_audios = random.sample(all_audio_files, num_audios) if all_audio_files else []
            
        for audio_idx, audio_path in enumerate(selected_audios):
            if not os.path.exists(audio_path): continue
            
            try:
                # 生成九宫格
                id_grid = generate_id_grid_with_smpl(image_path, id_image_paths, args, smpl_infer)
                if id_grid is not None:
                    id_grid = id_grid.to(device)

                # 计算等效面积目标尺寸
                target_h, target_w = args.height, args.width
                if target_h is None or target_w is None:
                    pil_img = Image.open(image_path)
                    w, h = pil_img.size
                    if w * h > args.max_pixels:
                        scale = (w * h / args.max_pixels) ** 0.5
                        target_h = int(h / scale)
                        target_w = int(w / scale)
                    else:
                        target_h, target_w = h, w
                    target_h = target_h // 16 * 16 # VAE 需要 16 的倍数
                    target_w = target_w // 16 * 16

                # 推理
                video = run_inference(pipe, image_path, audio_path, args, prompt, negative_prompt, id_grid, target_h, target_w)
                
                # 保存
                video_name = generate_video_name(image_path, audio_path)
                video_save_path = os.path.join(output_dir, video_name)
                if os.path.exists(video_save_path):
                    video_save_path = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_rank{rank}.mp4")
                    
                save_video_with_audio(video, video_save_path, audio_path, fps=args.fps, quality=args.quality)
                success_count += 1
                
            except Exception as e:
                print(f"[RANK {rank}] [ERROR] Failed: {e}")
                fail_count += 1
                
    print(f"\n[RANK {rank}] Inference done! Success: {success_count}, Fail: {fail_count}")
    print(f"[RANK {rank}] Times default prompt was used (pos or neg missing): {default_prompt_count}")

if __name__ == "__main__":
    main()
