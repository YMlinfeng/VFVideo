import torch, types
import numpy as np
from PIL import Image
from einops import repeat
from typing import Optional, Union
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal
from transformers import Wav2Vec2Processor

from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit

from ..models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from ..models.wan_video_dit_s2v import rope_precompute
from ..models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..models.wan_video_animate_adapter import WanAnimateAdapter
from ..models.wan_video_mot import MotWanModel
from ..models.wav2vec import WanS2VAudioEncoder
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

def visualize_tensor(tensor_data, save_prefix="debug_output_data", fps=15):
    """
    通用可视化函数，修复 Broken pipe 问题。
    """
    import torch
    import numpy as np
    import os
    import imageio
    from PIL import Image

    # 1. 准备目录
    save_dir = os.path.dirname(save_prefix)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # 2. 搬运数据
    data = tensor_data.detach().cpu()
    
    # === 视频处理 (C, T, H, W) ===
    if data.ndim == 4:
        # 维度变换: (C, T, H, W) -> (T, H, W, C)
        vis_np = data.permute(1, 2, 3, 0).numpy()
        
        # 反归一化
        vis_np = (vis_np + 1.0) / 2.0 * 255.0
        vis_np = np.clip(vis_np, 0, 255).astype(np.uint8)

        # 【核心修复 1】: 强制内存连续！
        # permute 后的数组在内存中是不连续的，直接喂给 ffmpeg 必挂
        vis_np = np.ascontiguousarray(vis_np)

        # 保存第一帧
        try:
            first_frame = vis_np[0]
            Image.fromarray(first_frame).save(f"{save_prefix}_frame0.jpg")
            print(f"✅ [Video] 第一帧已保存")
        except Exception as e:
            print(f"⚠️ 保存图片失败: {e}")

        # 保存完整视频
        video_path = f"{save_prefix}_video.mp4"
        try:
            # 【核心修复 2】: 添加 macro_block_size=1
            # 这告诉 imageio 不要因为尺寸不是16的倍数就报错，也不要尝试缩放
            imageio.mimsave(
                video_path, 
                vis_np, 
                fps=fps, 
                codec='libx264',
                macro_block_size=1  # 关键参数
            )
            print(f"✅ [Video] 完整视频已保存: {video_path}")
        except Exception as e:
            print(f"❌ [Video] MP4保存失败 ({e})，正在降级保存为 GIF...")
            # 备选方案: GIF 是纯 Python 生成，不依赖外部 FFmpeg 进程，最稳
            try:
                gif_path = f"{save_prefix}_video.gif"
                imageio.mimsave(gif_path, vis_np, fps=fps)
                print(f"✅ [Video] GIF 已保存: {gif_path}")
            except Exception as e_gif:
                print(f"❌ [Video] GIF 也保存失败: {e_gif}")

    # === 图片处理 (C, H, W) ===
    elif data.ndim == 3:
        vis_np = data.permute(1, 2, 0).numpy()
        vis_np = (vis_np + 1.0) / 2.0 * 255.0
        vis_np = np.clip(vis_np, 0, 255).astype(np.uint8)
        
        # 图片也最好连续化一下
        vis_np = np.ascontiguousarray(vis_np)

        Image.fromarray(vis_np).save(f"{save_prefix}_image.jpg")
        print(f"✅ [Image] 图片已保存")

    else:
        print(f"⚠️ [Visualizer] 跳过: 形状 {data.shape} 不支持")

def visualize_latents(latents, save_prefix="debug_latents", fps=8):
    """
    专门用于可视化 5D Latents (B, C, F, H, W)
    由于 Latents 有16个通道，无法直接显示，这里提供"前3通道伪彩色"和"热力图"两种视角。
    """
    # === 1. 内部导入 ===
    import torch
    import numpy as np
    import os
    import imageio
    import matplotlib.pyplot as plt

    # === 2. 准备目录 ===
    save_dir = os.path.dirname(save_prefix)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print(f"[LatentVis] Input shape: {latents.shape}")

    # === 3. 数据预处理 ===
    # 确保是 (C, F, H, W) 结构 (去掉 Batch 维度)
    # 如果是 (B, C, F, H, W)，取第0个样本
    if latents.ndim == 5:
        data = latents[0].detach().cpu().float()
    elif latents.ndim == 4:
        data = latents.detach().cpu().float()
    else:
        print(f"❌ [LatentVis] 维度不对，跳过: {latents.shape}")
        return

    # === 视点 A: 伪彩色 (取前3个通道模拟 RGB) ===
    # 形状: (3, F, H, W) -> (F, H, W, 3)
    rgb_proxy = data[:3, :, :, :].permute(1, 2, 3, 0).numpy()
    
    # Latents 数值通常在 -3 到 3 之间，我们需要重新映射到 0-255
    # 这里采用统计归一化，让画面对比度正常一点
    mean_val = np.mean(rgb_proxy)
    std_val = np.std(rgb_proxy)
    # 截断在 (均值 ± 2个标准差) 范围内
    rgb_proxy = np.clip(rgb_proxy, mean_val - 2*std_val, mean_val + 2*std_val)
    # 归一化到 0-255
    rgb_proxy = ((rgb_proxy - rgb_proxy.min()) / (rgb_proxy.max() - rgb_proxy.min() + 1e-6) * 255.0).astype(np.uint8)

    # === 视点 B: 能量热力图 (所有通道取绝对值平均) ===
    # (16, F, H, W) -> (F, H, W)
    activation = torch.mean(torch.abs(data), dim=0).numpy()
    # 归一化
    activation = ((activation - activation.min()) / (activation.max() - activation.min() + 1e-6) * 255.0).astype(np.uint8)

    # === 4. 保存为视频 ===
    try:
        # 保存伪彩色视频
        path_rgb = f"{save_prefix}_pseudo_rgb.mp4"
        imageio.mimsave(path_rgb, rgb_proxy, fps=fps, codec='libx264', macro_block_size=1)
        print(f"✅ [Latent] 伪彩色视频: {path_rgb}")

        # 保存热力图视频
        path_act = f"{save_prefix}_activation.mp4"
        imageio.mimsave(path_act, activation, fps=fps, codec='libx264', macro_block_size=1)
        print(f"✅ [Latent] 热力图视频: {path_act}")

    except Exception as e:
        print(f"❌ [Latent] 保存视频失败: {e}，尝试保存 GIF...")
        imageio.mimsave(f"{save_prefix}_activation.gif", activation, fps=fps)

    # === 5. 保存第一帧对比图 (JPG) ===
    try:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        ax[0].imshow(rgb_proxy[0])
        ax[0].set_title(f"Frame 0 (Pseudo RGB)\nShape: {rgb_proxy[0].shape}")
        ax[0].axis('off')

        ax[1].imshow(activation[0], cmap='inferno') # 使用热力图配色
        ax[1].set_title(f"Frame 0 (Activation Map)\nMean across {data.shape[0]} channels")
        ax[1].axis('off')

        jpg_path = f"{save_prefix}_frame0_check.jpg"
        plt.tight_layout()
        plt.savefig(jpg_path)
        plt.close()
        print(f"✅ [Latent] 第一帧对比图: {jpg_path}")
    except Exception as e:
        print(f"⚠️ 保存图片失败: {e}")


class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler("Wan")
        self.tokenizer: HuggingfaceTokenizer = None
        self.audio_processor: Wav2Vec2Processor = None
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None #!
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None #!
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.vace2: VaceWanModel = None
        self.vap: MotWanModel = None
        self.animate_adapter: WanAnimateAdapter = None
        self.audio_encoder: WanS2VAudioEncoder = None #!
        # self.in_iteration_models = ("dit", "motion_controller", "vace", "animate_adapter", "vap")
        self.in_iteration_models = ("dit",) # or ['dit']
        self.in_iteration_models_2 = ("dit2", "motion_controller", "vace2", "animate_adapter", "vap")
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_S2V(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
            # WanVideoUnit_FunControl(), 
            # WanVideoUnit_FunReference(), 
            # WanVideoUnit_FunCameraControl(), 
            # WanVideoUnit_SpeedControl(), 
            # WanVideoUnit_VACE(),
            # WanVideoUnit_AnimateVideoSplit(),
            # WanVideoUnit_AnimatePoseLatents(), 
            # WanVideoUnit_AnimateFacePixelValues(),
            # WanVideoUnit_AnimateInpaint(),
            # WanVideoUnit_VAP(),
            # WanVideoUnit_UnifiedSequenceParallel(),
            # WanVideoUnit_TeaCache(),
            # WanVideoUnit_CfgMerger(),
        ]
        self.post_units = [
            WanVideoPostUnit_S2V(),
        ]
        self.model_fn = model_fn_wan_video


    def enable_usp(self):
        from ..utils.xfuser import get_sequence_parallel_world_size, usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True


    @staticmethod # 静态方法不需要 self 或 cls 参数，可以直接通过类名调用
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [], #type:ignore
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = True,
        use_usp: bool = False,
        vram_limit: float = None,
    ):
        '''
        从硬盘把训练好的 .pth 或 .safetensors 文件读进来，填入上面定义的 vae、dit 等组件里
        '''
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"),
                "Wan2.1_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.1_VAE.safetensors"),
                "Wan2.2_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern][0]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to {redirect_dict[model_config.origin_file_pattern]}. You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern][0]
                    model_config.origin_file_pattern = redirect_dict[model_config.origin_file_pattern][1]
        
        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp:
            from ..utils.xfuser import initialize_usp
            initialize_usp(device)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("wan_video_text_encoder")
        dit = model_pool.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_pool.fetch_model("wan_video_vae")
        pipe.image_encoder = model_pool.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_pool.fetch_model("wan_video_motion_controller")
        vace = model_pool.fetch_model("wan_video_vace", index=2)
        if isinstance(vace, list):
            pipe.vace, pipe.vace2 = vace
        else:
            pipe.vace = vace
        # pipe.vap = model_pool.fetch_model("wan_video_vap")
        pipe.audio_encoder = model_pool.fetch_model("wans2v_audio_encoder")
        # pipe.animate_adapter = model_pool.fetch_model("wan_video_animate_adapter")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer and processor
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = HuggingfaceTokenizer(name=tokenizer_config.path, seq_len=512, clean='whitespace')
        if audio_processor_config is not None:
            audio_processor_config.download_if_necessary()
            pipe.audio_processor = Wav2Vec2Processor.from_pretrained(audio_processor_config.path)
        
        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None, #type:ignore
        denoising_strength: Optional[float] = 1.0,
        # Speech-to-video
        input_audio: Optional[np.array] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 16000,
        s2v_pose_video: Optional[list[Image.Image]] = None, #type:ignore
        s2v_pose_latents: Optional[torch.Tensor] = None, #type:ignore
        motion_video: Optional[list[Image.Image]] = None, #type:ignore
        # ControlNet
        control_video: Optional[list[Image.Image]] = None, #type:ignore
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]] = None,
        camera_control_speed: Optional[float] = 1/54,
        camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # VACE
        vace_video: Optional[list[Image.Image]] = None, #type:ignore
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Animate
        animate_pose_video: Optional[list[Image.Image]] = None, #type:ignore
        animate_face_video: Optional[list[Image.Image]] = None, #type:ignore
        animate_inpaint_video: Optional[list[Image.Image]] = None, #type:ignore
        animate_mask_video: Optional[list[Image.Image]] = None, #type:ignore
        # VAP
        vap_video: Optional[list[Image.Image]] = None, #type:ignore
        vap_prompt: Optional[str] = " ",
        negative_vap_prompt: Optional[str] = " ",
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # LongCat-Video
        longcat_video: Optional[list[Image.Image]] = None, #type:ignore
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52), #type:ignore
        tile_stride: Optional[tuple[int, int]] = (15, 26), #type:ignore
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
        output_type: Optional[Literal["quantized", "floatpoint"]] = "quantized",
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "vap_prompt": vap_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "negative_vap_prompt": negative_vap_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_video": control_video, "reference_image": reference_image,
            "camera_control_direction": camera_control_direction, "camera_control_speed": camera_control_speed, "camera_control_origin": camera_control_origin,
            "vace_video": vace_video, "vace_video_mask": vace_video_mask, "vace_reference_image": vace_reference_image, "vace_scale": vace_scale,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "longcat_video": longcat_video,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
            "input_audio": input_audio, "audio_sample_rate": audio_sample_rate, 
            "s2v_pose_video": s2v_pose_video, "audio_embeds": audio_embeds, "s2v_pose_latents": s2v_pose_latents, "motion_video": motion_video,
            "animate_pose_video": animate_pose_video, "animate_face_video": animate_face_video, "animate_inpaint_video": animate_inpaint_video, "animate_mask_video": animate_mask_video,
            "vap_video": vap_video, 
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if timestep.item() < switch_DiT_boundary * 1000 and self.dit2 is not None and not models["dit"] is self.dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
                models["vace"] = self.vace2
                
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device) #[1000.]
            
            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep) # torch.Size([1, 16, 15, H, W])
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep) # 一遍前向过程仅需一秒多
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler 利用预测出的噪声 noise_pred，计算出上一个时间步（噪声更少）的 Latents。这是画面逐渐清晰的过程
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
        

        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        # Decode
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        elif output_type == "floatpoint":
            pass
        # [img.save(f"/m2v_intern/mengzijie/DiffSynth-Studio/output/output_20260127_144600/frame_{i:03d}.png") for i, img in enumerate(video)]
        self.load_models_to_device([])
        return video



class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames"),
            output_params=("height", "width", "num_frames"),
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image"),
            output_params=("noise",)
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, vace_reference_image):
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            f = len(vace_reference_image) if isinstance(vace_reference_image, list) else 1
            length += f
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -f:], noise[:, :, :-f]), dim=2)
        
        return {"noise": noise} # ([1, 16, 15, 80, 70])
    


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "vace_reference_image"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, noise, tiled, tile_size, tile_stride, vace_reference_image):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(self.onload_model_names)
        # input_video = pipe.preprocess_video(input_video)
        input_video = input_video.unsqueeze(0)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        
        if vace_reference_image is not None:
            if not isinstance(vace_reference_image, list):
                vace_reference_image = [vace_reference_image]
            vace_reference_image = pipe.preprocess_video(vace_reference_image)
            vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0]) #todo
            return {"latents": latents}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    '''
    promptembedder
    '''
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            output_params=("context",),
            onload_model_names=("text_encoder",)
        )
    
    def encode_prompt(self, pipe: WanVideoPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = self.encode_prompt(pipe, prompt)
        return {"context": prompt_emb}



class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            output_params=("clip_feature",),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, height, width):
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"clip_feature": clip_context}
    


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("y",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        
        # 确保y张量不会在后续梯度计算中被in-place修改111
        y = y.clone().detach().requires_grad_(True)
        
        return {"y": y}



class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "fuse_vae_embedding_in_latents", "first_frame_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, latents, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        # i = input_image; i = (i + 1.0) / 2.0; plt.imsave("output0.png", F.to_pil_image(i.float()))
        if isinstance(input_image, torch.Tensor):
            resized_image = F.resize(input_image, (height, width)) 
            image = resized_image.unsqueeze(1)
        else:
            resized_image = input_image.resize((width, height))
            image = pipe.preprocess_image(resized_image).transpose(0, 1) #（C， B， H， W）
        # a = resized_image; a = (a + 1.0) / 2.0; plt.imsave("output1.png", F.to_pil_image(a.float()))
        # import torch; import matplotlib.pyplot as plt; import torchvision.transforms.functional as F; image = image.squeeze(1).to(torch.float32); plt.imsave("output1.png", F.to_pil_image(image))

        z = pipe.vae.encode([image], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        # print(f"DEBUG: latents version before assignment: {latents._version}, requires_grad: {latents.requires_grad}")

        latents[:, :, 0: 1] = z # z: ([1, 16, 1, 80, 70])
        # latents = torch.cat([z, latents[:, :, 1:]], dim=2)

        # === 插入这里进行观测 ===
        # 1. 看一眼被塞进去的 z (第一帧) 长啥样
        # 这里的 z 是 (B, C, 1, H, W)
        # visualize_latents(z, save_prefix="./debug_vis/z_first_frame")
        
        # 2. 看一眼整个 latents 现在的状态
        # 这里的 latents 是 (B, C, 81, H, W)
        # visualize_latents(latents, save_prefix="./debug_vis/latents_full")
        # =======================

        # 这一行之后
        # print(f"DEBUG: latents version after assignment: {latents._version}")
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}



class WanVideoUnit_FunControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("control_video", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "clip_feature", "y", "latents"),
            output_params=("clip_feature", "y"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, control_video, num_frames, height, width, tiled, tile_size, tile_stride, clip_feature, y, latents):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(control_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        y_dim = pipe.dit.in_dim-control_latents.shape[1]-latents.shape[1]
        if clip_feature is None or y is None:
            clip_feature = torch.zeros((1, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.zeros((1, y_dim, (num_frames - 1) // 4 + 1, height//8, width//8), dtype=pipe.torch_dtype, device=pipe.device)
        else:
            y = y[:, -y_dim:]
        y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}
    


class WanVideoUnit_FunReference(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_image", "height", "width", "reference_image"),
            output_params=("reference_latents", "clip_feature"),
            onload_model_names=("vae", "image_encoder")
        )

    def process(self, pipe: WanVideoPipeline, reference_image, height, width):
        if reference_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        reference_image = reference_image.resize((width, height))
        reference_latents = pipe.preprocess_video([reference_image])
        reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
        if pipe.image_encoder is None:
            return {"reference_latents": reference_latents}
        clip_feature = pipe.preprocess_image(reference_image)
        clip_feature = pipe.image_encoder.encode_image([clip_feature])
        return {"reference_latents": reference_latents, "clip_feature": clip_feature}



class WanVideoUnit_FunCameraControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "camera_control_direction", "camera_control_speed", "camera_control_origin", "latents", "input_image", "tiled", "tile_size", "tile_stride"),
            output_params=("control_camera_latents_input", "y"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, camera_control_direction, camera_control_speed, camera_control_origin, latents, input_image, tiled, tile_size, tile_stride):
        if camera_control_direction is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        camera_control_plucker_embedding = pipe.dit.control_adapter.process_camera_coordinates(
            camera_control_direction, num_frames, height, width, camera_control_speed, camera_control_origin)
        
        control_camera_video = camera_control_plucker_embedding[:num_frames].permute([3, 0, 1, 2]).unsqueeze(0)
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                control_camera_video[:, :, 1:]
            ], dim=2
        ).transpose(1, 2)
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
        control_camera_latents_input = control_camera_latents.to(device=pipe.device, dtype=pipe.torch_dtype)
        
        input_image = input_image.resize((width, height))
        input_latents = pipe.preprocess_video([input_image])
        input_latents = pipe.vae.encode(input_latents, device=pipe.device)
        y = torch.zeros_like(latents).to(pipe.device)
        y[:, :, :1] = input_latents
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)

        if y.shape[1] != pipe.dit.in_dim - latents.shape[1]:
            image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
            y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
            msk[:, 1:] = 0
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
            msk = msk.transpose(1, 2)[0]
            y = torch.cat([msk,y])
            y = y.unsqueeze(0)
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"control_camera_latents_input": control_camera_latents_input, "y": y}



class WanVideoUnit_SpeedControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("motion_bucket_id",),
            output_params=("motion_bucket_id",)
        )

    def process(self, pipe: WanVideoPipeline, motion_bucket_id):
        if motion_bucket_id is None:
            return {}
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"motion_bucket_id": motion_bucket_id}



class WanVideoUnit_VACE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("vace_video", "vace_video_mask", "vace_reference_image", "vace_scale", "height", "width", "num_frames", "tiled", "tile_size", "tile_stride"),
            output_params=("vace_context", "vace_scale"),
            onload_model_names=("vae",)
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        vace_video, vace_video_mask, vace_reference_image, vace_scale,
        height, width, num_frames,
        tiled, tile_size, tile_stride
    ):
        if vace_video is not None or vace_video_mask is not None or vace_reference_image is not None:
            pipe.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros((1, 3, num_frames, height, width), dtype=pipe.torch_dtype, device=pipe.device)
            else:
                vace_video = pipe.preprocess_video(vace_video)
            
            if vace_video_mask is None:
                vace_video_mask = torch.ones_like(vace_video)
            else:
                vace_video_mask = pipe.preprocess_video(vace_video_mask, min_value=0, max_value=1)
            
            inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
            reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
            inactive = pipe.vae.encode(inactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            reactive = pipe.vae.encode(reactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)
            
            vace_mask_latents = rearrange(vace_video_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')
            
            if vace_reference_image is None:
                pass
            else:
                if not isinstance(vace_reference_image,list):
                    vace_reference_image = [vace_reference_image]

                vace_reference_image = pipe.preprocess_video(vace_reference_image)

                bs, c, f, h, w = vace_reference_image.shape
                new_vace_ref_images = []
                for j in range(f):
                    new_vace_ref_images.append(vace_reference_image[0, :, j:j+1])
                vace_reference_image = new_vace_ref_images
                
                vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1)
                vace_reference_latents = [u.unsqueeze(0) for u in vace_reference_latents]

                vace_video_latents = torch.concat((*vace_reference_latents, vace_video_latents), dim=2)
                vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :f]), vace_mask_latents), dim=2)
            
            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return {"vace_context": None, "vace_scale": vace_scale}


class WanVideoUnit_VAP(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=("text_encoder", "vae", "image_encoder"),
            input_params=("vap_video", "vap_prompt", "negative_vap_prompt", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("vap_clip_feature", "vap_hidden_state", "context_vap")
        )
        
    def encode_prompt(self, pipe: WanVideoPipeline, prompt):
        ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if inputs_shared.get("vap_video") is None:
            return inputs_shared, inputs_posi, inputs_nega
        else:
            # 1. encode vap prompt
            pipe.load_models_to_device(["text_encoder"])
            vap_prompt, negative_vap_prompt = inputs_posi.get("vap_prompt", ""), inputs_nega.get("negative_vap_prompt", "")
            vap_prompt_emb = self.encode_prompt(pipe, vap_prompt)
            negative_vap_prompt_emb = self.encode_prompt(pipe, negative_vap_prompt)
            inputs_posi.update({"context_vap":vap_prompt_emb})
            inputs_nega.update({"context_vap":negative_vap_prompt_emb})
            # 2. prepare vap image clip embedding
            pipe.load_models_to_device(["vae", "image_encoder"])
            vap_video, end_image = inputs_shared.get("vap_video"), inputs_shared.get("end_image")

            num_frames, height, width = inputs_shared.get("num_frames"),inputs_shared.get("height"), inputs_shared.get("width")
            
            image_vap = pipe.preprocess_image(vap_video[0].resize((width, height))).to(pipe.device)

            vap_clip_context = pipe.image_encoder.encode_image([image_vap])
            if end_image is not None:
                vap_end_image = pipe.preprocess_image(vap_video[-1].resize((width, height))).to(pipe.device)
                if pipe.dit.has_image_pos_emb:
                    vap_clip_context = torch.concat([vap_clip_context, pipe.image_encoder.encode_image([vap_end_image])], dim=1)
            vap_clip_context = vap_clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
            inputs_shared.update({"vap_clip_feature":vap_clip_context})

            # 3. prepare vap latents            
            msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
            msk[:, 1:] = 0
            if end_image is not None:
                msk[:, -1:] = 1
                last_image_vap = pipe.preprocess_image(vap_video[-1].resize((width, height))).to(pipe.device)
                vae_input = torch.concat([image_vap.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image_vap.device), last_image_vap.transpose(0,1)],dim=1)
            else:
                vae_input = torch.concat([image_vap.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image_vap.device)], dim=1)
            
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
            msk = msk.transpose(1, 2)[0]

            tiled,tile_size,tile_stride = inputs_shared.get("tiled"), inputs_shared.get("tile_size"), inputs_shared.get("tile_stride")

            y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.concat([msk, y])
            y = y.unsqueeze(0)
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)

            vap_video = pipe.preprocess_video(vap_video)
            vap_latent = pipe.vae.encode(vap_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)

            vap_latent = torch.concat([vap_latent,y], dim=1).to(dtype=pipe.torch_dtype, device=pipe.device)
            inputs_shared.update({"vap_hidden_state":vap_latent})

            return inputs_shared, inputs_posi, inputs_nega



class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=(), output_params=("use_unified_sequence_parallel",))

    def process(self, pipe: WanVideoPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}



class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            output_params=("tea_cache",)
        )

    def process(self, pipe: WanVideoPipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}



class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_S2V(PipelineUnit):
    '''
    s2v
    '''
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=("audio_encoder", "vae",),
            input_params=("input_audio", "audio_embeds", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "audio_sample_rate", "s2v_pose_video", "s2v_pose_latents", "motion_video"),
            output_params=("audio_embeds", "motion_latents", "drop_motion_frames", "s2v_pose_latents"),
        )

    def process_audio(self, pipe: WanVideoPipeline, input_audio, audio_sample_rate, num_frames, fps=16, audio_embeds=None, return_all=False):
        if audio_embeds is not None:
            return {"audio_embeds": audio_embeds}
        pipe.load_models_to_device(["audio_encoder"])
        # 调用 pipe.audio_encoder (Wav2Vec2) 提取特征
        audio_embeds = pipe.audio_encoder.get_audio_feats_per_inference(input_audio, audio_sample_rate, pipe.audio_processor, fps=fps, batch_frames=num_frames-1, dtype=pipe.torch_dtype, device=pipe.device)
        if return_all:
            return audio_embeds
        else:
            return {"audio_embeds": audio_embeds[0]}
           

    def process_motion_latents(self, pipe: WanVideoPipeline, height, width, tiled, tile_size, tile_stride, motion_video=None):
        pipe.load_models_to_device(["vae"])
        motion_frames = 73
        kwargs = {}
        if motion_video is not None:
            assert motion_video.shape[2] == motion_frames, f"motion video must have {motion_frames} frames, but got {motion_video.shape[2]}"
            motion_latents = motion_video
            kwargs["drop_motion_frames"] = False
        else:
            motion_latents = torch.zeros([1, 3, motion_frames, height, width], dtype=pipe.torch_dtype, device=pipe.device)
            kwargs["drop_motion_frames"] = True
        motion_latents = pipe.vae.encode(motion_latents, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        
        kwargs.update({"motion_latents": motion_latents})
        return kwargs

    def process_pose_cond(self, pipe: WanVideoPipeline, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=None, num_repeats=1, return_all=False):
        if s2v_pose_latents is not None:
            return {"s2v_pose_latents": s2v_pose_latents}
        if s2v_pose_video is None:
            return {"s2v_pose_latents": None}
        pipe.load_models_to_device(["vae"])
        infer_frames = num_frames - 1
        input_video = pipe.preprocess_video(s2v_pose_video)[:, :, :infer_frames * num_repeats]
        # pad if not enough frames
        padding_frames = infer_frames * num_repeats - input_video.shape[2]
        input_video = torch.cat([input_video, -torch.ones(1, 3, padding_frames, height, width, device=input_video.device, dtype=input_video.dtype)], dim=2)
        input_videos = input_video.chunk(num_repeats, dim=2)
        pose_conds = []
        for r in range(num_repeats):
            cond = input_videos[r]
            cond = torch.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond], dim=2)
            cond_latents = pipe.vae.encode(cond, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            
            pose_conds.append(cond_latents[:,:,1:])

        if return_all:
            return pose_conds
        else:
            return {"s2v_pose_latents": pose_conds[0]}

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if (inputs_shared.get("input_audio") is None and inputs_shared.get("audio_embeds") is None) or pipe.audio_encoder is None or pipe.audio_processor is None:
            return inputs_shared, inputs_posi, inputs_nega
        num_frames, height, width, tiled, tile_size, tile_stride = inputs_shared.get("num_frames"), inputs_shared.get("height"), inputs_shared.get("width"), inputs_shared.get("tiled"), inputs_shared.get("tile_size"), inputs_shared.get("tile_stride")
        input_audio, audio_embeds, audio_sample_rate = inputs_shared.pop("input_audio", None), inputs_shared.pop("audio_embeds", None), inputs_shared.get("audio_sample_rate", 16000)
        # 如果没传 "s2v_pose_video"，这里得到的变量就是 None，不报错
        # s2v_pose_video, s2v_pose_latents, motion_video = inputs_shared.pop("s2v_pose_video", None), inputs_shared.pop("s2v_pose_latents", None), inputs_shared.pop("motion_video", None)

        audio_input_positive = self.process_audio(pipe, input_audio, audio_sample_rate, num_frames, audio_embeds=audio_embeds)
        # # 把真正的音频特征给“正向提示词”
        inputs_posi.update(audio_input_positive) # audio_input_positive["audio_embeds"].shape = [1, 25, 1024, 56]
        # 给“负向提示词”塞一个全 0 的静音特征 (CFG Guidance 的关键！)
        inputs_nega.update({"audio_embeds": 0.0 * audio_input_positive["audio_embeds"]})

        # inputs_shared.update(self.process_motion_latents(pipe, height, width, tiled, tile_size, tile_stride, motion_video))
        # inputs_shared.update(self.process_pose_cond(pipe, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=s2v_pose_latents))
        return inputs_shared, inputs_posi, inputs_nega

    @staticmethod # todo：这个函数什么时候调用
    def pre_calculate_audio_pose(pipe: WanVideoPipeline, input_audio=None, audio_sample_rate=16000, s2v_pose_video=None, num_frames=81, height=448, width=832, fps=16, tiled=True, tile_size=(30, 52), tile_stride=(15, 26)):
        assert pipe.audio_encoder is not None and pipe.audio_processor is not None, "Please load audio encoder and audio processor first."
        shapes = WanVideoUnit_ShapeChecker().process(pipe, height, width, num_frames)
        height, width, num_frames = shapes["height"], shapes["width"], shapes["num_frames"]
        unit = WanVideoUnit_S2V()
        audio_embeds = unit.process_audio(pipe, input_audio, audio_sample_rate, num_frames, fps, return_all=True)
        pose_latents = unit.process_pose_cond(pipe, s2v_pose_video, num_frames, height, width, num_repeats=len(audio_embeds), return_all=True, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        pose_latents = None if s2v_pose_video is None else pose_latents
        return audio_embeds, pose_latents, len(audio_embeds)


class WanVideoPostUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("latents", "motion_latents", "drop_motion_frames"))

    def process(self, pipe: WanVideoPipeline, latents, motion_latents, drop_motion_frames):
        if pipe.audio_encoder is None or motion_latents is None or drop_motion_frames:
            return {}
        latents = torch.cat([motion_latents, latents[:,:,1:]], dim=2)
        return {"latents": latents}


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x
        
        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + shift) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value



def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    vap: MotWanModel = None,
    animate_adapter: WanAnimateAdapter = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    vace_context = None,
    vace_scale = 1.0,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    vap_hidden_state = None,
    vap_clip_feature = None,
    context_vap = None,
    drop_motion_frames: bool = True,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    pose_latents=None,
    face_pixel_values=None,
    longcat_latents=None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
        
    # wan2.2 s2v
    if audio_embeds is not None:
        return model_fn_wans2v(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            audio_embeds=audio_embeds,
            motion_latents=motion_latents,
            s2v_pose_latents=s2v_pose_latents,
            drop_motion_frames=drop_motion_frames,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
        )

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)

    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents: #f and f
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    
    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context) #(1, 512, 1536)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
        
    # Camera control
    x = dit.patchify(x, control_camera_latents_input) # f
    
    # Animate
    if pose_latents is not None and face_pixel_values is not None:
        x, motion_vec = animate_adapter.after_patch_embedding(x, pose_latents, face_pixel_values)
    
    # Patchify
    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    
    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    # VAP 
    if vap is not None:
        # hidden state
        x_vap = vap_hidden_state
        x_vap = vap.patchify(x_vap)
        x_vap = rearrange(x_vap, 'b c f h w -> b (f h w) c').contiguous()
        # Timestep
        clean_timestep = torch.ones(timestep.shape, device=timestep.device).to(timestep.dtype)
        t = vap.time_embedding(sinusoidal_embedding_1d(vap.freq_dim, clean_timestep))
        t_mod_vap = vap.time_projection(t).unflatten(1, (6, vap.dim))

        # rope
        freqs_vap = vap.compute_freqs_mot(f,h,w).to(x.device)

        # context
        vap_clip_embedding = vap.img_emb(vap_clip_feature)
        context_vap = vap.text_embedding(context_vap)
        context_vap = torch.cat([vap_clip_embedding, context_vap], dim=1)
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if vace_context is not None: # f
        vace_hints = vace(
            x, vace_context, context, t_mod, freqs,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload
        )
    
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        def create_custom_forward_vap(block, vap):
            def custom_forward(*inputs):
                return vap(block, *inputs)
            return custom_forward
        
        for block_id, block in enumerate(dit.blocks):
            # Block
            if vap is not None and block_id in vap.mot_layers_mapping:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x, x_vap = torch.utils.checkpoint.checkpoint(
                            create_custom_forward_vap(block, vap),
                            x, context, t_mod, freqs, x_vap, context_vap, t_mod_vap, freqs_vap, block_id,
                            use_reentrant=False,
                        )
                elif use_gradient_checkpointing:
                    x, x_vap = torch.utils.checkpoint.checkpoint(
                        create_custom_forward_vap(block, vap),
                        x, context, t_mod, freqs, x_vap, context_vap, t_mod_vap, freqs_vap, block_id,
                        use_reentrant=False,
                    )
                else:
                    x, x_vap = vap(block, x, context, t_mod, freqs, x_vap, context_vap, t_mod_vap, freqs_vap, block_id)
            else:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                elif use_gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
                else:
                    x = block(x, context, t_mod, freqs) #vace
            
            # VACE
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                    current_vace_hint = torch.nn.functional.pad(current_vace_hint, (0, 0, 0, chunks[0].shape[1] - current_vace_hint.shape[1]), value=0)
                x = x + current_vace_hint * vace_scale
            
            # Animate
            if pose_latents is not None and face_pixel_values is not None:
                x = animate_adapter.after_transformer_block(block_id, x, motion_vec)
        if tea_cache is not None:
            tea_cache.store(x)
            
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))
    return x


def model_fn_wans2v(
    dit,
    latents,
    timestep,
    context,
    audio_embeds,
    motion_latents,
    s2v_pose_latents,
    drop_motion_frames=True,
    use_gradient_checkpointing_offload=False,
    use_gradient_checkpointing=False,
    use_unified_sequence_parallel=False,
):
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    origin_ref_latents = latents[:, :, 0:1]
    x = latents[:, :, 1:]

    # context embedding
    context = dit.text_embedding(context) # # [1, 512, 4096]->[1, 512, 5120] # （5120 是 DiT 的隐藏维度）

    # audio encode
    audio_emb_global, merged_audio_emb = dit.cal_audio_emb(audio_embeds) # [1, 14, 5, 5120]

    # x and s2v_pose_latents
    # s2v_pose_latents = torch.zeros_like(x) if s2v_pose_latents is None else s2v_pose_latents
    # Patchify: 将视频像素块变成 Token 序列
    # x, (f, h, w) = dit.patchify(dit.patch_embedding(x) + dit.cond_encoder(s2v_pose_latents))  #[1, 16800, 5120] # 
    x, (f, h, w) = dit.patchify(dit.patch_embedding(x))
    seq_len_x = seq_len_x_global = x.shape[1] # global used for unified sequence parallel 16800

    # reference image
    ref_latents, (rf, rh, rw) = dit.patchify(dit.patch_embedding(origin_ref_latents)) # [1, 1200, 5120]
    grid_sizes = dit.get_grid_sizes((f, h, w), (rf, rh, rw)) # todo
    x = torch.cat([x, ref_latents], dim=1)
    # mask
    mask = torch.cat([torch.zeros([1, seq_len_x]), torch.ones([1, ref_latents.shape[1]])], dim=1).to(torch.long).to(x.device)
    # freqs ROPE:计算 3D 位置信息 (时间、高、宽)
    pre_compute_freqs = rope_precompute(x.detach().view(1, x.size(1), dit.num_heads, dit.dim // dit.num_heads), grid_sizes, dit.freqs, start=None) # (1, 18000, 40, 64)
    # motion
    # x, pre_compute_freqs, mask = dit.inject_motion(x, pre_compute_freqs, mask, motion_latents, drop_motion_frames=drop_motion_frames, add_last_motion=2)

    x = x + dit.trainable_cond_mask(mask).to(x.dtype)  # 加上一个可训练的 Mask Embedding，让模型区分“生成区”和“参考区” #todo

    # tmod
    timestep = torch.cat([timestep, torch.zeros([1], dtype=timestep.dtype, device=timestep.device)]) # torch.Size([2]):[528., 0.]拼接两个时间步：实际时间步 + 0（给参考帧用）
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep)) # 时间嵌入dit.freq_dim=256;(2)->(2,256)->(2,5120)
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim)).unsqueeze(2).transpose(0, 2) # # 投影成调制参数 ([1, 6, 2, 5120]);(2,5120)->(2,30720)->(2,6,5120)->(2,6,1,5120)->...

    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        world_size, sp_rank = get_sequence_parallel_world_size(), get_sequence_parallel_rank()
        assert x.shape[1] % world_size == 0, f"the dimension after chunk must be divisible by world size, but got {x.shape[1]} and {get_sequence_parallel_world_size()}"
        x = torch.chunk(x, world_size, dim=1)[sp_rank] # # 只拿自己那一份
        seg_idxs = [0] + list(torch.cumsum(torch.tensor([x.shape[1]] * world_size), dim=0).cpu().numpy())
        seq_len_x_list = [min(max(0, seq_len_x - seg_idxs[i]), x.shape[1]) for i in range(len(seg_idxs)-1)]
        seq_len_x = seq_len_x_list[sp_rank]

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block_id, block in enumerate(dit.blocks): #这部分的off-load比较浪费时间:30s
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, seq_len_x, pre_compute_freqs[0],
                    use_reentrant=False,
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(lambda x: dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                    x,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing: 
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, seq_len_x, pre_compute_freqs[0],
                use_reentrant=False,
            )
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(lambda x: dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                x,
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, seq_len_x, pre_compute_freqs[0])
            x = dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x_global, use_unified_sequence_parallel)

    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        x = get_sp_group().all_gather(x, dim=1) # 如果用了多卡并行，这里把数据收回来拼好

    x = x[:, :seq_len_x_global]
    x = dit.head(x, t[:-1]) # torch.Size([1, 16800, 64]) head 是一个全连接层,把 Transformer 那个很厚（1536维）的抽象特征，投影回我们需要的像素特征维度
    x = dit.unpatchify(x, (f, h, w)) # torch.Size([1, 16, 14, 80, 60])
    # make compatible with wan video, # 把原始的首帧 (origin_ref_latents) 拼回去，组成完整的视频流返回
    x = torch.cat([origin_ref_latents, x], dim=2)
    return x # [1, 16, 15, 80, 60]
