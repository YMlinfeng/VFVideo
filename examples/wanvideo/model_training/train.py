import torch, os, argparse, accelerate, warnings
import numpy as np
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import numpy as np
import torchaudio
from moviepy.editor import VideoFileClip 
from diffsynth.core.data.operators import DataProcessingOperator
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
# plt.imsave("output0.png", F.to_pil_image(input_image))
import time
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DistributedType
import torch.distributed as dist

os.environ['http_proxy'] = 'http://oversea-squid1.jp.txyun:11080'
os.environ['https_proxy'] = 'http://oversea-squid1.jp.txyun:11080'
os.environ['no_proxy'] = 'localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com'
if "OMPI_COMM_WORLD_RANK" in os.environ:
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
    os.environ["LOCAL_RANK"] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]

# torch.autograd.set_detect_anomaly(True)

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

        
class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        if not use_gradient_checkpointing:
            use_gradient_checkpointing = True
        
        # Load models
        # 四个路径
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(path=tokenizer_path)
        audio_processor_config = ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/") if audio_processor_path is None else ModelConfig(path=audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)
        
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def debug_parameters(self):
        print("=== Debugging Parameters ===")
        for name, param in self.named_parameters():
            print(f"{name}: requires_grad={param.requires_grad}, grad_fn={param.grad_fn}")
        print("=== End Debug ===")
        

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video_path"][:,0] if isinstance(data["video_path"], torch.Tensor) else data["video_path"][0]
            # elif extra_input == "end_image":
            #     inputs_shared["end_image"] = data["video_path"][:,-1] if isinstance(data["video_path"], torch.Tensor) else data["video_path"][-1]
            elif extra_input == "input_audio":
                # 修改点：显式映射，把 data["audio_path"] 给 inputs_shared["input_audio"]
                inputs_shared["input_audio"] = data["audio_path"]
            else:
                warnings.warn(f"出现不明extra_input:{extra_input},请立即查看parse_extra_inputs函数")
                inputs_shared[extra_input] = data.get(extra_input)
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        # 在每次训练迭代前被调用，负责把 Dataset 读出来的原始字典，转换成模型 forward 接受的参数
        inputs_posi = {"prompt": data["target_video_caption"]}
        inputs_nega = {}
        inputs_shared = {
            # 修改点：映射 video
            "input_video": data["video_path"], #fp32
            # "height": data["video_path"][0].size[1],
            # "width": data["video_path"][0].size[0],
            "height": data["video_path"][0].shape[-2] if isinstance(data["video_path"][0], torch.Tensor) else data["video_path"][0].size[1],
            "width": data["video_path"][0].shape[-1] if isinstance(data["video_path"][0], torch.Tensor) else data["video_path"][0].size[0],
            "num_frames": len(data["video_path"][0]) if isinstance(data["video_path"][0], torch.Tensor) else len(data["video_path"]),
            # ... 以下参数保持不变 ...
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        # visualize_tensor(data["video_path"], save_prefix="./debug_vis/test_data")
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        # visualize_tensor(inputs[0]["input_video"], save_prefix="./debug_vis/test_input")
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)  #fp32->bf16
        
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        # 根据任务类型计算损失
        loss = self.task_to_loss[self.task](self.pipe, *inputs)

        # 调试: 检查 loss 是否有 grad_fn
        # print(f"Loss: {loss}, grad_fn: {loss.grad_fn if hasattr(loss, 'grad_fn') else 'None'}")
        print(f"Loss: {loss}")
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--offload_optimizer_device", type=str, default="cpu", help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--debug", default=False, action="store_true", help="Whether to debug")
    return parser


if __name__ == "__main__":
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(f"RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}, LOCAL_RANK={os.environ.get('LOCAL_RANK')}")
        print(f"OMPI_COMM_WORLD_RANK={os.environ.get('OMPI_COMM_WORLD_RANK')}")
        # import debugpy
        # debugpy.listen(("0.0.0.0", 5678))
        # print("=" * 50)
        # print("Waiting for debugger to attach on port 5678...")
        # print("=" * 50)
        # debugpy.wait_for_client()  
        # print("Debugger attached! Continuing...")
    parser = wan_parser()
    args = parser.parse_args()
    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        offload_optimizer_device=args.offload_optimizer_device, 
        offload_param_device="none",     
        zero3_init_flag=False,
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        deepspeed_plugin=deepspeed_plugin, 
        # kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)]
    )
    # if accelerator.distributed_type != DistributedType.DEEPSPEED:
    #     raise RuntimeError(
    #         f"Fatal error: DeepSpeed not enabled, Current mode: {accelerator.distributed_type}\n"
    #     )
    # if dist.is_initialized():
    #     print(f"\n[DEBUG] Rank {dist.get_rank()}: Distributed Backend is: {dist.get_backend()}")
    #     print(f"[DEBUG] Rank {dist.get_rank()}: Current Device: {torch.cuda.current_device()}")
    #     if dist.get_backend() != 'nccl':
    #         raise ValueError("Error: Backend is NOT NCCL!")
    # else:
    #     print("\n[DEBUG] Distributed not initialized yet!")

    accelerator.wait_for_everyone() 
    # print("Prepare the dataset...")
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        num_frames=args.num_frames,
        tgt_fps=args.tgt_fps,
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            # "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            # "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            # "audio_path": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(
            #     num_frames=args.num_frames, 
            #     tgt_fps=args.tgt_fps, 
            #     sr=16000
            # ),
            "audio_path": LoadAudio(
                num_frames=args.num_frames, 
                tgt_fps=args.tgt_fps, 
                sr=16000
            ),
        }
    )
    # print("Load Model...")
    # local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # if local_rank > 0:
    #     # 非主进程稍微等一下，让 rank 0 把文件句柄先拿住
    #     time.sleep(2 * local_rank) 
    # if accelerator.is_main_process:
    accelerator.wait_for_everyone()  # <--- 加这句，确保大家都活过来了
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    # else:
    #     model = None
    # # 等待 Rank 0 加载完成
    accelerator.wait_for_everyone()
    # model.debug_parameters()
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    # print("Start Launching...")
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)