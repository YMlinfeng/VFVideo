import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-720P", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-720P", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-720P", origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-720P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ],
)
# state_dict = load_state_dict("/ytech_m2v4_hdd/mengzijie/DiffSynth-Studio/models/train/i2v_v1.0/step-1400.safetensors")
# pipe.dit.load_state_dict(state_dict)

input_image = VideoData("/ytech_m2v2_hdd/livephoto-body/EMO_data/emo_avspeech_split_121f_250919/livephoto-body/EMO_data/自采data2_with_audio/list/bu/001043/007/video/video_output_30fps_split0.mp4", height=560, width=480)[0]

video = pipe(
    prompt="from sunset to night, a small town, light, house, river",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    input_image=input_image,
    height=560, width=480, num_frames=45,
    seed=1, 
    # tiled=True
)
save_video(video, "i2v-0step.mp4", fps=15, quality=5)
