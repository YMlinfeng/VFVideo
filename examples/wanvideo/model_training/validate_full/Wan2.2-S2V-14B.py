import torch
from PIL import Image
import librosa
from diffsynth.utils.data import VideoData, save_video_with_audio
from diffsynth.core import load_state_dict
from safetensors.torch import load_file as load_safetensors
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
import datetime
import os
# print("MODELSCOPE_CACHE:", os.environ.get("MODELSCOPE_CACHE"))

# print("start downloading...")
# pipe = WanVideoPipeline.from_pretrained(
#     torch_dtype=torch.bfloat16,
#     device="cuda",
#     model_configs=[
#         ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
#         ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/model.safetensors"),
#         ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
#         ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="Wan2.1_VAE.pth"),
#     ],
#     audio_processor_config=ModelConfig(model_id="Wan-AI/Wan2.2-S2V-14B", origin_file_pattern="wav2vec2-large-xlsr-53-english/"),
# )
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        # 第一个ModelConfig：应该是4个safetensors文件作为一个列表
        ModelConfig(
            path=[
                "/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/diffusion_pytorch_model-00001-of-00004.safetensors",
                "/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/diffusion_pytorch_model-00002-of-00004.safetensors",
                "/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/diffusion_pytorch_model-00003-of-00004.safetensors",
                "/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/diffusion_pytorch_model-00004-of-00004.safetensors"
            ]
        ),
        # 第二个ModelConfig：wav2vec2模型
        ModelConfig(
            path="/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/model.safetensors"
        ),
        # 第三个ModelConfig：T5模型
        ModelConfig(
            path="/m2v_intern/mengzijie/DiffSynth-Studio/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors"
        ),
        # 第四个ModelConfig：VAE模型
        ModelConfig(
            path="/m2v_intern/mengzijie/DiffSynth-Studio/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors"
        ),
    ],
    audio_processor_config=ModelConfig(
        path="/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english"
    ),
    tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"), #todo
)
# num_frames = 81 # 4n+1
# height = 448
# width = 832

# num_frames = 121 # 4n+1
# height = 224
# width = 448

num_frames = 81 # 4n+1
height = 368
width = 688

# num_frames = 81 # 4n+1
# height = 640
# width = 480

# num_frames = 57 # 4n+1
# height = 256
# width = 192

#原始

# ckpt_path = "/m2v_intern/mengzijie/DiffSynth-Studio/models/train/Wan2.2-S2V-14B_T16/initial.safetensors"
# ckpt_path = "/m2v_intern/mengzijie/DiffSynth-Studio/models/train/114/initial.safetensors" # 全去原始
# ckpt_path = "/m2v_intern/mengzijie/DiffSynth-Studio/models/train/initial_ori.safetensors" # 官方原始
# ckpt_path = "/m2v_intern/mengzijie/DiffSynth-Studio/models/train/单卡/step-800.safetensors" # 全去走800步
ckpt_path = "/m2v_intern/mengzijie/DiffSynth-Studio/models/train/Wan2.2-S2V-14B_32gpu/step-1750.safetensors" # 全去走800步
state_dict = load_state_dict(ckpt_path)
# state_dict = load_state_dict("/m2v_intern/mengzijie/DiffSynth-Studio/models/train/Wan2.2-S2V-14B_T16/step-1410.safetensors")

missing, unexpected = pipe.dit.load_state_dict(state_dict, strict=True)




# prompt = "a person is singing"
prompt = "FPS-30. The video plays at normal speed. A person speaks directly to the camera in a well-lit space, delivering content in a clear, steady tone that suggests a speech or casual address. The individual’s appearance details are not specified, with the focus entirely on their verbal expression. The background is not clearly defined, presenting a simple and uncluttered setting that does not distract from the speaker. The person maintains a relatively upright posture while talking, occasionally making slight head or hand gestures that match the rhythm of their speech to emphasize key points. realistic, appearing as a short clip, possibly from a personal talk, a class presentation or a casual online sharing video, with natural lighting that highlights the subject. The scene is characterized by medium saturation, moderate contrast, moderate brightness and neutral-toned colors. The camera is fixed, capturing a medium shot of the person, with the lens at eye level. The speaker is positioned in the center of the frame. The camera remains stationary, with a medium depth of field, keeping the person in sharp focus while the background has a subtle blur, emphasizing the verbal delivery of the subject."
negative_prompt = "画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
# input_image = Image.open("/m2v_intern/mengzijie/data/example_video_dataset/wans2v/pose.png").convert("RGB").resize((width, height))
image_path = "/m2v_intern/mengzijie/data/example_video_dataset/wans2v/pose.png"

from PIL import ImageOps
input_image = Image.open(image_path).convert("RGB")
input_image = ImageOps.fit(input_image, (width, height), Image.LANCZOS)

audio_path = '/m2v_intern/mengzijie/data/example_video_dataset/wans2v/sing.MP3'
# audio_path = '/m2v_intern/mengzijie/DiffSynth-Studio/debug_vis_output_debug/sample_0001_start3_temp_audio.wav'
input_audio, sample_rate = librosa.load(audio_path, sr=16000) # 16kHz sampling rate
# pose_video_path = '/m2v_intern/mengzijie/data/example_video_dataset/wans2v/pose.mp4'
# pose_video = VideoData(pose_video_path, height=height, width=width)


# if os.environ.get("LOCAL_RANK", "0") == "0":
#     import debugpy
#     debugpy.listen(("0.0.0.0", 5678))
#     print("=" * 50)
#     print("Waiting for debugger to attach on port 5678...")
#     print("=" * 50)
#     debugpy.wait_for_client()  
#     print("Debugger attached! Continuing...")


# Speech-to-video with pose
video = pipe(
    prompt=prompt,
    input_image=input_image,
    negative_prompt=negative_prompt,
    seed=0,
    num_frames=num_frames,
    height=height,
    width=width,
    audio_sample_rate=sample_rate,
    input_audio=input_audio,
    # s2v_pose_video=pose_video,
    # num_inference_steps=40,
    num_inference_steps=40
)
# save_video_with_audio(video[1:], "debugoutput/ori.mp4", audio_path, fps=16, quality=5)

# Save result logic
save_dir = os.path.join("debugoutput", datetime.datetime.now().strftime("%H%M_%d%m"))
os.makedirs(save_dir, exist_ok=True)

with open(os.path.join(save_dir, "config.txt"), "w", encoding="utf-8") as f:
    # f.write(f"ckpt_path: {ckpt_path}\n")
    f.write(f"num_frames: {num_frames}\n")
    f.write(f"height: {height}\n")
    f.write(f"width: {width}\n")
    f.write(f"prompt: {prompt}\n")
    f.write(f"negative_prompt: {negative_prompt}\n")
    f.write(f"input_image_path: {image_path}\n")
    f.write(f"audio_path: {audio_path}\n")
    # f.write(f"pose_video_path: {pose_video_path}\n")

print(f"Saved config to {save_dir}")

save_video_with_audio(video[1:], os.path.join(save_dir, "output.mp4"), audio_path, fps=16, quality=5)
print(f"Saved video to {save_dir}")


