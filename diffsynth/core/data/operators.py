import torch, torchvision, imageio, os
import imageio.v3 as iio
from PIL import Image
import numpy as np
from decord import VideoReader
import ffmpeg
import subprocess
import imageio
import io
import os
import cv2
import json
import torch,torchaudio
from moviepy.editor import VideoFileClip, AudioFileClip
import pandas as pd
import soundfile as sf
import random
from decord import VideoReader, cpu
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms


class DataProcessingPipeline:
    '''
    存储多个操作符，按顺序执行
    '''
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    # 如果管道里任何一个 Operator 需要 full_data，整个管道就需要
    @property
    def needs_full_data(self):
        for op in self.operators:
            if hasattr(op, "needs_full_data") and op.needs_full_data:
                return True
        return False
    
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)


class DataProcessingOperator:
    '''
    核心基类
    '''
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)


class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data


class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)


class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)


class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)


class LoadImage(DataProcessingOperator):
    '''
    输入: "/path/to/image.jpg" (字符串路径)
    输出: PIL.Image 对象 (RGB模式)
    '''
    def __init__(self, convert_RGB=True, convert_RGBA=False):
        self.convert_RGB = convert_RGB
        self.convert_RGBA = convert_RGBA
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        if self.convert_RGBA: image = image.convert("RGBA")
        return image


# class ImageCropAndResize(DataProcessingOperator):
#     '''
#     height 和 width 指定	直接缩放到指定尺寸
#     只指定 max_pixels	按比例缩放，保证总像素数不超过限制
#     '''
#     def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1):
#         self.height = height
#         self.width = width
#         self.max_pixels = max_pixels
#         self.height_division_factor = height_division_factor
#         self.width_division_factor = width_division_factor

#     def crop_and_resize(self, image, target_height, target_width):
#         '''
#         crop_and_resize 逻辑: 原图 1920x1080 → 目标 512x512
#             1. 计算缩放比例: scale = max(512/1920, 512/1080) = 0.474
#             2. 缩放: 910x512
#             3. 中心裁剪: 512x512
#         '''
#         width, height = image.size
#         scale = max(target_width / width, target_height / height)
#         image = torchvision.transforms.functional.resize(
#             image,
#             (round(height*scale), round(width*scale)),
#             interpolation=torchvision.transforms.InterpolationMode.BILINEAR
#         )
#         image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
#         return image
    
#     def get_height_width(self, image):
#         if self.height is None or self.width is None:
#             width, height = image.size
#             if width * height > self.max_pixels:
#                 scale = (width * height / self.max_pixels) ** 0.5
#                 height, width = int(height / scale), int(width / scale)
#             height = height // self.height_division_factor * self.height_division_factor
#             width = width // self.width_division_factor * self.width_division_factor
#         else:
#             height, width = self.height, self.width
#         return height, width
    
#     def __call__(self, data: Image.Image):
#         image = self.crop_and_resize(data, *self.get_height_width(data))
#         return image


class ImageCropAndResize(DataProcessingOperator):
    '''
    集成功能：
    1. 接收 (T, C, H, W) 的 uint8 Tensor (等价于原来逻辑的 List[PIL])
    2. 批量 Resize & Crop
    3. 集成了 preprocess_image 的数学逻辑和 pattern 变换
    '''
    def __init__(self, height=None, width=None, max_pixels=None, 
                 height_division_factor=1, width_division_factor=1,
                 min_value=-1, max_value=1, pattern="B C T H W"): 
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        
        #  preprocess_image/video 的参数
        self.min_value = min_value
        self.max_value = max_value
        self.pattern = pattern

    def get_target_size(self, h, w):
        if self.height is None or self.width is None:
            if w * h > self.max_pixels:
                scale = (w * h / self.max_pixels) ** 0.5
                target_h, target_w = int(h / scale), int(w / scale)
            else:
                target_h, target_w = h, w
            target_h = target_h // self.height_division_factor * self.height_division_factor
            target_w = target_w // self.width_division_factor * self.width_division_factor
        else:
            target_h, target_w = self.height, self.width
        return target_h, target_w

    def __call__(self, video_tensor):
        """
        Args:
            video_tensor: (T, C, H, W) uint8 Tensor
        Returns:
            video_tensor: Float Tensor
        """
        # --- 1. Resize & Crop (保持原有几何变换) ---
        _, _, h, w = video_tensor.shape
        target_h, target_w = self.get_target_size(h, w)
        scale = max(target_w / w, target_h / h)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        
        video_tensor = TF.resize(
            video_tensor, 
            [new_h, new_w], 
            interpolation=transforms.InterpolationMode.BILINEAR, 
            antialias=True
        )
        video_tensor = TF.center_crop(video_tensor, [target_h, target_w])

        # --- 2. 严格复现 preprocess_video/image 的逻辑 ---
        
        # 对应: image = torch.Tensor(np.array(image, dtype=np.float32))
        # 对应: image = image.to(dtype=torch_dtype...)
        video_tensor = video_tensor.to(dtype=torch.float32)

        # 对应: image = image * ((max_value - min_value) / 255) + min_value
        # 这一步数学逻辑完全一致
        video_tensor = video_tensor * ((self.max_value - self.min_value) / 255.0) + self.min_value

        # 对应: pattern="B C T H W", dim=pattern.index("T") // 2
        # pattern="B C T H W" -> T index=2 -> dim = 2 // 2 = 1
        # 原逻辑: 单帧为 (1, C, H, W), 在 dim 1 堆叠 T 个 -> (1, T, C, H, W)
        
        # 当前 video_tensor 形状为 (T, C, H, W)
        # permute(1, 0, 2, 3) 变成 (C, T, H, W)
        video_tensor = video_tensor.permute(1, 0, 2, 3)

        # 只需要在第 0 维增加 Batch 维度，即可得到 (1, C, T, H, W)
        # video_tensor = video_tensor.unsqueeze(0)

        return video_tensor


class ToList(DataProcessingOperator):
    '''
    把单个图片包装成列表，统一视频和图片的输出格式
    '''
    def __call__(self, data):
        return [data]
    

# class LoadVideo(DataProcessingOperator):
#     '''
#     num_frames % time_division_factor == time_division_remainder
#     video.mp4
#         │
#         ▼
#     ┌─────────────────────────────┐
#     │ 1. 打开视频                  │
#     │ 2. 计算实际帧数              │
#     │    (不超过视频长度，满足除法条件) │
#     │ 3. 逐帧读取                  │
#     │ 4. 每帧经过 frame_processor  │
#     │ 5. 收集成列表                │
#     └─────────────────────────────┘
#         │
#         ▼
#     [frame1, frame2, ..., frame81]  # List[PIL.Image]
#     '''
#     def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
#         self.num_frames = num_frames
#         self.time_division_factor = time_division_factor
#         self.time_division_remainder = time_division_remainder
#         # frame_processor is build in the video loader for high efficiency.
#         self.frame_processor = frame_processor
        
#     def get_num_frames(self, reader):
#         num_frames = self.num_frames
#         if int(reader.count_frames()) < num_frames:
#             num_frames = int(reader.count_frames())
#             while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
#                 num_frames -= 1
#         return num_frames
        
#     def __call__(self, data: str):
#         reader = imageio.get_reader(data)
#         num_frames = self.get_num_frames(reader)
#         frames = []
#         for frame_id in range(num_frames):
#             frame = reader.get_data(frame_id)
#             frame = Image.fromarray(frame)
#             frame = self.frame_processor(frame)
#             frames.append(frame)
#         reader.close()
#         return frames

class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x, tgt_fps=15.0):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        self.frame_processor = frame_processor
        self.tgt_fps = tgt_fps 

    def run_ffprobe_subprocess(self, video_path):
        command = ['ffprobe', '-v', 'verbose', '-show_streams', video_path]
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
            # 返回的结果是一个 CompletedProcess 对象，我们主要关注 stdout
            result_ffprobe = result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e.stderr}")
            result_ffprobe = ''

        result_ffprobe = result_ffprobe.strip().split('\n')
        duration = -1
        for x in result_ffprobe:
            if 'duration=' in x:
                duration = float(x.split('=')[1])
        return duration

    def __call__(self, input_video_pth: str):
        if True:
            videoreader = VideoReader(input_video_pth)
            duration_ffprobe = self.run_ffprobe_subprocess(input_video_pth)
            fps = float(videoreader.get_avg_fps())
            
            # 原始视频的总帧数（基于 ffprobe 修正）
            n_raw = min(len(videoreader), int(duration_ffprobe*fps))

            # 计算在目标帧率 tgt_fps 下，整个视频原本会有多少帧
            # total_tgt_frames = 原始总帧数 / 原始帧率 * 目标帧率
            total_tgt_frames = int(n_raw / fps * self.tgt_fps)

            # 确定实际需要截取的长度
            # 如果指定了 num_frames，则取 min(total, num_frames)，否则取全部
            if self.num_frames is not None:
                actual_n = (min(total_tgt_frames, self.num_frames) - 1) // 4 * 4 + 1
            else:
                actual_n = (total_tgt_frames - 1) // 4 * 4 + 1

            # 确定切片的起始位置 (在 tgt_fps 的时间轴上)
            # 如果视频足够长，随机选一个起点；否则从 0 开始
            if total_tgt_frames > actual_n:
                start_idx = random.randint(0, total_tgt_frames - actual_n)
            else:
                start_idx = 0

            # 生成全量的索引映射，然后只取我们需要的切片部分
            # 1. 生成从 0 到 n_raw-1 的全量均匀映射 (长度为 total_tgt_frames)
            # print(f"----------actual_n: {actual_n}------------")
            all_indexes = np.linspace(0, n_raw-1, total_tgt_frames, dtype=int)
            
            # 2. 根据随机起点和长度，截取需要的索引
            # 这样能保证取出的帧在时间上是连续且对齐的
            frame_indexes = all_indexes[start_idx : start_idx + actual_n].tolist()

            #---------------------------------------------------#
            # batch_size = 32
            # input_img_list = []
            # for i in range(0, len(frame_indexes), batch_size):
            #     batch = frame_indexes[i:i+batch_size]
            #     frames = videoreader.get_batch(batch)
            #     input_img_list.append(frames.asnumpy())
            
            # input_img_list = np.concatenate(input_img_list)
            #---------------------------------------------------#
            input_img_list_ori = videoreader.get_batch(frame_indexes).asnumpy() # 可优化上述框中内容 (T,496,1024,C)
            
            del videoreader

            video_tensor = torch.from_numpy(input_img_list_ori).permute(0, 3, 1, 2).contiguous()
            if self.frame_processor:
                processed_tensor = self.frame_processor(video_tensor)
                
                # 步骤 A: 去掉 Batch 维度 (1, C, T, H, W) -> (C, T, H, W)
                # vis_tensor = processed_tensor.squeeze(0)
                
                # 步骤 B: 逆运算数学公式
                # 原公式: val = pixel * ((max - min) / 255) + min
                # 逆公式: pixel = (val - min) / ((max - min) / 255)
                min_v = self.frame_processor.min_value
                max_v = self.frame_processor.max_value
                
                # 避免除以零
                scale = (max_v - min_v) / 255.0
                if abs(scale) < 1e-6: scale = 1.0 
                
                vis_tensor = (processed_tensor - min_v) / scale
                
                # 步骤 C: 截断并转回 uint8
                vis_tensor = torch.clamp(vis_tensor, 0, 255).byte()
                
                # 步骤 D: Permute (C, T, H, W) -> (T, H, W, C) 并转 numpy
                input_img_list = vis_tensor.permute(1, 2, 3, 0).numpy() # (57, H=640, W=480, 3)
                
            else:
                # Fallback 如果没有 processor
                processed_tensor = video_tensor
                input_img_list = video_tensor.permute(0, 2, 3, 1).numpy() # 对比是否相等

            # pil_frames = []
            # for i in range(len(input_img_list)):
            #     # ndarray -> PIL
            #     img = Image.fromarray(input_img_list[i])
                
            #     # 应用后续处理器 (resize/crop)
            #     if self.frame_processor:
            #         img = self.frame_processor(img)
            #     pil_frames.append(img)

            # === [Debug Start] ===
            # print(f"[Debug: LoadVideo] processed_tensor shape: {processed_tensor.shape}") #(C,T,H,W)
            # print(f"[Debug: LoadVideo] processed_tensor dtype: {processed_tensor.dtype}") #fp32
            # print(f"{input_img_list_ori.shape}")

            # === [Debug End] ===

            return {
                "frames": processed_tensor,  # torch.Size([3, 57, 640, 480])
                "start_idx": start_idx,  # 把算出来的随机起点传出去
                "actual_n": actual_n,
                "input_img_list": input_img_list # #(T,H,W,C)
            }
            # return pil_frames, input_img_list, start_idx, actual_n

        # except Exception as e:
        #     print(f"Error loading video {input_video_pth}: {str(e)}")
        #     return {}

class SequencialProcess(DataProcessingOperator):
    '''
    对列表中的每个元素应用相同的操作符
    '''
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]


class LoadGIF(DataProcessingOperator):
    '''
    与 LoadVideo 类似，专门处理 GIF 文件
    '''
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames


class RouteByExtensionName(DataProcessingOperator):
    '''
    根据文件扩展名选择不同的加载器
    '''
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")

'''
CSV: {'video': 'a.mp4', 'input_audio': 'b.mp3'}
                │
                ▼
┌───────────────────────────────────────────────────────────┐
│                    __getitem__                            │
│                                                           │
│  for key in data_file_keys:                               │
│      │                                                    │
│      ├── key="input_audio" (在 special_operator_map)      │
│      │   │                                                │
│      │   ▼                                                │
│      │   ToAbsolutePath >> LoadAudio                      │
│      │   "b.mp3" → "/data/b.mp3" → numpy array            │
│      │                                                    │
│      └── key="video" (用 main_data_operator)              │
│          │                                                │
│          ▼                                                │
│          RouteByType                                      │
│              │                                            │
│              ▼ (str类型)                                   │
│          ToAbsolutePath >> RouteByExtensionName           │
│              │                                            │
│              ▼ (.mp4扩展名)                                │
│          LoadVideo(frame_processor=ImageCropAndResize)    │
│          "a.mp4" → [frame1, frame2, ..., frame81]         │
└───────────────────────────────────────────────────────────┘
                │
                ▼
输出: {'video': [81个PIL.Image], 'input_audio': numpy array, 'prompt': '...'}

'''
class RouteByType(DataProcessingOperator):
    '''
    根据数据类型选择不同的处理管道
    '''
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")


class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)


class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)

# class ToAbsolutePath(DataProcessingOperator):
#     def __init__(self, base_path=""):
#         self.base_path = base_path
        
#     def __call__(self, data):
#         # 情况 1: 传入的是字典 (Audio 处理流程)
#         if isinstance(data, dict):
#             # 尝试找到 audio_path 并修改为绝对路径
#             # 也可以根据需要添加其他 key
#             target_key = "audio_path"
#             if target_key in data and data[target_key]:
#                 rel_path = data[target_key]
#                 if not os.path.isabs(rel_path):
#                     data[target_key] = os.path.join(self.base_path, rel_path)
#             return data
            
#         # 情况 2: 传入的是字符串 (Video 处理流程)
#         else:
#             return os.path.join(self.base_path, data)


# class LoadAudio(DataProcessingOperator):
#     '''
#     加载音频，重采样到指定采样率（默认16kHz）
#     '''
#     def __init__(self, sr=16000):
#         self.sr = sr
#     def __call__(self, data: str):
#         import librosa
#         input_audio, sample_rate = librosa.load(data, sr=self.sr)
#         return input_audio

class LoadAudio(DataProcessingOperator):
    '''
    加载音频，支持 .npy, .mp3/wav, .mp4 等格式，并自动对齐视频帧数
    '''
    def __init__(self, num_frames, tgt_fps, sr):
        self.sr = sr
        self.tgt_fps = tgt_fps
        self.video_start_idx = 0
        self.num_frames = num_frames # 截取长度等于训练的总帧数
        self.needs_full_data = True 

    def load_audio_from_audiofile(self, mp3_path, start_idx, end_idx, fps, sample_rate):
        audio, in_sr = torchaudio.load(mp3_path)
        if audio.shape[0] > 1:
            audio = audio[0]

        if in_sr != sample_rate:
            resample_tf = torchaudio.transforms.Resample(in_sr, sample_rate)
            audio = resample_tf(audio)

        audio_start_idx = int(start_idx / fps * sample_rate)
        audio_end_idx = int(end_idx / fps * sample_rate)
        
        if audio_end_idx > audio.shape[-1]:
            audio_end_idx = audio.shape[-1]
            
        audio = audio[audio_start_idx:audio_end_idx+1]
        audio = audio / (1.0 + 1e-6)
        audio = torch.clamp(audio, -1.0, 1.0)

        return audio.unsqueeze(0)  # shape = [1, T]

    def load_audio_from_video(self, video_path, video_start_idx, video_end_idx, fps, sample_rate):
        video = VideoFileClip(video_path)
        audio = video.audio.to_soundarray(fps=sample_rate)
        video.close()

        audio = torch.from_numpy(audio.T).float()
        if audio.ndim > 1:
            audio = audio[0]

        audio_start_idx = int(video_start_idx / fps * sample_rate)
        audio_end_idx = int(video_end_idx / fps * sample_rate)
        audio = audio[audio_start_idx:audio_end_idx + 1]

        audio = audio / (1.0 + 1e-6)
        audio = torch.clamp(audio, -1.0, 1.0)

        return audio.unsqueeze(0)
    
    def __call__(self, data: dict):
        path = data.get("audio_path")        
        # 使用 self 里存好的参数
        video_start_idx = data.get("video_start_idx", 0) 
        video_end_idx = video_start_idx + data.get("actual_n", self.num_frames)
        tgt_fps = self.tgt_fps
        sample_rate = self.sr
        
        path = None if (path == "None" or path is None) else path
        audio = None

        if path is not None and path.endswith('.npy'):
            try:
                audio_resample = np.load(path)
                if audio_resample.ndim == 2:
                    audio_resample = audio_resample[0]
                
                audio_start_idx = int(video_start_idx / tgt_fps * sample_rate)
                audio_end_idx = int(video_end_idx / tgt_fps * sample_rate)
                audio = audio_resample[audio_start_idx:audio_end_idx+1]
                audio = torch.from_numpy(audio).float().unsqueeze(0)
            except Exception as e:
                print(f"Load audio from npy fails: {path}. Error: {e}")
                audio = torch.zeros((1, int(sample_rate/tgt_fps*int(video_end_idx-video_start_idx)))).float()

        elif path is not None and path.split('.')[-1].lower() in {"mp3", "wav", "flac", "ogg", "m4a"}:
            try:
                audio = self.load_audio_from_audiofile(path, video_start_idx, video_end_idx, tgt_fps, sample_rate)
            except Exception as e:
                print(f"Load audio file fails: {e}")
                audio = torch.zeros((1, int(sample_rate/tgt_fps*int(video_end_idx-video_start_idx)))).float()
        
        elif path is not None:
            try:
                audio = self.load_audio_from_video(path, video_start_idx, video_end_idx, tgt_fps, sample_rate)
            except Exception as e:
                print(f"Load audio from video fails: {path}. Error: {e}")
                audio = torch.zeros((1, int(sample_rate/tgt_fps*int(video_end_idx-video_start_idx)))).float()
        else:
            audio = torch.zeros((1, int(sample_rate/tgt_fps*int(video_end_idx-video_start_idx)))).float()

        # 转为 (T,) 的 numpy 数组供 Processor 使用

        if torch.is_tensor(audio):
            return audio.squeeze(0).numpy()
        return audio
    

class LoadPose(DataProcessingOperator):
    def __init__(self) -> None:
        super().__init__()

    def json_to_keypoints_matrix(json_path):
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取instance_info
        instance_info = data['instance_info']
        
        # 检查是否有数据
        if len(instance_info) == 0:
            return None

        num_keypoints = 133
        num_frames = len(instance_info)
        
        # 初始化结果数组 [f, n, 3]
        result_array = np.zeros((num_frames, num_keypoints+1, 3))
        
        # 填充数据
        for frame_idx, frame_data in enumerate(instance_info):
            frame_id = frame_data['frame_id']
            instances = frame_data['instances']
            
            try:
                # 假设每帧只有一个实例
                instance = instances[0]
                keypoints = np.array(instance['keypoints'])  # [n, 2]
                keypoint_scores = np.array(instance['keypoint_scores'])  # [n,]
                # 将关键点坐标和分数组合成 [n, 3]
                result_array[frame_idx, 1:, :2] = keypoints
                result_array[frame_idx, 1:, 2] = keypoint_scores
            except:
                pass
        
        return result_array[:,np.newaxis]

    def __call__(self, pose_path, height, width, ori_fps, tgt_fps, clip_start_idx=0, clip_length=None):
        if '.json' in pose_path:
            dwpose_np = self.json_to_keypoints_matrix(pose_path)
        else:
            dwpose_np = np.load(pose_path) # (Frames, 1, 134, 3)
        
        print(f"Original pose shape: {dwpose_np.shape}") 
        dwpose_np[..., 0] *= width
        dwpose_np[..., 1] *= height

        # 1. 计算重采样后的总长度 (Pose 也要先变换到 tgt_fps 的时间轴)
        n_raw = dwpose_np.shape[0]
        total_tgt_frames = int(n_raw * (tgt_fps / ori_fps))
        
        # 2. 生成全量重采样索引 (from 0 to n_raw-1)
        # 使用 linspace 确保和视频加载时的采样逻辑完全一致
        all_indices = np.linspace(0, n_raw - 1, total_tgt_frames, dtype=int)

        if clip_length is not None:
            # 确保索引不越界
            end_idx = min(clip_start_idx + clip_length, total_tgt_frames)
            # 截取对应的索引段
            selected_indices = all_indices[clip_start_idx : end_idx]
        else:
            selected_indices = all_indices

        dwpose_np = dwpose_np[selected_indices]
        
        print(f"Resampled & Sliced shape: {dwpose_np.shape} (fps: {tgt_fps}, slice: {clip_start_idx}:{clip_start_idx+len(dwpose_np)})")

        return dwpose_np
        