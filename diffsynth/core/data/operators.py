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


class LoadIDGrid(DataProcessingOperator):
    """
    加载视频并生成九宫格ID参考图。
    
    该操作符调用 FaceGrid 类将视频处理成九宫格形式的ID参考，
    用于在I2V训练中注入身份信息。
    
    九宫格布局：
    ┌───┬───┬───┐
    │ 1 │ 2 │ 3 │  不同角度/表情的人脸
    ├───┼───┼───┤
    │ 4 │ 5 │ 6 │  裁剪并增强后的人脸
    ├───┼───┼───┤
    │ 7 │ 8 │ 9 │
    └───┴───┴───┘
    
    输入: 包含 video_path, dwpose_path, fps, ori_fps 等信息的字典
    输出: 九宫格ID参考视频 tensor (C, T, H, W)
    
    关键设计说明：
    - 九宫格的帧数与主视频帧数保持一致，便于后续 token concat
    - 输出值域为 [-1, 1]，与主视频一致
    - 九宫格的尺寸可以与主视频不同（会在 VAE 编码后统一处理）
    """
    def __init__(self, id_grid_num_frames=1, tgt_fps=15.0, height=None, width=None, max_pixels=268800, aug_intensity=1.9):
        """
        Args:
            num_frames: 九宫格视频的独立帧数 (1表示退化为静态九宫格图片)
            tgt_fps: 目标帧率
            height: 九宫格输出高度（固定宽高模式用）
            width: 九宫格输出宽度（固定宽高模式用）
            max_pixels: 九宫格输出等效面积（等效面积模式用）
            aug_intensity: 数据增强强度（控制亮度、对比度、饱和度等变化）
        """
        self.id_grid_num_frames = id_grid_num_frames
        self.tgt_fps = tgt_fps
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.aug_intensity = aug_intensity
        self.needs_full_data = True  # 需要访问完整的 data 字典
        
        # 延迟导入 FaceGrid，避免循环依赖
        self._face_grid = None
    
    @property
    def face_grid(self):
        """延迟初始化 FaceGrid"""
        if self._face_grid is None:
            from diffsynth.core.data.video_downsample import FaceGrid
            self._face_grid = FaceGrid()
        return self._face_grid

    def _get_id_img_list(self, data: dict) -> list:
        """从 data 字典中提取所有以 'id_img_list' 开头的字段，解析为有效路径列表"""
        id_images = []
        for key, val in data.items():
            if str(key).startswith('id_img_list'):
                if pd.isna(val) or val is None or val == "nan":
                    continue
                # 如果是字符串形式的列表 "['path1', 'path2']"，需要解析
                if isinstance(val, str) and val.startswith('['):
                    import ast
                    try:
                        parsed_list = ast.literal_eval(val)
                        id_images.extend(parsed_list)
                    except:
                        pass
                elif isinstance(val, list):
                    id_images.extend(val)
                elif isinstance(val, str) and os.path.exists(val):
                    id_images.append(val)
                    
        # 过滤掉不存在的路径
        valid_id_images = [p for p in id_images if isinstance(p, str) and os.path.exists(p)]
        return valid_id_images
    
    @property
    def smpl_infer(self):
        """延迟初始化 SmplInfer 及所需的补丁，确保只初始化一次"""
        if not hasattr(self, "_smpl_infer") or self._smpl_infer is None:
            import inspect
            import numpy as np
            from collections import namedtuple
            import sys
            import os

            if not hasattr(inspect, "getargspec"):
                ArgSpec = namedtuple(
                    "ArgSpec", ["args", "varargs", "keywords", "defaults"]
                )

                def getargspec(func):
                    spec = inspect.getfullargspec(func)
                    return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

                inspect.getargspec = getargspec
                inspect.ArgSpec = ArgSpec

            if not hasattr(np, "bool"):
                np.bool = bool

            get_smpl_motion_path = os.path.abspath(
                "/m2v_intern/mengzijie/DiffSynth-Studio/get_smpl_motion"
            )
            if get_smpl_motion_path not in sys.path:
                sys.path.append(get_smpl_motion_path)

            from get_smpl_motion.GVHMR.smpl_Infer_service_ljw import SmplInfer

            self._smpl_infer = SmplInfer(
                smpl_checkpoints_path="/ytech_milm/liujiwen/kling_motion_service/smpl_all_checkpoints",
                is_image=True,
            )

        return self._smpl_infer

    def _get_video_path(self, data: dict) -> str:
        """从 data 字典中获取视频路径，支持多种字段名"""
        # 优先使用原始视频路径字段
        for key in ["video_path"]:
            path = data.get(key)
            if path is not None and path != "None" and str(path) != "nan":
                # 如果是 tensor，说明已经被处理过了，需要找原始路径
                if isinstance(path, torch.Tensor):
                    continue
                return str(path)
        return None
    
    def _find_pose_path(self, video_path: str, data: dict) -> str:
        """查找姿态文件路径"""
        # 首先检查 data 中是否有 dwpose_path
        dwpose_path = data.get("dwpose_path")
        if dwpose_path is not None and dwpose_path != "None" and str(dwpose_path) != "nan":
            if os.path.exists(str(dwpose_path)):
                return str(dwpose_path)
        
        # 如果没有，尝试从 video_path 推断
        if video_path is None:
            return None
            
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # 尝试几种常见的 pose 文件路径模式
        possible_paths = [
            os.path.join(video_dir, f"{video_name}_dwpose.npy"),
            os.path.join(video_dir, f"{video_name}_Pose2d_dwpose.npy"),
            os.path.join(video_dir, "pose", f"{video_name}.npy"),
            os.path.join(video_dir, f"{video_name}.json"),
            # 常见的 pose 目录结构
            video_path.replace(".mp4", "_Pose2d_dwpose.npy"),
            video_path.replace(".mp4", ".json"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def load_and_expand_id_images(self, id_image_paths):
        """
        加载并扩充 ID 图片到 9 张。
        使用左右翻转或直接复制。
        """
        import cv2
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
                    np_img = cv2.imdecode(np.frombuffer(src_binary, np.uint8), cv2.IMREAD_COLOR)
                    flipped = cv2.flip(np_img, 1) # 水平翻转增强多样性
                    _, encoded = cv2.imencode('.png', flipped)
                    expanded_binaries.append(encoded.tobytes())
                except:
                    expanded_binaries.append(src_binary) # 失败则原样复制
            else:
                expanded_binaries.append(src_binary)
                
        return expanded_binaries[:9]
    
    def __call__(self, data: dict):
        """
        处理视频或图片列表生成九宫格ID参考。
        """
        import cv2
        
        # 提取 CSV 中的 id_img_list* 作为九宫格的基础
        id_image_paths = self._get_id_img_list(data)
        
        # 获取视频路径 (用作兜底或如果没给独立的 id images)
        video_path = self._get_video_path(data)
        
        # 如果 CSV 中提供了有效的 id 图片列表，则优先使用这些图片构建九宫格！
        input_id_image_list_binary = None
        if len(id_image_paths) > 0:
            input_id_image_list_binary = self.load_and_expand_id_images(id_image_paths)
            video_data = None
            if video_path and os.path.exists(video_path):
                # Optionally pass the video if the face_grid library requires it
                # But typically if input_id_image_list_binary is provided, we don't need the whole video
                pass
        else:
            # 兜底：如果 CSV 里没有合法的 id_img_list，只能退回到首帧提取
            pass

        # 准备传入 face_grid 的数据
        image_data = None
        if video_path and os.path.exists(video_path):
            with open(video_path, "rb") as f:
                image_data = f.read() # 取决于 face_grid 底层怎么用它

        # 确定尺寸
        face_ar = 480 / 560  # Default aspect ratio
        try:
            if video_path and os.path.exists(video_path):
                import imageio
                reader = imageio.get_reader(video_path)
                meta = reader.get_meta_data()
                w, h = meta['size']
                face_ar = w / h
        except:
            pass
            
        area = self.max_pixels
        grid_w_raw = (area * face_ar) ** 0.5
        grid_h_raw = (area / face_ar) ** 0.5
        final_w = int(round(grid_w_raw / 48) * 48) # 对齐到 48 的倍数
        final_h = int(round(grid_h_raw / 48) * 48)
        if final_w == 0: final_w = 48
        if final_h == 0: final_h = 48

        # TODO: 我们需要初始化 get_smpl_motion 库的 SmplInfer (因为原代码用它来做九宫格)
        # 这里统一调用底层的 face_grid 或 smpl_infer.get_face_grid
        # 我们这里复用 smpl_infer.get_face_grid 逻辑
        # 我们假设 self.face_grid 实际上就是封装了扣脸逻辑
        
        import sys
        sys.path.append(os.path.abspath("/m2v_intern/mengzijie/DiffSynth-Studio/get_smpl_motion"))
        try:
            from get_smpl_motion.GVHMR.smpl_Infer_service_ljw import SmplInfer
            smpl_infer = SmplInfer(smpl_checkpoints_path='/ytech_milm/liujiwen/kling_motion_service/smpl_all_checkpoints', is_image=True)
            
            res = smpl_infer.get_face_grid(
                image_data,
                id_video_data=None,
                input_id_image_list_binary=input_id_image_list_binary,
                output_dir="./debug_vis",
                target_size=[final_h, final_w],
                save_path_dir="./debug_vis"
            )
            
            frames = []
            for frame_np in res:
                frame_tensor = torch.from_numpy(frame_np.copy()).float()
                frame_tensor = frame_tensor.permute(2, 0, 1) # (C, H, W)
                frames.append(frame_tensor)
                
            video_tensor = torch.stack(frames, dim=1) # (C, T_raw, H, W)
            
            N = video_tensor.shape[1]
            T = self.id_grid_num_frames
            if N != T:
                if N == 1:
                    video_tensor = video_tensor.repeat(1, T, 1, 1)
                else:
                    indices = np.linspace(0, N - 1, T, dtype=int)
                    video_tensor = video_tensor[:, indices, :, :]
                    
            id_grid_tensor = video_tensor / 127.5 - 1.0 # [-1, 1]
            return id_grid_tensor
            
        except Exception as e:
            print(f"[Bug todo ][LoadIDGrid] Error building ID grid: {e}")
            return torch.zeros((3, self.id_grid_num_frames, final_h, final_w))
        
        # 转换为 tensor: (T, H, W, 3) -> (3, T, H, W)
        # 值域转换: [0, 255] -> [-1, 1]
        id_grid_tensor = torch.from_numpy(id_grid_frames.copy()).float()
        id_grid_tensor = id_grid_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)
        id_grid_tensor = id_grid_tensor / 127.5 - 1.0  # 归一化到 [-1, 1]
        
        return id_grid_tensor


class DebugVisualizer:
    """
    调试可视化工具类，用于保存中间结果。
    
    使用方法:
        visualizer = DebugVisualizer(enabled=args.debug, save_dir="./debug_output")
        visualizer.save_video(video_tensor, "id_grid", data_id=0)
        visualizer.save_image(image_tensor, "first_frame", data_id=0)
    """
    def __init__(self, enabled=False, save_dir="./debug_vis"):
        self.enabled = enabled
        self.save_dir = save_dir
        if enabled:
            os.makedirs(save_dir, exist_ok=True)
    
    def save_video(self, tensor, name, data_id=0, fps=8):
        """
        保存视频 tensor 为 mp4 文件。
        
        Args:
            tensor: (C, T, H, W) 或 (T, H, W, C) 格式的 tensor，值域 [-1, 1] 或 [0, 255]
            name: 文件名前缀
            data_id: 数据ID，用于区分不同样本
            fps: 帧率
        """
        if not self.enabled:
            return
        
        try:
            import imageio
            
            # 确保是 numpy array
            if torch.is_tensor(tensor):
                tensor = tensor.detach().cpu()
            
            # 处理维度
            if tensor.ndim == 4:
                if tensor.shape[0] == 3:  # (C, T, H, W)
                    tensor = tensor.permute(1, 2, 3, 0)  # -> (T, H, W, C)
                # else: 已经是 (T, H, W, C)
            
            tensor = tensor.numpy() if torch.is_tensor(tensor) else tensor
            
            # 处理值域
            if tensor.min() < 0:  # [-1, 1] -> [0, 255]
                tensor = (tensor + 1.0) / 2.0 * 255.0
            elif tensor.max() <= 1.0:  # [0, 1] -> [0, 255]
                tensor = tensor * 255.0
            
            tensor = np.clip(tensor, 0, 255).astype(np.uint8)
            tensor = np.ascontiguousarray(tensor)
            
            # 保存
            save_path = os.path.join(self.save_dir, f"{name}_{data_id:04d}.mp4")
            imageio.mimsave(save_path, tensor, fps=fps, codec='libx264', macro_block_size=1)
            print(f"[DebugVis] Saved video: {save_path}")
            
        except Exception as e:
            print(f"[DebugVis] Failed to save video {name}: {e}")
    
    def save_image(self, tensor, name, data_id=0):
        """
        保存图像 tensor 为 jpg 文件。
        
        Args:
            tensor: (C, H, W) 或 (H, W, C) 格式的 tensor
            name: 文件名前缀
            data_id: 数据ID
        """
        if not self.enabled:
            return
        
        try:
            from PIL import Image
            
            if torch.is_tensor(tensor):
                tensor = tensor.detach().cpu()
            
            # 处理维度
            if tensor.ndim == 3:
                if tensor.shape[0] == 3:  # (C, H, W)
                    tensor = tensor.permute(1, 2, 0)  # -> (H, W, C)
            
            tensor = tensor.numpy() if torch.is_tensor(tensor) else tensor
            
            # 处理值域
            if tensor.min() < 0:
                tensor = (tensor + 1.0) / 2.0 * 255.0
            elif tensor.max() <= 1.0:
                tensor = tensor * 255.0
            
            tensor = np.clip(tensor, 0, 255).astype(np.uint8)
            
            # 保存
            save_path = os.path.join(self.save_dir, f"{name}_{data_id:04d}.jpg")
            Image.fromarray(tensor).save(save_path)
            print(f"[DebugVis] Saved image: {save_path}")
            
        except Exception as e:
            print(f"[DebugVis] Failed to save image {name}: {e}")
    
    def save_grid_comparison(self, id_grid, first_frame, name, data_id=0):
        """
        保存九宫格和首帧的对比图。
        
        Args:
            id_grid: 九宫格 tensor (C, T, H, W)
            first_frame: 首帧 tensor (C, H, W)
            name: 文件名前缀
            data_id: 数据ID
        """
        if not self.enabled:
            return
        
        try:
            from PIL import Image
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # 处理九宫格 (取第一帧)
            if torch.is_tensor(id_grid):
                id_grid = id_grid.detach().cpu()
            if id_grid.ndim == 4 and id_grid.shape[0] == 3:
                id_grid_frame = id_grid[:, 0].permute(1, 2, 0).numpy()
            else:
                id_grid_frame = id_grid[0].numpy() if id_grid.ndim == 4 else id_grid.numpy()
            
            if id_grid_frame.min() < 0:
                id_grid_frame = (id_grid_frame + 1.0) / 2.0
            
            axes[0].imshow(np.clip(id_grid_frame, 0, 1))
            axes[0].set_title("ID Grid (Frame 0)")
            axes[0].axis('off')
            
            # 处理首帧
            if torch.is_tensor(first_frame):
                first_frame = first_frame.detach().cpu()
            if first_frame.ndim == 3 and first_frame.shape[0] == 3:
                first_frame = first_frame.permute(1, 2, 0).numpy()
            else:
                first_frame = first_frame.numpy()
            
            if first_frame.min() < 0:
                first_frame = (first_frame + 1.0) / 2.0
            
            axes[1].imshow(np.clip(first_frame, 0, 1))
            axes[1].set_title("First Frame (I2V Input)")
            axes[1].axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(self.save_dir, f"{name}_{data_id:04d}.jpg")
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"[DebugVis] Saved comparison: {save_path}")
            
        except Exception as e:
            print(f"[DebugVis] Failed to save comparison {name}: {e}")