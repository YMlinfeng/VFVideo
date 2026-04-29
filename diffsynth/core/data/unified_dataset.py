import operator
from .operators import *
import torch, json, pandas
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
import torch
import numpy as np
import torchaudio
from moviepy.editor import VideoFileClip 
from diffsynth.core.data.operators import DataProcessingOperator
import os
import io
import torch
import numpy as np
import soundfile as sf
import imageio
from diffsynth.core.data.operators import DataProcessingOperator, LoadIDGrid, DebugVisualizer, ImageCropAndResize

def images2video_buffer(images_list, kwargs):
    """将多个图像序列保存为对比视频"""
    fps = kwargs.get("fps", 30)
    format = kwargs.get("format", "mp4")
    codec = kwargs.get("codec", "libx264")
    ffmpeg_params = ["-crf", str(kwargs.get("crf", 12))] #crf越低画质越高
    pixelformat = kwargs.get("pixelformat", "yuv420p")
    
    video_stream = io.BytesIO()
    min_length = min(len(images) for images in images_list)
    
    with imageio.get_writer(video_stream, fps=fps, format=format, 
                          codec=codec, ffmpeg_params=ffmpeg_params, 
                          pixelformat=pixelformat) as writer:
        for idx in range(min_length):
            frame = np.concatenate([images[idx] for images in images_list], axis=1)
            writer.append_data(frame)
    return video_stream.getvalue()

def write_video_with_audio(video_res, audio_data, tgt_fps, save_path):
    """
    video_res: numpy array [T, H, W, C]
    audio_data: numpy array [N] or [1, N] or Tensor
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    temp_video_path = save_path.replace(".mp4", "_temp_silent.mp4")
    temp_audio_path = save_path.replace(".mp4", "_temp_audio.wav")

    # 1. 保存纯画面视频
    # 注意：video_res 是 uint8 格式
    video_data = images2video_buffer([video_res], {"fps": tgt_fps})
    with open(temp_video_path, "wb") as f:
        f.write(video_data)

    # 2. 保存音频 wav
    # 统一处理音频格式转为 numpy 1D
    if torch.is_tensor(audio_data):
        audio_data = audio_data.detach().cpu().numpy()
    if audio_data.ndim > 1:
        audio_data = audio_data.flatten() # [1, N] -> [N]
        
    if audio_data is not None and len(audio_data) > 0:
        sf.write(temp_audio_path, audio_data, 16000, 'PCM_24')
    else:
        print(f"Warning: No audio data for {save_path}")
        os.rename(temp_video_path, save_path)
        return

    # 3. 合并音画
    # -y: 覆盖
    # -shortest: 以最短的流为准（防止音频比视频长一点点导致黑屏）
    cmd = (
        f'ffmpeg -y -loglevel quiet '
        f'-i "{temp_video_path}" -i "{temp_audio_path}" '
        f'-c:v copy -c:a aac '
        f'"{save_path}"'
    )
    os.system(cmd)

    # 清理临时文件
    if os.path.exists(temp_video_path): os.remove(temp_video_path)
    # if os.path.exists(temp_audio_path): os.remove(temp_audio_path)
    print(f"[Debug] Saved check video: {save_path}")

class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        num_frames=57,
        tgt_fps=15,
        main_data_operator=lambda x: x,
        special_operator_map=None,
        # ========== 九宫格ID注入相关参数 ==========
        enable_id_grid=False,  # 是否启用九宫格ID注入
        id_grid_height=560,    # 九宫格输出高度
        id_grid_width=480,     # 九宫格输出宽度
        id_grid_max_pixels=268800, # 九宫格等效面积 (等效面积模式)
        id_grid_aug_intensity=1.9,  # 数据增强强度
        id_grid_num_frames=1,     # 九宫格参考视频帧数
        id_drop_rate=0.0,      # 九宫格随机丢弃率（用于 CFG）
        debug=False,           # 是否启用调试可视化
        debug_save_dir="./debug_vis",  # 调试输出目录
    ):
        self.ori_fps = 30
        self.video_start_idx = 0
        self.video_end_idx = 0
        self.sample_rate=16000
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.num_frames = num_frames
        self.tgt_fps = tgt_fps
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)
        # ========== 九宫格ID注入初始化 ==========
        self.enable_id_grid = enable_id_grid
        self.id_grid_height = id_grid_height
        self.id_grid_width = id_grid_width
        self.id_grid_max_pixels = id_grid_max_pixels
        self.id_grid_num_frames = id_grid_num_frames
        self.id_drop_rate = id_drop_rate
        self.id_grid_loader = None
        if enable_id_grid:
            self.id_grid_loader = LoadIDGrid(
                tgt_fps=tgt_fps,
                height=id_grid_height,
                width=id_grid_width,
                max_pixels=id_grid_max_pixels,
                aug_intensity=id_grid_aug_intensity,
                id_grid_num_frames=id_grid_num_frames,
            )
        # ========== 调试可视化初始化 ==========
        self.debug = debug
        self.debug_visualizer = DebugVisualizer(enabled=debug, save_dir=debug_save_dir)
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, RouteByExtensionName(operator_map=[
                # (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        self.use_dataframe = False
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        elif metadata_path.endswith(".txt"):
            print(f"[Dataset] Loading TXT: {metadata_path}")
            data_list =[]
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    row_dict = {}  
                    try:
                        line = line.strip()
                        if not line: continue
                        items = line.split('\t')
                        for item in items:
                            if ':' in item:
                                key, val = item.split(':', 1)
                                key = key.strip()
                                if val.startswith('"') and val.endswith('"'):
                                    val = val[1:-1]
                                val = val.strip()  
                                if key == "video_length" and val != "none":
                                    val = int(float(val))
                                elif key == "fps" and val != "none":
                                    val = float(val)
                                row_dict[key] = val
                        data_list.append(row_dict)
                    except Exception as e:
                        current_idx = row_dict.get("global_idx", "Unknown")
                        print(f"[Bug] Failed to parse line in global_idx: {current_idx}. Error: {e}")
                        continue
                        
            self.data = data_list
            print(f"[Dataset] Successfully loaded {len(self.data)} rows from TXT.")
        else:
            # 处理多个 CSV 混合 (用逗号分隔的路径)
            paths = metadata_path.split(',')
            all_metadata = []
            for path in paths:
                path = path.strip()
                if not path: continue
                print(f"[Dataset] Loading CSV: {path}")
                try:
                    metadata = pandas.read_csv(path)
                    # === 兼容性修正逻辑 ===
                    # 1. 统一 video_path
                    if 'video_path' not in metadata.columns and 'ceph_path' in metadata.columns:
                        metadata['video_path'] = metadata['ceph_path']
                        
                    # 2. 统一 caption (支持多种别名)
                    caption_candidates = ['target_video_caption', 'video_path_caption', '027_unreal_sp_1080p_caption', "ceph_path_caption"]
                    for col in caption_candidates:
                        if col in metadata.columns:
                            metadata['target_video_caption'] = metadata[col]
                            break
                            
                    # 3. 统一 video_length
                    if 'video_length' not in metadata.columns:
                        print(f"[Dataset Warning] {path} 缺少 video_length 列，将假设默认足够长 (9999)")
                        metadata['video_length'] = 9999
                    # 统一为 int 类型，避免因为 float 导致后续报错
                    metadata['video_length'] = metadata['video_length'].fillna(9999).astype(int)
                    # =====================
                    
                    # 过滤掉帧数不足的视频
                    metadata = metadata[metadata['video_length'] >= self.num_frames]
                    all_metadata.append(metadata)
                except Exception as e:
                    print(f"[Bug] Failed to load {path}: {e}")
            
            if all_metadata:
                # 执行交叉交织（Interleave）合并
                # 效果: CSV1_row1, CSV2_row1, ..., CSVN_row1, CSV1_row2, CSV2_row2...
                for i, df in enumerate(all_metadata):
                    df['_csv_idx'] = i
                    df['_row_idx'] = np.arange(len(df))
                    
                self.metadata_df = pandas.concat(all_metadata, ignore_index=True)
                # 优先按行号排序，行号相同则按 CSV 编号排序
                self.metadata_df.sort_values(by=['_row_idx', '_csv_idx'], inplace=True)
                self.metadata_df.drop(columns=['_csv_idx', '_row_idx'], inplace=True)
                self.metadata_df.reset_index(drop=True, inplace=True)
                
                print(f"[Dataset] Total loaded rows from all CSVs (Interleaved): {len(self.metadata_df)}")
            else:
                self.metadata_df = pandas.DataFrame()
                print("[Warning] No valid CSV data loaded!")
            self.use_dataframe = True

    def __getitem__(self, data_id):
        try:    
            if hasattr(self, "use_dataframe") and self.use_dataframe:
                data = self.metadata_df.iloc[data_id % len(self.metadata_df)].to_dict()
            else:
                data = self.data[data_id % len(self.data)].copy() # {'video': 'wans2v/s2v_video.mp4', 's2v_pose_video': 'wans2v/pose.mp4', 'input_audio': 'wans2v/sing.MP3', 'prompt': 'a person is singing'}
            # ================== 九宫格ID注入模块 ==================
            if self.enable_id_grid and self.id_grid_loader is not None:
                drop_rate = getattr(self, "id_drop_rate", 0.0)
                if random.random() < drop_rate:
                    id_grid_tensor = torch.zeros((3, self.id_grid_num_frames, self.id_grid_height, self.id_grid_width), dtype=torch.float32)
                else:
                    id_grid_tensor = self.id_grid_loader(data)
                data["id_grid"] = id_grid_tensor # ([3, 41, 原始高度, 原始宽度])
                
                # Debug 可视化
                if self.debug:
                    self.debug_visualizer.save_video(
                        id_grid_tensor, 
                        "id_grid", 
                        data_id=data_id, 
                        fps=20
                    )
                    # 如果有视频首帧，保存对比图
                    # if "video_path" in data and torch.is_tensor(data["video_path"]):
                    #     first_frame = data["video_path"][:, 0]  # (C, H, W)
                    #     self.debug_visualizer.save_grid_comparison(
                    #         id_grid_tensor,
                    #         first_frame,
                    #         "comparison",
                    #         data_id=data_id
                    #     )
            # ===================================================
            # 用于暂存 Debug 数据
            # debug_raw_video = None
            # debug_raw_audio = None

            for key in self.data_file_keys:
                # 只有audio需要特殊处理
                if key in self.special_operator_map: # {'animate_face_video': <diffsynth.core.data.operators.DataProcessingPipeline object at 0x7faa2d200f50>, 'input_audio': <diffsynth.core.data.operators.DataProcessingPipeline object at 0x7faa2d200fb0>}
                    operator = self.special_operator_map[key]
                elif key in self.data_file_keys: # ['video', 'input_audio', 's2v_pose_video']
                    operator = self.main_data_operator # diffsynth.core.data.operators.RouteByType
                else:
                    continue
                # 1. 判断是否需要传入全量 data (针对 LoadAudio)
                if hasattr(operator, "needs_full_data") and operator.needs_full_data:
                    # LoadAudio 走这里
                    processed_val = operator(data)
                    data[key] = processed_val
                    
                    # 捕获音频数据用于 Debug (假设 key 是 audio_path)
                    # if "audio" in key:
                    #     debug_raw_audio = processed_val
                
                # 2. 普通 Operator (针对 LoadVideo)
                else:
                    processed_val = operator(data[key])
                    
                    # 检测：如果 Operator 返回的是字典（且包含 start_idx），说明它是我们的 VideoLoader
                    if isinstance(processed_val, dict) and "start_idx" in processed_val:
                        # 提取 start_idx 存入 data 字典，供 Audio 使用
                        data["video_start_idx"] = processed_val["start_idx"]
                        data["actual_n"] = processed_val["actual_n"]
                        # 真正的视频数据赋值回 key
                        data[key] = processed_val["frames"]
                        
                        # 关键修改：创建视频数据的副本以避免in-place操作问题
                        # if torch.is_tensor(data[key]):
                        #     data[key] = data[key].clone().detach()
                        
                        # debug_raw_video = processed_val["input_img_list"]
                    else:
                        # 确保其他类型的数据也不会在训练过程中被in-place修改111
                        if torch.is_tensor(processed_val):
                            processed_val = processed_val.clone().detach()
                        data[key] = processed_val
            # ================== 可视化查验模块 ==================
            # print(f"data_id:{data_id}")
            # should_debug = False
            # if data_id % 200 == 0:
            #     should_debug = True 
                        
            # if should_debug and debug_raw_video is not None and debug_raw_audio is not None:
            #     save_dir = "./debug_vis_output_debug"
            #     # 文件名带上 data_id 和 start_idx 方便追溯
            #     start_idx = data.get("video_start_idx", 0)
            #     filename = f"sample_{data_id:04d}_start{start_idx}.mp4"
            #     save_path = os.path.join(save_dir, filename)
                
            #     try:
            #         write_video_with_audio(
            #             video_res=debug_raw_video, # numpy [T, H, W, C] # (57, 640, 480, 3)
            #             audio_data=debug_raw_audio, # numpy [N] (60800)
            #             tgt_fps=self.tgt_fps,              
            #             save_path=save_path
            #         )
            #     except Exception as e:
            #         print(f"[Debug Error] Failed to write video: {e}")
            # # ===================================================
            
            return data
        except Exception as e:
            print(f"[dataBug todo Dataset] Error processing data_id {data_id}: {e}. Retrying with another sample.")
            return self.__getitem__(random.randint(0, len(self) - 1))


    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            if hasattr(self, "use_dataframe") and self.use_dataframe:
                return len(self.metadata_df) * self.repeat
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True
