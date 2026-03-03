import cv2
import numpy as np
import imageio.v2 as imageio
import io
import os
from decord import VideoReader, cpu
import ffmpeg
import subprocess
import imageio
import json
import torch,torchaudio
from moviepy.editor import VideoFileClip, AudioFileClip
import pandas as pd
import soundfile as sf
import random
import csv
import ast


mouth_idx = [i for i in range(72,92)]+[i for i in range(55,60)]+[30,34]
def interpolate_bboxes(sparse_dict, kind='linear'):
    """
    对稀疏的 bbox 字典进行插值，返回连续的 numpy 矩阵。
    
    参数:
        sparse_dict (dict): 格式为 {frame_id: [x1, y1, x2, y2]} 的字典。
                            frame_id 必须是整数，value 可以是 list 或 numpy array。
        kind (str): 插值类型，目前主要支持 'linear' (线性插值)。
        
    返回:
        numpy.ndarray: 形状为 (n, 4) 的矩阵，包含从最小帧到最大帧的所有 bbox。
        list: 对应的帧号列表 (从 min_frame 到 max_frame)。
    """
    if not sparse_dict:
        return np.empty((0, 4)), []

    # 1. 获取排序后的帧号和对应的 bbox
    # sorted_frames 是一个列表: [1, 5, 10, ...]
    sorted_frames = sorted(sparse_dict.keys())
    
    # 2. 确定起止帧
    min_frame = sorted_frames[0]
    max_frame = sorted_frames[-1]
    num_frames = max_frame - min_frame + 1
    
    # 初始化结果矩阵，形状为 (总帧数, 4)
    # 使用 float 类型以保持插值精度
    dense_bboxes = np.zeros((num_frames, 4), dtype=np.float32)
    
    # 3. 执行插值
    # 我们遍历排序后的已知帧列表，处理每一段区间
    for i in range(len(sorted_frames) - 1):
        curr_f = sorted_frames[i]
        next_f = sorted_frames[i+1]
        
        curr_bbox = np.array(sparse_dict[curr_f])
        next_bbox = np.array(sparse_dict[next_f])
        
        # 计算这一段区间的长度
        gap = next_f - curr_f
        
        # 对 x1, y1, x2, y2 分别进行线性插值
        # np.linspace 生成从 curr_bbox 到 next_bbox 的均匀分布数值
        # endpoint=False 是因为 next_bbox 会在下一次循环作为起点被填入，
        # 但如果是最后一段，我们需要包含终点。
        # 这里为了逻辑简单，我们逐帧计算。
        
        for step in range(gap):
            # 相对位置索引 (0 到 gap-1)
            # 绝对帧号
            target_frame = curr_f + step
            
            # 线性插值公式: y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
            # alpha 是进度 (0.0 到 1.0)
            alpha = step / float(gap)
            interpolated_bbox = curr_bbox + (next_bbox - curr_bbox) * alpha
            
            # 填入矩阵
            # 注意：矩阵索引是从 0 开始的，所以要减去 min_frame
            idx = target_frame - min_frame
            dense_bboxes[idx] = interpolated_bbox

    # 4. 填入最后一帧 (循环中没有覆盖到最后一个关键帧本身)
    dense_bboxes[-1] = sparse_dict[max_frame]

    # 生成连续的帧号列表用于参考
    dense_frame_ids = list(range(min_frame, max_frame + 1))
    
    return dense_bboxes, dense_frame_ids

class FaceGrid:
    def __init__(self, model_path=None):
        self.face_idx = [i for i in range(24,92)] 

    def load_pose(self, pose_path, height, width, ori_fps, tgt_fps, clip_start_idx=0, clip_length=None):
        if '.json' in pose_path:
            dwpose_np = self.json_to_keypoints_matrix(pose_path)
        else:
            dwpose_np = np.load(pose_path) # (Frames, 1, 134, 3)
        
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

        return dwpose_np
    
    def generate_aug_params(self, intensity=0.5, pimple=False, wrinkle=False):
        """
        生成一组固定的增强参数。
        这些参数将在整个视频序列中保持不变，以防止闪烁。
        """
        if intensity <= 0:
            return {
                'b': 0, 'c': 1.0, 's': 1.0, 
                'flip': False, 'black': False,
                # ========== 新增参数 ==========
                'smooth': 0,           # 磨皮强度 (0=不做, 1-3=轻中重)
                'blemish_list': [],         # 痘痘/痣列表
                'wrinkle_list': [],       # 皱纹参数
            }


        # === 定义最大变化范围 ===
        MAX_BRIGHTNESS_DELTA = 30   
        MAX_CONTRAST_VAR = 0.2      
        MAX_SATURATION_VAR = 0.3   
        blemish_list = []
        wrinkle_list = []

        if pimple: #@jiwen: 痣的参数
            # 亮度
            b_limit = MAX_BRIGHTNESS_DELTA * intensity
            brightness_delta = random.uniform(-b_limit, 0)
            
            # 对比度
            c_limit = MAX_CONTRAST_VAR * intensity
            contrast_factor = random.uniform(1.0, 1.0 + c_limit)
            
            # 饱和度
            s_limit = MAX_SATURATION_VAR * intensity
            saturation_factor = random.uniform(1.0, 1.0 + s_limit)
            
            if random.uniform(0, 1) < 1: #@jiwen 痣的概率
                num_blemishes = random.randint(1, 6)
                for _ in range(num_blemishes):
                    blemish = {
                        'rel_x': random.uniform(-1, 1) / 2,
                        'rel_y': random.uniform(-1, 1) / 2,
                        'radius': random.randint(1, 6), # 1, 6
                        'blur_k': random.randint(5, 9),
                    }
                    blemish['blur_k'] =  blemish['radius'] // 2 + blemish["blur_k"]
                    blemish_list.append(blemish)
        if wrinkle:
            # 亮度
            b_limit = MAX_BRIGHTNESS_DELTA * intensity
            brightness_delta = random.uniform(-b_limit, 0)
            
            # 对比度
            c_limit = MAX_CONTRAST_VAR * intensity
            contrast_factor = random.uniform(1.0, 1.0 + c_limit)
            
            # 饱和度
            s_limit = MAX_SATURATION_VAR * intensity
            saturation_factor = random.uniform(1.0, 1.0 + s_limit)
            if random.uniform(0, 1) < 1:
                num_wrinkles = random.randint(1, 4)
                for _ in range(num_wrinkles):
                    wrinkle = {
                        'rel_x': random.uniform(-1, 1) / 2,
                        'rel_y': random.uniform(-1, 1) / 2,
                        'radius': random.randint(20, 40), # 1, 6
                        'blur_k': random.randint(1, 6),
                        'angle': random.uniform(0, 360),
                        'shift': random.uniform(-0.5, 0.5),
                    }
                    wrinkle['blur_k'] =  2*(wrinkle['radius'] //2 + wrinkle["blur_k"]) + 1
                    wrinkle_list.append(wrinkle)
        elif not pimple and not wrinkle:
            # 亮度
            b_limit = MAX_BRIGHTNESS_DELTA * intensity
            brightness_delta = random.uniform(-b_limit, b_limit)
            
            # 对比度
            c_limit = MAX_CONTRAST_VAR * intensity
            contrast_factor = random.uniform(1.0 - c_limit, 1.0 + c_limit)
            
            # 饱和度
            s_limit = MAX_SATURATION_VAR * intensity
            saturation_factor = random.uniform(1.0 - s_limit, 1.0 + s_limit)

        flip = random.uniform(0, 1) < 0.5
        black = random.uniform(0, 1) < 0.05

        # 30% 概率启用磨皮
        if random.uniform(0, 1) < 0.3:
            smooth_level = random.choice([1, 2, 3])  # 轻/中/重
            # smooth_level = 3  # 轻/中/重
        else:
            smooth_level = 0
        
        return {
            'b': brightness_delta,
            'c': contrast_factor,
            's': saturation_factor,
            'flip': flip,
            'black': black,
            'blemish_list': blemish_list,
            'wrinkle_list': wrinkle_list,
            'smooth': smooth_level,
        }
    def _apply_smooth(self, img, level):
        """
        快速磨皮效果
        
        方法：高斯模糊 + 高频细节保留混合
        比双边滤波快 5-10 倍，效果接近
        """
        if level == 0:
            return img
        
        # 根据等级设置参数
        blur_size = {1: 5, 2: 9, 3: 15}[level]
        blend_alpha = {1: 0.3, 2: 0.5, 3: 0.7}[level]  # 模糊图的权重
        
        # 方法1: 简单高斯模糊混合（最快）
        blurred = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
        
        # 混合原图和模糊图
        result = cv2.addWeighted(img, 1 - blend_alpha, blurred, blend_alpha, 0)
        
        return result
    
    def _apply_smooth_v2(self, img, level):
        """
        改进版磨皮：保留边缘的快速磨皮
        
        原理：只模糊低频部分，保留高频边缘
        速度：比双边滤波快约 3-5 倍
        """
        if level == 0:
            return img
        
        blur_size = {1: 5, 2: 11, 3: 21}[level]
        
        # 1. 获取模糊图
        blurred = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
        
        # 2. 计算高频细节 (原图 - 模糊图)
        # 先转float避免溢出
        img_f = img.astype(np.float32)
        blurred_f = blurred.astype(np.float32)
        high_freq = img_f - blurred_f
        
        # 3. 使用边缘检测生成mask（边缘区域保留更多细节）
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_mask = cv2.GaussianBlur(edges.astype(np.float32), (5, 5), 0)
        edge_mask = edge_mask / (edge_mask.max() + 1e-6)  # 归一化到 0-1
        edge_mask = edge_mask[:, :, np.newaxis]  # 扩展维度
        
        # 4. 边缘区域保留更多高频，平坦区域更模糊
        blend_factor = {1: 0.3, 2: 0.5, 3: 0.7}[level]
        detail_preserve = high_freq * (edge_mask * 0.8 + 0.2)  # 边缘保留更多
        
        result = blurred_f + detail_preserve * (1 - blend_factor)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        
        return result

    def _apply_blemish(self, img_ori, blemish_list, lms_i_new):
        """
        添加痘痘/痣/斑点
        
        速度：非常快（只是画几个圆）
        """
        img = img_ori.copy() * 0
        img0 = img.astype("float32")
        h, w = img.shape[:2]

        fw = np.max(lms_i_new[:,0]) - np.min(lms_i_new[:,0])
        fh = np.max(lms_i_new[:,1]) - np.min(lms_i_new[:,1])
        for blemish in blemish_list:
            result = img.copy()
            # 计算实际像素位置
            cx = int(blemish['rel_x'] * fw + lms_i_new[30, 0])
            cy = int(blemish['rel_y'] * fh + lms_i_new[30, 1])
            blur_k = blemish['blur_k'] 
            radius = blemish['radius']
         
            # color_var = blemish['color_var']
            color_var = 0
            
            # 确保在图像范围内
            if cx < radius or cx > w - radius or cy < radius or cy > h - radius:
                continue
            
            center_color = (255,255,255)
                
            cv2.circle(result, (cx, cy), radius, center_color[::-1], -1, cv2.LINE_AA)

            result = cv2.blur(result, (blur_k,blur_k)).astype("float32") / 255
            img0[result>0] = result[result>0]
        
        # cv2.imwrite("debug_mask.png", (img0 * 255).astype(np.uint8))
        return img0


    def process_pimple(self, img, params, lms_i_new=None, frame_idx=0):
        img_ori = img.copy()
        brightness_delta = params['b']
        contrast_factor = params['c']
        saturation_factor = params['s']
        blemish_list = params['blemish_list']
       
        # 如果参数是默认值，直接返回
        if brightness_delta == 0 and contrast_factor == 1.0 and saturation_factor == 1.0:
            return img

        # --- 步骤 A: 亮度和对比度 (快速矩阵运算) ---
        img = img.astype(np.float32)
        img = img * contrast_factor + brightness_delta
        img = np.clip(img, 0, 255)

        # --- 步骤 B: 饱和度 ---
        if abs(saturation_factor - 1.0) > 0.01:
            img = img.astype(np.uint8)
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            
            hsv_img[..., 1] *= saturation_factor
            hsv_img[..., 1] = np.clip(hsv_img[..., 1], 0, 255)
            
            hsv_img = hsv_img.astype(np.uint8)
            img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        else:
            img = img.astype(np.uint8)
        
        mask = (img_ori == 0).all(axis=2)  # 对于RGB/BGR图片，检查所有通道是否为0
        img[mask] = 0

        mask = self._apply_blemish(img, blemish_list, lms_i_new)

        img = (mask * img + (1-mask) * img_ori).astype(np.uint8)

        return img
    
    def apply_stable_augmentation(self, img, params, lms_i_new=None, frame_idx=0):
        """
        使用预先计算好的参数对图片进行增强。

        增强顺序：
        1. 亮度/对比度
        2. 饱和度
        3. 磨皮 (新增)
        4. 痘痘/痣 (新增)
        5. 皱纹 (新增)
        6. 翻转
        7. 全黑
        """
        img_ori = img.copy()
        brightness_delta = params['b']
        contrast_factor = params['c']
        saturation_factor = params['s']
        flip = params['flip']
        black = params['black']
        smooth_level = params.get('smooth', 0)
      
        # 如果参数是默认值，直接返回
        if brightness_delta == 0 and contrast_factor == 1.0 and saturation_factor == 1.0:
            return img

        # --- 步骤 A: 亮度和对比度 (快速矩阵运算) ---
        img = img.astype(np.float32)
        img = img * contrast_factor + brightness_delta
        img = np.clip(img, 0, 255)

        # --- 步骤 B: 饱和度 ---
        if abs(saturation_factor - 1.0) > 0.01:
            img = img.astype(np.uint8)
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            
            hsv_img[..., 1] *= saturation_factor
            hsv_img[..., 1] = np.clip(hsv_img[..., 1], 0, 255)
            
            hsv_img = hsv_img.astype(np.uint8)
            img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        else:
            img = img.astype(np.uint8)

        # img = img._apply_smooth_v2(img, smooth_level)
        
        mask = (img_ori == 0).all(axis=2)  # 对于RGB/BGR图片，检查所有通道是否为0
        img[mask] = 0

        if flip:
            img = img[:,::-1]

        if black:
            img[:,:,0] = img[:,:,0]*0
            img[:,:,1] = img[:,:,1]*0
            img[:,:,2] = img[:,:,2]*0

        return img



    # def _apply_wrinkle(self, img_ori, wrinkle_list, lms_i_new):
    #     """
    #     添加皱纹纹理
    #     返回一个 mask，表示皱纹的强度（0-1）
    #     """
    #     h, w = img_ori.shape[:2]
        
    #     if not wrinkle_list:
    #         return np.zeros((h, w, 3), dtype="float32")
        
    #     # 最终的皱纹mask
    #     img0 = np.zeros((h, w, 3), dtype="float32")
        
    #     # 加载并预处理皱纹纹理图
    #     wrinkle_mask = self.wrinkle_mask.copy()
    #     # wrinkle_mask = wrinkle_mask / wrinkle_mask.max()  # 归一化到 0-1
    #     wrinkle_mask = cv2.resize(wrinkle_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        
    #     # kernel = np.ones((3, 3), np.uint8)
    #     # wrinkle_mask = cv2.dilate(wrinkle_mask, kernel, iterations=1)
        
    #     # 平铺3x3，确保偏移采样时不越界
    #     wrinkle_mask = np.tile(wrinkle_mask, (3, 3, 1))
        
    #     fw = np.max(lms_i_new[:, 0]) - np.min(lms_i_new[:, 0])
    #     fh = np.max(lms_i_new[:, 1]) - np.min(lms_i_new[:, 1])
        
    #     for wrinkle in wrinkle_list:
    #         # 计算皱纹中心位置
    #         cx = int(wrinkle['rel_x'] * fw + lms_i_new[30, 0])
    #         cy = int(wrinkle['rel_y'] * fh + lms_i_new[30, 1])
            
    #         # 计算在平铺纹理中的采样偏移
    #         Cx = int(-cx + 2 * w + wrinkle["shift"] * w)
    #         Cy = int(-cy + 2 * h + wrinkle["shift"] * h)
            
    #         blur_k = wrinkle['blur_k']
    #         radius = wrinkle['radius']
            
    #         # 边界检查
    #         if cx < radius or cx > w - radius or cy < radius or cy > h - radius:
    #             continue
            
    #         # 检查采样索引是否有效
    #         y_start, y_end = Cy - h // 2, Cy + h - (h // 2)
    #         x_start, x_end = Cx - w // 2, Cx + w - (w // 2)
            
    #         if y_start < 0 or x_start < 0 or y_end > wrinkle_mask.shape[0] or x_end > wrinkle_mask.shape[1]:
    #             continue
            
    #         # 从平铺纹理中采样当前区域
    #         wrinkle_mask_i = wrinkle_mask[y_start:y_end, x_start:x_end].copy()
            
    #         angle = wrinkle.get('angle', (cx * 7 + cy * 13) % 360)  # 固定但看起来随机
    #         center = (w // 2, h // 2)
    #         rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    #         wrinkle_mask_i = cv2.warpAffine(wrinkle_mask_i, rotation_matrix, (w, h),
    #                                         borderMode=cv2.BORDER_WRAP)

    #         # wrinkle_mask_i = cv2.GaussianBlur(wrinkle_mask_i, (blur_k, blur_k), 0)
    #         # wrinkle_mask_i = wrinkle_mask_i - low_freq  # 高频部分
    #         # wrinkle_mask_i = np.clip(wrinkle_mask_i, 0, 1)  # 只保留正向（皱纹线条）
            
    #         # 创建圆形渐变mask，限制皱纹显示区域
    #         # circle_mask = np.zeros((h, w), dtype="float32")
    #         # cv2.circle(circle_mask, (cx, cy), radius, 1.0, -1, cv2.LINE_AA)
    #         circle_mask = np.zeros((h, w), dtype="float32")
    #         cv2.ellipse(circle_mask, (cx, cy), (radius*2, radius), angle, angle, 360-angle, 1.0, -1, cv2.LINE_AA)
    #         # # 高斯模糊使边缘平滑过渡
    #         # blur_k_odd = blur_k if blur_k % 2 == 1 else blur_k + 1
    #         # circle_mask = cv2.GaussianBlur(circle_mask, (blur_k_odd, blur_k_odd), 0)
    #         circle_mask = circle_mask[:, :, np.newaxis]  # 扩展维度 [h, w, 1]
            
    #         # 直接用皱纹纹理乘以圆形mask
    #         local_wrinkle = wrinkle_mask_i * circle_mask
            
    #         # 累加到总mask（取最大值避免重叠区域过暗）
    #         # img0[local_wrinkle>0] = local_wrinkle[local_wrinkle>0]
    #         img0 = np.maximum(img0, local_wrinkle)

    #     return img0
    
    # def process_wrinkle(self, img, params, lms_i_new=None, frame_idx=0):
    #     """
    #     仿照 process_pimple 的逻辑处理皱纹
    #     """
    #     img_ori = img.copy()
    #     brightness_delta = params['b']
    #     contrast_factor = params['c']
    #     saturation_factor = params['s']
    #     wrinkle_list = params['wrinkle_list']

    #     if brightness_delta == 0 and contrast_factor == 1.0 and saturation_factor == 1.0:
    #         return img

    #     # --- 步骤 A: 亮度和对比度 ---
    #     img = img.astype(np.float32)
    #     img = img * contrast_factor + brightness_delta
    #     img = np.clip(img, 0, 255)

    #     # --- 步骤 B: 饱和度 ---
    #     if abs(saturation_factor - 1.0) > 0.01:
    #         img = img.astype(np.uint8)
    #         hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    #         hsv_img[..., 1] *= saturation_factor
    #         hsv_img[..., 1] = np.clip(hsv_img[..., 1], 0, 255)
    #         hsv_img = hsv_img.astype(np.uint8)
    #         img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    #     else:
    #         img = img.astype(np.uint8)

    #     mask = (img_ori == 0).all(axis=2)
    #     img[mask] = 0

    #     # --- 步骤 C: 获取皱纹 mask 并混合（仿照 pimple）---
    #     wrinkle_mask = self._apply_wrinkle(img, wrinkle_list, lms_i_new)  
        
    #     cv2.imwrite("/m2v_intern/mengzijie/DiffSynth-Studio/data/traindataset/ID_Encoder/wrinklemask.png", (wrinkle_mask).astype(np.uint8))
        
    #     # 混合：mask=0 的地方用处理后的 img（变暗的肤色），mask=1 的地方用原图
    #     # mask = (wrinkle_mask > 0).astype(np.float32)
    #     # SUM = np.sum(wrinkle_mask, axis=(0,1))
    #     # NUM = np.sum(mask, axis=(0,1))
    #     # average_wrinkle_mask = SUM / NUM
    #     # # print(SUM, NUM, average_wrinkle_mask)
    #     # img = img + (wrinkle_mask-1.5*average_wrinkle_mask) * mask
    #     # img[img>255] = 255
    #     # img[img<0] = 0
    #     # img = img.astype(np.uint8)

    #     img = (wrinkle_mask / 256.0 * img + (1 - wrinkle_mask/256.0) * img_ori)
    #     return img
    
    
    def get_center_crop_face(self, img, landmarks, landmarksi, scale=1.5, target_w=None, target_h=None):
        """
        基于人脸中心和指定目标长宽比进行裁剪，超出边界则硬裁剪。
        如果目标长宽比未指定，则直接返回自然扩大的 bounding box (等效面积训练用)。
        """
        new_landmarksi = landmarksi.copy()
        h_img, w_img = img.shape[:2]
        
        # 1. 获取人脸关键点边界
        valid_pts = landmarks[:, :2]
        if valid_pts.shape[0] == 0:
            return img, new_landmarksi

        min_x, min_y = np.min(valid_pts, axis=0)
        max_x, max_y = np.max(valid_pts, axis=0)

        # 人脸中心
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        
        # 人脸原始宽高
        fw = max_x - min_x
        fh = max_y - min_y
        
        if target_w is not None and target_h is not None:
            # 兼容固定宽高：强行保持目标长宽比
            base_size = max(fw, fh) * scale
            target_ar = target_w / target_h
            if target_ar >= 1:
                crop_h = base_size
                crop_w = base_size * target_ar
            else:
                crop_w = base_size
                crop_h = base_size / target_ar
        else:
            # 等效面积：直接取人脸自然的 bounding box 扩大
            crop_w = fw * scale
            crop_h = fh * scale
            
        # 3. 计算坐标 (直接以 cx, cy 为中心)
        x1 = int(cx - crop_w / 2)
        y1 = int(cy - crop_h / 2)
        x2 = int(cx + crop_w / 2)
        y2 = int(cy + crop_h / 2)
        
        # 4. 硬裁切 (不使用 Padding 等奇技淫巧)
        safe_x1 = max(0, x1)
        safe_y1 = max(0, y1)
        safe_x2 = min(w_img, x2)
        safe_y2 = min(h_img, y2)
        
        # 5. 执行裁剪
        crop = img[safe_y1:safe_y2, safe_x1:safe_x2]
        
        # 兜底
        if crop.size == 0:
            return img, new_landmarksi
        
        new_landmarksi[:, 0] = landmarksi[:, 0] - safe_x1
        new_landmarksi[:, 1] = landmarksi[:, 1] - safe_y1

        return crop, new_landmarksi


    def compose_face_grid_frames_stable(self, source_frames, dwpose_np_face, dwpose_np_full, h, w, max_pixels=None, aug_intensity=1.9):
        """
        合成人脸九宫格视频 + 无闪烁(Stable)的可调节增强
        支持固定长宽比 (传 h, w) 和等效面积 (传 max_pixels, 不传 h, w) 两种模式。
        """
        # 1. 数据切分为9份
        grid_sources = np.split(source_frames, 9) # (9, 536, 1024, 3)
        grid_landmarks = np.split(dwpose_np_face, 9)
        grid_landmarks_full = np.split(dwpose_np_full, 9)
        
        target_length = grid_sources[0].shape[0]
        
        # 决定九宫格的最终尺寸和单格尺寸
        if max_pixels is not None:
            # 等效面积模式
            # 随机取一帧来估计人脸自然长宽比
            lms_idx_est = random.randint(0, len(grid_landmarks[0])-1)
            est_img = grid_sources[0][lms_idx_est]
            est_lms = grid_landmarks[0][lms_idx_est]
            est_lms_i = grid_landmarks[0][lms_idx_est]
            face_crop, _ = self.get_center_crop_face(est_img, est_lms, est_lms_i, scale=1.2, target_w=None, target_h=None)
            face_h, face_w = face_crop.shape[:2]
            if face_h == 0 or face_w == 0:
                face_ar = 1.0
            else:
                face_ar = face_w / face_h
            
            # 计算等效面积下的宽高
            # 3*cell_w * 3*cell_h = max_pixels
            # cell_w / cell_h = face_ar
            area = max_pixels if max_pixels else 268800 # 默认等效面积为 480*560
            grid_w = (area * face_ar) ** 0.5
            grid_h = (area / face_ar) ** 0.5
            
            # 为了 VAE 和 九宫格划分，宽高需要是 48 (16 * 3) 的整数倍
            final_w = int(round(grid_w / 48) * 48)
            final_h = int(round(grid_h / 48) * 48)
            
            if final_w == 0: final_w = 48
            if final_h == 0: final_h = 48
            
            cell_w = final_w // 3
            cell_h = final_h // 3
            
            # 告诉 get_center_crop_face 不要强行矫正比例，用自然比例
            target_w_for_crop, target_h_for_crop = None, None
        else:
            # 固定宽高模式
            final_h = h
            final_w = w
            cell_w = w // 3
            cell_h = h // 3
            target_w_for_crop, target_h_for_crop = cell_w, cell_h

        final_video_array = np.zeros((target_length, final_h, final_w, 3), dtype=np.uint8)
        
        # =========================================================
        # 关键点：在开始处理视频帧之前，为9个格子分别生成固定的参数
        # =========================================================
        cell_params_list = []
        for k in range(9):
            # 第k个格子获得一套专属的参数
            params = self.generate_aug_params(intensity=aug_intensity) #@zijie
            cell_params_list.append(params)
            # 打印一下参数，方便调试查看每个格子的风格
        pimple_params_list = []
        for k in range(9):
            # 第k个格子获得一套专属的参数
            params = self.generate_aug_params(intensity=aug_intensity, pimple=True) #@zijie
            pimple_params_list.append(params)
            # 打印一下参数，方便调试查看每个格子的风格
        
        if random.uniform(0,1)< 0.25:
            cell_params_list = [params]*9

        cells_idx_list = [j for j in range(9)]
        random.shuffle(cells_idx_list)
        lms_idx = random.randint(0,len(grid_landmarks[0])-1)
        scale=random.uniform(1.1,1.3)
        for i in range(target_length):
            cells = []
            for k in cells_idx_list:
                img = grid_sources[k][i]
                lms = grid_landmarks[k][lms_idx]
                lms_i = grid_landmarks[k][i]
                lms_i_full = grid_landmarks_full[k][i]
                
                # --- 裁剪与缩放逻辑 (模拟) ---
                # 如果是等效面积，target_* 为 None，使用人脸自然比例硬裁切。如果有拉伸那也是不同人脸本身的比例差异
                face_crop, lms_i_new = self.get_center_crop_face(img, lms, lms_i, scale=scale, target_w=target_w_for_crop, target_h=target_h_for_crop)
                
                if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
                    resized_face = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                else:
                    resized_face = cv2.resize(face_crop, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
                    lms_i_new[:,1] = lms_i_new[:,1]*(cell_h/face_crop.shape[0])
                lms_i_new[:,0] = lms_i_new[:,0]*(cell_w/face_crop.shape[1])
                # 每一帧 i，第 k 个格子都使用 cell_params_list[k]
                # 因为参数不变，所以画面不会闪烁
                augmented_face = self.process_pimple(resized_face, pimple_params_list[k], lms_i_new=lms_i_new, frame_idx=i) #@jiwen: 加痣入口
                augmented_face = self.apply_stable_augmentation(augmented_face, cell_params_list[k], lms_i_new=lms_i_new, frame_idx=i)
                cells.append(augmented_face)
                
            # 拼接
            row1 = np.hstack(cells[0:3])
            row2 = np.hstack(cells[3:6])
            row3 = np.hstack(cells[6:9])

            final_video_array[i][:cell_h*3, :cell_w*3] = np.vstack((row1, row2, row3))
        
        if random.uniform(0,1)<0.6:
            final_video_array = final_video_array[::-1]

        if random.uniform(0,1)<0.1:
            final_video_array[:] = final_video_array[lms_idx:lms_idx+1]

        return final_video_array

    def read_video_frames(self, input_path, target_length):
        try:
            vr = VideoReader(input_path, ctx=cpu(0))
        except Exception as e:
            raise ValueError(f"无法读取视频: {e}")

        original_frames_count = len(vr)
        fps = vr.get_avg_fps()
        
        total_samples = target_length * 9
        
        indices = np.linspace(0, original_frames_count - 1, total_samples, dtype=int).tolist()
        indices_new = indices+indices[::-1][1:]+indices[1:]+indices[::-1][1:]+indices[1:]+indices[::-1][1:]+indices[1:]+indices[::-1][1:]+indices[1:]+indices[::-1][1:]+indices[1:]+indices[::-1][1:]+indices[1:]

        indices2 = []  
        index = 0
        for i in range(total_samples):
            if i%9==0:
                if random.uniform(0,1)<0.2:
                    r = random.uniform(0.1,0.5) 
                else:
                    r = random.uniform(2,8)
            index = index+r

            indices2.append(indices_new[int(round(index))])

        #indices = [indices_new[int(i*r)] for i in range(total_samples)]
        indices = np.array(indices2)
        
        #参考角度不足的情况：
        if random.uniform(0,1)<0.1:
            indices[:] = random_number = random.choice(indices_new)

        try:
            source_frames_batch = vr.get_batch(indices).numpy()
        except:
            source_frames_batch = vr.get_batch(indices).asnumpy()
        
        return source_frames_batch, fps, indices


    def save_video_from_frames(self, video_array, output_path, fps):
        """
        阶段3：视频保存 (基于 ImageIO 重写)
        将内存中的视频数组编码为字节流，并写入文件。
        
        Args:
            video_array: [F, H, W, 3] RGB格式数组
            output_path: 最终保存的文件路径
            fps: 帧率
        """
        frame_count = video_array.shape[0]

        # 创建内存中的二进制流
        video_stream = io.BytesIO()
        
        # 初始化 writer
        # 注意：Decord 读取的是 RGB，ImageIO 默认期望也是 RGB，所以不需要像 OpenCV 那样转 BGR
        with imageio.get_writer(
            video_stream, 
            fps=fps, 
            format="mp4", 
            codec="libx264", 
            ffmpeg_params=["-crf", "12"],  # CRF 12 质量很高
            pixelformat="yuv420p"          # 确保兼容性
        ) as writer:
            
            for i in range(frame_count):
                # 直接写入 numpy 矩阵 [H, W, 3]
                writer.append_data(video_array[i])
                
        # 获取二进制数据
        video_data = video_stream.getvalue()
        
        # 写入物理文件
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, "wb") as f:
            f.write(video_data)
            
    def read_json_dwpose(self, json_dwpose_path):
        try:
        #if True:
            with open(json_dwpose_path, 'r', encoding='utf-8') as f:
                my_dict = json.load(f)

            bbox_dict = {}
            for data in my_dict['face_norm_bboxs_list']:
                idx = data['frame_index']
                if len(data['bboxes'])>0:
                    bbox = data['bboxes'][0]
                    bbox_dict[idx] = bbox


            n = max(bbox_dict)
            dwpose_np = np.zeros([n+1,1,134,3]).astype('float32')

            result_matrix, frame_ids = interpolate_bboxes(bbox_dict)

            for i in range(n):
                dwpose_np[i,0,24:50,0] = result_matrix[i,0]
                dwpose_np[i,0,24:50,1] = result_matrix[i,1]
                dwpose_np[i,0,50:92,0] = result_matrix[i,2]
                dwpose_np[i,0,50:92,1] = result_matrix[i,3]
                dwpose_np[i,0,24:92,2] = 1
                dwpose_np[i,0,mouth_idx,2] = 0

            dwpose_np = np.concatenate([dwpose_np, dwpose_np[::-1]], axis=0)
        except:
            dwpose_np = None

        return dwpose_np

    def json_to_keypoints_matrix(self, json_path):
        
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'instance_info' in data:
            # 提取instance_info
            instance_info = data['instance_info']
        else:
            return self.read_json_dwpose(json_path)
        
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

    # --- 主流程串联 ---
    def execute(self, input_path, output_path, dwpose_path, fps, ori_fps, h=None, w=None, max_pixels=None, target_length=1, save=False):
        # 1. 读
        raw_frames, fps, indices = self.read_video_frames(input_path, target_length)
        ori_h, ori_w = raw_frames[0].shape[:2]
        dwpose_np = self.load_pose(dwpose_path, ori_h, ori_w, ori_fps, fps) #[n,1,134,3]
        indices[indices>dwpose_np.shape[0]-1] = dwpose_np.shape[0]-1

        dwpose_np_face = dwpose_np[:,0,self.face_idx]
        dwpose_np_face = dwpose_np_face[indices]
        dwpose_np_full = dwpose_np[:,0][indices]  # 新增：完整134点

        # 2. 算
        result_frames = self.compose_face_grid_frames_stable(raw_frames, dwpose_np_face, dwpose_np_full, h=h, w=w, max_pixels=max_pixels)
            
        if save: #random.uniform(0,1)<0.05:
            # 3. 写 (使用新的 imageio 逻辑)
            #os.system('cp '+input_path+' '+output_path.replace('.mp4', '_in.mp4'))
            self.save_video_from_frames(result_frames, output_path, fps=8)

        return result_frames

if __name__ == "__main__":
    # import debugpy
    # debugpy.listen(("0.0.0.0", 5679))
    # print("=" * 50)
    # print("Waiting for debugger to attach on port 5678...")
    # print("=" * 50)
    # debugpy.wait_for_client()  
    # print("Debugger attached! Continuing...")
    
    face_grid = FaceGrid()
    #csv_file = "/ytech_m2v2_hdd/liujiwen/audio_v3/m2v-diffusers/id_data_480_720_1080_with_pose/filtered_AVspeech_data_human_check_8dot3w_0804_with_md5_vae_caption_te_193f_2d5w.csv"
    csv_file = "/ytech_m2v2_hdd/liujiwen/audio_v3/m2v-diffusers/id_data_480_720_1080_with_pose/720_filter_emo_40w_30fps_split_121f_with_md5_vae_caption_te_0925_40w_fix_pose.csv"

    # input_video_path = '/ytech_milm_disk2/Keling_HumanMotion/30fps/keling/kling_preprocess/qt_kungfu_motion_cut/gesture_dance/7fb207d6dfd5113a0d7c839470a2580e_241_0.mp4'
    # output_video_path = '/ytech_m2v2_hdd/liujiwen/ID_Encoder/motion/show/debug.mp4'
    # dwpose_path = '/ytech_m2v2_hdd/Pose_2D_Augmentation/0918_render_all_0603version/MultiModal_Feature_Results/Pose2d_0402/ytech_milm_disk2/Keling_HumanMotion/30fps/keling/qt_add_data_cut/0619_sport_data_1607/ski/719c30c570c22910f50b0ff07f79e9d2_305_0/719c30c570c22910f50b0ff07f79e9d2_305_0_Pose2d/719c30c570c22910f50b0ff07f79e9d2_305_0_Pose2d_dwpose.npy'
    # face_grid.execute(input_video_path, output_video_path, dwpose_path=dwpose_path, fps=30, ori_fps=30, h=720, w=1280, n=1, save=True)

    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            #if i<500: continue
            if i%5!=0: continue
            # 一行代码为每个字段设置默认值
            video_path1 = row.get("027_unreal_sp_1080p", None)
            video_path2 = row.get("video_path", None)
            video_path3 = row.get("ceph_path", None)
            dwpose_path = row.get("dwpose_path", None)
            vae_latent_path = row.get("target_video_vae", None)
            fps = float(row.get("fps", -1))
            ori_fps = float(row.get("ori_fps", -1))
            if ori_fps == -1 or str(ori_fps) == "nan":
                ori_fps = fps

            if video_path1 is not None:
                input_video_path = video_path1
            elif video_path2 is not None:
                input_video_path = video_path2
            elif video_path3 is not None:
                input_video_path = video_path3

            output_video_path = '/m2v_intern/mengzijie/DiffSynth-Studio/data/traindataset/ID_Encoder/show3/output_'+str(i)+'.mp4' #@jiwen
            print ('input_video_path', input_video_path)


            latent_dict = torch.load(vae_latent_path, map_location="cpu")
            video_latents = latent_dict["video_data"][0]   # [(b C f h w)] x L
            print ('video_latents', video_latents.shape)
            h, w = video_latents.shape[-2]*16, video_latents.shape[-1]*16

            face_grid.execute(input_video_path, output_video_path, dwpose_path=dwpose_path, fps=fps, ori_fps=ori_fps, h=h, w=w, n=1, save=True)

            if i>100: break

        
