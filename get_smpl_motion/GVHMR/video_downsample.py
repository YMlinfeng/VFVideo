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

class FaceGrid:
    def __init__(self, model_path=None):
        self.face_idx = [i for i in range(24,92)] 
        self.dwpose_idx_for_face_bbox = [i for i in range(24,92)]
        self.dwpose_idx_for_face_left = [i for i in range(24,33)]+[81,90,86,75,57,54,53,52,51]+[63,64,65,60]
        self.dwpose_idx_for_face_right = [i for i in range(32,41)]+[69,70,71,66]+[51, 52, 53, 54, 57, 75, 86, 90, 81]
        self.dwpose_idx_for_mouth_left = [72,73,74,75,86,90,81,82,83]
        self.dwpose_idx_for_mouth_right = [75,86,90,81,80,79,78,77,76]

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


    def generate_aug_params(self, intensity=0.5):
        """
        生成一组固定的增强参数。
        这些参数将在整个视频序列中保持不变，以防止闪烁。
        """
        if intensity <= 0:
            return {'b': 0, 'c': 1.0, 's': 1.0, 'flip': 0}

        # === 定义最大变化范围 ===
        MAX_BRIGHTNESS_DELTA = 20   
        MAX_CONTRAST_VAR = 0.2      
        MAX_SATURATION_VAR = 0.3   

        # === 计算随机值 ===
        # 亮度
        b_limit = MAX_BRIGHTNESS_DELTA * intensity
        brightness_delta = random.uniform(-b_limit, b_limit)
        
        # 对比度
        c_limit = MAX_CONTRAST_VAR * intensity
        contrast_factor = random.uniform(1.0 - c_limit, 1.0 + c_limit)
        
        # 饱和度
        s_limit = MAX_SATURATION_VAR * intensity
        saturation_factor = random.uniform(1.0 - s_limit, 1.0 + s_limit)

        if random.uniform(0,1)<0.4:
            flip = True
        else:
            flip = False
        return {
            'b': brightness_delta,
            'c': contrast_factor,
            's': saturation_factor,
            'flip': flip
        }

    def apply_stable_augmentation(self, img, params):
        """
        使用预先计算好的参数对图片进行增强。
        """
        brightness_delta = params['b']
        contrast_factor = params['c']
        saturation_factor = params['s']
        flip = params['flip']

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
        
        if flip:
            img = img[:,::-1]

        return img

    
    def calculate_iou(self, bbox1, bbox2):
        """
        计算两个边界框的 IoU (Intersection over Union)。
        
        参数:
            bbox1: [x1, y1, x2, y2] (左上角x, 左上角y, 右下角x, 右下角y)
            bbox2: [x1, y1, x2, y2]
            
        返回:
            float: IoU 值，范围 [0, 1]
        """
        # 1. 解析坐标
        b1_x1, b1_y1, b1_x2, b1_y2 = bbox1
        b2_x1, b2_y1, b2_x2, b2_y2 = bbox2

        # 2. 计算交集区域 (Intersection) 的坐标
        # 交集的左上角是两个框左上角坐标的较大值 (max)
        # 交集的右下角是两个框右下角坐标的较小值 (min)
        inter_x1 = max(b1_x1, b2_x1)
        inter_y1 = max(b1_y1, b2_y1)
        inter_x2 = min(b1_x2, b2_x2)
        inter_y2 = min(b1_y2, b2_y2)

        # 3. 计算交集区域的宽和高
        # 如果 x2 < x1 或 y2 < y1，说明没有重叠，宽度或高度为 0
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        
        inter_area = inter_w * inter_h

        # 如果交集面积为0，直接返回0，避免后续除法运算（虽然分母通常不为0，但这是个好习惯）
        if inter_area == 0:
            return 0.0

        # 4. 计算两个原始框的面积
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        # 5. 计算并集面积 (Union)
        # 并集 = A面积 + B面积 - 交集面积
        union_area = b1_area + b2_area - inter_area

        # 6. 计算 IoU
        # 防止分母为0的情况（极少数情况，例如两个框都是点）
        if union_area == 0:
            return 0.0
            
        iou = inter_area / union_area
        
        return iou


    
    def get_center_crop_face(self, img, landmarks, landmarksi, target_ar, scale0=1.1):
        """
        最简逻辑：计算人脸中心 -> 按比例外扩 -> 直接裁切(超出部分自动丢弃)
        """
        h_img, w_img = img.shape[:2]
        
        # 1. 获取人脸关键点边界
        valid_pts = landmarks[:, :2]
        if valid_pts.shape[0] == 0:
            return img 

        min_x, min_y = np.min(valid_pts, axis=0)
        max_x, max_y = np.max(valid_pts, axis=0)
        if scale0<1:
            min_y = min_y-(max_y-min_y)*0.0
        else:
            min_y = min_y-(max_y-min_y)*0.15

        valid_ptsi = landmarksi[:, :2]
        if valid_ptsi.shape[0] == 0:
            return imgi

        ori_x1, ori_y1 = np.min(valid_ptsi, axis=0)
        ori_x2, ori_y2 = np.max(valid_ptsi, axis=0)
        if scale0<1:
            ori_y1 = ori_y1-(ori_y2-ori_y1)*0.0
        else:
            ori_y1 = ori_y1-(ori_y2-ori_y1)*0.15

        iou = self.calculate_iou([ori_x1,ori_y1,ori_x2, ori_y2], [min_x,min_y,max_x, max_y])
        if iou<0.7:
            valid_pts = valid_ptsi
            min_x,min_y,max_x, max_y = ori_x1,ori_y1,ori_x2, ori_y2
            landmarks = landmarksi

        ori_y1 = ori_y1-(ori_y2-ori_y1)*0.25

        # 人脸中心
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        current_w = max_x - min_x
        current_h = max_y - min_y
        x1, x2, y1, y2 = min_x, max_x, min_y, max_y
        
        current_ar = current_w / current_h
        if current_ar < target_ar:
            final_w = current_w
            final_h = current_w / target_ar
        elif current_ar > target_ar:
            final_h = current_h
            final_w = current_h * target_ar
        else:
            final_w = current_w
            final_h = current_h
        
        #scale = max(current_h/final_h, current_w/final_w)*scale0
        scale = max(current_h/final_h, current_w/final_w*0.85, 1)*scale0

        x1 = cx - final_w / 2
        x2 = cx + final_w / 2
        y1 = cy - final_h / 2
        y2 = cy + final_h / 2
        current_w = x2-x1
        current_h = y2-y1
        cx = (x1+x2)/2
        cy = (y1+y2)/2

        x1 = max(cx-current_w/2*scale, 0)
        x2 = min(cx+current_w/2*scale, w_img)
        y1 = max(cy-current_h/2*scale, 0)
        y2 = min(cy+current_h/2*scale, h_img)
        current_w = x2-x1
        current_h = y2-y1
        cx = (x1+x2)/2
        cy = (y1+y2)/2

        current_ar = current_w / current_h
        if current_ar < target_ar:
            final_w = current_w
            final_h = current_w / target_ar
        elif current_ar > target_ar:
            final_h = current_h
            final_w = current_h * target_ar
        else:
            final_w = current_w
            final_h = current_h
        
        x1 = int(max(cx - final_w / 2, 0))
        x2 = int(min(cx + final_w / 2, w_img))
        y1 = int(max(cy - final_h / 2, 0))
        y2 = int(min(cy + final_h / 2, h_img))

        crop = img[y1:y2, x1:x2]

        return crop, landmarks
        

    
    #----------------------------------------------------------------------------------------------------------------------------------
    def get_bbox_from_dwpose(self, points):

        try:
            # 筛选出置信度大于阈值的点
            filtered_points = points[points[:, 2] > 0.25]

            # 计算边界框
            x1 = np.min(filtered_points[:, 0])  # x 的最小值
            y1 = np.min(filtered_points[:, 1])  # y 的最小值
            x2 = np.max(filtered_points[:, 0])  # x 的最大值
            y2 = np.max(filtered_points[:, 1])  # y 的最大值

            face_bbox = [int(y1), int(x1), int(y2), int(x2)]
        except:
            face_bbox = [0, 0, 0, 0]

        return face_bbox

    def get_face_mask_from_dwpose(self, points, image_shape):  #image_shape: 元组(height, width), 必须提供

        if image_shape is None or len(image_shape) < 2:
            raise ValueError("image_shape must be provided as (height, width)")
        
        # 初始化全0 mask
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        # 筛选出置信度大于阈值的点
        filtered_points = points[points[:, 2] > 0.25]
        
        if len(filtered_points) < 3:
            return mask  # 点数不足，返回全0 mask
        
        # 提取x,y坐标
        pts = filtered_points[:, :2].astype(np.int32)
        
        # 获取轮廓点并填充多边形
        cv2.fillPoly(mask, [pts], color=1)
        
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2).astype('uint8')
        
        return mask

    def bboxes_pad(self, tgt_crop_bbox, h, w, pad_range=1.5):
        y1, x1, y2, x2 = tgt_crop_bbox
        
        # 计算原始边界框的中心点和尺寸
        center_y = (y1 + y2) / 2.0
        center_x = (x1 + x2) / 2.0
        bbox_height = y2 - y1
        bbox_width = x2 - x1

        offset_y = 0-bbox_height*0.12
        center_y = center_y + offset_y
        
        # 按比例外扩
        new_height = bbox_height * pad_range
        new_width = bbox_width * (pad_range-0.15)
        
        # 计算外扩后的边界框坐标
        new_y1 = max(0, center_y - new_height / 2)
        new_y2 = min(h, center_y + new_height / 2)
        new_x1 = max(0, center_x - new_width / 2)
        new_x2 = min(w, center_x + new_width / 2)
        
        # 确保边界框不会超出图像范围
        new_y1 = int(round(new_y1))
        new_x1 = int(round(new_x1))
        new_y2 = int(round(new_y2))
        new_x2 = int(round(new_x2))
        
        return [new_y1, new_x1, new_y2, new_x2]

    def crop_face(self, img0, dwpose_np):
        h,w = img0.shape[:2]

        face_bbox = self.get_bbox_from_dwpose(dwpose_np[0, 0, self.dwpose_idx_for_face_bbox, :])
        face_ori = img0[face_bbox[0]:face_bbox[2], face_bbox[1]:face_bbox[3]]
        

        mask1 = self.get_face_mask_from_dwpose(dwpose_np[0, 0, self.dwpose_idx_for_face_left, :], [h,w])
        mask2 = self.get_face_mask_from_dwpose(dwpose_np[0, 0, self.dwpose_idx_for_face_right, :], [h,w])
        mask3 = self.get_face_mask_from_dwpose(dwpose_np[0, 0, self.dwpose_idx_for_mouth_left, :], [h,w])
        mask4 = self.get_face_mask_from_dwpose(dwpose_np[0, 0, self.dwpose_idx_for_mouth_right, :], [h,w])

        mask1[mask2==1] = 1
        mask1[mask3==1] = 0
        mask1[mask4==1] = 0
        mask1 = mask1[face_bbox[0]:face_bbox[2], face_bbox[1]:face_bbox[3]]

        face_ori = face_ori*mask1

        face_bbox = self.bboxes_pad(face_bbox, h, w)
        face = img0[face_bbox[0]:face_bbox[2], face_bbox[1]:face_bbox[3]]


        return face, face_ori


    def get_id_video_new(self, video_id_0, id_video, img0, dwpose_video_0, dwpose_img):
        
        print ('dwpose_img', dwpose_img.shape)
        print ('dwpose_video_0', dwpose_video_0.shape)
        """
        简化版本，保持三张人脸面积相同
        """
        img0_face, img0_face_ori  = self.crop_face(img0, dwpose_img)
        img1_face, img1_face_ori = self.crop_face(video_id_0, dwpose_video_0)
        
        n = len(id_video)
        transfer_stats = None
        for i in range(n):
            # if i==0:
            #     cv2.imwrite('/ytech_m2v2_hdd/liujiwen/ID_Encoder/motion/m2v-diffusers/get_smpl_motion/debug1_1.png', img0_face_ori[:,:,::-1])
            #     cv2.imwrite('/ytech_m2v2_hdd/liujiwen/ID_Encoder/motion/m2v-diffusers/get_smpl_motion/debug2_1.png', img1_face_ori[:,:,::-1])
            id_video[i], transfer_stats = self.color_transfer_robust(id_video[i], img0_face_ori, img1_face_ori, transfer_stats=transfer_stats)
            # if i==0: cv2.imwrite('/ytech_m2v2_hdd/liujiwen/ID_Encoder/v3/m2v-diffusers/i2v_audio_pre_post_rocessor/pre_processing/debug.png', id_video[i][:,:,::-1])
        
        return id_video

    def color_transfer_robust(self, target_img_real, ref_img, target_img, transfer_stats=None):
        
        # 确定实际要处理的图像
        # 如果传入了 real，就处理 real；否则处理 target_img 自己
        img_to_process = target_img_real if target_img_real is not None else target_img
        
        # 获取实际处理图像的 Mask (用于最后扣除背景)
        process_mask = np.any(img_to_process > 0, axis=2)
        h_proc, w_proc = img_to_process.shape[:2]

        # --- 分支 A: 计算统计量 (使用 ref_img 和 target_img) ---
        if transfer_stats is None:
            print("[ColorTransfer] Calculating stats (Ref vs Target Sample)...")
            
            # 1. 准备数据：归一化并转到 Lab
            # 注意：这里只处理 target_img 用于计算
            tgt_float = target_img.astype(np.float32) / 255.0
            tgt_lab = cv2.cvtColor(tgt_float, cv2.COLOR_RGB2LAB)
            
            ref_float = ref_img.astype(np.float32) / 255.0
            ref_lab = cv2.cvtColor(ref_float, cv2.COLOR_RGB2LAB)
            
            h_ref, w_ref = ref_img.shape[:2]
            h_tgt, w_tgt = target_img.shape[:2]
            
            # 统一大小方便计算 Mask (以 Ref 为基准)
            process_size = (w_ref, h_ref)
            
            if (w_tgt, h_tgt) != process_size:
                tgt_lab_resized = cv2.resize(tgt_lab, process_size, interpolation=cv2.INTER_LINEAR)
                tgt_mask_resized = np.any(cv2.resize(target_img, process_size, interpolation=cv2.INTER_NEAREST) > 0, axis=2)
            else:
                tgt_lab_resized = tgt_lab
                tgt_mask_resized = np.any(target_img > 0, axis=2)
                
            ref_mask = np.any(ref_img > 0, axis=2)
            
            # 计算共同 Mask
            common_mask = ref_mask & tgt_mask_resized
            
            # 生成可视化 Mask (用于调试，看的是计算区域)
            mask_vis = (common_mask.astype(np.uint8) * 255)
            # 如果实际处理图尺寸不同，resize mask 用于显示
            if mask_vis.shape[:2] != (h_proc, w_proc):
                mask_vis = cv2.resize(mask_vis, (w_proc, h_proc), interpolation=cv2.INTER_NEAREST)

            # 采样计算
            if np.sum(common_mask) == 0:
                stats = {'scale': np.ones(3, dtype=np.float32), 'offset': np.zeros(3, dtype=np.float32)}
                return img_to_process, mask_vis, stats

            ref_pixels = ref_lab[common_mask]
            tgt_pixels = tgt_lab_resized[common_mask]
            
            # 抗遮挡筛选
            diff = np.linalg.norm(ref_pixels - tgt_pixels, axis=1)
            num_keep = int(len(diff) * 0.5)
            if num_keep == 0:
                stats = {'scale': np.ones(3, dtype=np.float32), 'offset': np.zeros(3, dtype=np.float32)}
                return img_to_process, mask_vis, stats
                
            keep_indices = np.argsort(diff)[:num_keep]
            ref_samples = ref_pixels[keep_indices]
            tgt_samples = tgt_pixels[keep_indices]
            
            # 计算均值和方差
            ref_mean = np.mean(ref_samples, axis=0)
            ref_std = np.std(ref_samples, axis=0)
            tgt_mean = np.mean(tgt_samples, axis=0)
            tgt_std = np.std(tgt_samples, axis=0)
            tgt_std = np.clip(tgt_std, 1e-6, None)
            
            # 计算线性参数
            scale = ref_std / tgt_std
            offset = ref_mean - (tgt_mean * scale)
            
            stats = {
                'scale': scale,
                'offset': offset
            }

        # --- 分支 B: 复用参数 ---
        else:
            stats = transfer_stats
            # 可视化 Mask 直接展示实际处理图的前景
            mask_vis = (process_mask.astype(np.uint8) * 255)

        # 2. 应用变换 (对 img_to_process 进行操作)
        #    注意：这里处理的是 target_img_real (如果存在)
        proc_float = img_to_process.astype(np.float32) / 255.0
        proc_lab = cv2.cvtColor(proc_float, cv2.COLOR_RGB2LAB)
        
        l, a, b = cv2.split(proc_lab)
        
        scale = stats['scale']
        offset = stats['offset']
        
        l_new = l * scale[0] + offset[0]
        a_new = a * scale[1] + offset[1]
        b_new = b * scale[2] + offset[2]

        l_new = np.clip(l_new, 0, 100)
        a_new = np.clip(a_new, -128, 127)
        b_new = np.clip(b_new, -128, 127)
        
        result_lab = cv2.merge([l_new, a_new, b_new])

        # 3. 后处理
        result_rgb = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB).astype('float32')
        result_rgb = (result_rgb-0.5)*0.8+0.5
        result_rgb = np.clip(result_rgb * 255.0, 0, 255).astype(np.uint8)
        
        return result_rgb, stats

    #----------------------------------------------------------------------------------------------------------------------------------


    def split_list(self, lst, n):
        """将列表lst尽可能均匀地分成n份"""
        k, m = divmod(len(lst), n)
        return [lst[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]


    def img_list_pad(self, source_frames, ref_img, dwpose_np_face, dwpose_np_face_ref, target_n=9):

        flip_list = []
        sacle_list = []
        scale1, scale2 = 1.1, 1.2
        ori_source_frames_n = len(source_frames)
        flip_list = flip_list+[False]*ori_source_frames_n
        sacle_list = sacle_list+[scale1]*ori_source_frames_n
        
        source_frames = source_frames+[ref_img]*3
        dwpose_np_face = np.concatenate([dwpose_np_face]+[dwpose_np_face_ref]*3, axis=0)
        flip_list = flip_list+[False, False, True]
        sacle_list = sacle_list+[scale2,scale2,scale1]

        source_frames = source_frames+source_frames[:ori_source_frames_n]
        dwpose_np_face = np.concatenate([dwpose_np_face, dwpose_np_face[:ori_source_frames_n]], axis=0)
        flip_list = flip_list+[True]*ori_source_frames_n
        sacle_list = sacle_list+[scale1]*ori_source_frames_n

        source_frames = source_frames+source_frames[:ori_source_frames_n]
        dwpose_np_face = np.concatenate([dwpose_np_face, dwpose_np_face[:ori_source_frames_n]], axis=0)
        flip_list = flip_list+[False]*ori_source_frames_n
        sacle_list = sacle_list+[scale1]*ori_source_frames_n

        source_frames = source_frames+source_frames[:ori_source_frames_n]
        dwpose_np_face = np.concatenate([dwpose_np_face, dwpose_np_face[:ori_source_frames_n]], axis=0)
        flip_list = flip_list+[True]*ori_source_frames_n
        sacle_list = sacle_list+[scale1]*ori_source_frames_n

        source_frames = source_frames+source_frames[:ori_source_frames_n]
        dwpose_np_face = np.concatenate([dwpose_np_face, dwpose_np_face[:ori_source_frames_n]], axis=0)
        flip_list = flip_list+[False]*ori_source_frames_n
        sacle_list = sacle_list+[scale1]*ori_source_frames_n

        source_frames = source_frames+[ref_img]*3
        dwpose_np_face = np.concatenate([dwpose_np_face]+[dwpose_np_face_ref]*3, axis=0)
        flip_list = flip_list+[True, True, True]
        sacle_list = sacle_list+[scale1,scale1,scale1]

        return source_frames[:9], dwpose_np_face[:9], flip_list[:9], sacle_list[:9]



    def replace_spatial_grid_by_index_list(self, final_video_array, final_video_array_ori, index_list):
        result_array = final_video_array.copy()
        index_list = list(index_list)
        H = result_array.shape[1]
        W = result_array.shape[2]
        h_step = H // 3
        w_step = W // 3
        for idx in index_list:
            row = idx // 3
            col = idx % 3
            x1 = row * h_step
            x2 = (row + 1) * h_step
            y1 = col * w_step
            y2 = (col + 1) * w_step
            result_array[:, x1:x2, y1:y2] = final_video_array_ori[:, x1:x2, y1:y2]
            
        return result_array

    def compose_face_grid_frames_stable(self, source_frames, ref_img, dwpose_np, dwpose_np_ref, h, w, scale=1.15, aug_intensity=1,is_image=False):
        """
        合成人脸九宫格视频 + 无闪烁(Stable)的可调节增强
        """
        #0. id视频和首帧图进行脸部颜色校准

        video_id_0 = source_frames[0].copy()
        
        dwpose_np_face = dwpose_np[:,0,self.face_idx]
        dwpose_np_face_ref = dwpose_np_ref[:1,0,self.face_idx]

        ori_img_idx_list = []
        # 1. 数据切分为9份
        if is_image:
            source_frames = [self.get_id_video_new(source_frames[i], [source_frames[i]], ref_img, dwpose_np[i:i+1], dwpose_np_ref[0:1])[0] for i in range(len(source_frames))]
            print ('len(source_frames)', len(source_frames))
            if len(source_frames)<9:
                if len(source_frames)+1<9: ori_img_idx_list.append(len(source_frames)+1)
                if len(source_frames)+2<9: ori_img_idx_list.append(len(source_frames)+2)
                source_frames, dwpose_np_face, flip_list, sacle_list = self.img_list_pad(source_frames, ref_img, dwpose_np_face, dwpose_np_face_ref, target_n=9)

            grid_sources = self.split_list(source_frames, 9)
        else:
            grid_sources = self.split_list(source_frames, 9)
            flip_list = [False]*9
            sacle_list = [1.0]*9
            sacle_list[1] = 1.15
            sacle_list[4] = 0.9
        
        grid_landmarks = np.split(dwpose_np_face, 9)
        
        target_length = len(grid_sources[0])

        
        # 九宫格单格大小
        cell_w = w // 3
        cell_h = h // 3
        
        # 最终画布
        final_h = h
        final_w = w
        final_video_array = np.zeros((target_length, final_h, final_w, 3), dtype=np.uint8)
        
        # =========================================================
        # 关键点：在开始处理视频帧之前，为9个格子分别生成固定的参数
        # =========================================================
        cell_params_list = []
        for k in range(9):
            # 第k个格子获得一套专属的参数
            params = self.generate_aug_params(intensity=aug_intensity)
            cell_params_list.append(params)
            # 打印一下参数，方便调试查看每个格子的风格

        cells_idx_list = [j for j in range(9)]
        if is_image:
            random.shuffle(cells_idx_list)
            if len(ori_img_idx_list)==2:
                a, b = ori_img_idx_list[0], ori_img_idx_list[1]
                cells_idx_list = [i for i in range(9) if i not in [a,b]]
                random.shuffle(cells_idx_list)
                cells_idx_list.insert(1, a)
                cells_idx_list.insert(4, b)
            if len(ori_img_idx_list)==1:
                a = ori_img_idx_list[0]
                cells_idx_list = [i for i in range(9) if i not in [a]]
                random.shuffle(cells_idx_list)
                cells_idx_list.insert(1, a)
            
        #lms_idx = 0 #len(grid_landmarks[0])//2 #random.randint(0,len(grid_landmarks[0])-1)
        for i in range(target_length):
            cells = []
            for k in cells_idx_list:
                img = grid_sources[k][i]
                lms = grid_landmarks[k][0]
                lms_i = grid_landmarks[k][i]
                # --- 裁剪与缩放逻辑 (模拟) ---
                face_crop, lms = self.get_center_crop_face(img, lms, lms_i, target_ar=cell_w/cell_h, scale0=sacle_list[k]) 
                grid_landmarks[k][0] = lms
                resized_face = cv2.resize(face_crop, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
                
                # --- 核心逻辑: 应用固定的参数 ---
                # 每一帧 i，第 k 个格子都使用 cell_params_list[k]
                # 因为参数不变，所以画面不会闪烁
                #augmented_face = self.apply_stable_augmentation(resized_face, cell_params_list[k])
                if flip_list[k]:resized_face = resized_face[:,::-1]
                cells.append(resized_face)
                
            # 拼接
            row1 = np.hstack(cells[0:3])
            row2 = np.hstack(cells[3:6])
            row3 = np.hstack(cells[6:9])

            final_video_array[i][:cell_h*3, :cell_w*3] = np.vstack((row1, row2, row3))
        
        if not is_image:
            final_video_array_ori = final_video_array.copy()
            final_video_array = self.get_id_video_new(video_id_0, final_video_array, ref_img, dwpose_np[0:1], dwpose_np_ref[0:1])
            final_video_array = self.replace_spatial_grid_by_index_list(final_video_array, final_video_array_ori, index_list=[1])

        if target_length==1:
            final_video_array = np.concatenate([final_video_array]*9, axis=0)

        return final_video_array

    # def read_video_frames(self, id_video_path, id_image_path_list, n):
    #     #视频id参考
    #     if id_video_path is not None:
    #         is_image = False
    #         vr = VideoReader(id_video_path, ctx=cpu(0))

    #         original_frames_count = len(vr)
    #         fps = vr.get_avg_fps()
            
    #         target_length = 8 * n + 1
    #         total_samples = target_length * 9
            
    #         indices = np.linspace(1, original_frames_count - 1, total_samples, dtype=int)
    #         #source_frames_batch = vr.get_batch(indices).numpy()
    #         source_frames_batch = vr.get_batch(indices).asnumpy()
            
    #     #图像id参考
    #     else:
    #         is_image = True
    #         source_frames_batch = [cv2.imread(img_path)[:,:,::-1] for img_path in id_image_path_list]
    #         fps = 1
    #         indices = [i for i in range(len(source_frames_batch))]

    #     return source_frames_batch, fps, indices, is_image

    def read_video_frames(self, id_video_path, id_image_path_list, n, start_idx=None, end_idx=None):

        #视频id参考
        if id_video_path is not None:
            is_image = False
            vr = VideoReader(id_video_path, ctx=cpu(0))

            original_frames_count = len(vr)
            fps = vr.get_avg_fps()

            print ('++++++++++++++++++++start_idx, end_idx, original_frames_count', start_idx, end_idx, original_frames_count)
            
            # 如果提供了start_idx和end_idx，则使用校验后的有效范围
            if start_idx is not None and end_idx is not None:
                # 确保索引在有效范围内
                start_idx = max(0, min(start_idx, original_frames_count - 1))
                end_idx = max(0, min(end_idx, original_frames_count - 1))
                # 计算有效帧范围
                valid_frame_count = end_idx - start_idx + 1
            else:
                start_idx = 1
                valid_frame_count = original_frames_count - 1
            
            target_length = 8 * n + 1
            total_samples = target_length * 9
            
            # 在有效范围内进行采样
            indices = np.linspace(start_idx, start_idx + valid_frame_count - 1, total_samples, dtype=int)
            # 确保索引不越界
            indices = np.clip(indices, 0, original_frames_count - 1)
            
            source_frames_batch = vr.get_batch(indices).asnumpy()

            del vr
            
        #图像id参考
        else:
            is_image = True
            # 图片模式：读取图片列表（支持numpy数组或文件路径）
            source_frames_batch = []
            for img_item in id_image_path_list:
                if isinstance(img_item, np.ndarray):
                    # 如果已经是numpy数组（来自execute_input_validation的image_list）
                    # print(f"[Debug]img_item.shape:{img_item.shape}")
                    # cv2.imwrite("/ytech_m2v4_hdd/mengzijie/get_smpl_motion/debug_ori/debug.png", img_item[:,:,::-1])
                    source_frames_batch.append(img_item)
                elif isinstance(img_item, str):
                    # 如果是文件路径
                    img = cv2.imread(img_item)
                    if img is not None:
                        source_frames_batch.append(img[:,:,::-1])  # BGR to RGB
            
            fps = 1
            indices = [i for i in range(len(source_frames_batch))]

        return source_frames_batch, fps, indices, is_image




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
            

    def json_to_keypoints_matrix(self, json_path):
        
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


    # --- 主流程串联 ---
    def execute(self, input_path, output_path, dwpose_path, fps, ori_fps, h, w, n):
        # 1. 读
        raw_frames, fps, indices = self.read_video_frames(input_path, n)
        ori_h, ori_w = raw_frames[0].shape[:2]   
        dwpose_np = self.load_pose(dwpose_path, ori_h, ori_w, ori_fps, fps)
        indices[indices>dwpose_np.shape[0]-1] = dwpose_np.shape[0]-1
        dwpose_np_face = dwpose_np[:,0,self.face_idx]
        dwpose_np_face = dwpose_np_face[indices]

        # 2. 算
        result_frames = self.compose_face_grid_frames_stable(raw_frames, dwpose_np_face, h, w)
        
        if False: #random.uniform(0,1)<0.05:
            # 3. 写 (使用新的 imageio 逻辑)
            self.save_video_from_frames(result_frames, output_path, fps=8)

        return result_frames

if __name__ == "__main__":

    
    face_grid = FaceGrid()
    #csv_file = "/ytech_m2v2_hdd/liujiwen/audio_v3/m2v-diffusers/id_data_480_720_1080_with_pose/filtered_AVspeech_data_human_check_8dot3w_0804_with_md5_vae_caption_te_193f_2d5w.csv"
    csv_file = "/ytech_m2v2_hdd/liujiwen/audio_v3/m2v-diffusers/id_data_480_720_1080_with_pose/720_filter_emo_40w_30fps_split_121f_with_md5_vae_caption_te_0925_40w_fix_pose.csv"

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

            output_video_path = './show/output_'+str(i)+'.mp4'
            print ('input_video_path', input_video_path)


            latent_dict = torch.load(vae_latent_path, map_location="cpu")
            video_latents = latent_dict["video_data"][0]   # [(b C f h w)] x L
            print ('video_latents', video_latents.shape)
            h, w = video_latents.shape[-2]*16, video_latents.shape[-1]*16

            face_grid.execute(input_video_path, output_video_path, dwpose_path=dwpose_path, fps=fps, ori_fps=ori_fps, h=h, w=w, n=1)

            if i>500: break

