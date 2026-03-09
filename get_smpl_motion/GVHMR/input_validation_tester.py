# -*- coding: utf-8 -*-
"""
简化版输入校验测试类
仅保留execute_input_validation函数的核心逻辑，用于批量测试
"""

import os
from random import sample
from tracemalloc import start
import cv2
import numpy as np
from decord import VideoReader, cpu
from .hmr4d.utils.preproc import Tracker
from .dwpose_tools.dwpose import DWposeDetector


# 输入校验错误码
class InputErrorCode:
    SUCCESS: int = 0                    # 校验通过
    INVALID_FILE_TYPE: int = 1          # 输入文件类型错误（读取失败）
    RESOLUTION_TOO_SMALL: int = 2       # 分辨率过小
    RESOLUTION_TOO_LARGE: int = 3       # 分辨率过大
    IMAGE_LIST_TOO_LONG: int = 4        # 图片列表超过8张
    VIDEO_TOO_SHORT_OR_LONG: int = 5            # 视频过短（小于1秒）
    NO_FACE_DETECTED: int = 6           # 未检测到人脸
    NO_CLEAR_SUBJECT: int = 7           # 没有明显的主体


# 错误消息映射
INPUT_ERROR_MESSAGES = {
    InputErrorCode.SUCCESS: "校验通过",
    InputErrorCode.INVALID_FILE_TYPE: "文件类型错误，读取失败",
    InputErrorCode.RESOLUTION_TOO_SMALL: "分辨率过小，短边需大于128像素",
    InputErrorCode.RESOLUTION_TOO_LARGE: "分辨率过大，长边需小于4096像素",
    InputErrorCode.IMAGE_LIST_TOO_LONG: "图片列表过长，最多8张",
    InputErrorCode.VIDEO_TOO_SHORT_OR_LONG: "视频过短，最少1秒",
    InputErrorCode.NO_FACE_DETECTED: "未检测到人脸",
    InputErrorCode.NO_CLEAR_SUBJECT: "没有明显的主体",
}


# 任务类型
class TaskType:
    IMAGE_LIST = "image_list"
    VIDEO = "video"

class InputValidationTester:
    """
    简化版输入校验测试类
    仅用于测试execute_input_validation函数功能
    """
    
    def __init__(self, smpl_checkpoints_path='/ytech_milm/liujiwen/kling_motion_service/smpl_all_checkpoints',
                 min_resolution=127,
                 max_resolution=9060,
                 max_image_list_length=8,
                 min_video_duration=0.8,
                 max_video_duration=120.0,
                 sample_fps=1,
                 face_confidence_threshold=5,
                 subject_area_ratio=1.5):
        """初始化测试器"""
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.max_image_list_length = max_image_list_length
        self.min_video_duration = min_video_duration
        self.max_video_duration = max_video_duration
        self.sample_fps = sample_fps
        self.face_confidence_threshold = face_confidence_threshold
        self.subject_area_ratio = subject_area_ratio
        
        # 初始化检测器
        self.tracker = Tracker(smpl_checkpoints_path + '/checkpoints/yolo/yolo11x.pt')
        pose_config = smpl_checkpoints_path + '/checkpoints/rtmw-x/rtmw-x_8xb320-270e_cocktail14-384x288.py'
        pose_ckpt = smpl_checkpoints_path + '/checkpoints/rtmw-x/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth'
        self.xdwpose = DWposeDetector(pose_config, pose_ckpt, device='cuda')
    
    def _get_dwpose_np(self, img_list):
        """从图片列表获取dwpose结果"""
        res = []
        det_res = []
        for img in img_list:
            h, w = img.shape[:2]
            det_results = self.tracker.detect(img)
            bbox = [0, 0, w, h] if len(det_results) == 0 else det_results[0][0]
            
            xdwpose_result = self.xdwpose(
                image_np_hwc=img,
                show_body=False,
                show_face=False,
                show_hands=True,
                plot=False,
                box_ext=np.array([bbox])
            )
            dwpose = np.concatenate([xdwpose_result[0], xdwpose_result[1][:, :, np.newaxis]], axis=2)
            res.append(dwpose)
            det_res.append(det_results)
        return np.array(res), det_res

    def execute_input_validation(self, image_list=None, video_path=None):
        result = {
            'task_type': TaskType.VIDEO if video_path else TaskType.IMAGE_LIST,
            'image_list': None,
            'start_idx': None,
            'end_idx': None,
            'error_code': InputErrorCode.SUCCESS,
            'error_message': '',
            'removed_frames': []
        }
        
        # ========== 第一步：读取输入，统一转为image_list ==========
        vr = None
        sample_indices = None
        fps = 30.0
        
        if video_path:
            # 视频模式：读取视频
            try:
                vr = VideoReader(video_path, ctx=cpu(0))
                fps = vr.get_avg_fps()
                total_frames = len(vr)
            except Exception as e:
                result['error_code'] = InputErrorCode.INVALID_FILE_TYPE
                result['error_message'] = f"{INPUT_ERROR_MESSAGES[InputErrorCode.INVALID_FILE_TYPE]}: {e}"
                return result
            
            # 检查视频时长
            duration = total_frames / fps
            print ('total_frames, fps, duration', total_frames, fps, duration)
            if total_frames<4 or duration < self.min_video_duration or duration > self.max_video_duration:
                result['error_code'] = InputErrorCode.VIDEO_TOO_SHORT_OR_LONG
                result['error_message'] = INPUT_ERROR_MESSAGES[InputErrorCode.VIDEO_TOO_SHORT_OR_LONG]
                return result
            
            # 检查分辨率
            first_frame = vr[0].asnumpy()
            h, w = first_frame.shape[:2]
            if min(h, w) < self.min_resolution:
                result['error_code'] = InputErrorCode.RESOLUTION_TOO_SMALL
                result['error_message'] = INPUT_ERROR_MESSAGES[InputErrorCode.RESOLUTION_TOO_SMALL]
                return result
            if max(h, w) > self.max_resolution:
                result['error_code'] = InputErrorCode.RESOLUTION_TOO_LARGE
                result['error_message'] = INPUT_ERROR_MESSAGES[InputErrorCode.RESOLUTION_TOO_LARGE]
                return result
            
            
            # 采样帧用于校验
            sample_n = int(max(duration/1, 10))
            sample_indices = np.linspace(1, min(30*fps, total_frames-1), sample_n, dtype=int).tolist()
            # print(f"sample_indices:{sample_indices}")
            # print(type(sample_indices))
            # print(sample_indices.shape)

            
            sampled_frames = vr.get_batch(sample_indices).asnumpy()
            del vr
            image_list = [sampled_frames[i] for i in range(len(sampled_frames))]
            
        else:
            # 图片模式：直接使用image_list
            if not image_list or len(image_list) == 0:
                result['error_code'] = InputErrorCode.INVALID_FILE_TYPE
                result['error_message'] = INPUT_ERROR_MESSAGES[InputErrorCode.INVALID_FILE_TYPE]
                return result
            
            if len(image_list) > self.max_image_list_length:
                result['removed_frames'].append({'reason': f'TRUNCATED: {len(image_list) - self.max_image_list_length} images', 'error_code': InputErrorCode.IMAGE_LIST_TOO_LONG})
                image_list = image_list[:self.max_image_list_length]
            # 过滤无效图片和分辨率不符的图片
            valid_list = []
            sample_indices = []
            for i, img in enumerate(image_list):
                # try:
                img = cv2.imread(img)
                if img is None:
                    result['removed_frames'].append({'idx': i, 'reason': 'INVALID', 'error_code': InputErrorCode.INVALID_FILE_TYPE})
                    continue
                img = img[:,:,::-1] 
                # except Exception as e:

                #     result['removed_frames'].append({'idx': i, 'reason': 'INVALID', 'error_code': InputErrorCode.INVALID_FILE_TYPE})
                #     continue
                h, w = img.shape[:2]
                if min(h, w) < self.min_resolution:
                    result['removed_frames'].append({'idx': i, 'reason': 'TOO_SMALL', 'error_code': InputErrorCode.RESOLUTION_TOO_SMALL})
                    continue
                if max(h, w) > self.max_resolution:
                    result['removed_frames'].append({'idx': i, 'reason': 'TOO_LARGE', 'error_code': InputErrorCode.RESOLUTION_TOO_LARGE})
                    continue
                valid_list.append(img)
                sample_indices.append(i) #[0,2,4,7,6,8]
            image_list = valid_list
        
        if not image_list:
            result['error_code'] = InputErrorCode.INVALID_FILE_TYPE
            return result
        
        # ========== 第二步：人脸检测 ==========
        dwpose_results, det_list = self._get_dwpose_np(image_list)
        face_conf = dwpose_results[:, 0, 24:92, 2]  # 人脸关键点置信度
        # valid_face = np.min(face_conf, axis=1) > self.face_confidence_threshold
        valid_face = np.mean(face_conf, axis=1) > self.face_confidence_threshold
        if not np.any(valid_face):
            result['error_code'] = InputErrorCode.NO_FACE_DETECTED
            result['error_message'] = INPUT_ERROR_MESSAGES[InputErrorCode.NO_FACE_DETECTED]
            return result
        
        # 记录无人脸的帧
        for i, valid in enumerate(valid_face):
            if not valid:
                # sample是正确下标
                frame_info = {'idx': sample_indices[i] if sample_indices else i, 'reason': 'NO_FACE', 'error_code': InputErrorCode.NO_FACE_DETECTED}
                result['removed_frames'].append(frame_info)
        
        # 找有效帧范围
        valid_indices = np.where(valid_face)[0]
        # print(f"Valid face frames indices: {valid_indices}")
        start_valid, end_valid = valid_indices[0], valid_indices[-1]
        # print(f"Valid face frames time range: {start_valid} to {end_valid}") # 0 to 9

        # ========== 第三步：主体检测 ==========
        valid_subject = []
        subject_start = None
        subject_end = None
        
        for i in range(len(image_list)):
            det = det_list[i]
            if len(det) >= 2 and det[0][1] < det[1][1] * self.subject_area_ratio:
                valid_subject.append(False)
                frame_info = {'idx': sample_indices[i] if sample_indices else i, 'reason': 'NO_CLEAR_SUBJECT', 'error_code': InputErrorCode.NO_CLEAR_SUBJECT}
                result['removed_frames'].append(frame_info)
            else:
                valid_subject.append(True)
                if subject_start is None:
                    subject_start = i
                subject_end = i
        
        if subject_start is None:
            result['error_code'] = InputErrorCode.NO_CLEAR_SUBJECT
            result['error_message'] = INPUT_ERROR_MESSAGES[InputErrorCode.NO_CLEAR_SUBJECT]
            return result
        
        start_valid = max(start_valid, subject_start)
        end_valid = min(end_valid, subject_end)
        
        if start_valid > end_valid:
            result['error_code'] = InputErrorCode.NO_CLEAR_SUBJECT  
            result['error_message'] = INPUT_ERROR_MESSAGES[InputErrorCode.NO_CLEAR_SUBJECT]
            return result
        
        # ========== 第四步：生成最终image_list ==========
        final_mask = [f and s for f, s in zip(valid_face, valid_subject)]
        image_list = [img for img, valid in zip(image_list, final_mask) if valid]
        if not image_list:
            result['error_code'] = InputErrorCode.NO_CLEAR_SUBJECT  
            result['error_message'] = INPUT_ERROR_MESSAGES[InputErrorCode.NO_CLEAR_SUBJECT]
            return result
        
        result['image_list'] = image_list 
        
        if video_path:
            # 记录视频模式下对应的原始帧索引
            result['start_idx'] = sample_indices[start_valid]
            result['end_idx'] = sample_indices[end_valid]
        
        result['error_message'] = INPUT_ERROR_MESSAGES[InputErrorCode.SUCCESS]
        return result