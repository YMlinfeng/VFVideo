# -*- coding: utf-8 -*-

import os
import cv2
import torch
# import pytorch_lightning as pl
import numpy as np
import argparse
import hydra
from hydra import initialize_config_module, compose
from pytorch3d.transforms import quaternion_to_matrix

# 在解析路径时，显式处理 Unicode
from pathlib import Path

from .hmr4d.configs import register_store_gvhmr
from .hmr4d.utils.video_io_utils import (
    save_video,
    merge_videos_horizontal,
)
from .hmr4d.utils.vis.cv2_utils import (
    draw_bbx_xyxy_on_image_batch,
    draw_coco17_skeleton_batch,
)

from .hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SLAMModel  #很慢

from .hmr4d.utils.geo.hmr_cam import (
    get_bbx_xys_from_xyxy,
    estimate_K,
    convert_K_to_K4,
    create_camera_sensor,
)
from .hmr4d.utils.geo_transform import compute_cam_angvel
from .hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from .hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from .hmr4d.utils.vis.renderer import (
    Renderer,
    get_global_cameras_static,
    get_ground_params_from_points,
)
from tqdm import tqdm
from .hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from einops import einsum, rearrange
import json
import time

from .text2motion.run_t2m import text2motion
from .bvh2smplx.bvh2smplx_sample import bvh_to_smplx
from .text2motion.visualization.smpl2bvh import smplx2bvh

from decord import VideoReader,cpu
import imageio.v2 as imageio
import subprocess
import io
from .dwpose_tools.dwpose import DWposeDetector
import copy
import uuid
import pickle
import json
from .retarget_axis_t2m import bvh2hmr, hmr2bvh, gv_global2bvh

# 引入wilor相关
from .wilor.models.backbones.vit import ViT
from .wilor.configs import get_config
from .wilor.datasets.vitdet_dataset import ViTDetDataset
from .wilor.datasets.utils import (convert_cvimg_to_tensor,
                                    expand_to_aspect_ratio,
                                    generate_image_patch_cv2)
from .wilor.utils import recursive_to
from .wilor.models.heads import RefineNet
from .wilor.models.mano_wrapper import MANO
from .wilor.mano2smplx import convert_mano_to_smplx
from collections import OrderedDict
from scipy.spatial.transform import Rotation as R
import math
from skimage.filters import gaussian
from ultralytics import YOLO

from .video_downsample import FaceGrid
from  .pose_filter  import DwposeRefine
import random

USE_FPS = 30 #28

class WilorCrop:
    def __init__(self,
                 cfg,
                 img_cv2: np.array,
                 boxes: np.array,
                 right: np.array,
                 rescale_factor=2.5,
                 train: bool = False,
                 **kwargs):
        self.cfg = cfg
        self.img_cv2 = img_cv2
        # self.boxes = boxes

        assert train == False, "ViTDetDataset is only for inference"
        self.train = train
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        # Preprocess annotations
        boxes = boxes.astype(np.float32)
        self.center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        self.scale = rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        self.personid = np.arange(len(boxes), dtype=np.int32)
        self.right = right.astype(np.float32)

    def get_img(self, idx: int):

        center = self.center[idx].copy()
        center_x = center[0]
        center_y = center[1]

        scale = self.scale[idx]
        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()

        patch_width = patch_height = self.img_size

        right = self.right[idx].copy()
        flip = right == 0

        # 3. generate image patch
        # if use_skimage_antialias:
        cvimg = self.img_cv2.copy()
        if True:
            # Blur image to avoid aliasing artifacts
            downsampling_factor = ((bbox_size*1.0) / patch_width)
            #print(f'{downsampling_factor=}')
            downsampling_factor = downsampling_factor / 2.0
            if downsampling_factor > 1.1:
                cvimg  = gaussian(cvimg, sigma=(downsampling_factor-1)/2, channel_axis=2, preserve_range=True)


        img_patch_cv, trans = generate_image_patch_cv2(cvimg,
                                                    center_x, center_y,
                                                    bbox_size, bbox_size,
                                                    patch_width, patch_height,
                                                    flip, 1.0, 0,
                                                    border_mode=cv2.BORDER_CONSTANT)
        img_patch_cv = img_patch_cv[:, :, ::-1]
        img_patch = convert_cvimg_to_tensor(img_patch_cv)

        # apply normalization
        for n_c in range(min(self.img_cv2.shape[2], 3)):
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        return torch.tensor(img_patch)

# 用户上传的图片错误码
class ImageErrorCode:
    SUCCESS: int = 0  # 校验通过
    RESOLUTION_TOO_SMALL: int = 1  # 短边<=510
    RESOLUTION_TOO_LARGE: int = 2  # 长边>=2560
    NO_COMPLETE_UPPER_BODY: int = 3  # 没有检测到任何完整的上半身
    BODY_AREA_TOO_SMALL: int = 4  # 人体面积<=全图25%
    MULTIPLE_PEOPLE_IOU_TOO_HIGH: int = 5  # 多人检测框IOU>=0.2 和其他人有明显重叠
    TOO_MANY_SUBJECTS: int = 7  # 主体过多
    OTHER_ERRORS: int = 6  # 其他未定义错误


# 驱动信号错误码
class VideoErrorCode:
    SUCCESS: int = 0  # 校验通过
    RESOLUTION_TOO_SMALL: int = 1  # 短边<=510
    RESOLUTION_TOO_LARGE: int = 2  # 长边>=2560
    NO_COMPLETE_UPPER_BODY: int = 3  # 没有检测到任何完整的上半身
    NO_CLEAR_MAIN_BODY: int = 4  # 首帧无明确主体（主主体面积<=25% 或 次主体面积>=主体1/3）
    POSE_DIFFERENCE_TOO_LARGE: int = 5  # 首帧与输入图片姿态差异过大 例如驱动信号是站立的，被驱动的事躺着的
    FRAME_DROP_EXCEEDED: int = 6  # 连续丢帧数>=5，运动过快，太模糊等情况下导致检测不到，大量丢帧
    FRAME_IOU_TOO_LOW: int = 7  # 帧间检测框IOU<=0.1 人体运动过快，位置有跳变
    MOTION_DURATION_TOO_SHORT: int = 8  # 连续有效动作时长<=1s
    MOTION_DURATION_TOO_LONG: int = 9  # 连续有效动作时长>50s
    MOTION_TEXT_INVALID_INPUT: int=10 # text2motion输入非法
    MOTION_BVH_INVALID_INPUT: int=11 # bvh2motion输入非法
    OTHER_ERRORS: int = 12  # 其他未定义错误（兜底策略）

# 图片错误码与错误消息的映射
IMAGE_ERROR_MESSAGES = {
    ImageErrorCode.SUCCESS: "Validation passed",
    ImageErrorCode.RESOLUTION_TOO_SMALL: "Image resolution is too low; the shorter side must be greater than 510 pixels",
    ImageErrorCode.RESOLUTION_TOO_LARGE: "Image resolution is too high; the longer side must be less than 2560 pixels",
    ImageErrorCode.NO_COMPLETE_UPPER_BODY: "No complete upper body detected; ensure the upper body is clearly visible",
    ImageErrorCode.BODY_AREA_TOO_SMALL: "The body area is too small; it should occupy at least 25% of the image",
    ImageErrorCode.MULTIPLE_PEOPLE_IOU_TOO_HIGH: "Multiple detection boxes overlap too much; ensure the main subject is clear and non-overlapping",
    ImageErrorCode.TOO_MANY_SUBJECTS: "Too many subjects detected; please ensure there are no more than 2 clear subjects",
    ImageErrorCode.OTHER_ERRORS: "Other undefined errors; please contact technical support",
}

# 视频错误码与错误消息的映射
VIDEO_ERROR_MESSAGES = {
    VideoErrorCode.SUCCESS: "Validation passed",
    VideoErrorCode.RESOLUTION_TOO_SMALL: "Video resolution is too low; the shorter side must be greater than 510 pixels",
    VideoErrorCode.RESOLUTION_TOO_LARGE: "Video resolution is too high; the longer side must be less than 2560 pixels",
    VideoErrorCode.NO_COMPLETE_UPPER_BODY: "No complete upper body detected; ensure the upper body is clearly visible",
    VideoErrorCode.NO_CLEAR_MAIN_BODY: "No clear main subject in the first frame; ensure the main subject occupies at least 25% of the frame and the secondary subject is no more than 1/3 of the main subject",
    VideoErrorCode.POSE_DIFFERENCE_TOO_LARGE: "The pose difference between the first frame and the input image is too large; ensure the driving signal matches the pose of the subject",
    VideoErrorCode.FRAME_DROP_EXCEEDED: "Excessive frame drops; ensure the video is clear and the motion speed is moderate",
    VideoErrorCode.FRAME_IOU_TOO_LOW: "The intersection-over-union (IOU) between frames is too low; ensure the subject's motion speed is moderate",
    VideoErrorCode.MOTION_DURATION_TOO_SHORT: "The duration of continuous valid motion is too short; it should last at least 1 second",
    VideoErrorCode.MOTION_DURATION_TOO_LONG: "The duration of continuous valid motion is too long; it should not exceed 50 seconds",
    VideoErrorCode.MOTION_TEXT_INVALID_INPUT: "The text input is not valid.",
    VideoErrorCode.MOTION_BVH_INVALID_INPUT: "The bvh file input is not valid ",
    VideoErrorCode.OTHER_ERRORS: "Other undefined errors; please contact technical support",
}


# 输入校验错误码（用于execute函数）
class InputErrorCode:
    SUCCESS: int = 0                    # 校验通过
    INVALID_FILE_TYPE: int = 1          # 输入文件类型错误（读取失败）
    RESOLUTION_TOO_SMALL: int = 2       # 分辨率过小
    RESOLUTION_TOO_LARGE: int = 3       # 分辨率过大
    IMAGE_LIST_TOO_LONG: int = 4        # 图片列表超过8张
    VIDEO_TOO_SHORT: int = 5            # 视频过短（小于1秒）
    NO_FACE_DETECTED: int = 6           # 未检测到人脸
    NO_CLEAR_SUBJECT: int = 7           # 没有明显的主体（多人且面积差异不够大）


# 输入校验错误码与错误消息的映射
INPUT_ERROR_MESSAGES = {
    InputErrorCode.SUCCESS: "Validation passed",
    InputErrorCode.INVALID_FILE_TYPE: "Invalid file type; failed to read the file",
    InputErrorCode.RESOLUTION_TOO_SMALL: "Resolution too small; the shorter side must be greater than 128 pixels",
    InputErrorCode.RESOLUTION_TOO_LARGE: "Resolution too large; the longer side must be less than 4096 pixels",
    InputErrorCode.IMAGE_LIST_TOO_LONG: "Image list too long; maximum 8 images allowed, extra images removed",
    InputErrorCode.VIDEO_TOO_SHORT: "Video too short; minimum 1 second required",
    InputErrorCode.NO_FACE_DETECTED: "No face detected; ensure face keypoints are clearly visible",
    InputErrorCode.NO_CLEAR_SUBJECT: "No clear subject; the largest person's area should be at least 1.5x the second largest",
}

# 任务类型枚举
class TaskType:
    IMAGE_LIST = "image_list"
    VIDEO = "video"


def find_closest_number(lst, a):
    if not lst:
        return None
    return min(lst, key=lambda x: abs(x - a))

def rotate_pose_90(pose, k, image_shape):
    """
    将pose旋转k * 90度。
    k: 旋转次数，1表示90度，2表示180度，3表示270度。
    image_shape: 图像的形状 (h, w)。
    """
    h, w = image_shape
    rotated_pose = []
    for point in pose:
        x, y = point
        if k == 1:  # 90度
            rotated_pose.append([y, w - x])
        elif k == 2:  # 180度
            rotated_pose.append([w - x, h - y])
        elif k == 3:  # 270度
            rotated_pose.append([h - y, x])
        else:  # 0度
            rotated_pose.append([x, y])
    return rotated_pose

def calculate_orientation(pose, head_id=0, left_shoulder_id=8, right_shoulder_id=11):
    """
    计算人体朝向。
    返回值：0（竖直），1（向右），2（倒立），3（向左）。
    """
    head = pose[head_id]
    left_shoulder = pose[left_shoulder_id]
    right_shoulder = pose[right_shoulder_id]

    # 计算肩部向量
    shoulder_vector = np.array(right_shoulder) - np.array(left_shoulder)
    shoulder_angle = np.arctan2(shoulder_vector[1], shoulder_vector[0])  # 肩部向量与x轴的夹角

    # 计算头部到肩部中点的向量
    shoulder_mid = (np.array(left_shoulder) + np.array(right_shoulder)) / 2
    head_to_shoulder_vector = shoulder_mid-np.array(head)
    head_angle = np.arctan2(head_to_shoulder_vector[1], head_to_shoulder_vector[0])/3.14159*180-90  # 头部向量与x轴的夹角

    # 计算人体朝向
    angle_diff = head_angle  # 角度差
    if abs(angle_diff) <45:
        return 0  # 竖直
    elif angle_diff>=45 and angle_diff<45*3:
        return 1  # 向左
    elif angle_diff<-45 and angle_diff>= -45*3:
        return 3  # 向右
    else:
        return 2  # 倒立

def find_best_rotation(prev_pose, image_shape):
    """
    根据上一帧的pose，找到最佳旋转角度。
    """
    best_k = 0
    best_orientation = 0  # 默认竖直方向

    # 尝试三种旋转角度（90, 180, 270）
    for k in range(0, 4):
        rotated_pose = rotate_pose_90(prev_pose, k, image_shape)
        orientation = calculate_orientation(rotated_pose)
        if orientation == 0:  # 如果朝向竖直，直接选择
            best_k = k
            break
        # elif orientation < best_orientation:
        #     best_k = k
        #     best_orientation = orientation

    return best_k

def rotate_image_90(image, k):
    """
    将图像旋转k * 90度。
    k: 旋转次数，1表示90度，2表示180度，3表示270度。
    """
    return np.rot90(image, k)

def rotate_bbox_90(bbox, k, image_shape):
    """
    将边界框旋转k * 90度。
    k: 旋转次数，1表示90度，2表示180度，3表示270度。
    image_shape: 图像的形状 (h, w)。
    """
    h, w = image_shape
    x1, y1, x2, y2 = bbox[0].copy()
    if k == 1:  # 90度
        return np.array([[y1, w - x2, y2, w - x1]])
    elif k == 2:  # 180度
        return np.array([[w - x2, h - y2, w - x1, h - y1]])
    elif k == 3:  # 270度
        return np.array([[h - y2, x1, h - y1, x2]])
    else:  # 0度
        return np.array([[x1, y1, x2, y2]])


def xdwpose2vitpose(xdwpose):
    n = xdwpose.shape[0]
    id_pair_dict = {0:0,1:15,2:14,3:17,4:16,5:5,6:2,7:6,8:3,9:7,10:4,11:11,12:8,13:12,14:9,15:13,16:10}
    vit_pose = np.ones([n,17,3])
    for i in range(n):
        for j in range(17):
            vit_pose[i,j,0] = xdwpose[i,0,id_pair_dict[j],0]
            vit_pose[i,j,1] = xdwpose[i,0,id_pair_dict[j],1]
            vit_pose[i,j,2] = xdwpose[i,0,id_pair_dict[j],2]/10 

    return torch.from_numpy(vit_pose).float()

def xdwpose_filter(xdwpose_result_list):
    n = len(xdwpose_result_list)
    subset_list = [xdwpose_result_list[i][1] for i in range(n)]
    subset_list = np.array(subset_list) #[n,1,134]
    
    new_subset_list = []

    th = 3.0
    th2 = 5.0
    window_size = 6

    for i in range(n):
        # subset = np.max(subset_list[max(i-window_size, 0): min(i+window_size+1,n)], axis=0)
        subset_median = np.percentile(subset_list[max(i-window_size, 0): min(i+window_size+1,n)], 75, axis=0) # 75 %分位
        subset = np.maximum(subset_list[i], subset_median)

        subset_old = subset_list[i]
        subset[:,:92][subset_old[:,:92]<th] = 0
        subset[:,92:][subset_old[:,92:]<th2] = 0
        xdwpose_result_list[i][1] = subset

    return xdwpose_result_list


def images2video_buffer(images, fps=USE_FPS):
    format = "mp4" # 默认为 mp4 格式
    codec = "libx264"  # 默认为 libx264 编码器
    ffmpeg_params = ["-crf", '12']
    pixelformat =  "yuv420p"  # 视频像素格式

    # 创建一个 BytesIO 对象作为视频数据的内存存储
    video_stream = io.BytesIO()

    with imageio.get_writer(video_stream, fps=fps, format=format, codec=codec, ffmpeg_params=ffmpeg_params, pixelformat=pixelformat) as writer:
        for idx in range(len(images)):
            writer.append_data(images[idx])

    return video_stream.getvalue()


def run_ffprobe_subprocess(video_path):
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


class SmplInfer:
    def __init__(self, smpl_checkpoints_path='/ytech_milm/liujiwen/kling_motion_service/smpl_all_checkpoints', is_image=False):
        # id 相关筛选阈值
        min_resolution=127           # 分辨率下限（短边）
        max_resolution=9060          # 分辨率上限（长边）
        max_image_list_length=8      # 图片列表最大长度
        min_video_duration=0.8       # 视频最短时长（秒）
        max_video_duration=120.0
        sample_fps=1                 # 视频采样fps（每秒取多少帧）
        face_confidence_threshold=5  # 人脸关键点置信度阈值
        subject_area_ratio=1.5
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.max_image_list_length = max_image_list_length
        self.min_video_duration = min_video_duration
        self.max_video_duration = max_video_duration
        self.sample_fps = sample_fps
        self.face_confidence_threshold = face_confidence_threshold
        self.subject_area_ratio = subject_area_ratio
        
        self.face_grid = FaceGrid()
        self.dwpose_refine =  DwposeRefine()
        self.smpl_checkpoints_path = smpl_checkpoints_path
        self.tracker = Tracker(self.smpl_checkpoints_path+'/checkpoints/yolo/yolo11x.pt')
        ##self.tracker = Tracker(self.smpl_checkpoints_path+'/checkpoints/yolo/yolo11x.engine') #trt 需要切换镜像
        self.extractor = Extractor(self.smpl_checkpoints_path+'/checkpoints/hmr2/epoch=10-step=25000.ckpt')
        self.slam_height = None
        self.slam_width = None
        self.slam = None
        self.cfg = self.parse_args_to_cfg(self.smpl_checkpoints_path+'/checkpoints/gvhmr/gvhmr_siga24_release.ckpt')
        self.model: DemoPL = hydra.utils.instantiate(self.cfg.model, model_path=self.smpl_checkpoints_path+'/checkpoints/', _recursive_=False)
        print("model: DemoPL = hydra.utils.instantiate sucess")
        self.model.load_pretrained_model(self.cfg.ckpt_path)
        print("load_pretrained_model")
        self.model = self.model.eval().cuda()
        print("model.eval().cuda()")

        pose_config = self.smpl_checkpoints_path+'/checkpoints/rtmw-x/rtmw-x_8xb320-270e_cocktail14-384x288.py'
        pose_ckpt = self.smpl_checkpoints_path+'/checkpoints/rtmw-x/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth'
    
        self.xdwpose = DWposeDetector(pose_config, pose_ckpt, device='cuda')

        if is_image:
            self.error_code = ImageErrorCode
            self.error_messages = IMAGE_ERROR_MESSAGES
        else:
            self.error_code = VideoErrorCode
            self.error_messages = VIDEO_ERROR_MESSAGES
        self.is_image = is_image

        print ('self.is_image', self.is_image)

        # t2m 初始化
        self.t2m_model = text2motion(smpl_checkpoints_path+"/t2m")

        # wilor 初始化
        self.wilor_cfg = get_config(self.smpl_checkpoints_path+'/wilor_checkpoints/model_config.yaml', update_cachedir=True)                       
        self.wilor_backbone = ViT(
                mano_path=self.smpl_checkpoints_path+'/wilor_checkpoints/mano_mean_params.npz',
                img_size=(256, 192),
                patch_size=16,
                embed_dim=1280,
                depth=32,
                num_heads=16,
                ratio=1,
                use_checkpoint=False,
                mlp_ratio=4,
                qkv_bias=True,
                drop_path_rate=0.55,
                cfg = self.wilor_cfg
            )
        self.wilor_refine = RefineNet(self.wilor_cfg, feat_dim=1280, upscale=3)
        mano_cfg = {k.lower(): v for k,v in dict(self.wilor_cfg.MANO).items()}
        mano_cfg['data_dir'] = self.smpl_checkpoints_path+'/wilor_checkpoints/'
        mano_cfg['model_path'] = self.smpl_checkpoints_path+'/wilor_checkpoints/'
        mano_cfg['mano_params'] = self.smpl_checkpoints_path+'/wilor_checkpoints/mano_mean_params.npz'
        self.mano = MANO(**mano_cfg)

        self.hand_detector = YOLO(self.smpl_checkpoints_path+'/wilor_checkpoints/detector.pt')

        wilor_ckpt = torch.load(self.smpl_checkpoints_path+'/wilor_checkpoints/wilor_final.ckpt')['state_dict']                                         
        wilor_backbone_state_dict = OrderedDict()
        wilor_refine_state_dict = OrderedDict()
        mano_state_dict = OrderedDict()
        for k, v in wilor_ckpt.items():
            if k.startswith('backbone'):
                wilor_backbone_state_dict[k[9:]] = v
            if k.startswith('refine_net'):
                wilor_refine_state_dict[k[11:]] = v
            if k.startswith('mano'):
                mano_state_dict[k[5:]] = v

        
        self.wilor_backbone.load_state_dict(wilor_backbone_state_dict)   
        self.wilor_refine.load_state_dict(wilor_refine_state_dict)        
        self.mano.load_state_dict(mano_state_dict)   
        
        self.wilor_backbone.cuda()
        self.wilor_refine.cuda()
        self.mano.cuda()
        self.hand_detector.cuda()
        self.norm_hand_pose = np.load(self.smpl_checkpoints_path+'/wilor_checkpoints/norm_hand.npy')
        print("load_wilor_model")

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


    def parse_args_to_cfg(self, gvhmr_model_pth):
        current_file_path = os.path.abspath(__file__)
        # 获取父目录路径
        parent_directory = os.path.dirname(current_file_path)

        # Input
        verbose = False
        static_cam = False

        with initialize_config_module(
            version_base="1.3", config_module=f"GVHMR.hmr4d.configs"
        ):
            overrides = [
                f"static_cam={static_cam}",
                f"verbose={verbose}",
            ]
            register_store_gvhmr()
            cfg = compose(config_name="demo", overrides=overrides)

        cfg.ckpt_path = gvhmr_model_pth
        print ('cfg.ckpt_path', cfg.ckpt_path )
        return cfg


    def load_video_or_image_np(self, input_pth,  tgt_fps=USE_FPS, resize_min_w_h=720, is_image=False, min_video_length=1, max_video_length=241):
        if is_image:
            image = cv2.imread(input_pth)[:,:,::-1]
            h, w = image.shape[:2]

            if min(h, w)<128: return None, self.error_code.RESOLUTION_TOO_SMALL
            if max(h,w)>38500: return None, self.error_code.RESOLUTION_TOO_LARGE
            
            s = resize_min_w_h/min(h, w)
            new_h, new_w = int(h*s), int(w*s)
            h, w = new_h//16*16, new_w//16*16
            image = cv2.resize(image, (new_w, new_h))[:h, :w]

            return np.array([image]), self.error_code.SUCCESS

        input_img_list = []
        videoreader = VideoReader(input_pth)
        duration_ffprobe = run_ffprobe_subprocess(input_pth)
        fps = float(videoreader.get_avg_fps())
        n = min(len(videoreader), int(duration_ffprobe*fps)) #视频长度双重校验
        len_video = n/fps

        if len_video<1: return None, self.error_code.MOTION_DURATION_TOO_SHORT
        if len_video>50: return None, self.error_code.MOTION_DURATION_TOO_LONG

        sample_rate = fps/tgt_fps
        clip_length = int(n/sample_rate)

        frame_indexes = np.linspace(0, n-1, clip_length, dtype=int).tolist()[2:max_video_length+2] #TODO
        ##input_img_list = videoreader.get_batch(frame_indexes).numpy()
        input_img_list = videoreader.get_batch(frame_indexes).asnumpy()

        n, h, w = input_img_list.shape[:3]
        if min(h, w)<128: return None, self.error_code.RESOLUTION_TOO_SMALL
        if max(h,w)>38500: return None, self.error_code.RESOLUTION_TOO_LARGE

        s = resize_min_w_h/min(h, w)
        new_h, new_w = int(h*s), int(w*s)

        h, w = new_h//16*16, new_w//16*16

        input_img_list = [cv2.resize(input_img_list[i], (new_w, new_h))[:h, :w] for i in range(n)]
        input_img_list = np.array(input_img_list)

        del videoreader

        return input_img_list, self.error_code.SUCCESS

    @torch.no_grad()
    def find_first_frame_for_track(self, img_list, min_area=0.05):
        h, w = img_list[0].shape[:2]
        for i, img in enumerate(img_list):
            det_results = self.tracker.detect(img)
            human_num = len(det_results)
            if human_num==0:
                continue
            elif human_num==1:
                area1 = det_results[0][1]
                if area1/(w*h)>min_area:
                    return i, det_results
            else:
                area1 = det_results[0][1]
                area2 = det_results[1][1]
                if self.is_image:
                    if area1/(w*h)>min_area:
                        return i, det_results
                else:
                    if area1/(w*h)>min_area and area1/area2>2:
                        return i, det_results
        return -1, None

    def calculate_iou(self, bbox1, bbox2):
        """
        计算两个 bbox 的 IOU（交并比）。
        
        :param bbox1: 第一个 bbox，格式为 [x1, y1, x2, y2]
        :param bbox2: 第二个 bbox，格式为 [x1, y1, x2, y2]
        :return: IOU 值
        """
        # 计算交集区域的坐标
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        # 计算交集区域的面积
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        # 计算两个 bbox 的面积
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # 计算并集区域的面积
        union_area = bbox1_area + bbox2_area - inter_area
        
        # 计算 IOU
        iou = inter_area / union_area if union_area > 0 else 0
        return iou

    def calculate_iou_wilor(self, bbox_big, bbox_small):            # 大的框在前  小的框在后
        """
        计算两个 bbox 的 IOU（交并比）。
        
        :param bbox_big: 第一个 bbox，格式为 [x1, y1, x2, y2]
        :param bbox_small: 第二个 bbox，格式为 [x1, y1, x2, y2]
        :return: IOU 值
        """
        # 计算交集区域的坐标
        x1 = max(bbox_big[0], bbox_small[0])
        y1 = max(bbox_big[1], bbox_small[1])
        x2 = min(bbox_big[2], bbox_small[2])
        y2 = min(bbox_big[3], bbox_small[3])
        
        # 计算交集区域的面积
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        # 计算两个 bbox 的面积
        # bbox_big_area = (bbox_big[2] - bbox_big[0]) * (bbox_big[3] - bbox_big[1])
        bbox_small_area = (bbox_small[2] - bbox_small[0]) * (bbox_small[3] - bbox_small[1])
        
        # 计算并集区域的面积
        # union_area = bbox_big_area + bbox_small_area - inter_area
        
        # 计算 IOU
        iou = inter_area / bbox_small_area if bbox_small_area > 0 else 0
        return iou

    def filter_bboxes(self, bboxes, h, w, min_area_ratio=0.05, max_iou_threshold=0.1):
        # 计算全图的面积
        image_area = h * w
        
        # 筛选面积大于全图 min_area_ratio 的 bbox
        bboxes_area_filtered = [bbox for bbox in bboxes if bbox[1] > min_area_ratio * image_area]
        
        # 筛选与其他任何 bbox 的 IOU 小于 max_iou_threshold 的 bbox
        bboxes_new = []
        for i, bbox1 in enumerate(bboxes_area_filtered):
            # 检查与所有其他 bbox 的 IOU
            iou_too_large = False
            for j, bbox2 in enumerate(bboxes_area_filtered):
                if i == j:
                    continue
                iou = self.calculate_iou(bbox1[0], bbox2[0])
                if iou >= max_iou_threshold:
                    iou_too_large = True
                    break
            # 如果与所有其他 bbox 的 IOU 都小于 max_iou_threshold，则保留
            if not iou_too_large:
                bboxes_new.append(bbox1[0])
        
        return bboxes_new


    def check_upper_body_keypoints(self, xdwpose_np0, w, h, jump_threshold=0.3, min_valid_ratio=0.95):
        
        xdwpose_np = xdwpose_np0[:,0]
        jump_threshold = (w+h)/2*jump_threshold
        # 上半身关键点的索引
        # upper_body_indices = [0, 14, 15, 1, 2, 5, 8, 11]
        upper_body_indices = [0, 14, 15, 1] # TODO:
        n_frames = len(xdwpose_np)
        min_valid_frames = int(n_frames * min_valid_ratio)  # 满足条件的最小帧数
        if self.is_image: min_valid_frames=1
        visibility_id_list = []

        # 检查全程可见性
        valid_frames_visibility = 0
        for i, frame in enumerate(xdwpose_np):
            all_visible = True
            frame[8,2] = max(frame[8,2], frame[11,2])
            frame[11,2] = frame[8,2]
            for idx in upper_body_indices:
                x, y, s = frame[idx]
                if s <= 4.5:  # 置信度不足，不可见
                    all_visible = False
                    break
            if all_visible:
                valid_frames_visibility += 1
                visibility_id_list.append(i)
        
        # 如果可见性不满足 95%，直接返回 False
        print ('valid_frames_visibility < min_valid_frames', valid_frames_visibility < min_valid_frames)
        
        if valid_frames_visibility < min_valid_frames:
            print(f"Only {valid_frames_visibility}/{n_frames} frames have all upper body keypoints visible.")
            return self.error_code.NO_COMPLETE_UPPER_BODY, visibility_id_list
        
        # 检查帧间稳定性
        valid_frames_stability = 0
        for i in range(1, n_frames):
            frame_valid = True
            for idx in upper_body_indices:
                x_prev, y_prev, _ = xdwpose_np[i - 1][idx]
                x_curr, y_curr, _ = xdwpose_np[i][idx]
                
                # 计算帧间位置变化
                dx = abs(x_curr - x_prev)
                dy = abs(y_curr - y_prev)
                
                # 如果位置变化超过阈值，视为跳变
                if dx > jump_threshold or dy > jump_threshold:
                    frame_valid = False
                    break
            if frame_valid:
                valid_frames_stability += 1
        
        # 如果稳定性不满足 95%，返回 False
        print ('valid_frames_stability < min_valid_frames', valid_frames_stability < min_valid_frames)
        if valid_frames_stability < min_valid_frames and (not self.is_image):
            print(f"Only {valid_frames_stability}/{n_frames} frames have stable upper body keypoints.")
            return self.error_code.POSE_DIFFERENCE_TOO_LARGE, visibility_id_list
        
        # 通过检查
        return self.error_code.SUCCESS, visibility_id_list

    @torch.no_grad()
    def get_video_xdwpose(self, bbx_xyxy, video_np, is_image):

        output_xpose_video_np = []
        xdwpose_result_list = []
        n = bbx_xyxy.shape[0]
        m = len(video_np)
        for j in range(n):
            box_ext = bbx_xyxy[j:j+1].copy()
            frame = video_np[min(j,m-1)][:,:,::-1]  #图片模式，一图多bbox
            best_k=0
            
            if j<1:
                xdwpose_result = self.xdwpose(image_np_hwc=frame, show_body=True,show_face=False, show_hands=True, plot=True, box_ext=box_ext)
                xdwpose_result_list.append(xdwpose_result)
            else:
                # 找到最佳旋转角度
                best_k = find_best_rotation(xdwpose_result_prev[0][0], frame.shape[:2])
                # 旋转图像和bbox
                rotated_image = rotate_image_90(frame, best_k)
                rotated_bbox = rotate_bbox_90(box_ext, best_k, frame.shape[:2])
                # 检测pose
                ##img_test = rotated_image[int(rotated_bbox[0][1]):int(rotated_bbox[0][3]), int(rotated_bbox[0][0]):int(rotated_bbox[0][2])]
                ##cv2.imwrite('/ytech_milm/liujiwen/kling_motion_service/new2/get_smpl_motion/output/'+str(j)+'.png', img_test)
                xdwpose_result = self.xdwpose(image_np_hwc=rotated_image, show_body=True,show_face=False, show_hands=True, plot=True, box_ext=rotated_bbox)
                # 将pose旋转回原始朝向
                if best_k != 0:
                    xdwpose_result[0][0] = rotate_pose_90(xdwpose_result[0][0], (4 - best_k) % 4, rotated_image.shape[:2])
                    bbx_xyxy[j:j+1] = rotate_bbox_90(rotated_bbox, (4 - best_k) % 4, rotated_image.shape[:2])
                xdwpose_result_list.append(xdwpose_result)

            xdwpose_result_prev = copy.deepcopy(xdwpose_result)

        xdwpose_result_list = xdwpose_filter(xdwpose_result_list)
        
        # for j in range(bbx_xyxy.shape[0]):
        #     output_img = self.xdwpose(image_np_hwc=frame, show_body=True,show_face=False, show_hands=True, plot=True, box_ext=bbx_xyxy[j:j+1], xdwpose_result=copy.deepcopy(xdwpose_result_list[j]))
        #     output_xpose_video_np.append(output_img)
        
        # video_stream = images2video_buffer(output_xpose_video_np, fps=24)
        # with open('./test_xpose2.mp4', 'wb') as f:
        #     f.write(video_stream)

        n = len(xdwpose_result_list)
        subset_list = [xdwpose_result_list[i][1] for i in range(n)]
        subset_list = np.array(subset_list) #[n,1,134]
        candidate_list = [xdwpose_result_list[i][0] for i in range(n)]
        candidate_list = np.array(candidate_list) #[n,1,134,2]

        xdwpose_np = np.concatenate([candidate_list, subset_list[:,:,:,np.newaxis]], axis=3)

        if not is_image: xdwpose_np = self.dwpose_refine.refine(xdwpose_np, frame.shape[0], frame.shape[1])

        visible_list = [xdwpose_result_list[i][2] for i in range(n)]
        visible_list = np.array(visible_list) #[n,1,134]

        return xdwpose_np, bbx_xyxy, visible_list



    def bbox_to_json(self, bbox_xyxy0, w, h):
        """
        将 bbox_xyxy 转换为 0~1 的坐标，并生成 JSON 格式。
        :param bbox_xyxy: n*4 的矩阵，每个 bbox 是 [x1, y1, x2, y2]
        :param w: 图像的宽度
        :param h: 图像的高度
        :return: JSON 格式的字符串
        """
        # 将 Tensor 转换为 NumPy 数组
        bbox_xyxy = bbox_xyxy0.cpu().numpy().copy()  # 如果 Tensor 在 GPU 上，先移动到 CPU
        # 将 bbox_xyxy 转换为 0~1 的坐标
        bbox_normalized = bbox_xyxy / np.array([w, h, w, h])

        # 生成 JSON 格式
        result = []
        for i, bbox in enumerate(bbox_normalized):
            result.append({
                "id": i ,  # id 从 0 开始
                "x1": float(bbox[0]),
                "y1": float(bbox[1]),
                "x2": float(bbox[2]),
                "y2": float(bbox[3])
            })

        # 转换为 JSON 字符串
        return json.dumps(result)


    def get_start_end_from_bbox(self, bbx_xyxy, m=3):
        """
        找出第一段人持续出现的片段的开始和结尾索引，允许有 m 帧没有检测到人。
        在漏检时，将 bbox 赋值为上一帧的结果，并返回更新后的 bbx_xyxy。

        参数:
        bbx_xyxy (torch.Tensor): 形状为 (n, 4) 的张量，表示 n 个 bbox，每个 bbox 是 [x1, y1, x2, y2]。
        m (int): 允许连续没有检测到人的帧数，默认为 3。

        返回:
        tuple: (start_idx, end_idx, updated_bbx_xyxy)，表示第一段人持续出现的片段的开始和结束索引，
            以及更新后的 bbx_xyxy。如果没有检测到人，返回 (None, None, bbx_xyxy)。
        """
        n = bbx_xyxy.shape[0]  # 获取 bbox 的数量
        start_idx = None
        end_idx = None
        last_valid_bbox = None  # 记录最后一个有效的 bbox
        missed_frames = 0  # 记录连续没有检测到人的帧数
        updated_bbx_xyxy = bbx_xyxy.clone()  # 创建 bbx_xyxy 的副本，用于更新

        for i in range(n):
            bbox = updated_bbx_xyxy[i]
            # 检查 bbox 是否表示检测到人（不是 [0,0,0,0]）
            if not torch.all(bbox == torch.tensor([0, 0, 0, 0], device=bbox.device)):
                if start_idx is None:  # 找到片段的开始
                    start_idx = i
                end_idx = i  # 更新片段的结束
                last_valid_bbox = bbox  # 更新最后一个有效的 bbox
                missed_frames = 0  # 重置连续没有检测到人的帧数
            else:
                if start_idx is not None:  # 如果片段已经开始
                    if missed_frames < m:  # 允许有 m 帧没有检测到人
                        end_idx = i  # 更新片段的结束
                        missed_frames += 1
                        updated_bbx_xyxy[i] = last_valid_bbox  # 在漏检时，bbox 赋值为上一帧的结果
                    else:  # 超过 m 帧没有检测到人，片段结束
                        end_idx -= m
                        break

        if start_idx is not None and end_idx is not None:
            return start_idx, end_idx, updated_bbx_xyxy
        else:
            return None, None, updated_bbx_xyxy  # 没有检测到人，返回原始 bbx_xyxy


    @torch.no_grad()
    def run_wilor(self, video_length, video_np, xdwpose_np, xdwpose_visible):                                                # video_length: int      video_np:[N, H, W]      xdwpose_np:[N, 1, 134, 3]     xdwpose_visible: [N, 1, 134]
        is_right = np.array([0, 1])                                                                         # 左右手顺序固定
        wilor_out = {}
        all_global_orient = torch.ones([video_length, 2, 1, 3, 3])
        all_hand_pose = torch.ones([video_length, 2, 15, 3, 3])
        # iou_hand = np.zeros(video_length)
        left_vis = np.zeros(video_length)
        right_vis = np.zeros(video_length)
        for i in tqdm(range(video_length), desc="WiLor Tracking"):
            # 抽帧
            if i%2==1:
                continue
            
            img_np = video_np[i]
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            
            # 根据关键点的边界画左手的框, 并适当外扩
            left_hand_np = xdwpose_np[i][0][92: 113, :2]
            x_min, y_min = np.min(left_hand_np, axis=0)
            x_max, y_max = np.max(left_hand_np, axis=0)
            y_expand = (y_max-y_min)/4
            x_expand = (x_max-x_min)/4
            left_hand_box_noexpand = np.array([x_min, y_min, x_max, y_max])
            left_hand_box = np.array([x_min-x_expand, y_min-y_expand, x_max+x_expand, y_max+y_expand])
            
            # 右手的框, 类似
            right_hand_np = xdwpose_np[i][0][113:, :2]
            x_min, y_min = np.min(right_hand_np, axis=0)
            x_max, y_max = np.max(right_hand_np, axis=0)
            y_expand = (y_max-y_min)/4
            x_expand = (x_max-x_min)/4
            right_hand_box_noexpand = np.array([x_min, y_min, x_max, y_max])
            right_hand_box = np.array([x_min-x_expand, y_min-y_expand, x_max+x_expand, y_max+y_expand])

            # 用Wilor的框作辅助判断
            detections = self.hand_detector(img_np, conf = 0.3, verbose=False)

            if len(detections[0].boxes) == 0:
                continue

            Bbox_wilor = detections[0].boxes.data.cpu().detach().squeeze().numpy()
            
            if Bbox_wilor.ndim == 1:
                Bbox_wilor = Bbox_wilor.reshape(1, -1)

            # 只有两个框有交集，才认为存在完整的手
            left_hand_iou = []
            right_hand_iou = []
            for j in range(Bbox_wilor.shape[0]):
                left_hand_iou.append(self.calculate_iou_wilor(Bbox_wilor[j, :4], left_hand_box_noexpand))
                right_hand_iou.append(self.calculate_iou_wilor(Bbox_wilor[j, :4], right_hand_box_noexpand))
            if max(left_hand_iou)>0.5 and xdwpose_visible[i, 0, 92]>0.85 and xdwpose_np[i, 0, 92, 2]>5.5:
                left_vis[i] = True
            if max(right_hand_iou)>0.5 and xdwpose_visible[i, 0, 113]>0.85 and xdwpose_np[i, 0, 113, 2]>5.5:
                right_vis[i] = True

            # 防止两个手的点画到一起了 强大的边界条件:既考虑wilor预测的左右手也考虑置信度 宁愿双手都判断为不可见也不判断错
            iou_hand = self.calculate_iou(left_hand_box_noexpand, right_hand_box_noexpand)
            
            if left_vis[i] and right_vis[i] and iou_hand>0.3:
                left_wilor_box_index = left_hand_iou.index(max(left_hand_iou))
                right_wilor_box_index = right_hand_iou.index(max(right_hand_iou))
                if left_wilor_box_index==right_wilor_box_index:
                    left_right_conf_diff = xdwpose_np[i, 0, 94:113, 2].mean() - xdwpose_np[i, 0, 115:, 2].mean()
                    if left_right_conf_diff>1 or Bbox_wilor[left_wilor_box_index, -1]==0:
                        right_vis[i] = False
                    elif left_right_conf_diff<-1 or Bbox_wilor[left_wilor_box_index, -1]==1:
                        left_vis[i] = False

            hand_boxes = np.stack([left_hand_box, right_hand_box])
            
            crop_img = WilorCrop(self.wilor_cfg, img_np, hand_boxes, is_right, rescale_factor=2.0)  
            wilor_input = torch.stack([crop_img.get_img(0), crop_img.get_img(1)]).cuda()
            
            with torch.no_grad():
                temp_mano_params, pred_cam, pred_mano_feats, vit_out = self.wilor_backbone(wilor_input[:,:,:,32:-32]) 
                batch_size = wilor_input.shape[0]

                # Compute camera translation
                device = temp_mano_params['hand_pose'].device
                dtype = temp_mano_params['hand_pose'].dtype
                focal_length = self.wilor_cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
                
                # Temp MANO 
                temp_mano_params['global_orient'] = temp_mano_params['global_orient'].reshape(batch_size, -1, 3, 3)
                temp_mano_params['hand_pose'] = temp_mano_params['hand_pose'].reshape(batch_size, -1, 3, 3)
                temp_mano_params['betas'] = temp_mano_params['betas'].reshape(batch_size, -1)
                temp_mano_output  = self.mano(**{k: v.float() for k,v in temp_mano_params.items()}, pose2rot=False)
                #temp_keypoints_3d = temp_mano_output.joints
                temp_vertices     = temp_mano_output.vertices

                pred_mano_params, pred_cam = self.wilor_refine(vit_out, temp_vertices, pred_cam, pred_mano_feats, focal_length) 
                # Store useful regression outputs to the output dict
                
                pred_mano_params = {k: v.clone() for k,v in pred_mano_params.items()}                                             # output['pred_mano_params'].keys(): global_orient, hand_pose, betas

                # 如果手的可见度低, 例如被挡住了或出了屏幕, 用上一帧的pose代替         
                all_global_orient[i] = pred_mano_params['global_orient']
                all_hand_pose[i] = pred_mano_params['hand_pose']
        
        all_global_orient = all_global_orient.cpu().numpy()
        all_hand_pose = all_hand_pose.cpu().numpy()

        wilor_out['global_orient'] = all_global_orient
        wilor_out['hand_pose'] = all_hand_pose
        wilor_out['left_vis'] = left_vis
        wilor_out['right_vis'] = right_vis

        return wilor_out


    @torch.no_grad()
    def get_dwpose_np_from_img_list(self, img_list):
        res = []
        for i, img in enumerate(img_list):
            h, w = img.shape[:2]
            # 为了不修改原图，拷贝一份用于可视化
            vis_img = img.copy()
            
            # 1. 人体检测与跟踪
            det_results = self.tracker.detect(img)
            human_num = len(det_results)
            
            if human_num == 0:
                # 如果没人，使用全图作为 bbox，或者跳过
                bbox = [0, 0, w, h]
            else:
                # 取第一个检测到的人
                bbox = det_results[0][0]

            # 2. 运行 DWPose
            # 注意：这里假设 xdwpose 返回的是 list 或 tuple，索引 0 是 keypoints，索引 1 是 scores
            xdwpose_result = self.xdwpose(
                image_np_hwc=img, 
                show_body=False,
                show_face=False, 
                show_hands=True, 
                plot=False, 
                box_ext=np.array([bbox])
            )
                
            dwpose = np.concatenate([xdwpose_result[0], xdwpose_result[1][:,:,np.newaxis]], axis=2)
            res.append(dwpose)

        return np.array(res)

    def replace_frames_by_index_list(self, image, raw_frames, index_list, flip_list=[]):
        new_frames = list(raw_frames)
        index_list = list(index_list)
        for idx in index_list:
            start_frame = idx * 9
            end_frame = start_frame + 9
            
            for i in range(start_frame, end_frame):
                if i < len(new_frames):
                    if idx in flip_list:
                        new_frames[i] = image[:,::-1]
                    else:
                        new_frames[i] = image
                    
        return new_frames


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
            sample_indices = np.linspace(1, min(30*fps, total_frames-1), sample_n, dtype=int)
            
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


    @torch.no_grad()
    def get_face_grid(self, image_data, id_video_data=None, input_id_image_list_binary=None, output_dir='./output/', target_size=[720, 1280], save_path_dir=None):

        output_dir = os.path.abspath(output_dir)
        output_result_path = output_dir+'/'+str(uuid.uuid4())+'/'
        os.makedirs(output_result_path)
        print ('output_result_path', output_result_path)
        image_path = output_result_path+'/image.png'
        id_video_path = output_result_path+'/video.mp4'
        id_image_path_list = [output_result_path+'/id_image_'+str(i)+'.png' for i in range(9)]

        with open(image_path, "wb") as f:
            f.write(image_data)

        if id_video_data is not None:
            with open(id_video_path, "wb") as f:
                f.write(id_video_data)
        else:
            id_video_path = None


        if input_id_image_list_binary is not None:
            id_image_path_list = id_image_path_list[:len(input_id_image_list_binary)]
            for i in range(min(9,len(input_id_image_list_binary))):
                with open(id_image_path_list[i], "wb") as f:
                    f.write(input_id_image_list_binary[i])
        else:
            id_image_path_list = None

        if id_image_path_list is None and id_video_path is None:
            id_image_path_list = [image_path]
        
        image = cv2.imread(image_path)[:,:,::-1] #RGB
        h, w = image.shape[:2]
        tgt_size, tgt_size2 = target_size
        area = tgt_size * tgt_size2
        area1 = h*w

        s = math.sqrt(area / area1)
        new_w = int(w*s)
        new_h = int(h*s)
        
        resize_width = new_w//16*16
        resize_height = new_h//16*16

        result = self.execute_input_validation(image_list=id_image_path_list, video_path=id_video_path)

        id_image_path_list = result["image_list"]
        if id_image_path_list is None or len(id_image_path_list) == 0:
            id_image_path_list = [image_path]


        try:
        #if True:
            #raw_frames, fps, indices, is_image = self.face_grid.read_video_frames(id_video_path, id_image_path_list, n=1)
            raw_frames, fps, indices, is_image = self.face_grid.read_video_frames(
                id_video_path=id_video_path,
                id_image_path_list=id_image_path_list,
                n=1,
                start_idx=result["start_idx"],
                end_idx=result["end_idx"]
            )    

            if not is_image:
                raw_frames = self.replace_frames_by_index_list(image, raw_frames, index_list=[1], flip_list=[4])
            
            dwpose_face_video=self.get_dwpose_np_from_img_list(raw_frames)
            dwpose_face_img=self.get_dwpose_np_from_img_list([image])

            result_frames = self.face_grid.compose_face_grid_frames_stable(raw_frames, image, dwpose_face_video, dwpose_face_img, resize_height, resize_width,is_image=is_image)
        except:
            image = cv2.resize(image, (resize_width, resize_height))
            result_frames = np.array([image]*9)

        if save_path_dir is not None:
            self.face_grid.save_video_from_frames(result_frames, save_path_dir+'/id.mp4', fps=8)
            np.save(save_path_dir+'/id.npy', result_frames)
        
        os.system('rm -rf '+output_result_path) ##清空临时文件
        return result_frames



    @torch.no_grad()
    def run_preprocess(self, cfg, tgt_fps=USE_FPS, resize_min_w_h=720, video_length=241):
        video_path = cfg.video_path
        paths = cfg.paths
        static_cam = cfg.static_cam
        verbose = cfg.verbose
        verbose = False
        result_data_dict = {}
        try:
            video_np, load_video_error_code = self.load_video_or_image_np(video_path, is_image=self.is_image, tgt_fps=tgt_fps, resize_min_w_h=resize_min_w_h, max_video_length=video_length)
        except:
            result_data_dict['error_code'] = self.error_code.OTHER_ERRORS
            result_data_dict['error_message'] = self.error_messages.get(result_data_dict['error_code'])
            return result_data_dict, None, None, None
        result_data_dict['error_code'] = load_video_error_code
        result_data_dict['error_message'] = self.error_messages.get(result_data_dict['error_code'])
        if load_video_error_code != self.error_code.SUCCESS: return result_data_dict, None, None, None

        find_first_frame_id, det_results = self.find_first_frame_for_track(video_np)
        if find_first_frame_id==-1:
            result_data_dict['error_code'] = self.error_code.NO_COMPLETE_UPPER_BODY
            result_data_dict['error_message'] = self.error_messages.get(result_data_dict['error_code'])
            return result_data_dict, None, None, None

        video_np = video_np[find_first_frame_id:]
        video_width = video_np[0].shape[1]
        video_height = video_np[0].shape[0]

        if self.is_image:
            ##满足要求的所有bbox 都要跑出来smpl和2Dpose，用来前端用户选择，最多跑面积最大的5个人
            if len(det_results)>2 and det_results[2][1]/det_results[0][1] > 0.5:
                result_data_dict['error_code'] = self.error_code.TOO_MANY_SUBJECTS
                result_data_dict['error_message'] = self.error_messages.get(result_data_dict['error_code'])
                return result_data_dict, None, None, None
            bbx_xyxy = self.filter_bboxes(det_results, video_height, video_width, min_area_ratio=0.05, max_iou_threshold=0.1)[:5]
            bbox_num_of_image = len(bbx_xyxy)
            if bbox_num_of_image==0:
                result_data_dict['error_code'] = self.error_code.NO_COMPLETE_UPPER_BODY
                result_data_dict['error_message'] = self.error_messages.get(result_data_dict['error_code'])
                return result_data_dict, None, None, None
        else:
            video_stream = images2video_buffer(video_np, fps=USE_FPS)
            with open(self.cfg.output_root+'/temp.mp4', 'wb') as f:
                f.write(video_stream)

        video_path = self.cfg.output_root+'/temp.mp4'
        cfg.video_path = video_path
        self.cfg.video_path = video_path


        # Get bbx tracking result
        if self.is_image:
            bbx_xyxy = torch.from_numpy(np.array(bbx_xyxy)).float() # (L, 4)
        else:
            bbx_xyxy = self.tracker.get_one_track(str(video_path))
            if bbx_xyxy is None:
                result_data_dict['error_code'] = self.error_code.NO_COMPLETE_UPPER_BODY
                result_data_dict['error_message'] = self.error_messages.get(result_data_dict['error_code'])
                return result_data_dict, None, None, None
            else:
                bbx_xyxy = bbx_xyxy.float()  # (L, 4)

        if not self.is_image:
            start_idx, end_idx, bbx_xyxy = self.get_start_end_from_bbox(bbx_xyxy)
            if start_idx is None or end_idx is None:
                result_data_dict['error_code'] = self.error_code.NO_COMPLETE_UPPER_BODY
                result_data_dict['error_message'] = self.error_messages.get(result_data_dict['error_code'])
                return result_data_dict, None, None, None
            if end_idx-start_idx+1< 30:
                result_data_dict['error_code'] = self.error_code.NO_COMPLETE_UPPER_BODY
                result_data_dict['error_message'] = self.error_messages.get(result_data_dict['error_code'])
                return result_data_dict, None, None, None
            bbx_xyxy = bbx_xyxy[start_idx:end_idx+1]
            video_np = video_np[start_idx:end_idx+1]
        video_length = len(video_np)

        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()  # (L, 3) apply aspect ratio and enlarge
        json_bbox_with_ID = self.bbox_to_json(bbx_xyxy, video_width, video_height)
        result_data_dict['json_bbox_with_ID'] = json_bbox_with_ID

        # Get VitPose
        if True:
            xdwpose_np, bbx_xyxy, xdwpose_visible = self.get_video_xdwpose(bbx_xyxy.numpy(), video_np, self.is_image)

            xdwpose_error_code, visibility_id_list = self.check_upper_body_keypoints(xdwpose_np, video_width, video_height, jump_threshold=0.3, min_valid_ratio=0.9)
            result_data_dict['error_code'] = xdwpose_error_code
            result_data_dict['error_message'] = self.error_messages.get(result_data_dict['error_code'])
            if xdwpose_error_code != self.error_code.SUCCESS: return result_data_dict, None, None, None
            
            if self.is_image: #选出dwpose 符合要求的人
                bbx_xyxy = bbx_xyxy[visibility_id_list]
                bbx_xys = bbx_xys[visibility_id_list]
                xdwpose_np = xdwpose_np[visibility_id_list]
            vitpose = xdwpose2vitpose(xdwpose_np.copy())
            bbox_num_of_image = vitpose.shape[0]
        
        # Get Wilor results
        wilor_out = self.run_wilor(video_length=video_length, video_np=video_np, xdwpose_np=xdwpose_np, xdwpose_visible=xdwpose_visible)

        if verbose:
            video_overlay = draw_bbx_xyxy_on_image_batch(bbx_xyxy, video_np.copy())
            save_video(video_overlay, cfg.paths.bbx_xyxy_video_overlay)

        if verbose:
            video_overlay = draw_coco17_skeleton_batch(video_np.copy(), vitpose, 0.5)
            save_video(video_overlay, paths.vitpose_video_overlay)

        # Get vit features
        if True:
            vit_features = self.extractor.extract_video_features(video_np, bbx_xys)

        # Get DPVO results
        length, width, height = video_length, video_width, video_height
        if not static_cam:  # use slam to get cam rotation
            if True:
                K_fullimg = estimate_K(width, height)
                intrinsics = convert_K_to_K4(K_fullimg)
                self.slam = SLAMModel(
                    video_np.copy(),
                    width,
                    height,
                    intrinsics,
                    buffer=4000,
                    resize=0.5,
                    model_path=self.smpl_checkpoints_path+'/checkpoints/dpvo/dpvo.pth',
                )
                self.slam.track()
                slam_results = self.slam.process()  # (L, 7), numpy
                ##torch.save(slam_results, paths.slam)

        if cfg.static_cam:
            R_w2c = torch.eye(3).repeat(length, 1, 1)
        else:
            traj = slam_results #torch.load(cfg.paths.slam)
            traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
            R_w2c = quaternion_to_matrix(traj_quat).mT
        K_fullimg = estimate_K(width, height).repeat(length, 1, 1)
        # K_fullimg = create_camera_sensor(width, height, 26)[2].repeat(length, 1, 1)

        result_data_dict['data_list'] = []

        xdwpose_np[:,:,:,0] = xdwpose_np[:,:,:,0]/width
        xdwpose_np[:,:,:,1] = xdwpose_np[:,:,:,1]/height

        if self.is_image:
            result_data_dict['image_ori'] = video_np[0]
            for i in range(bbox_num_of_image):
                data = {
                    "length": torch.tensor(length)*3,
                    "bbx_xys": bbx_xys[[i]*3],
                    "kp2d": vitpose[[i]*3],
                    "K_fullimg": K_fullimg[[0]*3],
                    "cam_angvel": compute_cam_angvel(R_w2c[[0]*3]),
                    "f_imgseq": vit_features[[i]*3],
                    'width': width,
                    'height': height,
                    'xdwpose_np': xdwpose_np[i:i+1],
                }
                result_data_dict['data_list'].append(data)
        else:
            data = {
                "length": torch.tensor(length),
                "bbx_xys": bbx_xys,
                "kp2d": vitpose,
                "K_fullimg": K_fullimg,
                "cam_angvel": compute_cam_angvel(R_w2c),
                "f_imgseq": vit_features,
                'width': width,
                'height': height,
                'xdwpose_np': xdwpose_np,
                'video_np': video_np
            }
            result_data_dict['data_list'].append(data)

        result_data_dict['error_code'] = self.error_code.SUCCESS
        result_data_dict['error_message'] = self.error_messages.get(result_data_dict['error_code'])
        return result_data_dict, wilor_out, xdwpose_np, xdwpose_visible
        
    def input_decode_and_save(self, input_data_binary, output_result_path, task_mode):
        if task_mode=='video2motion':
            input_path = output_result_path+'/'+'input.mp4'
            with open(input_path, "wb") as f:
                f.write(input_data_binary)
        elif task_mode=='image2motion':
            input_path = output_result_path+'/'+'input.png'
            with open(input_path, "wb") as f:
                f.write(input_data_binary)
        elif task_mode=='BVH2motion':
            input_path = output_result_path+'/'+'input.bvh'
            with open(input_path, "wb") as f:
                f.write(input_data_binary)
        else:
            input_path = None
        
        return input_path

    def smpl2bvh(self, smpl_pt_ori, out_path, task_mode):
        smpl_pt = copy.deepcopy(smpl_pt_ori)
        if task_mode not in ['image2motion', 'video2motion']:
            bvh_pt = hmr2bvh(smpl_pt)
        else:
            bvh_pt = gv_global2bvh(smpl_pt)
        ##bvh_binary = smplx2bvh(model_path=self.smpl_checkpoints_path+'/checkpoints/', poses=smpl_pt, output=out_path+'/'+'motion.bvh', use_trans=False)
        bvh_binary = smplx2bvh(model_path=self.smpl_checkpoints_path+'/checkpoints/', poses=smpl_pt, output=out_path+'/'+'motion.bvh', use_trans=False, fps=USE_FPS, max_f=300)
        return bvh_binary

    def infer(self, input_data_binary, input_text=None, output_dir='./output/', tgt_fps=USE_FPS, resize_min_w_h=640, duration=10000, task_mode='video2motion', input_id_video_binary=None):
        '''5s:145, 10s:289'''
        # tgt_video_length = find_closest_number([4800, 9600], float(duration))
        # tgt_video_length = int(float(tgt_video_length)/1000*tgt_fps+1)
        '''5s:145, 10s:289'''

        '''5s:153, 10s:305'''
        # tgt_video_length = find_closest_number([5100, 10160], float(duration))
        tgt_video_length = float(duration)
        # 严格向上取整帧数
        tgt_video_length = math.ceil(float(tgt_video_length)/1000*tgt_fps)
        tgt_video_length = ((tgt_video_length-1)//8)*8+1
        '''5s:153, 10s:305'''

        print ('tgt_video_length', tgt_video_length)
        result_data_dict = {} #初始输出
        # 设置输入
        print ('-------------------------task_mode--------------------------')
        print (task_mode)
        if task_mode not in ['image2motion', 'video2motion', 'text2motion', 'BVH2motion']:
            result_data_dict['error_code'] = self.error_code.OTHER_ERRORS
            result_data_dict['error_message'] = self.error_messages.get(result_data_dict['error_code'])
            os.system('rm -rf '+output_result_path) ##清空临时文件
            return result_data_dict['error_code'], result_data_dict['error_message'], None, json.dumps({}), None
        print ('-------------------------task_mode--------------------------')

        
        #生成临时文件夹，存放中间结果：
        output_dir = os.path.abspath(output_dir)
        output_result_path = output_dir+'/'+str(uuid.uuid4())+'/'
        os.makedirs(output_result_path)
        print ('output_result_path', output_result_path)

        input_file = self.input_decode_and_save(input_data_binary, output_result_path, task_mode)

        ##self.cfg.video_name = video_name_without_ext
        self.cfg.video_path = input_file
        self.cfg.output_root = output_result_path

        paths = self.cfg.paths
        print("paths", paths)

        if task_mode == 'text2motion':
            # t2m 推理 结果仅为肢体！pose为22*3!
            input_text = str(input_text)
            if not input_text:
                result_data_dict['smpl'] = []
                result_data_dict['error_code'] = self.error_code.MOTION_TEXT_INVALID_INPUT
                return result_data_dict['error_code'], result_data_dict['error_message'], None, json.dumps({}), None
            else:
                result = self.t2m_model.infer(text=input_text, duration=None, target_fps=USE_FPS) # 不转帧率则target_fps设置-1
                result=bvh2hmr(result)
                result_data_dict['smpl'] = [result]
                result_data_dict['error_code'] = self.error_code.SUCCESS
            result_data_dict['error_message'] = self.error_messages.get(result_data_dict['error_code'])
            ##torch.save(result_data_dict, paths.hmr4d_results.replace('/hmr4d_results.pt', '/all_result.pt'))
            bvh_binary = self.smpl2bvh(result_data_dict['smpl'][0], output_result_path, task_mode)
            print("Done!")
            os.system('rm -rf '+output_result_path) ##清空临时文件
            return result_data_dict['error_code'], result_data_dict['error_message'], pickle.dumps(result_data_dict), json.dumps({}), bvh_binary

        if task_mode == 'BVH2motion':
            # bvh2smplx 结果是带手部的！pose为54*3！
            bvh_file = input_file
            if not bvh_file or not os.path.exists(bvh_file):
                result_data_dict['smpl'] = []
                result_data_dict['error_code'] = self.error_code.MOTION_BVH_INVALID_INPUT
                return result_data_dict['error_code'], result_data_dict['error_message'], None, json.dumps({}), None
            else:
                result = bvh_to_smplx(bvh_file, target_fps=USE_FPS) # 不转帧率则target_fps设置-1
                result=bvh2hmr(result)
                result_data_dict['smpl'] = [result]
                result_data_dict['error_code'] = self.error_code.SUCCESS
            result_data_dict['error_message'] = self.error_messages.get(result_data_dict['error_code'])
            ##torch.save(result_data_dict, paths.hmr4d_results.replace('/hmr4d_results.pt', '/all_result.pt'))
            bvh_binary = self.smpl2bvh(result_data_dict['smpl'][0], output_result_path, task_mode)
            print("Done!")
            os.system('rm -rf '+output_result_path) ##清空临时文件
            return result_data_dict['error_code'], result_data_dict['error_message'], pickle.dumps(result_data_dict), json.dumps({}), bvh_binary

        # ===== Preprocess and save to disk ===== #
        result_data_dict, wilor_out, xdwpose_np, xdwpose_visible = self.run_preprocess(self.cfg, tgt_fps=tgt_fps, resize_min_w_h=resize_min_w_h, video_length=tgt_video_length)
        if result_data_dict['error_code'] != self.error_code.SUCCESS: 
            print (result_data_dict['error_message'])
            os.system('rm -rf '+output_result_path) ##清空临时文件
            return result_data_dict['error_code'], result_data_dict['error_message'], None, json.dumps({}), None

        result_data_dict['smpl'] = []
        for i, data in enumerate(result_data_dict['data_list']):
            # ===== HMR4D ===== #
            if True:
                pred = self.model.predict(data, static_cam=self.cfg.static_cam)
                pred = detach_to_cpu(pred)
                data_time = data["length"] / USE_FPS
                pred['smpl_params_incam'] = convert_mano_to_smplx(pred['smpl_params_incam'], wilor_out, xdwpose_np, xdwpose_visible, self.norm_hand_pose.copy())    
                pred['smpl_params_global']['left_hand_pose'] = pred['smpl_params_incam']['left_hand_pose'] 
                pred['smpl_params_global']['right_hand_pose'] = pred['smpl_params_incam']['right_hand_pose']  
                pred['smpl_params_global']['body_pose'][:, 57:60] = pred['smpl_params_incam']['body_pose'][:, 57:60]  
                pred['smpl_params_global']['body_pose'][:, 60:63] = pred['smpl_params_incam']['body_pose'][:, 60:63]                                                                   # 将mano的参数融合到smplx中
                result_data_dict['smpl'].append(pred)
                # torch.save(pred, paths.hmr4d_results.replace('/hmr4d_results.pt', '/hmr4d_results_'+str(i)+'.pt'))
                # print (paths.hmr4d_results.replace('/hmr4d_results.pt', '/hmr4d_results_'+str(i)+'.pt'))
        
        if task_mode in ['image2motion']:
            ##返回一个list
            bvh_binary = [self.smpl2bvh(smpl_data, output_result_path, task_mode) for smpl_data in result_data_dict['smpl']]
        else:
            bvh_binary = self.smpl2bvh(result_data_dict['smpl'][0], output_result_path, task_mode)

        print("Done!")
        os.system('rm -rf '+output_result_path) ##清空临时文件
        ##torch.save(result_data_dict, paths.hmr4d_results.replace('/hmr4d_results.pt', '/all_result.pt'))
        json_bbox_with_ID = result_data_dict.pop("json_bbox_with_ID", {})
        if task_mode != 'image2motion':
            json_bbox_with_ID = json.dumps({}) #防止json过大导致传输负担
        
        print (result_data_dict['error_message'])
        return result_data_dict['error_code'], result_data_dict['error_message'], pickle.dumps(result_data_dict), json_bbox_with_ID, bvh_binary#, output_result_path





# 确保只有当直接运行此脚本时才执行 main 函数
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="gRPC server")
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument('--is_image_or_video', type=int, default=0) # required=True, help="0:video, 1:image")

    args = parser.parse_args()

    ##初始化
    smpl_infer = SmplInfer(smpl_checkpoints_path='/ytech_milm/liujiwen/kling_motion_service/smpl_all_checkpoints', is_image=args.is_image_or_video)

    input_file = args.input_file
    output_dir = args.output_dir
    

    start_time = time.time()
    #推理
    error_code, result_path = smpl_infer.infer(input_file, output_dir, tgt_fps=24.0, resize_min_w_h=640, video_length=10)
    print (result_path)
    

    print ('time:', time.time()-start_time)
