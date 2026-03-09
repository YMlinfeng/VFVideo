import cv2
import torch
import sys
import os
import json
import math
import matplotlib
import pytorch_lightning as pl
import numpy as np
import argparse
import matplotlib
import shutil
import hydra
from pytorch3d.transforms import quaternion_to_matrix
from .text2motion.visualization.smpl2bvh import smplx2bvh
from .retarget_axis_t2m import bvh2hmr, hmr2bvh, gv_global2bvh
import matplotlib.cm as cm

from .hmr4d.utils.video_io_utils import (
    get_video_lwh,
    get_writer,
    get_video_reader,
)

from .hmr4d.utils.net_utils import to_cuda
from .hmr4d.utils.smplx_utils import make_smplx
from .hmr4d.utils.vis.renderer import Renderer, perspective_projection

from tqdm import tqdm
import subprocess
import numpy as np
import copy
import uuid
import pickle

import networkx as nx
import numpy as np
import pickle
import math
from .dwpose_draw_tools import draw_dwpose_2d, bodypose_24to20
from scipy.spatial.transform import Rotation as R
from .pose_retargeter import retargeter as retargeter_pose_dit
from .pose_blender import retargeter as pose_blender_dit

import random
from torchvision import transforms
from typing import List, Literal, Optional, Type, Union, Tuple
from PIL import Image
import onnxruntime as ort
from skimage.filters import gaussian
import statistics

USE_FPS = 30 #28
USE_interpolate_N = 25 #1 for self-reconstruction #17 #25
OptHeadTail = True
Headrepeat = 2 # 3 2 3 2
Tailrepeat = 2 # 2 3 3 2

DEFAULT_COLOR = [0.5, 0.5, 0.5]
# 错误码定义
class ErrorCode:
    SUCCESS: int = 0               # 成功，表示操作正常完成
    INVALID_INPUT_FORMAT: int = 1  # 输入格式异常，输入数据不符合预期格式要求
    INVALID_POSE: int = 2          # 无效姿态，视频首帧人物姿态不符合要求（如倒立）
    OTHER_ERRORS: int = 3          # 其他错误，未明确分类的异常情况

# 错误码与错误消息的映射（用于向用户展示友好的错误信息）
ERROR_MESSAGES = {
    ErrorCode.SUCCESS: "Success",  # 成功，操作已成功完成
    ErrorCode.INVALID_INPUT_FORMAT: "Invalid input format; please check your input data.",  
    ErrorCode.INVALID_POSE: "The characters in the first frame of the motion video cannot be standing upside down.", 
    ErrorCode.OTHER_ERRORS: "Other undefined errors; please contact technical support.",  
}

def make_neg_pose_full(video_xdwpose_np_final_list):

    # new_list = []
    # for arr in video_xdwpose_np_final_list:
    #     F = arr.shape[0]
    #     # 复制一份，避免修改原始数据
    #     new_arr = arr.copy()
    #     # 替换 F,1,134,0:2 部分为随机数
    #     new_arr[:, :, :, 0:2] = np.random.rand(F, 1, 134, 2)
    #     new_list.append(new_arr)
    # return new_list

    conf_thres = 5.0

    new_list = []
    for arr in video_xdwpose_np_final_list:
        F = arr.shape[0]
        new_arr = arr.copy()

        # 定义左右手范围
        ranges = [(0, 20), (92, 113), (113, 134)]  # [start, end)

        for start, end in ranges:
            for f in range(F):
                # 当前帧的手部 (N,2)
                hand_coords = new_arr[f, 0, start:end, 0:3]

                # 置信度 mask
                mask = hand_coords[:, 2] > conf_thres
                valid_coords = hand_coords[mask, :2]

                if valid_coords.shape[0] > 0:
                    # 只统计高置信度点的范围
                    x_min, y_min = valid_coords.min(axis=0)
                    x_max, y_max = valid_coords.max(axis=0)
                else:
                    # 如果没有高置信度点，就退化成用全体
                    x_min, y_min = hand_coords[:, :2].min(axis=0)
                    x_max, y_max = hand_coords[:, :2].max(axis=0)

                # 在这个范围内随机生成
                rand_x = np.random.uniform(x_min, x_max, size=(end-start,))
                rand_y = np.random.uniform(y_min, y_max, size=(end-start,))
                rand_coords = np.stack([rand_x, rand_y], axis=-1)

                # 覆盖原来的手部关键点
                new_arr[f, 0, start:end, 0:2] = rand_coords

        new_list.append(new_arr)
    return new_list


# def make_neg_pose(video_xdwpose_np_final_list):
#     new_list = []
#     for arr in video_xdwpose_np_final_list:
#         F = arr.shape[0]
#         new_arr = arr.copy()

#         # 定义两个范围
#         ranges = [(92, 113), (113, 134)]  # 注意 Python 切片右边是不包含的
        
#         for start, end in ranges:
#             # 取出该范围的子数组 (F,1,end-start,2)
#             sub_arr = new_arr[:, :, start:end, 0:2]

#             # 展平后打乱索引
#             flat = sub_arr.reshape(F, -1, 2)   # (F, N, 2)
#             for f in range(F):
#                 idx = np.arange(flat.shape[1])
#                 np.random.shuffle(idx)
#                 flat[f] = flat[f, idx]  # 打乱当前帧
#             # 放回去
#             new_arr[:, :, start:end, 0:2] = flat.reshape(F, 1, end-start, 2)

#         new_list.append(new_arr)
#     return new_list


def overlap_split_save(video_raw, output_result_path, fps, infer_group_len = 297, overlap = 25):

    infer_max_len = len(video_raw)

    stride = infer_group_len - overlap
    start_indices = list(range(0, infer_max_len, stride))
    segments = []
    for start in start_indices:
        end = start + infer_group_len
        segments.append((start, end))
        if end >= infer_max_len:
            break
        
    count = 0
    for start,end in segments:
        video_split = video_raw[start:end]
        save_name = "motion_video_{}".format(str(count))
        writer = get_writer(output_result_path+'/{}.mp4'.format(save_name), fps=fps, crf=23)
        for img in video_split:
            writer.write_frame(img)
        writer.close()
        count = count + 1


def make_neg_pose(video_xdwpose_np_final_list):
    conf_thres = 5.0
    new_list = []
    for arr in video_xdwpose_np_final_list:
        F = arr.shape[0]
        new_arr = arr.copy()

        # 定义左右手范围
        ranges = [(92, 113), (113, 134)]  # [start, end)

        for start, end in ranges:
            for f in range(F):
                # 当前帧的手部 (N,2)
                hand_coords = new_arr[f, 0, start:end, 0:2]

                # 获取当前帧手部坐标范围
                x_min, y_min = hand_coords.min(axis=0)
                x_max, y_max = hand_coords.max(axis=0)

                # hand_coords = new_arr[f, 0, start:end, 0:3]
                # # 置信度 mask
                # mask = hand_coords[:, 2] > conf_thres
                # valid_coords = hand_coords[mask, :2]

                # if valid_coords.shape[0] > 0:
                #     # 只统计高置信度点的范围
                #     x_min, y_min = valid_coords.min(axis=0)
                #     x_max, y_max = valid_coords.max(axis=0)
                # else:
                #     # 如果没有高置信度点，就退化成用全体
                #     x_min, y_min = hand_coords[:, :2].min(axis=0)
                #     x_max, y_max = hand_coords[:, :2].max(axis=0)

                # 在这个范围内随机生成
                rand_x = np.random.uniform(x_min, x_max, size=(end-start,))
                rand_y = np.random.uniform(y_min, y_max, size=(end-start,))
                rand_coords = np.stack([rand_x, rand_y], axis=-1)

                # 覆盖原来的手部关键点
                new_arr[f, 0, start:end, 0:2] = rand_coords

        head_idx_list = [0,14,15,16,17]
        for f in range(F):
            hand_coords = new_arr[f, 0, head_idx_list, 0:2]

            x_min, y_min = hand_coords.min(axis=0)
            x_max, y_max = hand_coords.max(axis=0)

            # 中心点
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2

            # 宽和高
            w = x_max - x_min
            h = y_max - y_min

            # 取最长边
            side = max(w, h)

            # 重新定义边界框（保持中心，正方形）
            x_min = cx - side / 2
            x_max = cx + side / 2
            y_min = cy - side / 2
            y_max = cy + side / 2

            rand_x = np.random.uniform(x_min, x_max, size=(5,))
            rand_y = np.random.uniform(y_min, y_max, size=(5,))
            rand_coords = np.stack([rand_x, rand_y], axis=-1)

            new_arr[f, 0, head_idx_list, 0:2] = rand_coords

        new_list.append(new_arr)
    return new_list


def rotation_matrix_from_vectors(v1, v2):
    """
    计算将向量v1旋转到向量v2方向的旋转矩阵。
    
    参数:
    v1 -- 三维向量，numpy数组
    v2 -- 三维向量，numpy数组
    
    返回:
    R -- 3x3旋转矩阵，使得 R @ v1 与 v2 同方向
    """
    # 归一化输入向量
    u = v1 / np.linalg.norm(v1)
    v = v2 / np.linalg.norm(v2)
    
    # 处理共线情况
    if np.allclose(u, v):
        return np.eye(3)  # 相同方向，返回单位矩阵
    if np.allclose(u, -v):
        # 寻找任意垂直于u的旋转轴
        if abs(u[0]) < 0.9:
            arbitrary = np.array([1, 0, 0])
        else:
            arbitrary = np.array([0, 1, 0])
        rotation_axis = np.cross(u, arbitrary)
        if np.linalg.norm(rotation_axis) < 1e-10:  # 仍然共线
            rotation_axis = np.cross(u, np.array([0, 0, 1]))
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        theta = np.pi
    else:
        # 计算旋转轴和角度
        rotation_axis = np.cross(u, v)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        theta = np.arccos(np.dot(u, v))
    
    # 使用罗德里格斯公式构建旋转矩阵
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    R = np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)
    return R
def find_closest_number(lst, a):
    if not lst:
        return None
    return min(lst, key=lambda x: abs(x - a))

def composeRT(R, T):
    RT = np.eye(4)
    RT[:3, :3] = R  # 设置旋转部分
    RT[:3, 3:] = T   # 设置平移部分
    return RT
def decomposeRT(RT):
    R = RT[:3, :3].copy()  # 提取旋转部分
    T = RT[:3, 3:].copy()   # 提取平移部分
    return R, T

def rodrigues_to_matrix(rot_vec):
    rot_vec = np.asarray(rot_vec, dtype=np.float64).flatten()
    theta = np.linalg.norm(rot_vec)
    
    if theta < 1e-10:
        return np.eye(3)
    
    axis = rot_vec / theta
    K = np.array([
        [0,        -axis[2], axis[1]],
        [axis[2],  0,        -axis[0]],
        [-axis[1], axis[0],  0]
    ])
    
    I = np.eye(3)
    R = I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R

def rotate_points(points, R):
    return np.dot(points, R.T)  # 等价于 (R @ points.T).T

def rodrigues_to_vector(R):
    R = np.asarray(R, dtype=np.float64)
    assert R.shape == (3, 3), "Rotation matrix must be 3x3"
    
    # 计算旋转角度theta
    trace = np.trace(R)
    theta_cos = (trace - 1) / 2.0
    theta = np.arccos(np.clip(theta_cos, -1.0, 1.0))  # 避免数值误差导致超出范围
    
    # 处理角度为0的情况（无旋转）
    if np.isclose(theta, 0.0):
        return np.zeros(3)
    
    # 一般情况：theta不为0或π
    if not np.isclose(theta, np.pi):
        # 计算旋转轴
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) / (2 * np.sin(theta))
        rotation_vector = theta * axis
    else:
        # 处理theta=π的特殊情况
        axis_squared = (np.diag(R) + 1) / 2.0
        axis_squared = np.maximum(axis_squared, 0.0)  # 避免负值
        max_index = np.argmax(axis_squared)  # 选择最大分量
        
        n = np.zeros(3)
        n[max_index] = np.sqrt(axis_squared[max_index])
        
        # 根据非对角元素计算其他分量
        if max_index == 0:
            n[1] = R[0, 1] / (2 * n[0])
            n[2] = R[0, 2] / (2 * n[0])
        elif max_index == 1:
            n[0] = R[0, 1] / (2 * n[1])
            n[2] = R[1, 2] / (2 * n[1])
        else:
            n[0] = R[0, 2] / (2 * n[2])
            n[1] = R[1, 2] / (2 * n[2])
        
        # 归一化并计算旋转向量
        n = n / np.linalg.norm(n)
        rotation_vector = theta * n
    
    return rotation_vector


def scale_part_batch(keypoints, parent_fixed_index, parent_move_index, child_indices, scale_factor):
    """批量处理脖子缩放 (形状保持为 [batch, 2, 134, 2])"""
    # 提取父节点坐标 [batch, 2]
    parent_fixed = keypoints[..., parent_fixed_index, :]
    parent_move = keypoints[..., parent_move_index, :]
    
    if child_indices is not None:
        # 记录孩子节点相对父节点偏移
        child_offset = keypoints[..., child_indices, :] - parent_move[..., None, :]

    # 父节点移动
    parent_vecs = parent_move - parent_fixed
    keypoints[..., parent_move_index, :] = parent_fixed + parent_vecs * scale_factor
    
    if child_indices is not None:
        # 孩子节点移动
        keypoints[..., child_indices, :] = parent_move[..., None, :] + child_offset
    

def scale_fingers_batch(keypoints, scale_factor, finger_list=[113, 114, 115, 116, 117]):

    for i in range(len(finger_list)-1):
        parent_fixed_index= finger_list[i]
        parent_move_index= finger_list[i+1]
        if i+2 < len(finger_list):
            child_indices=finger_list[i+2:]
        else:
            child_indices = None
        scale_part_batch(keypoints, parent_fixed_index, parent_move_index, child_indices, scale_factor)


def scale_hands_batch(keypoints, scale_factor):
    # keypoints (F, 1, 134, 2)

    scale_fingers_batch(keypoints, scale_factor, finger_list=[113] + [i for i in range(114, 118)])
    scale_fingers_batch(keypoints, scale_factor, finger_list=[113] + [i for i in range(118, 122)])
    scale_fingers_batch(keypoints, scale_factor, finger_list=[113] + [i for i in range(122, 126)])
    scale_fingers_batch(keypoints, scale_factor, finger_list=[113] + [i for i in range(126, 130)])
    scale_fingers_batch(keypoints, scale_factor, finger_list=[113] + [i for i in range(130, 134)])


    scale_fingers_batch(keypoints, scale_factor, finger_list=[92] + [i for i in range(93, 97)])
    scale_fingers_batch(keypoints, scale_factor, finger_list=[92] + [i for i in range(97, 101)])
    scale_fingers_batch(keypoints, scale_factor, finger_list=[92] + [i for i in range(101, 105)])
    scale_fingers_batch(keypoints, scale_factor, finger_list=[92] + [i for i in range(105, 109)])
    scale_fingers_batch(keypoints, scale_factor, finger_list=[92] + [i for i in range(109, 113)])



def interpolate_poses(a, b, n, confidence_threshold=5):
    """
    Interpolate between two poses a and b for visible keypoints, including confidence.
    Keypoints with visible=False must have coordinates (-1, -1) in both a and b.
    
    Args:
        a: numpy array of shape [1, 1, 134, 3], pose a
        b: numpy array of shape [1, 1, 134, 3], pose b
        n: int, number of interpolation frames
        confidence_threshold: float, threshold for keypoint visibility
        
    Returns:
        numpy array of shape [n, 1, 134, 3], interpolated poses from a to b
    """
    # Remove singleton dimensions for easier handling
    a = a.squeeze()  # shape [134, 3]
    b = b.squeeze()  # shape [134, 3]
    
    # Ensure invisible points have coordinates (-1, -1)
    a_low_confidence = a[:, 2] < confidence_threshold
    b_low_confidence = b[:, 2] < confidence_threshold
    low_confidence = a_low_confidence+b_low_confidence

    a[low_confidence, :] = [-1, -1, 0]  # Set invisible points to (-1, -1)
    b[low_confidence, :] = [-1, -1, 0]  # Set invisible points to (-1, -1)
    
    # Determine visible keypoints (where both a and b are not (-1, -1))
    a_not_invalid = (a[:, 0] != -1) & (a[:, 1] != -1)
    b_not_invalid = (b[:, 0] != -1) & (b[:, 1] != -1)
    visible = a_not_invalid & b_not_invalid
    
    # Initialize output array
    interpolated = np.zeros((n, 1, 134, 3))
    
    # Generate interpolation weights (n steps from 0 to 1)
    weights = np.linspace(0, 1, n)
    
    for i, alpha in enumerate(weights):
        # Start with a copy of pose a
        frame = a.copy()
        
        # Interpolate visible keypoints (x, y, s)
        frame[visible, :] = (1 - alpha) * a[visible, :] + alpha * b[visible, :]
        
        # Reshape to match output dimensions
        interpolated[i, 0] = frame
    
    return interpolated


def slerp_pose(img_pose, video_pose, N=24, use_3d=True):
    # 对body pose插值
    # origin_size = video_pose['body_pose'].shape[0]
    inter_pose = torch.zeros(N, 63)
    inter_global = torch.zeros(N, 3)
    img_body_pose = img_pose['body_pose'][0]
    video_body_pose = video_pose['body_pose']
    img_global_pose = img_pose['global_orient'][0]
    video_global_pose = video_pose['global_orient']

    # if use_3d:
    #     for frame_id in range(N):
    #         for joint_id in range(21):
    #             img_joint_pose = img_body_pose[joint_id*3: joint_id*3+3]
    #             video_joint_pose = video_body_pose[0, joint_id*3: joint_id*3+3]
    #             inter_pose[frame_id, joint_id*3: joint_id*3+3] = torch.from_numpy(slerp_vec(img_joint_pose, video_joint_pose, 0, N, frame_id))
    #         inter_global[frame_id] = torch.from_numpy(slerp_vec(img_global_pose, video_global_pose[0], 0, N, frame_id))
    # else:
    #     for frame_id in range(N-10):
    #         for joint_id in range(21):
    #             img_joint_pose = img_body_pose[joint_id*3: joint_id*3+3]
    #             video_joint_pose = video_body_pose[0, joint_id*3: joint_id*3+3]
    #             inter_pose[frame_id+5, joint_id*3: joint_id*3+3] = torch.from_numpy(slerp_vec(img_joint_pose, video_joint_pose, 0, N-10, frame_id)) 
    #         inter_global[frame_id+5] = torch.from_numpy(slerp_vec(img_global_pose, video_global_pose[0], 0, N-10, frame_id))
    #     for frame_id in range(5):
    #         inter_pose[frame_id] = inter_pose[5]
    #         inter_pose[N-frame_id-1] = inter_pose[N-6]
    #         inter_global[N-frame_id-1] = inter_global[N-6]
    for frame_id in range(N):
        for joint_id in range(21):
            img_joint_pose = img_body_pose[joint_id*3: joint_id*3+3]
            video_joint_pose = video_body_pose[0, joint_id*3: joint_id*3+3]
            inter_pose[frame_id, joint_id*3: joint_id*3+3] = torch.from_numpy(slerp_vec(img_joint_pose, video_joint_pose, 0, N, frame_id))
        inter_global[frame_id] = torch.from_numpy(slerp_vec(img_global_pose, video_global_pose[0], 0, N, frame_id))
    video_pose['body_pose'] = torch.concatenate((inter_pose, video_body_pose), dim=0)
    video_pose['global_orient'] = torch.concatenate((inter_global, video_global_pose), dim=0)

    
    # 其他的key也相应的插值
    video_pose['transl'] = torch.concatenate((video_pose['transl'][0].repeat(N).reshape(N,-1), video_pose['transl']), dim=0)
    video_pose['betas'] = torch.concatenate((video_pose['betas'][0].repeat(N).reshape(N,-1), video_pose['betas']), dim=0)
    video_pose['left_hand_pose'] = torch.concatenate((video_pose['left_hand_pose'][0].repeat(N).reshape(N,-1), video_pose['left_hand_pose']), dim=0)
    video_pose['right_hand_pose'] = torch.concatenate((video_pose['right_hand_pose'][0].repeat(N).reshape(N,-1), video_pose['right_hand_pose']), dim=0)
    return video_pose


def slerp_vec(v1, v2, i_before, i_after, i):
    t = (i-i_before)/(i_after-i_before)
    q1 = R.from_rotvec(v1).as_quat()
    q2 = R.from_rotvec(v2).as_quat()
    res = slerp_q(q1, q2, t)
    res = R.from_quat(res).as_rotvec()
    return res


def slerp_q(q1, q2, t):
    cos_theta = clamp(np.dot(q1, q2) / (np.linalg.norm(q1) * np.linalg.norm(q2)), -1, 1)
    abs_cos_theta = abs(cos_theta)
    theta = math.acos(abs_cos_theta)

    # 退化至lerp
    if abs_cos_theta >= 1.0:
        q = (1-t) * q1 + t * q2
        return q / np.linalg.norm(q)

    a_t = math.sin((1-t)*theta) / math.sin(theta)
    b_t = math.sin(t*theta) / math.sin(theta)
    b_t = b_t if cos_theta>0 else -b_t
    q = a_t * q1 + b_t * q2
    return q / np.linalg.norm(q)

def clamp(value, min_val, max_val):
    return max(min(value, max_val), min_val)


class Retarget:
    def __init__(self, smpl_model_path='/ytech_milm/liujiwen/kling_motion_service/smpl_all_checkpoints/checkpoints', face_width=224, face_height=224):
        self.retargeter_pose_dit = retargeter_pose_dit(smpl_model_path+'/motion_retarget/config_noconf.yaml', smpl_model_path+'/motion_retarget/noconf.pt', smpl_model_path+'/motion_retarget/config_forconf.yaml', smpl_model_path+'/motion_retarget/forconf.pt', 'cuda')
        self.pose_blender_dit = pose_blender_dit(smpl_model_path+'/motion_retarget/config_blend.yaml', smpl_model_path+'/motion_retarget/blend.pt', 'cuda')
        self.smpl_model_path = smpl_model_path
        self.smplx = make_smplx("supermotion",smpl_model_path=smpl_model_path).cuda()
        self.smplx2smpl = torch.load(os.path.join(smpl_model_path,"body_models/smplx2smpl_sparse.pt")).cuda()
        self.faces_smpl = make_smplx("smpl", smpl_model_path=smpl_model_path).faces
        occ_vert = np.load(os.path.join(smpl_model_path, 'body_models/smpl_hand_verts.npz'))
        self.left_hand_inds =occ_vert['smpl_left_hand_verts'].astype(np.int16)
        self.right_hand_inds =occ_vert['smpl_right_hand_verts'].astype(np.int16)
        self.add_lmk_ids=[(331,15),(2800,57),(6262,56),(550,59),(4036,58)] #nose, left/right eyes, left/right ears
        self.colormap = np.array(cm.get_cmap("inferno").colors)
        self.error_code = ErrorCode()
        self.error_messages = ERROR_MESSAGES
        self.dw_points =[15,12,17,19,21,16,18,20,2,5,8,1,4,7,56,57,58,59,10,11]
        self.hand_point=[20,37,38,39,66,25,26,27,67,28,29,30,68,34,35,36,69,31,32,33,70, 
                    21,52,53,54,71,40,41,42,72,43,44,45,73,49,50,51,74,46,47,48,75]
        self.dw_bones = [ [12,17],[12,16],[17,19],[19,21],[16,18],[18,20],[12,2],[2,5],[5,8],[12,1],[1,4],[4,7],[12,15],[15,56],[56,58],[15,57],[57,59],[7,10],[8,11], \
                [20, 37],   # left hand from my view
                [37, 38],   # left_thumb
                [38, 39],
                [39, 66],   # finger tips
                [20, 25],   
                [25, 26],   # left_index
                [26, 27],   
                [27, 67],   # finger tips
                [20, 28],
                [28, 29],   # left_middle
                [29, 30],   
                [30, 68],
                [20, 34],
                [34, 35],   # left_ring
                [35, 36],
                [36, 69],
                [20, 31],
                [31, 32],   # left_pinky
                [32, 33],
                [33, 70],
                [21, 52],   # right hand from my view
                [52, 53],   # right_thumb
                [53, 54],
                [54, 71],   # finger tips
                [21, 40],   
                [40, 41],   # right_index
                [41, 42],   
                [42, 72],   # finger tips
                [21, 43],
                [43, 44],   # right_middle
                [44, 45],   
                [45, 73],   # finger tips
                [21, 49],
                [49, 50],   # right_ring
                [50, 51],
                [51, 74],   # finger tips
                [21, 46],   
                [46, 47],   # right_pinky
                [47, 48],
                [48, 75],   # finger tips
            ]
        
        self.face_width = face_width
        self.face_height = face_height

    def render_depth(self,verts,width,height,k):
        renderer = Renderer(width, height, device="cuda", faces=self.faces_smpl, K=k)
        depth_list =[]
        # _list = []
        for i in tqdm(range(verts.shape[0])):
            raw0=np.zeros([height,width,3])
            verts[i,self.left_hand_inds]=verts[i,1962]
            verts[i,self.right_hand_inds]=verts[i,5705]
            # _, depth = renderer.render_mesh(torch.tensor(verts[i]).cuda(), raw0, [0.8, 0.8, 0.8])
            depth = renderer.render_mesh(torch.tensor(verts[i]).cuda(), raw0, [0.8, 0.8, 0.8])
            # _list.append(_)
            depth_list.append(depth)
        return np.array(depth_list)

    def draw_dwpose(self, img, start, end, height, width, colors, stickwidth = 6):
        if start[0] == -1 or start[1] == -1 or end[0] == -1 or end[1] == -1:
            return img
        X = np.array([start[1],end[1]])
        X *= width
        Y = np.array([start[0],end[0]])
        Y *= height
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(img, polygon, colors)
        return img

    
    def draw_lines(self, img, start, end, height, width, depthx, depthy, thickness=6):
        if start[0] == -1 or start[1] == -1 or end[0] == -1 or end[1] == -1:
            return img
        if int(start[0]) ==0 and int(start[1]) ==0:
            return img
        if int(end[0]) ==0 and int(end[1]) ==0:
            return img

        colorx = (self.colormap[(depthx*255).astype(np.uint8)]*255).astype(np.uint8)
        colory = (self.colormap[(depthy*255).astype(np.uint8)]*255).astype(np.uint8)
        #print(tuple(colorx.tolist()))
        img = cv2.circle(img, (int(start[0]*height), int(start[1]*width)), thickness, tuple(colorx.tolist()), thickness=-1)
    
        max_span = np.max(np.abs(start-end))
        span = int(max_span//thickness+1)
        for i in range(span):
            x,y = ((span-i)*start+i*end)//span
            cx = ((span-i)*depthx+i*depthy)/span
            colorcx = (self.colormap[(cx*255).astype(np.uint8)]*255).astype(np.uint8)
            img = cv2.circle(img, (int(x*height), int(y*width)), thickness, tuple(colorcx.tolist()), thickness=-1)
        img = cv2.circle(img, (int(end[0]*height), int(end[1]*width)), thickness, tuple(colory.tolist()), thickness=-1)
        return img    
    
    def cal_rot_diff(self,ori1,ori2):
        '''
        orient1:np rot mat of video first frame
        orient2: np rot mat of image
        
        return mot_or_img :bool 
        '''
        '''项量测试法
        test_vec = np.array([0,0,1])
        dir1_vec = ori1@test_vec
        dir2_vec = ori2@test_vec
        '''
        yaw1 = np.arctan2(ori1[1, 0], ori1[0, 0])
        yaw2 = np.arctan2(ori2[1, 0], ori2[0, 0])
        angle_diff = yaw2 - yaw1
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        print('*#'*20, angle_diff/np.pi*180, '*#'*20)
        if np.abs(angle_diff) > (np.pi/18): # 4.5->40度, 18->10度
            print('*#'*20, 'False', '*#'*20)
            return False
        else:
            print('*#'*20, 'True', '*#'*20)
            return True


    def angle_between_vectors(self, v1, v2, degrees=True, directional=False):

        v1 = np.asarray(v1)
        v2 = np.asarray(v2)
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm == 0 or v2_norm == 0:
            raise ValueError("输入向量不能为零向量")
        dot_product = np.dot(v1, v2)
        
        cos_theta = dot_product / (v1_norm * v2_norm)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        if directional:
            cross_product = np.cross(v1, v2)
            angle = np.arctan2(np.abs(cross_product), dot_product)
            if cross_product < 0:
                angle = -angle
        else:
            angle = np.arccos(cos_theta)
        
        if degrees:
            angle = np.degrees(angle)
        
        return angle


    def cal_rot_diff_v2(self,vid,img):
        '''
        orient1:np rot mat of video first frame
        orient2: np rot mat of image
        
        return mot_or_img :bool 
        '''

        test_vec = np.array([[0],[0],[1]])

        inds=np.array([0,2])
        angle_diff = self.angle_between_vectors((img@test_vec)[inds,0],(vid@test_vec)[inds,0]) #角度


        print('*#'*20, angle_diff, '*#'*20)
        if np.abs(angle_diff) > 15:
            print('*#'*20, 'False', '*#'*20)
            return False
        else:
            print('*#'*20, 'True', '*#'*20)
            return True


    def retarget_rt(self, pred_src, pred_img, maintain_orient=True, force_direction=False):
        
        print(pred_img.keys())
        if "smpl_params_incam" not in pred_src.keys():
            pred_src["smpl_params_incam"] = pred_src["smpl_params_global"]
        pred_res = copy.deepcopy(pred_src)
        pred_res["K_fullimg"] = pred_img["K_fullimg"]
        print(pred_img["smpl_params_incam"].keys())
        print(pred_src["smpl_params_global"].keys())
        total_len = pred_src["smpl_params_global"]["body_pose"].shape[0]
        #判定image 朝向与 video朝向 差距
        orient_video = rodrigues_to_matrix(pred_src["smpl_params_incam"]["global_orient"][0,...].numpy())
        orient_image = rodrigues_to_matrix(pred_img["smpl_params_incam"]["global_orient"][0,...].numpy())
        mot_or_img = ~maintain_orient
        if not force_direction:
            mot_or_img = self.cal_rot_diff_v2(orient_video,orient_image)#True为motion朝向
        if maintain_orient:
            for i in range(0, total_len):
                #r0 = rodrigues_to_matrix(pred_src["smpl_params_global"]["global_orient"][0,...].numpy())
                r0 = rodrigues_to_matrix(pred_src["smpl_params_incam"]["global_orient"][0,...].numpy())
                #rimg = rodrigues_to_matrix(pred_img["smpl_params_incam"]["global_orient"][0,...].numpy())
                rimg = rodrigues_to_matrix(pred_img["smpl_params_incam"]["global_orient"][0,...].numpy())
                R0 = rodrigues_to_matrix(pred_src["smpl_params_global"]["global_orient"][0,...].numpy())
                R1 = rodrigues_to_matrix(pred_src["smpl_params_global"]["global_orient"][i,...].numpy())
                if i ==0:
                    
                    #判断首帧朝向
                    test_vec = np.array([0,1,0])
                    test_res = r0@test_vec
                    if False:
                        return pred_res, True
                    norm_vec =np.array([0,-1,0])
                    curr_vec = rimg @ norm_vec
                    fy_rimg = rotation_matrix_from_vectors(curr_vec,np.array([0,1,0]))
                    rfy_rimg = fy_rimg @ rimg
                    
                    norm_vec =np.array([0,-1,0])
                    curr_vec = r0 @ norm_vec
                    fy_r0 = rotation_matrix_from_vectors(curr_vec,np.array([0,1,0]))
                    rfy_r0 = fy_r0 @ r0

                    norm_vec =np.array([0,0,1])
                    curr1_vec = rfy_r0 @ norm_vec
                    curr2_vec = rfy_rimg @norm_vec
                    rot_y = rotation_matrix_from_vectors(curr1_vec,curr2_vec)
                r0 = rot_y @ r0

                timg  = pred_img["smpl_params_incam"]["transl"][0,...].view(3,1).numpy()
                t0 = pred_src["smpl_params_incam"]["transl"][0,...].view(3,1).numpy()
                T0 = pred_src["smpl_params_global"]["transl"][0,...].view(3,1).numpy()
                T1 = pred_src["smpl_params_global"]["transl"][i,...].view(3,1).numpy()
                rt0=composeRT(r0,t0)
                RT0=composeRT(R0,T0)
                RT1=composeRT(R1,T1)
                r1,t1= decomposeRT(rt0@np.linalg.inv(RT0)@RT1)
                if i ==0:
                    t_start =t1
                r1 =rodrigues_to_vector(r1)
                # 用一个比较胖的betas，模拟穿衣服的情况，避免穿模
                # pred_res["smpl_params_incam"]["betas"][i,...] = pred_img["smpl_params_incam"]["betas"][0,...].clone()
                pred_res["smpl_params_incam"]["betas"][i,...] = torch.tensor([ 0.5662, -0.1761,  0.4324,  1.0520,  0.7279,  0.0582,  0.2008,  0.4091, 0.2780,  0.4773])
                pred_res["smpl_params_incam"]["global_orient"][i,...] =torch.tensor(r1)
                pred_res["smpl_params_incam"]["transl"][i,...] = torch.tensor(t1).squeeze() - torch.tensor(t_start).squeeze() + pred_img["smpl_params_incam"]["transl"][0,...]
        else:
            for i in range(0, total_len):
                r0 = rodrigues_to_matrix(pred_src["smpl_params_incam"]["global_orient"][0,...].numpy())
                R0 = rodrigues_to_matrix(pred_src["smpl_params_global"]["global_orient"][0,...].numpy())
                R1 = rodrigues_to_matrix(pred_src["smpl_params_global"]["global_orient"][i,...].numpy())
            
                t0 = pred_src["smpl_params_incam"]["transl"][0,...].view(3,1).numpy()
                T0 = pred_src["smpl_params_global"]["transl"][0,...].view(3,1).numpy()
                T1 = pred_src["smpl_params_global"]["transl"][i,...].view(3,1).numpy()
                rt0=composeRT(r0,t0)
                RT0=composeRT(R0,T0)
                RT1=composeRT(R1,T1)
                r1,t1= decomposeRT(rt0@np.linalg.inv(RT0)@RT1)
                r1 =rodrigues_to_vector(r1)
                pred_res["smpl_params_incam"]["betas"][i,...] = pred_img["smpl_params_incam"]["betas"][0,...].clone()
                # pred_res["smpl_params_incam"]["betas"][i,...] = torch.tensor([ 0.5662, -0.1761,  0.4324,  1.0520,  0.7279,  0.0582,  0.2008,  0.4091, 0.2780,  0.4773])
                pred_res["smpl_params_incam"]["global_orient"][i,...] =torch.tensor(r1)
                pred_res["smpl_params_incam"]["transl"][i,...]= torch.tensor(t1).view(1,3)
                pred_res["smpl_params_incam"]["transl"][i,...] = pred_res["smpl_params_incam"]["transl"][i,...] \
                                                            - pred_src["smpl_params_incam"]["transl"][0,...] \
                                                            + pred_img["smpl_params_incam"]["transl"][0,...]
        pred_res["smpl_params_global"] = pred_res["smpl_params_incam"]

        return pred_res, False, mot_or_img

    # 修复depth     
    def joint_occ(self, joint_id, depth, x, y ,z_buff):
        if x<0 or x>z_buff.shape[1] or y<0 or y>z_buff.shape[0] or depth<0:
            return 1
        # 认为手腕点都可见      不考虑手指顶点和大拇指不可见的情况
        if joint_id in [20, 21, 37, 38, 39, 27, 30, 36, 33, 52, 53, 54, 42, 45, 51, 48]+[i for i in range(66, 76)]:
            return 1
        if depth< z_buff[int(y),int(x)]:
            return 1
        return 0

    def painter_smpl(self, out_path,pt_list,video_shape,feat_list, is_motion_space=False, interpolate_N=24):
        vid_len, width,height,fps = video_shape
        if not is_motion_space:
            ori_width = feat_list[0]['width']
            ori_height =feat_list[0]['height']
            width_ratio = width / ori_width
            height_ratio = height / ori_height
        ######## 
        img_list = []
        joints_list = []
        z_buff_ori_list = []
        min_frame_num = 1000
        K = None
        for pred_pt in pt_list:
            min_frame_num = min(pred_pt["smpl_params_incam"]["body_pose"].shape[0], min_frame_num)
        print ('min_frame_num', min_frame_num)

        for pred_pt in pt_list:
            K = pred_pt["K_fullimg"][0]
            smplx_out = self.smplx(**to_cuda(pred_pt["smpl_params_incam"]))
            pred_c_joints = smplx_out.joints[:min_frame_num,:,:]
            joints = perspective_projection(pred_c_joints,K=K.cuda())
            joints = joints.detach().cpu().numpy()
            joints[:,12] = (joints[:,16]+joints[:,17])//2
            newA = (3*joints[:,1]-joints[:,2])/2
            newB = (3*joints[:,2]-joints[:,1])/2
            joints[:,1] =newA
            joints[:,2] =newB

            ### replace five facial landmarks
            pred_c_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smplx_out.vertices[:min_frame_num]])
            verts = perspective_projection(pred_c_verts,K=K.cuda())
            verts = verts.detach().cpu().numpy()
            pred_c_verts = pred_c_verts.detach().cpu().numpy()

            if not is_motion_space:
                z_buff_ori = self.render_depth(pred_c_verts,ori_width,ori_height,K.cuda())
                z_buff_ori_list.append(z_buff_ori)
                print('z_buff.shape', '\t', z_buff_ori.shape)
 
            for vert_id,joint_id in self.add_lmk_ids:
                joints[:,joint_id] = verts[:,vert_id]
            #################################
            pred_c_joints = pred_c_joints.detach().cpu().numpy()
            for iframe in range(pred_c_joints.shape[0]):
                for ijoint in range(pred_c_joints.shape[1]):
                    if pred_c_joints[iframe,ijoint,2]<0:
                        joints[iframe,ijoint,0]=-1
                        joints[iframe,ijoint,1]=-1
            
            joints_list.append(joints)

            if joints.shape[0] < min_frame_num:
                min_frame_num= joints.shape[0]

        # 如果重定向时有两个人，考虑深度图的重叠
        if not is_motion_space:
            if len(z_buff_ori_list)==1:
                z_buff_ori = z_buff_ori_list[0]
            else:
                z_buff_ori = np.min(np.stack(z_buff_ori_list), axis=0)

        if OptHeadTail:
            # TODO：尾帧崩坏和ending-pose兼容
            vid_len = min(min_frame_num, vid_len)
        else:
            print('min_frame_num', min_frame_num)
            ##vid_len = min((((min_frame_num-1)//8)*8+1),vid_len)
            vid_len = min((((min_frame_num-1)//8)*8+1),vid_len)
            print("vid_len--------------", vid_len)

        # 首帧dwpose
        dwposes_list = []
        for ibody in range(len(joints_list)):
            dwpose_134 = np.ones((134, 2), dtype=np.float32) * -1.0
            if is_motion_space:
                # ori_width = feat_list[ibody]['data_list'][0]['width']
                # ori_height = feat_list[ibody]['data_list'][0]['height']
                width_ratio = 1.0
                height_ratio = 1.0
            for ijoint in range(20):
                x, y, _ = joints_list[ibody][0, self.dw_points[ijoint]]
                dwpose_134[ijoint][0] = x * width_ratio
                dwpose_134[ijoint][1] = y * height_ratio
            for ijoint in range(len(self.hand_point)):
                x, y, _ = joints_list[ibody][0, self.hand_point[ijoint]]
                dwpose_134[ijoint+92][0] = x * width_ratio
                dwpose_134[ijoint+92][1] = y * height_ratio
            dwposes_list.append(dwpose_134)
        if is_motion_space:
            return vid_len, dwposes_list
        
        dwposes_list = []
        for ibody in range(len(joints_list)):
            dwpose_all_frames_134 = np.ones((vid_len, 1, 134, 3), dtype=np.float32) * 0.0 # (F 1 134 3)
            for iframe in tqdm(range(vid_len)):
                dwpose_134 = dwpose_all_frames_134[iframe, 0] # (134, 2)

                # 判断手部关键点是否遮挡，通过关键点的遮挡来判断整手是否可见
                joints_vis = np.zeros([2, 21])
                hand_vis = np.zeros(2)
                for iright_idx in range(2):
                    for ijoint_idx in range(21):
                        ijoint = self.hand_point[21*iright_idx+ijoint_idx]
                        x,y,d = joints_list[ibody][iframe,ijoint]
                        joints_vis[iright_idx, ijoint_idx] = self.joint_occ(ijoint, d, x, y, z_buff_ori[iframe])
                    if not 0 in joints_vis[iright_idx]:
                        hand_vis[iright_idx] = 1

                for ijoint in range(20):
                    x, y, _ = joints_list[ibody][iframe,self.dw_points[ijoint]]
                    if x == -1 or y ==-1:
                        dwpose_134[ijoint][2] = 0.0
                    else:
                        dwpose_134[ijoint][2] = 10.0
                        dwpose_134[ijoint][0] = x * width_ratio
                        dwpose_134[ijoint][1] = y * height_ratio
                for ijoint in range(len(self.hand_point)):
                    if (ijoint<21 and not hand_vis[0]) or (ijoint>=21 and not hand_vis[1]):
                        x, y = -1, -1
                    else:
                        x, y, _ = joints_list[ibody][iframe, self.hand_point[ijoint]]
                    if x == -1 or y ==-1:
                        dwpose_134[ijoint+92][2] = 0.0
                    else:
                        dwpose_134[ijoint+92][2] = 10.0
                        dwpose_134[ijoint+92][0] = x * width_ratio
                        dwpose_134[ijoint+92][1] = y * height_ratio
            dwposes_list.append(dwpose_all_frames_134)

        return vid_len, dwposes_list

    def painter_smpl_depth(self, out_path, pt_list, video_shape, feat_list):
        vid_len, width,height,fps = video_shape
        ori_width = feat_list[0]['width']
        ori_height =feat_list[0]['height']
        width_ratio = width / ori_width
        height_ratio = height / ori_height

        img_list = []
        joints_list =[]
        depth_list =[]
        min_frame_num = 1000
        K = None
        for pred_pt in pt_list:
            K = pred_pt["K_fullimg"][0]
            smplx_out = self.smplx(**to_cuda(pred_pt["smpl_params_incam"]))
            pred_c_joints = smplx_out.joints[:,:,:]

            
            joints = perspective_projection(pred_c_joints, K=K.cuda())
            joints = joints.detach().cpu().numpy()
            pred_c_joints = pred_c_joints.detach().cpu().numpy()
            depth = pred_c_joints[...,2].copy()
            depth = 1/depth
            depth = (depth-depth.min())/(depth.max()-depth.min())

            joints[:,12] = (joints[:,16]+joints[:,17])//2
            newA = (3*joints[:,1]-joints[:,2])/2
            newB = (3*joints[:,2]-joints[:,1])/2
            joints[:,1] =newA
            joints[:,2] =newB

            ### replace five facial landmarks
            pred_c_verts = torch.stack([torch.matmul(self.smplx2smpl, v_) for v_ in smplx_out.vertices])
            verts = perspective_projection(pred_c_verts,K=K.cuda())
            verts = verts.detach().cpu().numpy()
            pred_c_verts = pred_c_verts.detach().cpu().numpy()

            for vert_id,joint_id in self.add_lmk_ids:
                joints[:,joint_id] = verts[:,vert_id]

            for iframe in range(pred_c_joints.shape[0]):
                for ijoint in range(pred_c_joints.shape[1]):
                    if pred_c_joints[iframe,ijoint,2]<0:
                        joints[iframe,ijoint,0]=-1
                        joints[iframe,ijoint,1]=-1
            
            joints_list.append(joints)
            depth_list.append(depth)

            if joints.shape[0] < min_frame_num:
                min_frame_num= joints.shape[0]
        print(min_frame_num)
        vid_len = min((((min_frame_num-1)//8)*8+1),vid_len)
        print(vid_len, 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        for iframe in tqdm(range(vid_len)):
            img =np.zeros([height,width,3])
            if iframe >= min_frame_num:
                img_list.append(img)
                continue
            for ibody in range(len(joints_list)):
                for index,ibones in enumerate(self.dw_bones):
                    if index > 18:
                        img = self.draw_lines(img,joints_list[ibody][iframe,ibones[0]],joints_list[ibody][iframe, ibones[1]], \
                                 height_ratio, width_ratio, depth_list[ibody][iframe,ibones[0]], depth_list[ibody][iframe,ibones[1]], 3)
                    else:
                        img = self.draw_lines(img,joints_list[ibody][iframe,ibones[0]],joints_list[ibody][iframe, ibones[1]], \
                                 height_ratio, width_ratio, depth_list[ibody][iframe,ibones[0]], depth_list[ibody][iframe,ibones[1]], 6)

            img_list.append(img)
        writer = get_writer(out_path+'/motion_video.mp4', fps=fps, crf=23)
        for img in img_list:

            writer.write_frame(img)
        writer.close()

        with open(out_path+'/motion_video.mp4', "rb") as f:
            motion_video_binary = f.read()
        
        return motion_video_binary


    def smpl2bvh(self, retarget_result_list, out_path, use_trans=False):
        result = []
        for index, pred_pt in enumerate(retarget_result_list):
            bvh_pt = copy.deepcopy(pred_pt)
            bvh_pt = hmr2bvh(bvh_pt)
            #bvh_binary = smplx2bvh(model_path=self.smpl_model_path, poses=bvh_pt, output=out_path+'/'+str(index)+'_motion.bvh', use_trans=False)
            bvh_binary = smplx2bvh(model_path=self.smpl_model_path, poses=bvh_pt, output=out_path+'/'+str(index)+'_motion.bvh', use_trans=False, fps=USE_FPS, max_f=300)
            result.append(bvh_binary)
        return result


    def get_image_for_i2v(self, image, tgt_size, spatial_scale_factor=16):
        h, w = image.shape[:2]
        tgt_size2_dict = {720:1280, 1080:1920}
        if tgt_size not in tgt_size2_dict:
            tgt_size, tgt_size2 = 720, 1280
        else:
            tgt_size2 = tgt_size2_dict[tgt_size]

        area = tgt_size * tgt_size2
        # area = 640200
        area1 = h*w

        s = math.sqrt(area / area1)
        new_w = int(w*s)
        new_h = int(h*s)
        
        resize_w = new_w
        resize_h = new_h
        image = cv2.resize(image, (new_w, new_h))[:new_h//spatial_scale_factor*spatial_scale_factor, :new_w//spatial_scale_factor*spatial_scale_factor]

        return image, new_w//spatial_scale_factor*spatial_scale_factor, new_h//spatial_scale_factor*spatial_scale_factor, resize_w, resize_h


    def get_bbox_from_dwpose(self, face_coordinate):

        x1 = int(np.min(face_coordinate[:,0]))
        y1 = int(np.min(face_coordinate[:,1]))
        x2 = int(np.max(face_coordinate[:,0]))
        y2 = int(np.max(face_coordinate[:,1]))
        face_bbox = [y1, x1, y2, x2]

        return face_bbox


    def bbox_list_size_filter2(self, bbox_list, w, h):

        bbox_w_list = self.moving_average_filter([bbox[2]-bbox[0] for bbox in bbox_list], 7)
        bbox_h_list = self.moving_average_filter([bbox[3]-bbox[1] for bbox in bbox_list], 7)
        bbox_cx_list = self.moving_average_filter([(bbox[2]+bbox[0])/2 for bbox in bbox_list], 3)
        bbox_cy_list = self.moving_average_filter([(bbox[3]+bbox[1])/2 for bbox in bbox_list], 3)

        bbox_list = [self.reshape_bbox2(w, h, bbox_w_list[i], bbox_h_list[i], bbox_cx_list[i], bbox_cy_list[i]) for i, bbox in enumerate(bbox_list)]

        return bbox_list


    def moving_average_filter(self, input_list, window_size):
        filtered_list = []

        for i in range(len(input_list)):
            if i < window_size // 2:
                window = input_list[:i+window_size]
            elif i >= len(input_list) - window_size // 2:
                window = input_list[i-window_size+1:]
            else:
                window = input_list[i-window_size//2:i+window_size//2+1]
            average = sum(window) / len(window)
            filtered_list.append(average)

        return filtered_list

    def moving_max_filter(self, input_list, window_size):
        filtered_list = []

        for i in range(len(input_list)):
            if i < window_size // 2:
                window = input_list[:i+window_size]
            elif i >= len(input_list) - window_size // 2:
                window = input_list[i-window_size+1:]
            else:
                window = input_list[i-window_size//2:i+window_size//2+1]
            max_value = max(window)
            filtered_list.append(max_value)

        return filtered_list


    def square2center_crop_bbox(self, bbox, w, h, dx=0, dy=0, crop_radio=2.5, w_h_radio=1, cx=None, cy=None, black_edge_bbox=None):

        if black_edge_bbox is None:
            min_w, min_h = 0,0
        else:
            min_w, min_h, w, h = black_edge_bbox

        width_length_ratio = w_h_radio
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]

        face_x = (x_min+y_min)/2
        if cx is None:
            cx, cy = (x_min+x_max)/2, (y_min+y_max)/2

        cx, cy = cx+dx, cy+dy

        bbox_w, bbox_h = x_max-x_min, y_max-y_min

        x_min = int(max(cx-bbox_w/2, min_w))
        x_max = int(min(cx+bbox_w/2, w))
        y_min = int(max(cy-bbox_h/2, min_h))
        y_max = int(min(cy+bbox_h/2, h))

        bbox_w, bbox_h = x_max-x_min, y_max-y_min

        # print (bbox_h/bbox_w,1)

        # 以人体框为基础，等比例外扩长或者宽，使得框接近目标比例,这是为了保证人尽可能居中
        if bbox_w/bbox_h > width_length_ratio:
            new_bbox_w = bbox_w
            new_bbox_h = new_bbox_w/width_length_ratio
        else:
            new_bbox_h = bbox_h
            new_bbox_w = new_bbox_h*width_length_ratio

        dx, dy = new_bbox_w-bbox_w, new_bbox_h-bbox_h
        x_min = max(int(x_min-dx/2), min_w)
        y_min = max(int(y_min-dy/2), min_h)
        x_max = min(int(x_max+dx/2), w)
        y_max = min(int(y_max+dy/2), h)

        # 上一步如果没有达到目标比例：非等比例外扩，使得框进一步接近目标比例
        bbox_w, bbox_h = x_max-x_min, y_max-y_min
        # print (bbox_h/bbox_w,2)
        dx, dy = new_bbox_w-bbox_w, new_bbox_h-bbox_h

        x_min = max(int(x_min-dx), min_w)
        y_min = max(int(y_min-dy), min_h)
        x_max = min(int(x_max+dx), w)
        y_max = min(int(y_max+dy), h)

        bbox_w, bbox_h = x_max-x_min, y_max-y_min

        # print (bbox_h/bbox_w,3)

        # 上一步得到了不crop人体区域情况下最接近目标比例的bbox，
        # 下面对bbox进行crop，使其比例达到精确的目标比例
        # 这一步会破会人体完整性，所以只能保证脸不出画
        # y方向保上面，x方向保脸
        if bbox_w/bbox_h > width_length_ratio:
            new_bbox_h = bbox_h
            new_bbox_w = new_bbox_h*width_length_ratio
        else:
            new_bbox_w = bbox_w
            new_bbox_h = new_bbox_w/width_length_ratio

        dx, dy = bbox_w-new_bbox_w, bbox_h-new_bbox_h

        cx, cy = face_x, y_max

        # (cx-x_min)/new_bbox_w

        x_min = max(int(x_min+dx*(cx-x_min)/bbox_w), min_w)
        y_min = max(int(y_min+dy*(cy-y_min)/bbox_h), min_h)
        x_max = min(int(x_min+new_bbox_w), w)
        y_max = min(int(y_min+new_bbox_h), h)

        bbox_w, bbox_h = x_max-x_min, y_max-y_min

        # print (bbox_h/bbox_w,4)
        # 最后一步，比例不变情况下扩大画幅，尽可能保背景
        crop_ratio = min([h/bbox_h, w/bbox_w, crop_radio])

        new_bbox_w = int(bbox_w*crop_ratio)//2*2
        new_bbox_h = new_bbox_w * w_h_radio

        dx, dy = new_bbox_w-bbox_w, new_bbox_h-bbox_h
        x_min = max((x_min-dx/2), min_w)
        y_min = max((y_min-dy/2), min_h)
        x_max = min((x_max+dx/2), w)
        y_max = min((y_max+dy/2), h)
        bbox_w, bbox_h = x_max-x_min, y_max-y_min
        dx, dy = new_bbox_w-bbox_w, new_bbox_h-bbox_h
        x_min = max(int(x_min-dx), min_w)
        y_min = max(int(y_min-dy), min_h)
        x_max = min(int(x_max+dx), w)
        y_max = min(int(y_max+dy), h)

        bbox_w, bbox_h = x_max-x_min, y_max-y_min

        bbox_w, bbox_h = x_max-x_min, y_max-y_min

        bbox = [x_min, y_min, x_max, y_max]

        return bbox


    def reshape_bbox2(self, w, h, bw, bh, cx, cy):

        x1 = int(max(cx-bw/2, 0))
        y1 = int(max(cy-bh/2, 0))
        x2 = int(min(cx+bw/2, w))
        y2 = int(min(cy+bh/2, h))

        face_bbox = [x1, y1, x2, y2]

        return face_bbox


    def crop_bbox_smooth(self, crop_bbox_list, k=7):
        #平滑crop_bbox_list，使得tgt视频是个平滑的视频片段

        n = len(crop_bbox_list)
        x1_list = self.moving_average_filter([bbox[0] for bbox in crop_bbox_list], k)
        y1_list = self.moving_average_filter([bbox[1] for bbox in crop_bbox_list], k)
        x2_list = self.moving_average_filter([bbox[2] for bbox in crop_bbox_list], k)
        y2_list = self.moving_average_filter([bbox[3] for bbox in crop_bbox_list], k)

        crop_bbox_list = [[int(x1_list[i]),int(y1_list[i]),int(x2_list[i]),int(y2_list[i])] for i in range(n)]

        return crop_bbox_list


    def pose_filter_ljw(self, dwpose_np, filter_strength=0.1):
        n = dwpose_np.shape[0]
        last_pose_arr = dwpose_np[0].copy()
        for i in range(n):
            pose_arr = dwpose_np[i].copy()
            last_candidate, last_subset = last_pose_arr[:, :, :2], last_pose_arr[:, :, 2]
            candidate, subset = pose_arr[:, :, :2], pose_arr[:, :, 2]

            candidate_diff = candidate-last_candidate
            k = filter_strength + ((1-filter_strength) / (np.exp(3-np.abs(candidate_diff)*500) + 1))
            #un_visible = subset < 0.3
            #k[un_visible] = 0.1
            k[:,14] = 1
            k[:,15] = 1
            k_63 = k[:,63].copy()
            k_66 = k[:,66].copy()

            k[:,60:] *= 0.7
            k[:,63] = k_63
            k[:,66] = k_66

            candidate = last_candidate + candidate_diff*k

            pose_arr = np.concatenate((candidate, subset[:, :, np.newaxis]), axis=2)
            dwpose_np[i] = pose_arr

            last_pose_arr = pose_arr.copy()

        return dwpose_np


    def get_crop_bbox_tsl(self, dwpose_np, batch_index, w, h, black_edge_bbox=None, crop_radio=None):
        dwpose_idx_for_face_bbox = [i for i in range(24,92)]
        dwpose_idx_for_face_center = [i for i in range(24,92)] #[i for i in range(72,92)]


        tgt_face_bbox_list_ori = [self.get_bbox_from_dwpose(dwpose_np[tgt_img_idx, 0, dwpose_idx_for_face_bbox, :]) for tgt_img_idx in batch_index]
        tgt_face_bbox_list = self.bbox_list_size_filter2(tgt_face_bbox_list_ori, h, w)

        tgt_face_center_bbox_list = [self.get_bbox_from_dwpose(dwpose_np[tgt_img_idx, 0, dwpose_idx_for_face_center, :]) for tgt_img_idx in batch_index]

        if crop_radio is None:
            # crop_radio_motion = random.uniform(1.2, 1.6)
            crop_radio_motion = random.uniform(1.1, 1.2) # TODO: 0926
        else:
            crop_radio_motion = crop_radio
        # dxy_s_motoin = 1
        # dx_list_motion, dy_list_motion = self.get_random_pad_dxy_list((tgt_face_bbox_list[0][2]-tgt_face_bbox_list[0][0])*dxy_s_motoin, len(batch_index), without_random_crop)
        n = len(tgt_face_bbox_list)
        dx_list_motion = [0 for i in range(n)]
        dy_list_motion = [0 for i in range(n)]
        # tgt_crop_bbox_list_motion = [self.square2center_crop_bbox(tgt_face_bbox, h, w, dx=dx_list_motion[i], dy=dy_list_motion[i], crop_radio=crop_radio_motion, cx=(tgt_face_center_bbox_list[i][0]+tgt_face_center_bbox_list[i][2])/2 ,cy=(tgt_face_center_bbox_list[i][1]+tgt_face_center_bbox_list[i][3])/2, black_edge_bbox=black_edge_bbox) for i, tgt_face_bbox in enumerate(tgt_face_bbox_list)]
        tgt_crop_bbox_list_motion = [self.square2center_crop_bbox(tgt_face_bbox, h, w, dx=dx_list_motion[i], dy=dy_list_motion[i], crop_radio=crop_radio_motion) for i, tgt_face_bbox in enumerate(tgt_face_bbox_list)]
        tgt_crop_bbox_list_motion = self.crop_bbox_smooth(tgt_crop_bbox_list_motion, 5)

        return tgt_crop_bbox_list_motion, crop_radio_motion


    def get_crop_bbox_tsl_hand(self, dwpose_np, batch_index, w, h, black_edge_bbox=None, crop_radio=None, dwpose_idx_for_hand_bbox=None, dwpose_idx_for_hand_center=None):

        tgt_hand_bbox_list_ori = [self.get_bbox_from_dwpose(dwpose_np[tgt_img_idx, 0, dwpose_idx_for_hand_bbox, :]) for tgt_img_idx in batch_index] # 原始手部bbox
        # tgt_hand_bbox_list = self.bbox_list_size_filter2(tgt_hand_bbox_list_ori, h, w) # 中心点和尺寸平滑
        tgt_hand_bbox_list = copy.deepcopy(tgt_hand_bbox_list_ori)

        tgt_hand_center_bbox_list = [self.get_bbox_from_dwpose(dwpose_np[tgt_img_idx, 0, dwpose_idx_for_hand_center, :]) for tgt_img_idx in batch_index]


        if crop_radio is None:
            crop_radio_motion = random.uniform(1.2, 1.6)
        else:
            crop_radio_motion = crop_radio
        n = len(batch_index)
        dx_list_motion = [0 for i in range(n)]
        dy_list_motion = [0 for i in range(n)]
        tgt_crop_bbox_list = [self.square2center_crop_bbox(tgt_hand_bbox, h, w, dx=dx_list_motion[i], dy=dy_list_motion[i], crop_radio=crop_radio_motion) for i, tgt_hand_bbox in enumerate(tgt_hand_bbox_list)]
        
        # tgt_crop_bbox_list = self.crop_bbox_smooth(tgt_crop_bbox_list, 5)

        return tgt_crop_bbox_list

    def painter_dwpose(self, out_path,dwpose_np_list,video_shape,target_resolution,threshold, input_image=None, is_show=False, is_ori_motion=False, is_neg=False, neg_pose_full=None):
        vid_len, width,height,fps = video_shape

        target_resolution_list = [720, 1080]
        if target_resolution not in target_resolution_list:
            target_resolution = 720
        if target_resolution == 720:
            min_idx = 1
        elif target_resolution == 1080:
            min_idx = 2


        video_xdwpose_np_multi_persons = np.concatenate(dwpose_np_list, axis=1) #(F, 2, 134, 3)
        # mask_point_idx = list(set(range(0, video_xdwpose_np_multi_persons.shape[2])) - set([1,8,11])) # 1,8,11
        # mask_point_idx = list(set(range(0, video_xdwpose_np_multi_persons.shape[2])))
        # mask_point_idx = list(set(range(0, video_xdwpose_np_multi_persons.shape[2])) - set([1])) # 1,8,11

        a = video_xdwpose_np_multi_persons[0, 0, 1, 0:2]
        b = video_xdwpose_np_multi_persons[0, 0, 8, 0:2]
        c = video_xdwpose_np_multi_persons[0, 0, 11, 0:2]
        d = video_xdwpose_np_multi_persons[USE_interpolate_N, 0, 1, 0:2]
        e = video_xdwpose_np_multi_persons[USE_interpolate_N, 0, 8, 0:2]
        f = video_xdwpose_np_multi_persons[USE_interpolate_N, 0, 11, 0:2]
        # 计算中点
        mid_bc = (b + c) / 2
        mid_ef = (e + f) / 2
        # 向量
        v1 = a - mid_bc
        v2 = d - mid_ef
        # 计算夹角余弦
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # 判断是否大于90度
        is_greater_than_90 = cos_theta < 0

        # 向量
        v1 = b - c
        v2 = e - f
        # 计算夹角余弦
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # 判断是否大于90度
        is_greater_than_90_v2 = cos_theta < 0

        # 向量
        v1 = a - b
        v2 = a - c
        # 计算夹角余弦
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # 判断是否大于 15 度
        theta_deg = math.degrees(math.acos(np.clip(cos_theta, -1.0, 1.0)))
        is_greater_than_15 = theta_deg > 15

        mask_point_idx = list(set(range(0, video_xdwpose_np_multi_persons.shape[2])) - set([1,8,11])) #TODO:走查
        # mask_point_idx = list(set(range(0, video_xdwpose_np_multi_persons.shape[2])) - set([1])) # 画图用的
        # '''2025/11/27:自测更优配置'''
        # if is_greater_than_90: # 上半身转角大于90度，3点辅助
        #     mask_point_idx = list(set(range(0, video_xdwpose_np_multi_persons.shape[2])) - set([1,8,11])) # 1,8,11
        # elif is_greater_than_90_v2 and is_greater_than_15:# 转身，3点辅助
        #     # mask_point_idx = list(set(range(0, video_xdwpose_np_multi_persons.shape[2])) - set([1])) # 1
        #     mask_point_idx = list(set(range(0, video_xdwpose_np_multi_persons.shape[2])) - set([1,8,11])) # 1,8,11
        # else:
        #     mask_point_idx = list(set(range(0, video_xdwpose_np_multi_persons.shape[2])))

        if OptHeadTail:
            # TODO: 首帧模糊优化
            USE_interpolate_N_clear = USE_interpolate_N-Headrepeat
        else:
            USE_interpolate_N_clear = USE_interpolate_N
        if is_neg is False:
            video_xdwpose_np_multi_persons[:USE_interpolate_N_clear, :, mask_point_idx, 2] = 0
            # mask_point_idx_2hexu = list(
            #     (set(range(92, video_xdwpose_np_multi_persons.shape[2])) | {14,15,16,17, 7})
            #     - {113,114,118,122,126,130} #- {92,93,97,101,105,109} 
            # ) # 画图用的
            # video_xdwpose_np_multi_persons[USE_interpolate_N:, :, mask_point_idx_2hexu, 2] = 0 # 画图用的
            # video_xdwpose_np_multi_persons[:USE_interpolate_N, :, :, 2] = 0
            pass
        else:
            # mask_point_idx = list(set(range(0, video_xdwpose_np_multi_persons.shape[2])) - set(range(92, video_xdwpose_np_multi_persons.shape[2]))) # 保留手部的点
            mask_point_idx = list(set(range(0, video_xdwpose_np_multi_persons.shape[2])) - set(range(92, video_xdwpose_np_multi_persons.shape[2])) - set([14,15,16,17,0])) 
            video_xdwpose_np_multi_persons[USE_interpolate_N_clear:, :, mask_point_idx, 2] = 0
            if neg_pose_full is not None:
                neg_pose_full_tmp = np.concatenate(neg_pose_full, axis=1) #(F, 2, 134, 3)
                video_xdwpose_np_multi_persons[1:USE_interpolate_N_clear, :, :, :] = neg_pose_full_tmp[1:USE_interpolate_N_clear, :, :, :]

                mask_point_idx_nobody = list(set(range(0, video_xdwpose_np_multi_persons.shape[2])) - set(range(0, 20)))
                video_xdwpose_np_multi_persons[0:1, :, :, 2] = 0
                video_xdwpose_np_multi_persons[1:USE_interpolate_N_clear, :, mask_point_idx_nobody, 2] = 0
                video_xdwpose_np_multi_persons[1:USE_interpolate_N_clear:2, :, :, 2] = 0
            else:
                video_xdwpose_np_multi_persons[0:1, :, :, 2] = 0
                video_xdwpose_np_multi_persons[1:USE_interpolate_N_clear, :, :, 2] = 0
                pass
        img_list = []
        for iframe in tqdm(range(vid_len)):
            dwpose_i = video_xdwpose_np_multi_persons[iframe]
            img = draw_dwpose_2d(dwpose_i, width, height, threshold=threshold, min_idx=min_idx)
            img_list.append(img)

        if is_ori_motion:
            save_name = "motion_video_ori"
        else:
            save_name = "motion_video"

        writer = get_writer(out_path+'/{}.mp4'.format(save_name), fps=fps, crf=23)
        for img in img_list:
            if is_show:
                if not is_ori_motion:
                    img = 0.3 * input_image[:, :, ::-1] + 0.7 * img
            writer.write_frame(img)
        writer.close()

        with open(out_path+'/{}.mp4'.format(save_name), "rb") as f:
            motion_video_binary = f.read()
        
        return motion_video_binary


    def retarget_scale_coarse(self, dwpose3d_list_motion, dwpose3d_list_image, video_xdwpose_np_list, image_xdwpose_np_list, threshold=5.0, interpolate_N=24):
        # dwpose3d_image (134, 2)

        body_indices = [1, 2, 5, 8, 9, 10, 11, 12, 13]

        # 关键点检查逻辑
        for video_dwpose, image_dwpose in zip(video_xdwpose_np_list, image_xdwpose_np_list):
            score_motion = video_dwpose[0, 0, body_indices, 2] # (N,)
            score_image = image_dwpose[0, 0, body_indices, 2] # (N,)

            valid_indices = np.where((score_motion > threshold) & (score_image > threshold))[0]
            filtered_body_indices = [body_indices[i] for i in valid_indices]
            if len(filtered_body_indices) <= 1:
                threshold = 4.0
                break
        # 关键点检查逻辑

        for dwposed3d_motion, dwpose3d_image, video_dwpose, image_dwpose in zip(dwpose3d_list_motion, dwpose3d_list_image, video_xdwpose_np_list, image_xdwpose_np_list):
            score_motion = video_dwpose[0, 0, body_indices, 2] # (N,)
            score_image = image_dwpose[0, 0, body_indices, 2] # (N,)

            valid_indices = np.where((score_motion > threshold) & (score_image > threshold))[0]
            filtered_body_indices = [body_indices[i] for i in valid_indices]

            # point_motion = dwposed3d_motion[filtered_body_indices, :2] # (M, 2)
            # point_image = dwpose3d_image[filtered_body_indices, :] # (M, 2)

            point_image = dwpose3d_image[interpolate_N, 0, filtered_body_indices, :2] # (M, 2)
            # point_image = image_dwpose[0, 0, filtered_body_indices, :2] # (M, 2)
            point_motion = video_dwpose[0, 0, :, :2]
            point_motion = point_motion[filtered_body_indices, :2] # (M, 2)

            ymax_motion = np.max(point_motion[:, 1])
            ymin_motion = np.min(point_motion[:, 1])

            ymax_image = np.max(point_image[:, 1])
            ymin_image = np.min(point_image[:, 1])

            yscale_motion2image = (ymax_image-ymin_image) /  (ymax_motion-ymin_motion)

            xmax_motion = np.max(point_motion[:, 0])
            xmin_motion = np.min(point_motion[:, 0])

            xmax_image = np.max(point_image[:, 0])
            xmin_image = np.min(point_image[:, 0])

            xscale_motion2image =  (xmax_image-xmin_image) / (xmax_motion-xmin_motion)
            if (ymax_motion-ymin_motion) > (xmax_motion-xmin_motion):
                motion2image_scale = (yscale_motion2image, yscale_motion2image)
            else:
                motion2image_scale = (xscale_motion2image, xscale_motion2image)

            # 各帧以身体中心为锚点缩放
            # scale_ancor_point = np.mean(video_xdwpose_np[:, 0, [2, 5, 8, 11], :2], axis=-2, keepdims=True) # (F, 1, 2)
            # scale_ancor_point = np.mean(video_dwpose[:, 0, [1], :2], axis=-2, keepdims=True) # (F, 1, 2)
            scale_ancor_point = video_dwpose[:, 0, [1], :2] # (F, 1, 2)
            video_dwpose[:, 0, :, 0] = (video_dwpose[:, 0, :, 0] - scale_ancor_point[:, :, 0]) * motion2image_scale[0] + scale_ancor_point[:, :, 0]
            video_dwpose[:, 0, :, 1] = (video_dwpose[:, 0, :, 1] - scale_ancor_point[:, :, 1]) * motion2image_scale[1] + scale_ancor_point[:, :, 1]


    def retarget_scale_fine(self, dwpose3d_list_motion, dwpose3d_list_image, video_xdwpose_np_list, image_xdwpose_np_list, threshold=5.0, interpolate_N=24, image_height=1):
        # dwpose3d_image (134, 2)

        body_indices = [1, 2, 5, 8, 9, 10, 11, 12, 13]
        retartget_scale_list = []
        parent_index = [1, 1,  1, 2, 3,  1, 5, 6,  1, 8, 9,  1, 11, 12,  0, 0, 14, 15, 13, 10] #脚 13, 10
        parent_index_for_scale = [[0,1], [1,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,8], [8,9], [9,10], [1,11], [11,12], [12,13], [0,14], [0,15], [14,16], [15,17], [13, 18], [10,19]] # 脚 [13, 18], [10,19]

        for dwposed3d_motion, dwpose3d_image, video_dwpose, image_dwpose in zip(dwpose3d_list_motion, dwpose3d_list_image, video_xdwpose_np_list, image_xdwpose_np_list):
            dwpose3d_image = dwpose3d_image[interpolate_N, 0, :, :2] # (134, 2)
            dwposed3d_motion = dwposed3d_motion[:, :2] # (134, 2)
            lens_image = []
            lens_motion = []
            for i in range(len(parent_index_for_scale)):
                pair = parent_index_for_scale[i]
                len_image = np.linalg.norm(dwpose3d_image[pair[0]] - dwpose3d_image[pair[1]])
                len_motion = np.linalg.norm(dwposed3d_motion[pair[0]] - dwposed3d_motion[pair[1]])
                lens_image.append(len_image)
                lens_motion.append(len_motion)

            # 考虑对称性
            max_pair_list = [[2, 5], [3, 6], [4, 7], [8, 11], [9, 12], [10, 13], [18, 19]]
            for i in range(len(max_pair_list)):
                pair = max_pair_list[i]
                tmp = max(lens_image[pair[0]], lens_image[pair[1]])
                lens_image[pair[0]] = lens_image[pair[1]] = tmp
                tmp = max(lens_motion[pair[0]], lens_motion[pair[1]])
                lens_motion[pair[0]] = lens_motion[pair[1]] = tmp

            scales = []
            for i in range(len(parent_index)):
                if lens_image[i] > 0.001:
                    scales.append((lens_image[i]+0.001)/(lens_motion[i]+0.001))
                else:
                    scales.append(1.0)
            # print("scales", scales)

            # 并行缩放
            # video_dwpose [F, 1, 134, 3]
            lefthand_pre = copy.deepcopy(video_dwpose[:, 0, 4:5, :2])
            righthand_pre = copy.deepcopy(video_dwpose[:, 0, 7:8, :2])

            lefthand_pre2 = copy.deepcopy(video_dwpose[:, 0, 113:114, :2])
            righthand_pre2 = copy.deepcopy(video_dwpose[:, 0, 92:93, :2])

            y_coords = image_dwpose[0, 0, [1, 8, 11], 1]
            max_diff = (y_coords.max(axis=0) - y_coords.min(axis=0)) / image_height
            print('body max_diff---------------------------------------------', max_diff)
            if max_diff <= 0.4:
                # hand_scale = scales[4]*1.15
                # hand_scale = scales[4]*1.05
                hand_scale = statistics.mean(scales)*1.05
            else:
                # hand_scale = scales[4]*1. #TODO：确定手部缩放比例
                hand_scale = statistics.mean(scales)

            dirs = []
            for i in range(len(parent_index)):
                dirs.append(video_dwpose[:, 0, i, :2] - video_dwpose[:, 0, parent_index[i], :2])

            for i in range(len(parent_index)):
                video_dwpose[:, 0, i, :2] = dirs[i] * scales[i] + video_dwpose[:, 0, parent_index[i], :2]
            
            # 平移手部
            # video_dwpose[:, 0, 113:, :2] += (video_dwpose[:, 0, 4:5, :2] - lefthand_pre)
            # video_dwpose[:, 0, 92:113, :2] += (video_dwpose[:, 0, 7:8, :2] - righthand_pre)

            video_dwpose[:, 0, 113:, :2] += ((video_dwpose[:, 0, 113:114, :2] - lefthand_pre) * hand_scale + video_dwpose[:, 0, 4:5, :2] - video_dwpose[:, 0, 113:114, :2])
            video_dwpose[:, 0, 92:113, :2] += ((video_dwpose[:, 0, 92:93, :2] - righthand_pre) * hand_scale + video_dwpose[:, 0, 7:8, :2] - video_dwpose[:, 0, 92:93, :2])

            # 缩放手部
            scale_hands_batch(video_dwpose[:,:,:,:2], hand_scale)

            # video_dwpose[:, 0, 113:, :2] += video_dwpose[:, 0, 4:5, :2] - lefthand_pre
            # video_dwpose[:, 0, 92:113, :2] += video_dwpose[:, 0, 7:8, :2] - righthand_pre

            '''
            video_dwpose[:, 0, 113:114, :2] = (video_dwpose[:, 0, 113:114, :2] - lefthand_pre) * 1 + video_dwpose[:, 0, 4:5, :2]
            video_dwpose[:, 0, 92:93, :2] = (video_dwpose[:, 0, 92:93, :2] - righthand_pre) * 1 + video_dwpose[:, 0, 7:8, :2]


            video_dwpose[:, 0, 114:, :2] = (video_dwpose[:, 0, 114:, :2] - lefthand_pre2) * hand_scale + video_dwpose[:, 0, 113:114, :2]
            video_dwpose[:, 0, 93:113, :2] = (video_dwpose[:, 0, 93:113, :2] - righthand_pre2) * hand_scale + video_dwpose[:, 0, 92:93, :2]
            '''


    def retarget_offset(self, dwpose3d_list_motion, dwpose3d_list_image, video_xdwpose_np_list, image_xdwpose_np_list, video_frame0_ori_list, threshold=5.0):
        # dwpose3d_image (134, 2)

        body_indices = [1, 2, 5, 8, 9, 10, 11, 12, 13]
        upper_body_indices = [1, 2, 5, 8, 11]

        # 关键点检查逻辑
        for video_dwpose, image_dwpose in zip(video_xdwpose_np_list, image_xdwpose_np_list):
            score_motion = video_dwpose[0, 0, body_indices, 2] # (N,)
            score_image = image_dwpose[0, 0, body_indices, 2] # (N,)

            valid_indices = np.where((score_motion > threshold) & (score_image > threshold))[0]
            filtered_body_indices = [body_indices[i] for i in valid_indices]
            if len(filtered_body_indices) <= 1:
                threshold = 4.0
                break
        # 关键点检查逻辑

        for dwposed3d_motion, dwpose3d_image, video_dwpose, image_dwpose, video_dwpose_frame0_ori in zip(dwpose3d_list_motion, dwpose3d_list_image, video_xdwpose_np_list, image_xdwpose_np_list, video_frame0_ori_list):
            score_motion = video_dwpose[0, 0, body_indices, 2] # (N,)
            score_image = image_dwpose[0, 0, body_indices, 2] # (N,)

            valid_indices = np.where((score_motion > threshold) & (score_image > threshold))[0]
            filtered_body_indices = [body_indices[i] for i in valid_indices]

            point_motion = video_dwpose[0, 0, :, :2] # (134, 2)
            point_image = image_dwpose[0, 0, :, :2] # (134, 2)


            '''
            valid_indices = np.where((score_motion > threshold) & (score_image > threshold))[0]
            upper_body_indices_filtered = [upper_body_indices[i] for i in valid_indices]
            # 水平对齐上半身中心点
            x_offset = np.mean(point_image[upper_body_indices_filtered, 0]) - np.mean(point_motion[upper_body_indices_filtered, 0])
            '''
            # 水平对齐脖子点
            x_offset = point_image[1, 0] - point_motion[1, 0]


            # # 垂直对齐脖子点
            y_offset = point_image[1, 1] - point_motion[1, 1]
            # 垂直对齐最低点
            #y_offset = np.max(point_image[filtered_body_indices, 1]) - np.max(point_motion[filtered_body_indices, 1])
            
            # 对齐首帧位置
            motion2image_offset = (x_offset, y_offset)
            video_dwpose[:, 0, :, 0] += motion2image_offset[0]
            video_dwpose[:, 0, :, 1] += motion2image_offset[1]


            # 逐帧位移scale重定向
            score_motion = video_dwpose[0, 0, body_indices, 2] # (N,)
            valid_indices = np.where(score_motion > threshold)[0]
            filtered_body_indices = [body_indices[i] for i in valid_indices]

            point_motion = video_dwpose_frame0_ori[0, 0, filtered_body_indices, :2] # (M, 2) # 原始motion视频空间
            point_image = video_dwpose[0, 0, :, :2] # 已经重定向到image space
            point_image = point_image[filtered_body_indices, :2] # (M, 2)

            ymax_motion = np.max(point_motion[:, 1])
            ymin_motion = np.min(point_motion[:, 1])

            ymax_image = np.max(point_image[:, 1])
            ymin_image = np.min(point_image[:, 1])

            yscale_motion2image = (ymax_image-ymin_image) /  (ymax_motion-ymin_motion)

            xmax_motion = np.max(point_motion[:, 0])
            xmin_motion = np.min(point_motion[:, 0])

            xmax_image = np.max(point_image[:, 0])
            xmin_image = np.min(point_image[:, 0])

            xscale_motion2image =  (xmax_image-xmin_image) / (xmax_motion-xmin_motion)
            if (ymax_motion-ymin_motion) > (xmax_motion-xmin_motion):
                motion2image_scale = yscale_motion2image
            else:
                motion2image_scale = xscale_motion2image

            # 逐帧位移缩放
            video_len = video_dwpose.shape[0]
            parent_index = [0] + list(range(video_len-1))
            dirs = []
            for i in range(len(parent_index)):
                dirs.append(video_dwpose[i, 0, 1, :2] - video_dwpose[parent_index[i], 0, 1, :2])

            for i in range(len(parent_index)):
                video_dwpose[i, 0, :, :2] += (dirs[i] * motion2image_scale + video_dwpose[parent_index[i], 0, 1, :2]) - video_dwpose[i, 0, 1, :2]


    def retartet_2d_coarse(self, point_src, pose_dst, is_align_height):
        # point_src (1, 134, 2) # 参考帧
        # pose_dst (n, 1, 134, 2) # 视频帧
        
        point_src = point_src[0] # (134, 2)
        point_dst_frame0 = pose_dst[0, 0] # (134, 2)

        all_i = [0,1,2,5,8,9,10,11,12,13,14,15,16,17]

        ymax_src = max(np.max(point_src[all_i, 1]), np.mean(point_src[[18, 19], 1]), np.mean(point_src[[21, 22], 1]))
        ymin_src = min(np.min(point_src[all_i, 1]), np.mean(point_src[[18, 19], 1]), np.mean(point_src[[21, 22], 1]))

        ymax_dst = max(np.max(point_dst_frame0[all_i, 1]), np.mean(point_dst_frame0[[18, 19], 1]), np.mean(point_dst_frame0[[21, 22], 1]))
        ymin_dst = min(np.min(point_dst_frame0[all_i, 1], ), np.mean(point_dst_frame0[[18, 19], 1]), np.mean(point_dst_frame0[[21, 22], 1]))

        if is_align_height:
            s = (ymax_src-ymin_src) / (ymax_dst-ymin_dst)
        else:
            s = 1

        cur_dst = pose_dst # (n, 1, 134, 2)

        # 对齐首帧脖子 x
        cur_dst[:, 0, :, 0] += (point_src[None, 1:2, 0] - cur_dst[0:1, 0, 1:2, 0]) #(F 1 2)

        # 对齐首帧脖子 x, y
        # cur_dst[:, 0, :, 0:2] += (point_src[None, 1:2, 0:2] - cur_dst[0:1, 0, 1:2, 0:2]) #(F 1 2)


        # 各帧以脖子为中心缩放
        neck_dst = cur_dst[:, 0, 1:2, :]
        cur_dst[:, 0, :, :] = (cur_dst[:, 0, :, :] - neck_dst) * s + neck_dst

        
        # 对齐首帧触地点 y
        point_src_max_y = max((point_src[18, 1]+point_src[19, 1])/2.0, (point_src[21, 1]+point_src[22, 1])/2.0)
        point_dst_max_y = max((cur_dst[0, 0, 18, 1]+cur_dst[0, 0, 19, 1])/2.0, (cur_dst[0, 0, 21, 1]+cur_dst[0, 0, 22, 1])/2.0)
        cur_dst[:, 0, :, 1] += (point_src_max_y - point_dst_max_y) #(F 1 2)


    # def dwpose_ori2new(self, image_xdwpose_np_list, ori_image_h, ori_image_w, resize_h, resize_w, crop_h, crop_w):
    #     image_xdwpose_np_list_new = []
    #     # image_xdwpose (1, 1, 134, 3)
    #     for image_xdwpose in image_xdwpose_np_list:
    #         image_xdwpose[:, :, :, 0] = (image_xdwpose[:, :, :, 0].copy() * ori_image_w).astype(np.int32) * (resize_w/ori_image_w) * (crop_w / resize_w) / crop_w
    #         image_xdwpose[:, :, :, 1] = (image_xdwpose[:, :, :, 1].copy() * ori_image_h).astype(np.int32) * (resize_h/ori_image_h) * (crop_h / resize_h) / crop_h
    #         # image_xdwpose[:, :, :, 0] = (((image_xdwpose[:, :, :, 0].copy() * ori_image_w).astype(np.int32) * (resize_w/ori_image_w)).astype(np.int32) * 1.0 / resize_w).clip(0, crop_w) / crop_w
    #         # image_xdwpose[:, :, :, 1] = (((image_xdwpose[:, :, :, 1].copy() * ori_image_h).astype(np.int32) * (resize_h/ori_image_h)).astype(np.int32) * 1.0 / resize_h).clip(0, crop_h) / crop_h

    def dwpose_ori2new(self, image_xdwpose_np_list, ori_image_h, ori_image_w, resize_h, resize_w, crop_h, crop_w):
        image_xdwpose_np_list_new = []
        # image_xdwpose (1, 1, 134, 3)
        for image_xdwpose in image_xdwpose_np_list:
            # image_xdwpose[:, :, :, 0] = (image_xdwpose[:, :, :, 0].copy() * ori_image_w).astype(np.int32) * (resize_w/ori_image_w) * (crop_w / resize_w) / crop_w
            # image_xdwpose[:, :, :, 1] = (image_xdwpose[:, :, :, 1].copy() * ori_image_h).astype(np.int32) * (resize_h/ori_image_h) * (crop_h / resize_h) / crop_h
            image_xdwpose[:, :, :, 0] = (image_xdwpose[:, :, :, 0].copy() * resize_w).clip(0, crop_w) / crop_w
            image_xdwpose[:, :, :, 1] = (image_xdwpose[:, :, :, 1].copy() * resize_h).clip(0, crop_h) / crop_h

    def retarget_with_data_list(self, input_image_binary, img_data_binary, video_data_binary_list, human_ID_list=[0], change_direction_list=[0], duration=10000, target_resolution=720, output_fps=USE_FPS,output_dir='./output/',use_dwpose=False, render_depth=False, save_split_path=None):
        '''5s:145, 10s:289'''
        # output_length = find_closest_number([4800, 9600], float(duration))
        # output_length = int(float(output_length)/1000*output_fps+1)
        '''5s:145, 10s:289'''

        '''5s:153, 10s:305'''
        # output_length = find_closest_number([5100, 10160], float(duration))
        output_length = float(duration)
        # 严格向上取整帧数
        output_length = math.ceil(float(output_length)/1000*output_fps)
        output_length = ((output_length-1)//8)*8+1
        '''5s:153, 10s:305'''

        print ('output_length', output_length)
        output_size = target_resolution
        #生成临时文件夹，存放中间结果：
        output_dir = os.path.abspath(output_dir)
        output_result_path = output_dir+'/'+str(uuid.uuid4())+'/'
        os.makedirs(output_result_path)
        print ('output_result_path', output_result_path)
        result_data_dict = {}

        #输入解析
        with open(output_result_path+'/input.png', 'wb') as f:
            f.write(input_image_binary)
        input_image = cv2.imread(output_result_path+'/input.png')
        img_data = pickle.loads(img_data_binary)
        #video_data = pickle.loads(video_data_binary)
        video_data_list = [pickle.loads(video_data_binary) for video_data_binary in video_data_binary_list] 
        video_smpl_list = [video_data["smpl"][0] for video_data in video_data_list] #多个驱动视频

        if OptHeadTail:
            output_length = video_data_list[0]['data_list'][0]['xdwpose_np'].shape[0] + USE_interpolate_N # 改为视频实际长度

        #############################################xdwpose_np#########################################################
        '''
        try:
            video_xdwpose_np_list = [video_data['data_list'][0]['xdwpose_np'] for video_data in video_data_list] #多个驱动视频
            print ('video_xdwpose_np_list[0]', video_xdwpose_np_list[0].shape)

            image_xdwpose_np_list = [datai['xdwpose_np'] for i, datai in enumerate(img_data['data_list']) if i in human_ID_list] #图片里面有多个可驱动的人,例如有4个,抽出 用户选的人1~2
            print('image_xdwpose_np_list[0].shape)', image_xdwpose_np_list[0].shape)
        except:
            pass
        '''
        ################################################################################################################
        
        ori_image_h, ori_image_w = input_image.shape[:2]
        input_image, new_w, new_h, resize_w, resize_h = self.get_image_for_i2v(input_image, output_size)
        cv2.imwrite(output_result_path+'/input.png', input_image)
        with open(output_result_path+'/input.png', "rb") as f:
            input_image_binary = f.read()
        
        n1 = len(human_ID_list)
        n2 = len(video_smpl_list)
        n3 = len(change_direction_list)
        n = min(n1, n2, n3)
        if n1 != n2 or n2 != n3 or n<1: 
            result_data_dict['error_code'] = self.error_code.INVALID_INPUT_FORMAT
            result_data_dict['error_message'] = self.error_messages.get(result_data_dict['error_code'])

            return None, None, None, result_data_dict['error_code'], result_data_dict['error_message']

        retarget_result_list = []
        retarget_3d_list = []
        img_smpl_list = img_data["smpl"]
        
        for i in range(n):
            ##result = self.retarget_rt(video_smpl_list[i], img_smpl_list[human_ID_list[i]], maintain_orient=change_direction_list[i])
            result, err_flag, mot_or_img = self.retarget_rt(copy.deepcopy(video_smpl_list[i]), copy.deepcopy(img_smpl_list[human_ID_list[i]]), maintain_orient=False)
            result3d, _, _ = self.retarget_rt(copy.deepcopy(video_smpl_list[i]), copy.deepcopy(img_smpl_list[human_ID_list[i]]), maintain_orient=change_direction_list[i],force_direction=True)
            
            if mot_or_img:
                change_direction_list[i] = False
            if err_flag:
                result_data_dict['error_code'] = self.error_code.INVALID_POSE
                result_data_dict['error_message'] = self.error_messages.get(result_data_dict['error_code'])
                return None, None, None, result_data_dict['error_code'], result_data_dict['error_message']
            retarget_result_list.append(result)
            retarget_3d_list.append(result3d)

        bvh_result = self.smpl2bvh(retarget_3d_list, output_result_path, use_trans=True)

        feat_list = img_data['data_list']

        interpolate_N = USE_interpolate_N                          # @tsl @zhx 需要插的帧数
        for i in range(n):
            retarget_result_list[i]["smpl_params_incam"] = slerp_pose(img_smpl_list[human_ID_list[i]]["smpl_params_incam"], retarget_result_list[i]["smpl_params_incam"], interpolate_N, change_direction_list[i])

        video_len0, dwpose3d_list_image = self.painter_smpl(output_result_path,retarget_result_list,[output_length, new_w, new_h ,output_fps],feat_list, interpolate_N=interpolate_N) # 目标视频像素空间

        if OptHeadTail:
            # TODO：尾帧崩坏和ending-pose兼容
            video_len = math.ceil((video_len0-1)/8)*8+1
            Redundant_frame = max(video_len - video_len0, 0)
            print('OPT ending-pose / dwpose3d_list_image / before / video_len0, video_len, Redundant_frame:', video_len0, video_len, Redundant_frame)
            for i in range(len(dwpose3d_list_image)):
                dwpose3d_list_image[i] = dwpose3d_list_image[i][:video_len0]
                x = dwpose3d_list_image[i]
                if Redundant_frame>0:
                    last = np.repeat(x[-1:], Redundant_frame, axis=0)  
                    x = np.concatenate([x, last], axis=0)
                x[-Tailrepeat:] = x[-(Tailrepeat+1):-Tailrepeat]  
                dwpose3d_list_image[i] = x
                print('OPT ending-pose / dwpose3d_list_image / after',dwpose3d_list_image[i].shape)
        else:
            video_len = video_len0

        print (output_length, video_len, 'output_length, video_len')
        ##depth_video_binary =  self.painter_smpl_depth(output_result_path, retarget_result_list,  [output_length, new_w, new_h ,output_fps],feat_list)
        ##depth_video_binary = None
        is_bvh_list =[]
        for i, video_data in enumerate(video_data_list):
            if 'data_list' not in video_data:
                change_direction_list[i] = True
                is_bvh_list.append(True)
            else:
                is_bvh_list.append(False)

        # motion朝向替换为2D关键点, 重定向
        threshold = 5.0
        ##image_xdwpose_np_list = [bodypose_24to20(datai['xdwpose_np'].copy(), threshold) for i, datai in enumerate(img_data['data_list']) if i in human_ID_list] #图片里面有多个可驱动的人,例如有4个,抽出 用户选的人1~2
        image_xdwpose_np_list = [bodypose_24to20(img_data['data_list'][id]['xdwpose_np'].copy(), threshold) for id in human_ID_list] #图片里面有多个可驱动的人,例如有4个,抽出 用户选的人1~2
        
        new_image_xdwpose_np_list = copy.deepcopy(image_xdwpose_np_list)
        # new_image_xdwpose_np_list = [img_data['data_list'][id]['xdwpose_np'].copy() for id in human_ID_list]
        self.dwpose_ori2new(new_image_xdwpose_np_list, ori_image_h, ori_image_w, resize_h, resize_w, new_h, new_w)
        new_image_h_w_array = np.array([[new_h, new_w]])
        new_image_dwpose_array = np.concatenate(new_image_xdwpose_np_list, axis=1)
        
        self.dwpose_ori2new(image_xdwpose_np_list, ori_image_h, ori_image_w, resize_h, resize_w, new_h, new_w)
        for imagei in range(len(image_xdwpose_np_list)):
            image_xdwpose_np = image_xdwpose_np_list[imagei]
            ori_width = img_data['data_list'][0]['width']
            ori_height = img_data['data_list'][0]['height']
            image_xdwpose_np[:, :, :, 0] = (image_xdwpose_np[:, :, :, 0].copy() * ori_width).astype(np.int32) * (new_w/ori_width) # inplace 操作 # 目标视频像素空间
            image_xdwpose_np[:, :, :, 1] = (image_xdwpose_np[:, :, :, 1].copy() * ori_height).astype(np.int32) * (new_h/ori_height) 

        video_xdwpose_np_final_list = []
        video_np_final_list_ori = []
        video_xdwpose_np_final_list_ori = []
        for i, change_flag in enumerate(change_direction_list):
            print('9'*20,' ', change_flag)
             
            if not is_bvh_list[i]:#not change_flag:
                feat_list = [video_data_list[i]] # 原始视频像素空间
                _, dwpose3d_list_motion = self.painter_smpl(output_result_path,[video_smpl_list[i]],[output_length, new_w, new_h ,output_fps],feat_list, is_motion_space=True, interpolate_N=interpolate_N) # dwpose3d_list_motion_pixel 原始视频像素空间
                video_xdwpose_np_list = [bodypose_24to20(video_data_list[i]['data_list'][0]['xdwpose_np'][:video_len].copy(), threshold)] #多个驱动视频

                if OptHeadTail:
                    # TODO：尾帧崩坏和ending-pose兼容
                    for i in range(len(video_xdwpose_np_list)):
                        video_xdwpose_np_list[i] = video_xdwpose_np_list[i]
                        x = video_xdwpose_np_list[i]
                        if Redundant_frame>0:
                            last = np.repeat(x[-1:], Redundant_frame, axis=0) 
                            x = np.concatenate([x, last], axis=0)
                        x[-Tailrepeat:] = x[-(Tailrepeat+1):-Tailrepeat]  
                        video_xdwpose_np_list[i] = x
                        print('OPT ending-pose / video_xdwpose_np_list shape',video_xdwpose_np_list[i].shape)
               
                # to pixel space
                video_frame0_ori_list = []
                for videoi in range(len(video_xdwpose_np_list)):
                    video_xdwpose_np = video_xdwpose_np_list[videoi]
                    ori_width = feat_list[videoi]['data_list'][0]['width']
                    ori_height = feat_list[videoi]['data_list'][0]['height']
                    video_xdwpose_np[:, :, :, 0] *= ori_width # inplace 操作 # 原始视频像素空间
                    video_xdwpose_np[:, :, :, 1] *= ori_height
                    video_frame0_ori_list.append(video_xdwpose_np[0:1, :, :, :].copy()) # (1 1 134 3)

                video_xdwpose_np_list_ori = copy.deepcopy(video_xdwpose_np_list)
                for videoi in range(len(video_xdwpose_np_list_ori)):
                    video_xdwpose_np_ori = video_xdwpose_np_list_ori[videoi]
                    ori_width = feat_list[videoi]['data_list'][0]['width']
                    ori_height = feat_list[videoi]['data_list'][0]['height']
                    video_xdwpose_np_ori[:, :, :, 0] /= ori_width # inplace 操作
                    video_xdwpose_np_ori[:, :, :, 1] /= ori_height

                print(video_data_list[i]['data_list'][0].keys())
                video_np_list = [video_data_list[i]['data_list'][0]['video_np'][:video_len]] #多个驱动视频
                for videoi in range(len(video_np_list)):
                    video_np = video_np_list[videoi]
                    video_np_interpolate_N = np.concatenate((video_np[:interpolate_N]*0, video_np), axis=0)
                    video_np_interpolate_N = video_np_interpolate_N[:video_len]
                    video_np_final_list_ori.append(video_np_interpolate_N)
                print("video_xdwpose_np_list", video_xdwpose_np_list[0].shape, "video_np_list", video_np_final_list_ori[0].shape, "video_len", video_len)

                # 重定向scalle
                # 逐关节缩放, 改变动作语义，拍手拍不上
                self.retarget_scale_fine(dwpose3d_list_motion, [dwpose3d_list_image[i]], video_xdwpose_np_list, [image_xdwpose_np_list[i]], interpolate_N=interpolate_N, image_height=new_h) #inplace操作 # 像素空间重定向
                # 等比整体缩放, 双人动作交叉
                self.retarget_scale_coarse(dwpose3d_list_motion, [dwpose3d_list_image[i]], video_xdwpose_np_list, [image_xdwpose_np_list[i]], interpolate_N=interpolate_N) #inplace操作 # 像素空间重定向
                # 重定向location
                self.retarget_offset(dwpose3d_list_motion, [dwpose3d_list_image[i]], video_xdwpose_np_list, [image_xdwpose_np_list[i]], video_frame0_ori_list) #inplace操作
            
                dwpose2d_fuse_3d_pose = np.array([-1.0, -1.0, 0.0]).repeat(interpolate_N*134).reshape(interpolate_N, 1, 134, 3)
                # weight_2d_3d = np.concatenate((np.linspace(1,0,interpolate_N//2), np.linspace(0,1,interpolate_N//2)))
                # weight_2d_3d = np.concatenate((np.linspace(1,0,interpolate_N//2), np.linspace(0,1,interpolate_N-(interpolate_N//2))))
                weight_2d_3d = np.concatenate((np.linspace(1,0.25,interpolate_N//2), np.linspace(0.25,1,interpolate_N-(interpolate_N//2)))) # debugTransition5:0.5 / debugTransition6:0.25
                
                if OptHeadTail and Headrepeat:
                    # TODO: 首帧模糊优化
                    interpolate_N_tmp = interpolate_N - (Headrepeat-1)
                    weight_2d_3d = np.concatenate((np.linspace(1,0.25,interpolate_N_tmp//2), np.linspace(0.25,1,interpolate_N_tmp-(interpolate_N_tmp//2))))
                    weight_2d_3d = np.concatenate((weight_2d_3d, np.array([1]*(Headrepeat-1))))
                    print('OPT head frame blur - weight_2d_3d:', weight_2d_3d)

                video_xdwpose_np = video_xdwpose_np_list[0]
                video_xdwpose_np[:, :, :, 0] /= new_w # inplace 操作
                video_xdwpose_np[:, :, :, 1] /= new_h

                image_xdwpose_np_list[i][:, :, :, 0] /= new_w # inplace 操作
                image_xdwpose_np_list[i][:, :, :, 1] /= new_h

                if OptHeadTail and Headrepeat:
                    # TODO: 首帧模糊优化
                    fix_num = Headrepeat-1
                    interpolate_xdwpose2d = interpolate_poses(image_xdwpose_np_list[i], video_xdwpose_np[0], interpolate_N-fix_num, confidence_threshold=5)
                    last = np.repeat(video_xdwpose_np[0:1], fix_num, axis=0)
                    interpolate_xdwpose2d = np.concatenate((interpolate_xdwpose2d, last), axis=0)
                else:
                    interpolate_xdwpose2d = interpolate_poses(image_xdwpose_np_list[i], video_xdwpose_np[0], interpolate_N, confidence_threshold=5)
                
                dwpose3d_list_image[i][:, :, :, 0] /= new_w # inplace 操作
                dwpose3d_list_image[i][:, :, :, 1] /= new_h
                video_w, video_h = video_data_list[i]['data_list'][0]['width'], video_data_list[i]['data_list'][0]['height']
                
                for frame_id in range(interpolate_N):
                    a_valid = interpolate_xdwpose2d[frame_id, 0, :, 2]>5
                    b_valid = dwpose3d_list_image[i][frame_id, 0, :, 2]>5
                    visible = a_valid & b_valid
                    if OptHeadTail and Headrepeat:
                        if frame_id>=(interpolate_N-Headrepeat): # TODO: 首帧模糊优化
                            visible = a_valid

                    dwpose2d_fuse_3d_pose[frame_id, 0, visible, :] = weight_2d_3d[frame_id] * interpolate_xdwpose2d[frame_id, 0, visible, :] + (1-weight_2d_3d[frame_id]) * dwpose3d_list_image[i][frame_id, 0, visible, :]
                    # dwpose2d_fuse_3d_pose[frame_id, 0, visible, :] = interpolate_xdwpose2d[frame_id, 0, visible, :] # not work
                    # dwpose2d_fuse_3d_pose[frame_id, 0, visible, :] = dwpose3d_list_image[i][frame_id, 0, visible, :] # not work # debugTransition4
                
                out_2d = np.concatenate((dwpose2d_fuse_3d_pose, video_xdwpose_np_list[0]), axis=0)[:video_len]

                video_xdwpose_np_final_list.append(out_2d)
                video_xdwpose_np_final_list_ori.append(np.concatenate((np.zeros_like(dwpose2d_fuse_3d_pose), video_xdwpose_np_list_ori[0]), axis=0)[:video_len]) # unuse: dwpose2d_fuse_3d_pose
                # video_xdwpose_np_final_list.append(video_xdwpose_np_list[0])
            else:
                dwpose3d_list_image[i][:, :, :, 0] /= new_w # inplace 操作
                dwpose3d_list_image[i][:, :, :, 1] /= new_h
                video_xdwpose_np_final_list.append(dwpose3d_list_image[i])
        

        for videoi in range(len(video_xdwpose_np_final_list)):
            video_xdwpose_np = video_xdwpose_np_final_list[videoi]
            # video_xdwpose_np[:, :, :, 0] /= new_w # inplace 操作
            # video_xdwpose_np[:, :, :, 1] /= new_h

            # image_xdwpose_np_list[videoi][:, :, :, 0] /= new_w # inplace 操作
            # image_xdwpose_np_list[videoi][:, :, :, 1] /= new_h
            # 如果不转朝向，需要进行2D插值
            # if not change_direction_list[videoi]:
                # interpolate_xdwpose_before = interpolate_poses(image_xdwpose_np_list[videoi], video_xdwpose_np[5], 5, confidence_threshold=5)
                # interpolate_xdwpose_after = interpolate_poses(video_xdwpose_np[interpolate_N-6], video_xdwpose_np[interpolate_N], 5, confidence_threshold=5)
                # video_xdwpose_np[:5] = interpolate_xdwpose_before
                # video_xdwpose_np[interpolate_N-5: interpolate_N] = interpolate_xdwpose_after

            for iframe in range(video_xdwpose_np.shape[0]):
                for ichar in range(video_xdwpose_np.shape[1]):
                    for ij2d in range(video_xdwpose_np.shape[2]):
                        if video_xdwpose_np[iframe,ichar,ij2d,0]>1.2 or video_xdwpose_np[iframe,ichar,ij2d,0]<-.2 or \
                           video_xdwpose_np[iframe,ichar,ij2d,1]>1.2 or video_xdwpose_np[iframe,ichar,ij2d,1]<-.2 :
                           video_xdwpose_np[iframe,ichar,ij2d,2]=0
            video_xdwpose_np_final_list[videoi] = video_xdwpose_np
            #print ('video_xdwpose_np', video_xdwpose_np.shape)
        
        for videoi in range(len(video_xdwpose_np_final_list)): #多人
            if not is_bvh_list[i]:
                video_dwpose = np.array(video_xdwpose_np_final_list[videoi]) #[24:]
                image_dwpose = np.array(video_xdwpose_np_final_list[videoi][0:1])
                video_hw = np.array([[new_h, new_w]]) # 已经重定向到image
                image_hw = np.array([[new_h, new_w]])
                print('new'*20)
                if change_direction_list[videoi]:
                    new_dwpose = self.retargeter_pose_dit.inf(video_dwpose, image_dwpose, video_hw, image_hw)
                else:
                    # new_dwpose = self.pose_blender_dit.inf(video_dwpose, image_dwpose, video_hw, image_hw)
                    new_dwpose = video_dwpose.copy()
                
                video_xdwpose_np_final_list[videoi] = new_dwpose[:video_xdwpose_np_final_list[videoi].shape[0]]
                #new_dwpose = video_dwpose.copy()
                #new_dwpose = np.load('re.npy')
                # 出框保护逻辑
                video_xdwpose_np = video_xdwpose_np_final_list[videoi]
                for iframe in range(video_xdwpose_np.shape[0]):
                    for ichar in range(video_xdwpose_np.shape[1]):
                        for ij2d in range(video_xdwpose_np.shape[2]):
                            if video_xdwpose_np[iframe,ichar,ij2d,0]>1.2 or video_xdwpose_np[iframe,ichar,ij2d,0]<-.2 or video_xdwpose_np[iframe,ichar,ij2d,1]>1.2 or video_xdwpose_np[iframe,ichar,ij2d,1]<-.2 :
                                video_xdwpose_np[iframe,ichar,ij2d,2]=0
                video_xdwpose_np_final_list[videoi] = video_xdwpose_np
                # 出框保护逻辑
        video_len = video_xdwpose_np_final_list[videoi].shape[0]
        print('video_len', video_len)

        video_xdwpose_np_multi_persons = np.concatenate(video_xdwpose_np_final_list, axis=1) #(F, 2, 134, 3)
        motion_video_binary = self.painter_dwpose(output_result_path,video_xdwpose_np_final_list,[video_len, new_w, new_h ,output_fps],target_resolution,threshold=threshold, input_image=input_image, is_show=False)
        
        video_xdwpose_np_final_list_neg = make_neg_pose(video_xdwpose_np_final_list)
        # video_xdwpose_np_final_list_neg_full = make_neg_pose_full(video_xdwpose_np_final_list)
        video_xdwpose_np_final_list_neg_full = None
        motion_video_binary_neg = self.painter_dwpose(output_result_path,video_xdwpose_np_final_list_neg,[video_len, new_w, new_h ,output_fps],target_resolution,threshold=threshold, input_image=input_image, is_show=False, is_neg=True, neg_pose_full=video_xdwpose_np_final_list_neg_full)

        ''' RGB motion info '''
        vid_len, width, height, fps = video_len, new_w, new_h ,output_fps
        video_np_list_ori=video_np_final_list_ori
        dwpose_np_list_ori=video_xdwpose_np_final_list_ori
        image_dwpose_array=new_image_dwpose_array.copy()
        
        video_xdwpose_np_ori = dwpose_np_list_ori[0][interpolate_N:] #(F-interpolate_N, 1, 134, 3)
        video_np_ori = video_np_list_ori[0][interpolate_N:] #(F-interpolate_N, h, w, 3)
        _, ori_height, ori_width, _ = video_np_ori.shape

        video_xdwpose_np = self.pose_filter_ljw(video_xdwpose_np_ori)
        # video_xdwpose_np = np.clip(video_xdwpose_np, 0, 1)
        video_xdwpose_np[..., 0] = np.clip(video_xdwpose_np[..., 0], 0, 1)
        video_xdwpose_np[..., 1] = np.clip(video_xdwpose_np[..., 1], 0, 1)
        video_xdwpose_np[...,0] *= ori_width
        video_xdwpose_np[...,1] *= ori_height

        frame_indexes = list(range(video_xdwpose_np_ori.shape[0]))

        '''脸部整体置信度'''
        dwpose_idx_for_face_center = [i for i in range(41,92)] #[i for i in range(72,92)]
        face_confidence_mean = np.mean(video_xdwpose_np[:, 0, dwpose_idx_for_face_center, 2])
        face_confidence_max = np.max(video_xdwpose_np[:, 0, dwpose_idx_for_face_center, 2])
        ratio_below_9 = np.mean(video_xdwpose_np[:, 0, dwpose_idx_for_face_center, 2] < 8)
        # face_confidence = [face_confidence_mean, face_confidence_max, ratio_below_9]
        scores = video_xdwpose_np[:, 0, dwpose_idx_for_face_center, 2]
        face_confidence = [scores[i:i+8].mean() for i in range(0, len(scores), 8)] 
        # print(output_result_path, '---------------------------', face_confidence)
        # exit()

        # 脸部
        tgt_crop_bbox_list, crop_radio = self.get_crop_bbox_tsl(video_xdwpose_np, frame_indexes, ori_width, ori_height)
        face_motion_img_list = [cv2.resize(tgt_img[tgt_crop_bbox_list[i][0]:tgt_crop_bbox_list[i][2], tgt_crop_bbox_list[i][1]:tgt_crop_bbox_list[i][3]], (self.face_width, self.face_height), interpolation=cv2.INTER_LINEAR) for i, tgt_img in enumerate(video_np_ori)]
        
        face_motion_img_list = [np.zeros_like(face_motion_img_list[0])] * interpolate_N + face_motion_img_list

        if OptHeadTail:
            print('face motion len / before',len(face_motion_img_list))
            if Redundant_frame>0:
                face_motion_img_list = face_motion_img_list + [face_motion_img_list[-1]] * Redundant_frame
            face_motion_img_list[-Tailrepeat:] = [face_motion_img_list[-(Tailrepeat+1)]] * Tailrepeat
            print('face motion len / after',len(face_motion_img_list))
            Redundant_frame = max(Redundant_frame-Tailrepeat, 0) # 返回最后截掉的帧数，补帧大于2帧去掉，只要最后3帧一致
        else:
            Redundant_frame = 0

        with open(save_split_path+"/Redundant_frame.txt", "w") as f:
            f.write(str(Redundant_frame))

        with open(save_split_path+"/Redundant_frame.txt", "r") as f:
            Redundant_frame = float(f.read())
        print('drop Redundant_frame:', Redundant_frame)

        save_name = "motion_video_ref"
        writer = get_writer(output_result_path+'/{}.mp4'.format(save_name), fps=fps, crf=23)
        for img in face_motion_img_list:
            writer.write_frame(img)
        writer.close()

        with open(output_result_path+'/{}.mp4'.format(save_name), "rb") as f:
            motion_video_ref_binary = f.read()
            
        if save_split_path is not None:
            save_name = "motion_video_raw"
            writer = get_writer(save_split_path+'/{}.mp4'.format(save_name), fps=fps, crf=17) #crf=23
            for img in video_np_ori:
                writer.write_frame(img)
            writer.close()
            # overlap_split_save(video_np_ori, save_split_path, fps)

        # image face motion
        image_dwpose_array = np.clip(image_dwpose_array, 0, 1)
        image_dwpose_array[...,0] *= width
        image_dwpose_array[...,1] *= height

        tgt_crop_bbox_list_ref_img, _ = self.get_crop_bbox_tsl(image_dwpose_array, [0], width, height, crop_radio=crop_radio)
        imgae_face_motion = cv2.resize(input_image[tgt_crop_bbox_list_ref_img[0][0]:tgt_crop_bbox_list_ref_img[0][2], tgt_crop_bbox_list_ref_img[0][1]:tgt_crop_bbox_list_ref_img[0][3]], (self.face_width, self.face_height), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(output_result_path+'/input_motion.png', imgae_face_motion)
        with open(output_result_path+'/input_motion.png', "rb") as f:
            input_image_motion_binary = f.read()

        result_data_dict['error_code'] = self.error_code.SUCCESS
        result_data_dict['error_message'] = self.error_messages.get(result_data_dict['error_code'])

        os.system('rm -rf '+output_result_path) ##清空临时文件
        return bvh_result, motion_video_binary, motion_video_binary_neg, input_image_binary, new_image_dwpose_array, new_image_h_w_array, video_xdwpose_np_multi_persons, result_data_dict['error_code'], result_data_dict['error_message'], \
            motion_video_ref_binary, input_image_motion_binary, face_confidence

        
