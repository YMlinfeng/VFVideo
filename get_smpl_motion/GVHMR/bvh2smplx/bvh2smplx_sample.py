import argparse
import os
import glob
from pathlib import Path
import torch
import numpy as np
import sys
import torch
import numpy as np
# sys.path.append(".")

# from beat.pymo.parsers import BVHParser
# from beat.pymo.preprocessing import *
# from beat.pymo.viz_tools import *
# from beat.pymo.writers import *
# from beat.tool import *
from .beat.anim import bvh, quat, txform
# from beat.beat_data_proc.MyBVH import load_bvh_data

# from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F

import io
import string
import h5py
import smplx
import trimesh

from scipy.spatial.transform import Rotation

# from .geometry import aa_to_rotation_6d, rot6d_to_aa, matrix_to_rotation_6d, rot6d_to_rotation_matrix, rotation_matrix_to_axis_angle
# MODEL_PATH = "/share/group_wenziyu/chenming/spin/human_models"

import pdb
from collections import OrderedDict

def extract_rotation_orders(bvh_file):
    """
    解析BVH文件，提取每个关节的欧拉角旋转顺序信息。
    返回一个字典，键为关节名称，值为旋转顺序字符串（例如 'ZXY'）。
    """
    rotation_orders = OrderedDict()
    current_joint = None

    with open(bvh_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 识别关节（ROOT 或 JOINT）
            if line.startswith("ROOT") or line.startswith("JOINT"):
                parts = line.split()
                if len(parts) >= 2:
                    current_joint = parts[1]
            # 查找CHANNELS行
            elif line.startswith("CHANNELS"):
                parts = line.split()
                # parts[0]为"CHANNELS"，parts[1]为通道数量，其余为通道名称
                channels = parts[2:]
                # 提取旋转通道，注意一般通道名称以"rotation"结尾（大小写可能不同）
                rot_order = [ch[0].upper() for ch in channels if "rotation" in ch.lower()]
                if current_joint and rot_order:
                    rotation_orders[current_joint] = "".join(rot_order)
    return rotation_orders

target_joints = ["pelvis",
            "left_hip",
            "right_hip",
            "spine1",
            "left_knee",
            "right_knee",
            "spine2",
            "left_ankle",
            "right_ankle",
            "spine3",
            "left_foot",
            "right_foot",
            "neck",
            "left_collar",
            "right_collar",
            "head",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "jaw",
            "left_eye_smplhf",
            "right_eye_smplhf",
            "left_index1",
            "left_index2",
            "left_index3",
            "left_middle1",
            "left_middle2",
            "left_middle3",
            "left_pinky1",
            "left_pinky2",
            "left_pinky3",
            "left_ring1",
            "left_ring2",
            "left_ring3",
            "left_thumb1",
            "left_thumb2",
            "left_thumb3",
            "right_index1",
            "right_index2",
            "right_index3",
            "right_middle1",
            "right_middle2",
            "right_middle3",
            "right_pinky1",
            "right_pinky2",
            "right_pinky3",
            "right_ring1",
            "right_ring2",
            "right_ring3",
            "right_thumb1",
            "right_thumb2",
            "right_thumb3",]

bone_names = target_joints
order = 'zyx'       # 'XYZ', 'ZXY'

'''
def smplx_fk(poses, trans):
    betas = np.zeros((300,))
    expression = np.zeros((1, 100))
    smplx_model = smplx.create(model_path=MODEL_PATH, 
                        num_betas=betas.shape[0],
                        model_type="smplx",
                        gender="neutral", 
                        num_expression_coeffs=expression.shape[1],
                        batch_size=1, use_face_contour=False, use_pca=False).cuda()
    

    trans_ = trans[0:1]
    pose_aa_ = poses[0:1]
    pose_aa_ = torch.from_numpy(pose_aa_).float().cuda()
    shape_ = torch.from_numpy(betas).unsqueeze(0).float().repeat((pose_aa_.shape[0], 1)).cuda()
    expr_ = torch.from_numpy(expression[0:1]).float().cuda()

    out = smplx_model(betas=shape_, 
                    expression=expr_,
                    jaw_pose=pose_aa_[:, 22], 
                    global_orient=pose_aa_[:, 0], 
                    body_pose=pose_aa_[:, 1:22], 
                    left_hand_pose=pose_aa_[:, 25:40],
                    right_hand_pose=pose_aa_[:, 40:55],
                    leye_pose=pose_aa_[:, 23],
                    reye_pose=pose_aa_[:, 24], 
                    return_verts=True)

    verts = out.vertices.detach().cpu().numpy()[0]

    mesh = trimesh.Trimesh(vertices=verts, faces=smplx_model.faces, process=False)
    mesh.export("mesh.obj")
'''

def smplx_bvh_to_params(animation_file, bvh_order="zxy"):

    anim_data = bvh.load(animation_file, order=bvh_order.lower())       #  'rotations' (8116, 75, 3), 'positions', 'offsets' (75, 3), 'parents', 'names' (75,), 'order' 'zyx', 'frametime' 0.016667
    nframes = len(anim_data["rotations"])

    # print(anim_data["rotations"].shape, anim_data["positions"].shape)
    rerange_idx = np.array([anim_data["names"].index(name) for name in target_joints]).astype(np.int32)
    anim_data["rotations"] = anim_data["rotations"][:, rerange_idx, :]
    anim_data["positions"] = anim_data["positions"][:, rerange_idx, :]
    anim_data["parents"] = np.array([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
                                    16, 17, 18, 19, 15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34,
                                    35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52,
                                    53])
    
    anim_data["names"] = target_joints
    dt = anim_data["frametime"]

    # print(anim_data.keys())
    njoints = len(anim_data["parents"])
    # print(anim_data["order"])
    # print(anim_data["rotations"].shape, anim_data["positions"].shape)
    # print(anim_data["parents"])
    # print(anim_data["names"])
    
    # TODO: check the rotation values of bvh
    anim_data["rotations"][np.isnan(anim_data["rotations"])] = 0.0

    lrot = quat.unroll(quat.from_euler(np.radians(anim_data["rotations"]), anim_data["order"]))
    lpos = anim_data["positions"]
    grot, gpos = quat.fk(lrot, lpos, anim_data["parents"])
    # Find root (Projected hips on the ground)

    transl = gpos[:, anim_data["names"].index("pelvis")] / 100.0
    angle, axis = quat.to_angle_axis(lrot)
    lrot_aa = angle * axis

    return lrot_aa, transl, 1 / dt

from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
def slerp(result, old_fps=20, target_fps=25):
    N = result["smpl_params_global"]["transl"].shape[0]
    t_old = np.linspace(0, (N-1)/old_fps, N)
    t_new = np.linspace(0, (N-1)/old_fps, int(N * target_fps/old_fps))

    transl_old = result["smpl_params_global"]["transl"].cpu().numpy()  # [N, 3]
    interp_transl = interp1d(t_old, transl_old, axis=0, kind='linear')
    transl_new = interp_transl(t_new)  # [M, 3]
    
    # 全局旋转插值
    global_orient_old = result["smpl_params_global"]["global_orient"].cpu().numpy()  # [N, 3]
    rots_global = Rotation.from_rotvec(global_orient_old)
    slerp_global = Slerp(t_old, rots_global)
    global_orient_new = slerp_global(t_new).as_rotvec()  # [M, 3]

    # 身体姿势插值（逐关节处理）
    body_pose_old = result["smpl_params_global"]["body_pose"].cpu().numpy()  # [N, 63]
    joints_num = int(body_pose_old.shape[1]/3)
    body_pose_new = np.zeros((len(t_new), 3*joints_num))

    for j in range(joints_num):  # 21个关节
        joint_rots = Rotation.from_rotvec(body_pose_old[:, j*3 : (j+1)*3])
        slerp_joint = Slerp(t_old, joint_rots)
        body_pose_new[:, j*3 : (j+1)*3] = slerp_joint(t_new).as_rotvec()

    # 手姿势插值（逐关节处理）
    left_hand_pose_old = result["smpl_params_global"]["left_hand_pose"].cpu().numpy()  # [N, 63]
    joints_num = int(left_hand_pose_old.shape[1]/3)
    left_hand_pose_new = np.zeros((len(t_new), 3*joints_num))
    for j in range(joints_num):  # 15个关节
        joint_rots = Rotation.from_rotvec(body_pose_old[:, j*3 : (j+1)*3])
        slerp_joint = Slerp(t_old, joint_rots)
        left_hand_pose_new[:, j*3 : (j+1)*3] = slerp_joint(t_new).as_rotvec()
    
    right_hand_pose_old = result["smpl_params_global"]["right_hand_pose"].cpu().numpy()  # [N, 63]
    joints_num = int(right_hand_pose_old.shape[1]/3)
    right_hand_pose_new = np.zeros((len(t_new), 3*joints_num))
    for j in range(joints_num):  # 15个关节
        joint_rots = Rotation.from_rotvec(body_pose_old[:, j*3 : (j+1)*3])
        slerp_joint = Slerp(t_old, joint_rots)
        right_hand_pose_new[:, j*3 : (j+1)*3] = slerp_joint(t_new).as_rotvec()
    
    betas_old = result["smpl_params_global"]["betas"].cpu().numpy()  # [N, 10]
    interp_betas = interp1d(t_old, betas_old, axis=0, kind='linear')
    betas_new = interp_betas(t_new)  # [M, 10]

    result_25fps = {
        "smpl_params_global": {
            "transl": torch.tensor(transl_new, dtype=torch.float32),
            "global_orient": torch.tensor(global_orient_new, dtype=torch.float32),
            "betas": torch.tensor(betas_new, dtype=torch.float32),
            "body_pose": torch.tensor(body_pose_new, dtype=torch.float32),
            "left_hand_pose": torch.tensor(left_hand_pose_new, dtype=torch.float32),
            "right_hand_pose": torch.tensor(right_hand_pose_new, dtype=torch.float32)
        }
    }
    return result_25fps

def bvh_to_smplx(bvh_file, target_fps=25):
    order = extract_rotation_orders(bvh_file)
    order = list(order.values())[0].lower()

    poses, trans, fps = smplx_bvh_to_params(bvh_file, order)
    N = min(poses.shape[0], int(10*fps))
    poses = poses[:N]
    trans = trans[:N]
    print (poses.shape, trans.shape, fps)
    body_pose_N = 21
    result={
        "smpl_params_global": {
            "transl": torch.tensor(trans, dtype=torch.float32).cpu(),
            "global_orient": torch.tensor(poses[:, 0, :], dtype=torch.float32).cpu(),
            "betas": torch.zeros(N, 10).cpu(),
            "body_pose": torch.tensor(poses[:, 1:body_pose_N+1, :].reshape(N, -1), dtype=torch.float32).cpu()
        }
    }
    if N > 54:
        result["smpl_params_global"]["left_hand_pose"] = torch.tensor(poses[:, body_pose_N+4:body_pose_N+4+15, :].reshape(N, -1), dtype=torch.float32).cpu()
        result["smpl_params_global"]["right_hand_pose"] = torch.tensor(poses[:, body_pose_N+4+15:body_pose_N+4+30, :].reshape(N, -1), dtype=torch.float32).cpu()
    # print("result : ",result["smpl_params_global"]["body_pose"].shape,  result["smpl_params_global"]["left_hand_pose"].shape , result["smpl_params_global"]["right_hand_pose"].shape)
    # 此处插值
    new_result = slerp(result, fps, target_fps) if target_fps > 0 else result
    print("result : ",new_result["smpl_params_global"]["body_pose"].shape,  new_result["smpl_params_global"]["left_hand_pose"].shape , new_result["smpl_params_global"]["right_hand_pose"].shape)
    return new_result

if __name__ == "__main__":
    result = bvh_to_smplx("/ytech_milm/lixiaohan/code/kelingHumanMotion/get_smpl_motion/bvh2smplx/001_Neutral_0_mirror_x_0_9.bvh", target_fps=24)
