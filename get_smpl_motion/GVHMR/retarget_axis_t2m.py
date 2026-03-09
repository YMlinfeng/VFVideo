import numpy as np
import torch

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

def bvh2hmr(pred_src):
    for i in range(pred_src["smpl_params_global"]["global_orient"].shape[0]):
        gr = pred_src["smpl_params_global"]["global_orient"][i].cpu().numpy()
        gr = rodrigues_to_matrix(gr)
        gr = rodrigues_to_matrix(np.array([0, np.pi, 0]))@rodrigues_to_matrix(np.array([0, 0, np.pi]))@gr
        pred_src["smpl_params_global"]["global_orient"][i] = torch.tensor(rodrigues_to_vector(gr))
    pred_src["smpl_params_global"]["transl"][...,1] = - pred_src["smpl_params_global"]["transl"][...,1].cpu()
    pred_src["smpl_params_global"]["transl"][...,2] = - pred_src["smpl_params_global"]["transl"][...,2].cpu()

    return pred_src

def hmr2bvh(pred_src):
    for i in range(pred_src["smpl_params_global"]["global_orient"].shape[0]):
        gr = pred_src["smpl_params_global"]["global_orient"][i].cpu().numpy()
        gr = rodrigues_to_matrix(gr)
        gr = rodrigues_to_matrix(np.array([0, 0, np.pi]))@rodrigues_to_matrix(np.array([0, np.pi, 0]))@gr
        pred_src["smpl_params_global"]["global_orient"][i] = torch.tensor(rodrigues_to_vector(gr))
    pred_src["smpl_params_global"]["transl"][...,1] = - pred_src["smpl_params_global"]["transl"][...,1].cpu()
    pred_src["smpl_params_global"]["transl"][...,2] = - pred_src["smpl_params_global"]["transl"][...,2].cpu()

    return pred_src

def gv_global2bvh(pred_src):
    for i in range(pred_src["smpl_params_global"]["global_orient"].shape[0]):
        gr = pred_src["smpl_params_global"]["global_orient"][i].cpu().numpy()
        gr = rodrigues_to_matrix(gr)
        gr = rodrigues_to_matrix(np.array([0, np.pi, 0]))@gr
        pred_src["smpl_params_global"]["global_orient"][i] = torch.tensor(rodrigues_to_vector(gr))
    pred_src["smpl_params_global"]["transl"][...,0] = - pred_src["smpl_params_global"]["transl"][...,0].cpu()
    pred_src["smpl_params_global"]["transl"][...,2] = - pred_src["smpl_params_global"]["transl"][...,2].cpu()

    return pred_src