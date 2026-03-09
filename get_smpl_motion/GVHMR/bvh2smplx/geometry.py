import torch
from torch.nn import functional as F
import numpy as np
import math

import torchgeometry as tgm
from scipy.spatial.transform import Rotation as R
"""
Useful geometric operations, e.g. Perspective projection and a differentiable Rodrigues formula
Parts of the code are taken from https://github.com/MandyMo/pytorch_HMR
"""

def flip_pose(pose, flip_pairs):
    """Flip the symmetric body part and multiply y, z-axis with -1.0"""
    pose = pose.copy()
    flip_pairs = np.array(flip_pairs)

    if len(pose.shape) == 2:
        pose[flip_pairs[:, 0], :], pose[flip_pairs[:, 1], :] = pose[flip_pairs[:, 1], :].copy(), pose[flip_pairs[:, 0], :].copy()
        pose[:, 1:3] *= -1
    else:
        pose[:, flip_pairs[:, 0], :], pose[:, flip_pairs[:, 1], :] = pose[:, flip_pairs[:, 1], :], pose[:, flip_pairs[:, 0], :]
        pose[:, :, 1:3] *= -1
    return pose

def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def camera_setup(alpha, phi, height):
    """
    The same virtual camera set-up with the paper.
    alpha, phi is in degree format.
    """

    # euler_degree = np.array([90, 180, 180])
    euler_degree = np.array([-90, 0, 0])
    Rc = R.from_euler("xyz", euler_degree, degrees=True).as_matrix()
    Rc_ = R.from_euler("xyz", np.array([alpha, 0, phi]), degrees=True).as_matrix()
    # Rc = R.from_euler("XYZ", euler_degree, degrees=True).as_matrix()
    # Rc_ = R.from_euler("XYZ", np.array([alpha, 0, phi]), degrees=True).as_matrix()
    Rc = np.matmul(Rc, Rc_)

    R_ = Rc.T

    Tc = np.array([0, 0, height])
    T_ = -np.dot(R_, Tc.T).T

    return R_, T_

def camera_world2camera(alpha, phi, height):
    euler_degree = np.array([90, 180, 180])
    Rc = R.from_euler("xyz", euler_degree, degrees=True).as_matrix()
    Rc_ = R.from_euler("xyz", np.array([alpha, 0, phi]), degrees=True).as_matrix()
    Rc = np.matmul(Rc, Rc_)
    Tc = np.array([0, 0, height])
    return Rc, Tc

def side_view_camera(R_wc, T_wc, R_old2new, T_old2new):
    R_new_wc = np.matmul(R_old2new, R_wc)
    T_new_wc = T_wc + T_old2new
    R_ = R_new_wc.T
    T_ = -np.dot(R_, T_new_wc.T).T
    return R_, T_
    

def world_to_camera(cam_rotation, cam_trans, pose, trans, bind_trans):

    root_pose = pose[0]
    root_rotmat = R.from_rotvec(root_pose).as_matrix()

    root_rotmat_cam = np.matmul(cam_rotation, root_rotmat)
    root_pose_cam = R.from_matrix(root_rotmat_cam).as_rotvec()

    # trans_cam = np.matmul(trans + bind_trans - cam_trans, np.transpose(cam_rotation)) - bind_trans
    trans_cam = np.dot(cam_rotation, trans + bind_trans) + cam_trans - bind_trans

    pose_cam = pose.copy()
    pose_cam[0, :] = root_pose_cam

    return pose_cam, trans_cam


def batch_euler2matrix(r):
    return quaternion_to_rotation_matrix(euler_to_quaternion(r))


def euler_to_quaternion(r):
    x = r[..., 0]
    y = r[..., 1]
    z = r[..., 2]

    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = torch.cos(z)
    sz = torch.sin(z)
    cy = torch.cos(y)
    sy = torch.sin(y)
    cx = torch.cos(x)
    sx = torch.sin(x)
    quaternion = torch.zeros_like(r.repeat(1,2))[..., :4].to(r.device)
    quaternion[..., 0] += cx*cy*cz - sx*sy*sz
    quaternion[..., 1] += cx*sy*sz + cy*cz*sx
    quaternion[..., 2] += cx*cz*sy - sx*cy*sz
    quaternion[..., 3] += cx*cy*sz + sx*cz*sy
    return quaternion

def rotation_matrix_to_axis_angle(rotmats):

    batch_size = rotmats.shape[0]
    rotmats = torch.cat([rotmats,torch.zeros((batch_size,3,1)).cuda().float()], 2)
    axis_angle = tgm.rotation_matrix_to_angle_axis(rotmats).reshape(-1,3) # axis-angle
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle

### NOTE: the rotation 6D to matrix is incorrect, 
# ref to https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#rotation_6d_to_matrix.
def rot6d_to_axis_angle(x):
    batch_size = x.shape[0]

    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    rot_mat = torch.stack((b1, b2, b3), dim=-1) # 3x3 rotation matrix

    rot_mat = torch.cat([rot_mat,torch.zeros((batch_size,3,1)).cuda().float()],2) # 3x4 rotation matrix
    axis_angle = tgm.rotation_matrix_to_angle_axis(rot_mat).reshape(-1,3) # axis-angle
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle

def rot6d_to_aa(x):
    x = x.reshape((-1, 6))
    batch_size = x.shape[0]
    rot_mat = rot6d_to_rotation_matrix(x).reshape((-1, 3, 3))
    rot_mat = torch.cat([rot_mat,torch.zeros((batch_size,3,1)).cuda().float()],2) # 3x4 rotation matrix
    axis_angle = tgm.rotation_matrix_to_angle_axis(rot_mat).reshape(-1,3) # axis-angle
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle

def quaternion_to_rotation_matrix(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)

def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """ 
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat    

def rot6d_to_rotation_matrix(x):

    return rotation_6d_to_matrix(x.reshape((-1, 6))).reshape((-1, 3, 3))

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    # return torch.stack((b1, b2, b3), dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

def aa_to_rotation_6d(theta):

    # rotmat = batch_rodrigues(theta.view(-1, 3)).view(-1, 3, 3)
    rotmat = axis_angle_to_matrix(theta.reshape((-1, 3)))

    return matrix_to_rotation_6d(rotmat)

def swing_twist_decompose(local_rotmat, par_global_rotmat, rel_joint, rel_joint_rest):
    p_vec = np.dot(par_global_rotmat.T, rel_joint)
    t_vec = rel_joint_rest.copy()

    p_vec_norm = np.linalg.norm(p_vec)
    t_vec_norm = np.linalg.norm(t_vec)

    axis = np.cross(t_vec, p_vec)
    axis_norm = np.linalg.norm(axis)

    cos = (p_vec * t_vec).sum() / (p_vec_norm * t_vec_norm + 1e-9)
    sin = axis_norm / (p_vec_norm * t_vec_norm + 1e-9)

    axis = axis / (axis_norm + 1e-9)
    
    rx, ry, rz = axis[0], axis[1], axis[2]
    K = np.array([[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]])
    I = np.eye(3)

    rotmat_swing = I + sin * K + (1 - cos) * np.dot(K, K)
    rotmat_twist = np.dot(rotmat_swing.T, local_rotmat)
    
    rotvec_twist = R.from_matrix(rotmat_twist).as_rotvec()
    
    angle_twist = np.linalg.norm(rotvec_twist)

    sin_twist = np.sin(angle_twist)
    cos_twist = np.cos(angle_twist)

    twist_phi = np.array([cos_twist, sin_twist])
    return twist_phi

