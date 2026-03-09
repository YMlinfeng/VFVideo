import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def filter_wilor(filter, rot_mat, i, beta):
    rot_mat = rot_mat.reshape(3,3)
    quan = R.from_matrix(rot_mat).as_quat()
    filter_out, _ = filter.filter(quan, 0.033*i, beta)
    smoothed_rot = R.from_quat(filter_out).as_matrix()
    return smoothed_rot

def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev


def vis2beta(vis):
    vis = vis.min()
    if vis<0.8:
        beta = 0.0
    elif vis<0.9:
        beta = 0.2
    else:
        beta = 0.5
    return beta

def diff2beta(diff):
    if diff<0.025:
        beta = 0.0
    elif diff<0.05:
        beta = 0.1
    elif diff<0.2:
        beta = 0.2
    else:
        beta = 0.5
    return beta



class OneEuroFilter:
    def __init__(self, x0=0.0, t0=None,  dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        if t0 is not None:
            self.t_prev = float(t0)
        self.init_flag = False

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


    def filter(self, x, t=None, reset=False, prev=None):
        """Compute the filtered signal."""
        if reset:
            self.init_flag = False
        if not self.init_flag:
            self.init_flag = True
            if prev is None:
                self.x_prev = x
                self.dx_prev = 0
                self.t_prev = t     
                return x, (self.x_prev, self.t_prev, self.dx_prev)
            else:
                self.x_prev, self.t_prev, self.dx_prev = prev

        if t is not None:
            t_e = t - self.t_prev
        else:
            t_e = 1/20 #20fps

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)
        # print("dx_hat : ", self.d_cutoff, a_d, dx, self.dx_prev, dx_hat)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)
        # print("x_hat : ", self.min_cutoff, cutoff, a, x, self.x_prev, x_hat)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat, (self.x_prev, self.t_prev, self.dx_prev)

def clamp(value, min_val, max_val):
    return max(min(value, max_val), min_val)

# compute angle 
def compute_angle(q1, q2):
    cos_theta = np.dot(q1, q2) / (np.linalg.norm(q1) * np.linalg.norm(q2))
    theta = math.acos(clamp(cos_theta, -1, 1))
    return theta

# q1 --- t * theta --- r --- (1-t) * theta --- q2
# 求一个向量 r = t * q1 + (1-t) * q2 注意：非实际数学定义，实际为球面线性插值
# 参考四元数球面插值slerp计算方法，当夹角较小时可直接退化为nlerp
# 参考：https://zhuanlan.zhihu.com/p/538653027
def slerp_matrix(r1, r2, i_before, i_after, i):
    if not i_before:
        return r2
    if not i_after:
        return r1
    t = (i-i_before)/(i_after-i_before)
    q1 = R.from_matrix(r1).as_quat()
    q2 = R.from_matrix(r2).as_quat()
    res = slerp_q(q1, q2, t)
    res = R.from_quat(res).as_matrix()
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

def exponential_smoothing_q(a, x, x_prev):
    return slerp_q(x_prev, x, a)

# 13+75*3 - 13+75*9 process_zeggs txform
class OneEuroFilterQuaternion:
    def __init__(self, x0=None, t0=None,  dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        # Previous values.
        self.x_prev = x0
        self.dx_prev = float(dx0)
        if t0 is not None:
            self.t_prev = float(t0)
        self.init_flag = False
    
    def filter(self, x,  t=None, reset=False, prev=None, beta=0.0):  #(x,y,z,w)
        """Compute the filtered signal."""
        self.beta = float(beta)

        if reset:
            self.init_flag = False
        if not self.init_flag:
            self.init_flag = True
            if prev is None:
                self.x_prev = x
                self.dx_prev = 0.
                self.t_prev = t
                return x, (self.x_prev, self.t_prev, self.dx_prev)
            else: 
                self.x_prev, self.t_prev, self.dx_prev = prev
                

        if t is not None:
            t_e = t - self.t_prev
        else:
            t_e = 1./20. #20fps

        # if np.random.randn()>0.5:
        #     pass
        # else:
        #     pass
        #     #x = -x

        #like unroll function
        #四元数完全相反的/W完全相反的，在这个非数值滤波中，需要unroll吗？-- 【不需要】，因为取了dot 绝对值算theta?
        #取相反是简单数值滤波中需要
        #这其实也不是右臂twist的原因
        # zy, 0516
        # d0 = np.sum(y[i] * y[i - 1], axis=-1)
        # d1 = np.sum(-y[i] * y[i - 1], axis=-1)

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = compute_angle(self.x_prev, x) / t_e # 角速度
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev) # 平滑角速度
        # print("dx_hat : ", self.d_cutoff, a_d, dx, self.dx_prev, dx_hat)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing_q(a, x, self.x_prev) # 平滑四元数
        # print("x_hat : ", self.min_cutoff, cutoff, a, x, self.x_prev, x_hat)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat, (self.x_prev, self.t_prev, self.dx_prev)


def find_nearest_true(lst, index):
    if lst[index]:
        return index, index
    
    # 向前查找最近的True值
    nearest_true_before = None
    for i in range(index - 1, -1, -1):
        if lst[i]:
            nearest_true_before = i
            break
    
    # 向后查找最近的True值
    nearest_true_after = None
    for i in range(index + 1, len(lst)):
        if lst[i]:
            nearest_true_after = i
            break
    
    return nearest_true_before, nearest_true_after


def compute_global_rotation(pose_axis_anges, joint_idx):
    """
    calculating joints' global rotation
    Args:
        pose_axis_anges (np.array): SMPLX's local pose (22,3)
    Returns:
        np.array: (3, 3)
    """
    global_rotation = np.eye(3)
    parents = [-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14, 16, 17, 18, 19]
    while joint_idx != -1:
        joint_rotation = R.from_rotvec(pose_axis_anges[joint_idx]).as_matrix()
        global_rotation = joint_rotation @ global_rotation
        joint_idx = parents[joint_idx]
    return global_rotation

def isolate_ones_np_optimized(arr, n):
    """
    向量化实现，将孤立的1变成0
    
    参数:
    arr -- 输入的NumPy数组
    n -- 前后需要检查的0的个数
    
    返回:
    修改后的NumPy数组
    """
    arr = arr.copy()
    length = len(arr)
    
    # 找到所有1的位置
    ones_pos = np.where(arr == 1)[0]
    
    for pos in ones_pos:
        # 检查前后n个元素是否都是0
        start = max(0, pos - n)
        end = min(length, pos + n + 1)
        
        # 提取前后n个元素（不包括当前元素）
        surrounding = np.concatenate([arr[start:pos], arr[pos+1:end]])
        
        if 1 not in surrounding:
            arr[pos] = 0
    
    return arr

def convert_mano_to_smplx(gvhmr_smplx_params, hamer_mano_params, xdwpose_np, xdwpose_visible, norm_hand_pose):                                          # xdwpose_np:[N, 1, 134, 3]     xdwpose_visible: [N, 1, 134]

    # load normal hand poses
    # norm_hand_pose = np.load('./GVHMR/wilor/norm_hand.npy')

    video_length = hamer_mano_params["global_orient"].shape[0]
    
    M = np.diag([-1, 1, 1])                                                                                                             # Preparing for the left hand switch

    head_id_list = [i for i in range(24,92)]+[0, 14, 15, 16, 17]
    head_np = xdwpose_np[0][0][head_id_list, :2]
    x_min, y_min = np.min(head_np, axis=0)
    x_max, y_max = np.max(head_np, axis=0)
    head_size = np.linalg.norm(np.array([x_max, y_max])-np.array([x_min, y_min]))

    
    left_vis = hamer_mano_params['left_vis']
    right_vis = hamer_mano_params['right_vis']
    left_vis = isolate_ones_np_optimized(left_vis, 2)
    right_vis = isolate_ones_np_optimized(right_vis, 2)

    # iou_hand = hamer_mano_params['iou_hand']

    # left_vis = [True if x>=0.85 else False for x in xdwpose_visible[:, 0, 92]]
    # right_vis = [True if x>=0.85 else False for x in xdwpose_visible[:, 0, 113]]

    # left_right_conf_diff = xdwpose_np[:, 0, [94, 97, 101, 105, 109], 2].mean(axis=1) - xdwpose_np[:, 0, [115, 118, 122, 126, 130], 2].mean(axis=1)

    # left_conf_finger = [True if x >=5.5 else False for x in xdwpose_np[:, 0, [94, 97, 101, 105, 109], 2].mean(axis=1)]
    # right_conf_finger = [True if x >=5.5 else False for x in xdwpose_np[:, 0, [115, 118, 122, 126, 130], 2].mean(axis=1)]

    # left_conf_wrist = [True if x>=5.5 else False for x in xdwpose_np[:, 0, 92, 2]]
    # right_conf_wrist = [True if x>=5.5 else False for x in xdwpose_np[:, 0, 113, 2]]

    # left_vis = left_vis and left_conf_finger and left_conf_wrist
    # right_vis = right_vis and right_conf_finger and right_conf_wrist


    # for i in range(1, video_length, 2):
    #     left_vis[i] = False
    #     right_vis[i] = False
    
    # Assuming that your data are stored in gvhmr_smplx_params and hamer_mano_params
    
    gvhmr_smplx_params["left_hand_pose"] = torch.ones([video_length, 45])
    gvhmr_smplx_params["right_hand_pose"] = torch.ones([video_length, 45])
    
    all_local_wrist_orient = np.ones([video_length, 2, 3, 3])
    all_local_hand_pose = np.ones([video_length, 2, 15, 3, 3])
    all_local_hand_pose[:, 1, ...] = hamer_mano_params["hand_pose"][:, 1, ...]

    for frame_id in range(video_length):
        full_body_pose = torch.concatenate((gvhmr_smplx_params["global_orient"][frame_id].unsqueeze(0), gvhmr_smplx_params["body_pose"][frame_id].reshape(21, 3)), dim=0)     # gvhmr_smplx_params["global_orient"]: (3, 3)
        
        if left_vis[frame_id]:
            left_elbow_global_rot = compute_global_rotation(full_body_pose, 18) # left elbow IDX: 18
            left_wrist_global_rot = hamer_mano_params["global_orient"][frame_id][0]                                                     # hamer_mano_params["global_orient"]: (2, 3, 3)
            left_wrist_global_rot = M @ left_wrist_global_rot @ M                                                                                # mirror switch
            left_wrist_pose = np.linalg.inv(left_elbow_global_rot) @ left_wrist_global_rot        
            all_local_wrist_orient[frame_id, 0] = left_wrist_pose  
            for i in range(15):
                left_finger_pose = M @ hamer_mano_params["hand_pose"][frame_id][0][i] @ M                                                    # hamer_mano_params["hand_pose"]: (2, 15, 3, 3)
                all_local_hand_pose[frame_id, 0, i] = left_finger_pose      
        
        if right_vis[frame_id]:
            right_elbow_global_rot = compute_global_rotation(full_body_pose, 19) # left elbow IDX: 19
            right_wrist_global_rot = hamer_mano_params["global_orient"][frame_id][1]
            right_wrist_pose = np.linalg.inv(right_elbow_global_rot) @ right_wrist_global_rot
            all_local_wrist_orient[frame_id, 1] = right_wrist_pose

    
    # 如果首尾帧不可见 无法插值 用平均手
    if not left_vis[0]:
        all_local_wrist_orient[0, 0] = np.eye(3, 3)
        # all_local_hand_pose[0, 0] = np.tile(np.eye(3, 3), (15, 1, 1))
        all_local_hand_pose[0, 0] = norm_hand_pose[0]
    if not right_vis[0]:
        all_local_wrist_orient[0, 1] = np.eye(3, 3)
        all_local_hand_pose[0, 1] = norm_hand_pose[1]
    if not left_vis[-1]:
        all_local_wrist_orient[-1, 0] = np.eye(3, 3)
        all_local_hand_pose[-1, 0] = norm_hand_pose[0]
    if not right_vis[-1]:
        all_local_wrist_orient[-1, 1] = np.eye(3, 3)
        all_local_hand_pose[-1, 1] = norm_hand_pose[1]

    # 保证首尾帧可见
    left_vis[0], right_vis[0] = True, True
    left_vis[-1], right_vis[-1] = True, True
    
    # 如果连续很多帧不可见 中间的帧用平均手
    for frame_id in range(video_length):
        # if iou_hand[frame_id]>0.5 and left_right_conf_diff[frame_id]<-1:
        #     all_local_wrist_orient[frame_id, 0] = np.eye(3, 3)
        #     all_local_hand_pose[frame_id, 0] = norm_hand_pose[0]
        #     left_vis[frame_id] = True
        # if iou_hand[frame_id]>0.5 and left_right_conf_diff[frame_id]>1:
        #     all_local_wrist_orient[frame_id, 1] = np.eye(3, 3)
        #     all_local_hand_pose[frame_id, 1] = norm_hand_pose[1]
        #     right_vis[frame_id] = True
        
        i_before, i_after = find_nearest_true(left_vis, frame_id)
        if frame_id-i_before>=4 and i_after-frame_id>=4:
            all_local_wrist_orient[frame_id, 0] = np.eye(3, 3)
            all_local_hand_pose[frame_id, 0] = norm_hand_pose[0]
            left_vis[frame_id] = True
        
        i_before, i_after = find_nearest_true(right_vis, frame_id)
        if frame_id-i_before>=4 and i_after-frame_id>=4:
            all_local_wrist_orient[frame_id, 1] = np.eye(3, 3)
            all_local_hand_pose[frame_id, 1] = norm_hand_pose[1]
            right_vis[frame_id] = True
    
    # 对抽帧过的和看不见的手进行插值
    for frame_id in range(1, video_length):
        if not left_vis[frame_id]:
            i_before, i_after = find_nearest_true(left_vis, frame_id)
            all_local_wrist_orient[frame_id, 0] = slerp_matrix(all_local_wrist_orient[i_before, 0], all_local_wrist_orient[i_after, 0], i_before, i_after, frame_id)
            for i in range(15):
                all_local_hand_pose[frame_id, 0, i] = slerp_matrix(all_local_hand_pose[i_before, 0, i], all_local_hand_pose[i_after, 0, i], i_before, i_after, frame_id)

        if not right_vis[frame_id]:
            i_before, i_after = find_nearest_true(right_vis, frame_id)
            all_local_wrist_orient[frame_id, 1] = slerp_matrix(all_local_wrist_orient[i_before, 1], all_local_wrist_orient[i_after, 1], i_before, i_after, frame_id)
            for i in range(15):
                all_local_hand_pose[frame_id, 1, i] = slerp_matrix(all_local_hand_pose[i_before, 1, i], all_local_hand_pose[i_after, 1, i], i_before, i_after, frame_id)
            
            

    # 滤波
    filter_list_global = [OneEuroFilterQuaternion(), OneEuroFilterQuaternion()]
    filter_list_hand = [[OneEuroFilterQuaternion() for i in range(15)], [OneEuroFilterQuaternion() for i in range(15)]]
    for frame_id in range(video_length):
        # start_frame = i-1 if i-1>=0 else 0
        # end_frame = i+1 if i+1<=video_length-1 else video_length-1
        # print(xdwpose_visible.shape)
        # beta_list = [vis2beta(xdwpose_visible[start_frame:end_frame, 0, 92]),vis2beta(xdwpose_visible[start_frame:end_frame, 0, 113])]
        left_global_smooth = filter_wilor(filter_list_global[0], all_local_wrist_orient[frame_id, 0], frame_id, 0.0)
        right_global_smooth = filter_wilor(filter_list_global[1], all_local_wrist_orient[frame_id, 1], frame_id, 0.0)
        all_local_wrist_orient[frame_id,0] = left_global_smooth[np.newaxis, ...]
        all_local_wrist_orient[frame_id,1] = right_global_smooth[np.newaxis, ...]
        for i in range(15):
            if i<=2:
                joint_idx = i+1
            elif i<=5:
                joint_idx = i+2
            elif i<=8:
                joint_idx = i+3
            elif i<=11:
                joint_idx = i+4
            else:
                joint_idx = i+5
            
            left_diff = np.linalg.norm(xdwpose_np[frame_id, 0, 92+joint_idx, :2] - xdwpose_np[frame_id-1, 0, 92+joint_idx, :2])/head_size
            right_diff = np.linalg.norm(xdwpose_np[frame_id, 0, 113+joint_idx, :2] - xdwpose_np[frame_id-1, 0, 113+joint_idx, :2])/head_size

            all_local_hand_pose[frame_id, 0, i] = filter_wilor(filter_list_hand[0][i], all_local_hand_pose[frame_id, 0, i], frame_id, diff2beta(left_diff))
            all_local_hand_pose[frame_id, 1, i] = filter_wilor(filter_list_hand[1][i], all_local_hand_pose[frame_id, 1, i], frame_id, diff2beta(right_diff))

    for frame_id in range(video_length):

        # 给local wrist pose 赋值
        left_wrist_pose_vec = R.from_matrix(all_local_wrist_orient[frame_id, 0, ...]).as_rotvec()
        right_wrist_pose_vec = R.from_matrix(all_local_wrist_orient[frame_id, 1, ...]).as_rotvec()
        gvhmr_smplx_params["body_pose"][frame_id][57: 60] = torch.tensor(left_wrist_pose_vec)
        gvhmr_smplx_params["body_pose"][frame_id][60: 63] = torch.tensor(right_wrist_pose_vec)
        
        # 给local finger pose 赋值
        left_hand_pose = np.ones(45)
        right_hand_pose = np.ones(45)
        for i in range(15):
            left_finger_pose = all_local_hand_pose[frame_id, 0, i]
            right_finger_pose = all_local_hand_pose[frame_id, 1, i]

            left_finger_pose_vec = R.from_matrix(left_finger_pose).as_rotvec()
            left_hand_pose[i*3: i*3+3] = left_finger_pose_vec
            right_finger_pose_vec = R.from_matrix(right_finger_pose).as_rotvec()
            right_hand_pose[i*3: i*3+3] = right_finger_pose_vec
        gvhmr_smplx_params["left_hand_pose"][frame_id] = torch.tensor(left_hand_pose)
        gvhmr_smplx_params["right_hand_pose"][frame_id] = torch.tensor(right_hand_pose)

    return gvhmr_smplx_params