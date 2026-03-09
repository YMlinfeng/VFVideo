import numpy as np
import cv2
import os
import io
import imageio
import colorsys # 引入colorsys库用于生成颜色



import numpy as np

import numpy as np

# 假设 PoseKinematics 类定义同上...
import numpy as np
from numba import njit

import numpy as np
from numba import njit

import numpy as np
from numba import njit

@njit
def fix_hand_rigid_motion(keypoints, hand_indices, thr, decay, head_root_idx):
    """
    使用刚体运动一致性修复手部。
    
    逻辑：
    1. 统计当前帧手部可见点的平均位移 (Motion Vector) 和平均置信度 (Avg Score)。
    2. 如果手部部分可见：不可见点的位移 = 平均位移，置信度 = 平均置信度。
    3. 如果手部全不可见：视为遮挡，不补全坐标，仅衰减置信度。
    
    Args:
        keypoints: [N, K, 3] 单人的所有帧关键点数据 (x, y, score)
        hand_indices: [H] 手部关键点的索引数组
        thr: 可见阈值
        decay: 全遮挡时的置信度衰减系数
    """
    N, K, _ = keypoints.shape
    
    # 从第1帧开始遍历 (第0帧无法计算位移，跳过)
    for t in range(1, N):
        
        # --- 步骤 1: 统计可见信息 ---
        sum_dx = 0.0
        sum_dy = 0.0
        sum_score = 0.0
        vis_count = 0
        
        head_root_idx
        head_root_score = keypoints[t, head_root_idx, 2]
        if head_root_score<thr:
            keypoints[t, hand_indices, 2] = 0

        for idx in hand_indices:
            curr_score = keypoints[t, idx, 2]
            prev_score = keypoints[t-1, idx, 2]
            
            # 当前点可见，且上一帧该点也存在（这样才能算位移）
            if curr_score >= thr and prev_score > 0.01:
                dx = keypoints[t, idx, 0] - keypoints[t-1, idx, 0]
                dy = keypoints[t, idx, 1] - keypoints[t-1, idx, 1]
                
                sum_dx += dx
                sum_dy += dy
                sum_score += curr_score
                vis_count += 1
        
        # --- 步骤 2: 决策与补全 ---
        
        # 情况 A: 手部完全不可见 (全遮挡)
        if vis_count == 0:
            continue

        # 情况 B: 手部部分可见 (补全逻辑)
        avg_dx = sum_dx / vis_count
        avg_dy = sum_dy / vis_count
        # 核心修改：计算可见部分的平均置信度
        avg_vis_score = sum_score / vis_count 
        
        for idx in hand_indices:
            # 针对当前帧不可见的点进行补全
            if keypoints[t, idx, 2] < thr:
                # 只有当上一帧该点有效时，才能基于上一帧叠加位移
                if keypoints[t-1, idx, 2] > 0.01:
                    
                    # 1. 坐标补全：上一帧坐标 + 可见部分的平均位移
                    keypoints[t, idx, 0] = keypoints[t-1, idx, 0] + avg_dx
                    keypoints[t, idx, 1] = keypoints[t-1, idx, 1] + avg_dy
                    
                    # 2. 置信度更新：赋予可见部分的平均分
                    # (这里可以乘以一个系数如 0.9 表示它是推算的，但在你的需求中直接用平均值即可)
                    keypoints[t, idx, 2] = avg_vis_score

    return keypoints

def process_hands(dwpose_seq, thr=0.3, decay=0.8):
    """
    对 dwpose 序列进行手部优化。
    
    Args:
        dwpose_seq: [N, M, K, 3] (帧数, 人数, 关键点, xy+score)
        hand_config: 字典 {'left': [...], 'right': [...]}
        thr: 判断可见的阈值
        decay: 全遮挡时的分数衰减率
    """

    # ================= 使用示例 =================
    hand_config = {
        'left': list(range(92, 113)),
        'right': list(range(113, 134))
    }

    # 复制一份数据，避免修改原始输入（如果需要原地修改可去掉 .copy()）
    # 建议保留 copy 以防调试时数据混乱，生产环境如果内存紧张可去掉
    seq_data = dwpose_seq.copy()
    
    N, M, K, C = seq_data.shape
    
    # 转换为 int32 数组以适配 Numba
    left_hand_idx = np.array(hand_config['left'], dtype=np.int32)
    right_hand_idx = np.array(hand_config['right'], dtype=np.int32)
    
    for m in range(M):
        # 提取单人数据 [N, K, 3]
        # 注意：这里切片取出的是视图，但为了保险起见，我们在 numba 内部直接操作传入的数组
        person_kps = seq_data[:, m, :, :]
        
        # 处理左手
        fix_hand_rigid_motion(person_kps, left_hand_idx, thr, decay, head_root_idx=7)
        
        # 处理右手
        fix_hand_rigid_motion(person_kps, right_hand_idx, thr, decay, head_root_idx=4)
        
        # 将处理后的数据写回 (虽然 numpy 切片通常是视图，但这步显式赋值更安全)
        seq_data[:, m, :, :] = person_kps
        
    return seq_data


import numpy as np

def fix_dwpose_missing_joints(pose_data, visibility_threshold=0.05, decay_factor=0.95):
    """
    修复 DWPose 视频序列中的缺失关键点。
    
    Args:
        pose_data (np.ndarray): 形状为 [n, 1, 134, 3] 的数组。
                                n=帧数, 1=人数, 134=关键点数, 3=(x, y, score)。
        visibility_threshold (float): 判断关键点是否可见的置信度阈值。低于此值视为不可见。
        decay_factor (float): 补全时置信度的衰减系数，防止误差累积。
    
    Returns:
        np.ndarray: 修复后的 pose_data，形状不变。
    """
    # 复制一份数据，避免修改原数据
    fixed_pose = pose_data.copy()
    
    n_frames, n_people, n_joints, n_dims = fixed_pose.shape
    
    # 确保输入符合预期
    assert n_dims == 3, "最后一维必须是 3 (x, y, score)"
    assert n_people == 1, "目前代码逻辑仅针对单人场景优化 (n_people=1)"
    
    # 脖子点的索引 (DWPose/OpenPose Body_25 格式通常 1 号位是 Neck)
    NECK_INDEX = 1
    
    # 遍历每一帧 (从第二帧开始，因为第一帧没有前一帧参考)
    for f in range(1, n_frames):
        # 获取当前帧和上一帧的数据 (去除人数维度，变成 [134, 3])
        current_frame = fixed_pose[f, 0]
        prev_frame = fixed_pose[f-1, 0]
        
        # 获取当前帧和上一帧的脖子坐标 (x, y)
        # 注意：这里假设脖子是必可见的，直接取值
        curr_neck_pos = current_frame[NECK_INDEX, :2]
        prev_neck_pos = prev_frame[NECK_INDEX, :2]
        
        # 获取当前帧所有点的置信度
        current_scores = current_frame[:, 2]
        
        # 找出当前帧中不可见的关键点 (score <= threshold)
        # 返回的是布尔索引掩码
        invisible_mask = current_scores <= visibility_threshold
        
        # 还要确保上一帧对应的点是"有效"的（即上一帧该点置信度不能太低，否则补全没有意义）
        # 这里设定上一帧的置信度至少要大于 0 (或者也可以用 visibility_threshold * 0.5)
        prev_scores = prev_frame[:, 2]
        valid_prev_mask = prev_scores > 0
        
        # 最终需要修复的点的掩码：当前不可见 AND 上一帧有效
        points_to_fix_mask = invisible_mask & valid_prev_mask
        points_to_fix_mask[92:] = 0
        # 如果没有需要修复的点，跳过当前帧
        if not np.any(points_to_fix_mask):
            continue
            
        # --- 开始修复逻辑 ---
        
        # 1. 计算上一帧这些点相对于上一帧脖子的向量 (Relative Vector)
        # prev_frame[mask, :2] shape is (K, 2), prev_neck_pos shape is (2,)
        rel_vecs = prev_frame[points_to_fix_mask, :2] - prev_neck_pos
        
        # 2. 将相对向量应用到当前帧的脖子位置
        # predicted_pos = curr_neck + rel_vec
        predicted_pos = curr_neck_pos + rel_vecs
        
        # 3. 计算新的置信度：上一帧置信度 * 衰减系数
        new_scores = prev_frame[points_to_fix_mask, 2] * decay_factor
        
        # 4. 赋值回 current_frame (注意：current_frame 是 fixed_pose 的视图，修改它会修改 fixed_pose)
        # 更新坐标 (x, y)
        fixed_pose[f, 0, points_to_fix_mask, :2] = predicted_pos
        # 更新置信度
        fixed_pose[f, 0, points_to_fix_mask, 2] = new_scores

    return fixed_pose



def save_video_from_frames(video_array, output_path, fps=30):
    """
    保存视频函数 (基于 ImageIO)
    """
    frame_count = video_array.shape[0]
    video_stream = io.BytesIO()
    
    # ImageIO 期望 RGB 格式
    with imageio.get_writer(
        video_stream, 
        fps=fps, 
        format="mp4", 
        codec="libx264", 
        ffmpeg_params=["-crf", "18"], 
        pixelformat="yuv420p"
    ) as writer:
        for i in range(frame_count):
            writer.append_data(video_array[i])
            
    video_data = video_stream.getvalue()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(video_data)
    print(f"Video saved to: {output_path}")

def get_color(index, total):
    """根据索引生成唯一的颜色 (RGB格式)"""
    # 使用 HSV 空间：色相(Hue)根据索引变化，饱和度(S)和亮度(V)设为最大
    hue = index / total
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    # 转换到 0-255 范围
    return (int(r * 255), int(g * 255), int(b * 255))

def draw_debug_pose(pose_data, H=720, W=720):
    """
    调试模式绘制：只画点和编号
    Args:
        pose_data: (F, 1, 134, 3)
    """
    # 移除 person 维度 -> (F, 134, 3)
    if pose_data.ndim == 4:
        pose_data = pose_data[:, 0, :, :]
        
    frames, num_kpts, _ = pose_data.shape
    # 预先生成 134 种颜色，避免循环里重复计算
    colors = [get_color(i, num_kpts) for i in range(num_kpts)]
    
    # 初始化视频数组 (黑色背景)
    video_frames = np.zeros((frames, H, W, 3), dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for f in range(frames):
        # 当前帧画布
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        kpts = pose_data[f] # (134, 3)
        
        for i in range(num_kpts):
            if i>23 and i<92: continue
            x, y, score = kpts[i]
            
            # 过滤掉置信度太低的点 (避免噪点)
            if score < 5:
                continue
            
            # 反归一化坐标
            px, py = int(x * W), int(y * H)
            
            # 1. 画点 (绿色 RGB: 0, 255, 0)
            cv2.circle(canvas, (px, py), 3, colors[i], -1)
            
            # 2. 写编号 (白色 RGB: 255, 255, 255)
            # 字体大小 0.3，避免文字太挤
            cv2.putText(canvas, str(i), (px + 4, py), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
        video_frames[f] = canvas

    return video_frames



def smooth_dwpose_outliers(dwpose_data, iou_stable_thresh=0.4, score_th=5):

    iou_jitter_thresh=iou_stable_thresh/2,
    # 复制一份数据以免修改原始输入
    processed_pose = dwpose_data.copy()
    
    n_frames = processed_pose.shape[0]
    
    # 如果帧数少于3，无法进行前后帧比较，直接返回
    if n_frames < 3:
        return processed_pose

    # 定义用于计算BBox的关键点索引
    # 0: Nose, 1: Neck(if available in 134 format usually index 1 is neck or between shoulders)
    # 2: R_Shoulder, 5: L_Shoulder
    # 8: R_Hip, 11: L_Hip
    # 14: R_Eye, 15: L_Eye, 16: R_Ear, 17: L_Ear
    # 注意：DWPose 134点格式通常兼容COCO-Wholebody。
    # 身体部分是前18个点 (0-17)。
    # 我们选取头部和躯干，排除胳膊(3,4,6,7)和腿(9,10,12,13)
    torso_head_indices = [0, 1, 2, 5, 8, 11, 14, 15]
    
    def get_bbox(points):
        """计算一组点的 [x1, y1, x2, y2]"""
        if len(points) == 0:
            return None
        x = points[:, 0]
        y = points[:, 1]
        return [np.min(x), np.min(y), np.max(x), np.max(y)]

    def compute_iou(bbox1, bbox2):
        """计算两个bbox的IOU"""
        if bbox1 is None or bbox2 is None:
            return 0.0
            
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        xx1 = max(x1_1, x1_2)
        yy1 = max(y1_1, y1_2)
        xx2 = min(x2_1, x2_2)
        yy2 = min(y2_1, y2_2)

        w = max(0, xx2 - xx1)
        h = max(0, yy2 - yy1)
        inter_area = w * h

        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = area1 + area2 - inter_area
        
        if union_area <= 1e-6:
            return 0.0
            
        return inter_area / union_area

    # 遍历每一帧（跳过第一帧和最后一帧）
    for i in range(1, n_frames - 1):
        prev_idx = i - 1
        curr_idx = i
        next_idx = i + 1
        
        # 获取三帧的关键点数据 [134, 3]
        # 这里假设 batch 维度为 1，直接取 [0]
        pose_prev = processed_pose[prev_idx, 0]
        pose_curr = processed_pose[curr_idx, 0]
        pose_next = processed_pose[next_idx, 0]
        
        # 1. 找出三帧都存在的有效点 (score > 0)
        # 这是一个mask操作
        valid_prev = pose_prev[torso_head_indices, 2] > score_th
        valid_curr = pose_curr[torso_head_indices, 2] > score_th
        valid_next = pose_next[torso_head_indices, 2] > score_th
        
        # 取交集：只有三帧里都有效的点才用来算BBox
        common_valid_mask = valid_prev & valid_curr & valid_next
        
        # 如果没有公共有效点，无法判断，跳过
        if not np.any(common_valid_mask):
            continue
            
        # 提取这些公共点的索引
        valid_indices_in_subset = np.where(common_valid_mask)[0]
        # 映射回原始的 0-133 索引
        final_indices = [torso_head_indices[k] for k in valid_indices_in_subset]
        
        # 2. 计算三个 BBox
        bbox_prev = get_bbox(pose_prev[final_indices, :2])
        bbox_curr = get_bbox(pose_curr[final_indices, :2])
        bbox_next = get_bbox(pose_next[final_indices, :2])
        
        # 3. 计算 IOU
        iou_prev_next = compute_iou(bbox_prev, bbox_next)
        iou_curr_prev = compute_iou(bbox_curr, bbox_prev)
        iou_curr_next = compute_iou(bbox_curr, bbox_next)
        
        # 4. 判断逻辑
        # 如果前后帧很相似 (IOU > 0.7)
        # 但是中间帧和前一帧差别大 (IOU < 0.4) 或者 和后一帧差别大
        # 注意：通常跳变会导致和前后都不太一样，或者只偏离了一边。
        # 这里严格按照你的要求：中间帧和前一帧 OR 后一帧 IOU 小于 0.4
        is_stable_context = iou_prev_next > iou_stable_thresh
        is_jitter_frame = (iou_curr_prev < iou_jitter_thresh) or (iou_curr_next < iou_jitter_thresh)
        
        if is_stable_context and is_jitter_frame:
            print ('iou_prev_next, iou_curr_prev, iou_curr_next', iou_prev_next, iou_curr_prev, iou_curr_next)
            # 5. 插值修复
            # 使用前后帧的均值代替中间帧 (包括 x, y, score)
            # 注意：这里是对整个人体（所有134个点）进行插值，不仅仅是躯干
            avg_pose = (pose_prev + pose_next) / 2.0
            processed_pose[curr_idx, 0] = avg_pose
            
            # 打印日志方便调试（可选）
            # print(f"Fixed jitter at frame {i}: Prev-Next IOU={iou_prev_next:.3f}, Curr IOU={min(iou_curr_prev, iou_curr_next):.3f}")

    return processed_pose
    

def fix_hand_jitter(dwpose_data, score_thr=5):
    """
    修复 dwpose 手部飞点 (带全可见性保护)
    :param dwpose_data: shape [n, 1, 134, 3]
    :param score_thr: 置信度阈值 (如果你的原始数据阈值是5，请在此传入5)
    """
    data = dwpose_data.copy()
    n_frames = data.shape[0]
    
    # 关键点索引
    hands_idx = [list(range(92, 113)), list(range(113, 134))] # 0:Left, 1:Right
    
    # 预计算：面积(areas) 和 全可见性(vis_flags)
    # shape: [n_frames, 2] -> col 0: Left, col 1: Right
    areas = np.zeros((n_frames, 2))
    vis_flags = np.zeros((n_frames, 2), dtype=bool)

    for i in range(n_frames):
        kps = data[i, 0] # (134, 3)
        for hand_id, indices in enumerate(hands_idx):
            hand_pts = kps[indices]
            
            # 1. 检查该手部是否“所有点”可见
            # 只有所有关键点分数都 > score_thr 才标记为 True
            vis_flags[i, hand_id] = np.all(hand_pts[:, 2] > score_thr)
            
            # 2. 计算面积 (仅当有有效点时)
            valid_mask = hand_pts[:, 2] > score_thr
            if np.sum(valid_mask) > 3:
                pts = hand_pts[valid_mask, :2]
                wh = pts.max(axis=0) - pts.min(axis=0)
                areas[i, hand_id] = wh[0] * wh[1]

    frames_to_interp = set()

    # 遍历帧寻找异常
    for i in range(1, n_frames - 1):
        for hand_id in range(2): # 0:Left, 1:Right
            # --- 保护机制 ---
            # 只有当 前一帧 和 后一帧 的【这只手】所有点都可见时，才允许插值
            if not (vis_flags[i-1, hand_id] and vis_flags[i+1, hand_id]):
                continue

            p_a, c_a, n_a = areas[i-1, hand_id], areas[i, hand_id], areas[i+1, hand_id]
            
            # 避免除零
            if p_a < 1e-6 or n_a < 1e-6: continue

            # 逻辑1: 前后帧面积稳定 (差异 < 50%)
            if abs(p_a / n_a - 1) >= 0.5:
                continue
            
            # 逻辑2: 当前帧面积突变 (大于前或后 3倍以上)
            if (c_a / p_a - 1 > 2) or (c_a / n_a - 1 > 2):
                frames_to_interp.add(i)
                break # 只要有一只手异常，整帧标记，跳出当前帧的手部循环

    # 执行插值
    for i in sorted(list(frames_to_interp)):
        # 整体线性插值
        data[i] = (data[i-1] + data[i+1]) / 2
        print(f"Frame {i} interpolated (Protected).")

    return data



def filter_isolated_points(pose_data, score_thr=5):
    """
    剔除孤立的关键点：如果一个点可见，但它所有直接相连的点都不可见，则剔除该点。
    
    参数:
        pose_data (np.ndarray): DWPose 输出数据 [n, 1, 134, 3]。
        score_thr (float): 置信度阈值。
    
    返回:
        np.ndarray: 处理后的数据。
    """
    filtered_pose = pose_data.copy()
    num_people = filtered_pose.shape[0]
    
    # =========================================================================
    # 1. 定义骨架拓扑结构 (基于你提供的 OpenPose Body-25 图片)
    # =========================================================================
    # 格式：(点A, 点B) 表示 A 和 B 相连
    # 注意：这里只列出了身体躯干和面部的主干，手部(25以后)如果需要也可以加
    skeleton_pairs = [
        # --- 躯干 (Torso) ---
        (1, 2),   (1, 5),   # Neck -> Shoulders
        (1, 8),   (1, 11),  # Neck -> Hips (注意：OpenPose中脖子常作为中心)
        (0, 1),             # Nose -> Neck
        
        # --- 右臂 (Right Arm) ---
        (2, 3), (3, 4),     # Shoulder -> Elbow -> Wrist
        
        # --- 左臂 (Left Arm) ---
        (5, 6), (6, 7),     # Shoulder -> Elbow -> Wrist
        
        # --- 右腿 (Right Leg) ---
        (8, 9), (9, 10),    # Hip -> Knee -> Ankle
        (10, 19),           # Ankle -> BigToe (图片中的19)
        
        # --- 左腿 (Left Leg) ---
        (11, 12), (12, 13), # Hip -> Knee -> Ankle
        (13, 18),           # Ankle -> BigToe (图片中的18)
        
        # --- 面部 (Face) ---
        (0, 14), (14, 16),  # Nose -> R_Eye -> R_Ear
        (0, 15), (15, 17)   # Nose -> L_Eye -> L_Ear
    ]
    
    # 将连接对转换为邻接表 {point_idx: [neighbor_idx1, neighbor_idx2, ...]}
    adjacency = {}
    for p1, p2 in skeleton_pairs:
        # 确保 p1 在字典中
        if p1 not in adjacency: adjacency[p1] = []
        adjacency[p1].append(p2)
        
        # 确保 p2 在字典中 (无向图)
        if p2 not in adjacency: adjacency[p2] = []
        adjacency[p2].append(p1)

    # =========================================================================
    # 2. 执行过滤
    # =========================================================================
    for i in range(num_people):
        # 取出当前人的关键点 [134, 3]
        keypoints = filtered_pose[i, 0]
        
        # 记录需要剔除的点的索引
        points_to_remove = []
        
        # 遍历邻接表中定义的所有点
        for point_idx, neighbors in adjacency.items():
            # 边界检查：防止索引超出 134 范围 (虽然 Body-25 都在范围内)
            if point_idx >= keypoints.shape[0]:
                continue
                
            current_score = keypoints[point_idx, 2]
            
            # 只有当前点是“可见”的，才需要检查它是否孤立
            if current_score > score_thr:
                
                all_neighbors_invisible = True
                
                # 检查所有邻居
                for neighbor_idx in neighbors:
                    if neighbor_idx < keypoints.shape[0]:
                        neighbor_score = keypoints[neighbor_idx, 2]
                        # 只要有一个邻居可见，它就不是孤立的
                        if neighbor_score > score_thr:
                            all_neighbors_invisible = False
                            break
                
                # 如果循环结束，所有邻居都不可见，则标记为删除
                if all_neighbors_invisible:
                    points_to_remove.append(point_idx)
        
        # 统一执行剔除 (避免在循环中修改数据影响后续判断)
        for idx in points_to_remove:
            filtered_pose[i, 0, idx] = [0, 0, 0] # 坐标和置信度都归零
            # print(f"Person {i}: Removed Isolated Point {idx}")
            
    return filtered_pose



def filter_flying_knees(pose_data, img_h, img_w, score_thr=5):
    
    # 也就是复制一份数据，避免修改原数据
    filtered_pose = pose_data.copy()
    
    # 获取人数 n
    num_people = filtered_pose.shape[0]
    
    # 定义底部区域的阈值 (图片底部 20%)
    bottom_threshold = img_h * 0.8
    
    # 关键点索引
    # 右侧
    R_HIP = 8
    R_KNEE = 9
    R_ANKLE = 10
    
    # 左侧
    L_HIP = 11
    L_KNEE = 12
    L_ANKLE = 13
    
    for i in range(num_people):
        # 获取当前人的关键点数据，形状 [134, 3]
        # 这里假设第二个维度 1 是固定的，直接取索引 0
        keypoints = filtered_pose[i, 0] 
        
        # ---------------------------------------------------------
        # 规则 1: 处理右腿 (8, 9, 10)
        # ---------------------------------------------------------
        y_r_hip = keypoints[R_HIP, 1]
        score_r_hip = keypoints[R_HIP, 2]
        
        y_r_knee = keypoints[R_KNEE, 1]
        score_r_knee = keypoints[R_KNEE, 2]
        
        score_r_ankle = keypoints[R_ANKLE, 2]
        
        # 条件分解：
        # 1. Hip 8 接近图片底部 (y > img_h * 0.8) 且 Hip 自身是可见的(可选，但在边缘通常置信度还可以)
        is_hip_at_bottom = y_r_hip > bottom_threshold
        
        # 2. Knee 9 可见 (置信度 > 阈值)
        is_knee_visible = score_r_knee > score_thr
        
        # 3. Knee 9 比 Hip 8 高 (在图像坐标系中，高意味着 y 值更小)
        #    即: y_knee < y_hip
        is_knee_higher_than_hip = y_r_knee < y_r_hip
        
        # 4. Ankle 10 不可见 (置信度 < 阈值)
        is_ankle_invisible = score_r_ankle < score_thr
        
        if is_hip_at_bottom and is_knee_visible and is_knee_higher_than_hip and is_ankle_invisible:
            # 删除 9 (将置信度置为0，坐标通常也可以置为0以防误用)
            filtered_pose[i, 0, R_KNEE] = 0.0
            # print(f"Person {i}: Removed Right Knee (Flying Knee detected)")

        # ---------------------------------------------------------
        # 规则 2: 处理左腿 (11, 12, 13)
        # ---------------------------------------------------------
        y_l_hip = keypoints[L_HIP, 1]
        score_l_hip = keypoints[L_HIP, 2]
        
        y_l_knee = keypoints[L_KNEE, 1]
        score_l_knee = keypoints[L_KNEE, 2]
        
        score_l_ankle = keypoints[L_ANKLE, 2]
        
        # 1. Hip 11 接近图片底部
        is_l_hip_at_bottom = y_l_hip > bottom_threshold
        
        # 2. Knee 12 可见
        is_l_knee_visible = score_l_knee > score_thr
        
        # 3. Knee 12 比 Hip 11 高 (y_knee < y_hip)
        is_l_knee_higher_than_hip = y_l_knee < y_l_hip
        
        # 4. Ankle 13 不可见
        is_l_ankle_invisible = score_l_ankle < score_thr
        
        if is_l_hip_at_bottom and is_l_knee_visible and is_l_knee_higher_than_hip and is_l_ankle_invisible:
            # 删除 12
            filtered_pose[i, 0, L_KNEE] = 0.0
            # print(f"Person {i}: Removed Left Knee (Flying Knee detected)")
            
    return filtered_pose




from collections import deque

def smooth_big_face(data, img_w, img_h, window_size=5, conf_thresh=5, ratio_thresh=0.25):
    """
    1. 全局判断：计算整个视频中有效人脸的平均占比，判断是否为大脸视频。
    2. 条件平滑：如果是大脸视频，则对全序列进行平滑；否则返回原数据。
    
    参数:
    - data: 原始数据 [n, 1, 134, 3]
    - img_w, img_h: 图像宽高
    - window_size: 平滑窗口大小
    - conf_thresh: 置信度阈值 (要求所有点都大于此值)
    - ratio_thresh: 判定为大脸的平均占比阈值
    
    返回:
    - result_data: 处理后的数据 (平滑后 或 原始数据)
    - is_big_face_video: 布尔值，表示是否判定为大脸视频
    """
    n_frames = data.shape[0]
    
    # 人脸关键点索引 (24到92)
    FACE_START, FACE_END = 24, 92
    
    # ==========================================
    # 第一步：全局判断 (是否为大脸视频)
    # ==========================================
    valid_ratios = [] # 存储所有有效帧的人脸占比
    valid_idx = []
    for t in range(n_frames):
        # 取出单帧人脸数据 [69, 3]
        face_data = data[t, 0, FACE_START:FACE_END, :]
        coords = face_data[:, :2]
        scores = face_data[:, 2]
        
        # 只有当该帧所有人脸点置信度都达标，才纳入统计
        if np.all(scores > conf_thresh):
            valid_idx.append(t)
            # 计算外接框面积
            x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
            y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
            
            face_area = (x_max - x_min) * (y_max - y_min)
            img_area = min(img_w, img_h)**2
            
            if img_area > 0:
                valid_ratios.append(face_area / img_area)

    # 计算平均占比
    avg_ratio = 0.0
    if len(valid_ratios) > 0:
        avg_ratio = np.mean(valid_ratios)
    
    print ('-----------------------avg_ratio------------------------', avg_ratio)
    # 判定结果
    is_big_face_video = (avg_ratio > ratio_thresh)

    # ==========================================
    # 第二步：条件平滑
    # ==========================================
    if not is_big_face_video:
        # 如果不是大脸视频，不做处理，直接返回原始数据
        return data
    
    # 如果是大脸视频，进行窗口平滑
    smoothed_data_list = []
    history_buffer = deque(maxlen=window_size)
    
    for t in range(n_frames):
        curr_frame = data[t] # [1, 134, 3]
        
        # 加入队列
        if t in valid_idx:
            history_buffer.append(curr_frame)
        
        # 计算队列均值 (平滑坐标和置信度)
        # axis=0 表示在时间维度(队列长度)上取平均
        smoothed_frame = np.mean(history_buffer, axis=0)
        
        smoothed_data_list.append(smoothed_frame)
        
    return np.array(smoothed_data_list)

class DwposeRefine:    
    def __init__(self):
        """初始化"""
        pass
    def refine(self, dwpose, img_h, img_w):
        dwpose_ori = dwpose.copy()
        try:
            dwpose = process_hands(dwpose, thr=5, decay=0.95)
            dwpose = fix_dwpose_missing_joints(dwpose, visibility_threshold=5, decay_factor=0.99)
            dwpose = smooth_dwpose_outliers(dwpose)
            dwpose = fix_hand_jitter(dwpose)
            dwpose = smooth_big_face(dwpose, img_h, img_w)
            dwpose = filter_flying_knees(dwpose, img_h, img_w)
            dwpose = filter_isolated_points(dwpose)
            dwpose[0] = dwpose_ori[0]
            return dwpose
        except:
            return dwpose_ori

        
# dwpose_refine = DwposeRefine()


# # --- 主流程 ---

# list_file = '/ytech_m2v2_hdd/liujiwen/ID_Encoder/motion/dwpose_debug/list.txt'

# # 读取文件列表
# if os.path.exists(list_file):
#     lines = open(list_file).read().strip().split('\n')
# else:
#     print("List file not found.")
#     lines = []


# for l in lines:
#     dwpose_path = l.strip()
#     if not dwpose_path or not os.path.exists(dwpose_path):
#         continue

#     #try:
#     if True:
#         print(f"Processing: {dwpose_path}")
#         # 加载数据 (F, 1, 134, 3)
#         dwpose = np.load(dwpose_path)
#         dwpose[:,:,:,0] = dwpose[:,:,:,0]/np.max(dwpose[:,:,:,0])
#         dwpose[:,:,:,1] = dwpose[:,:,:,1]/np.max(dwpose[:,:,:,1])

#         dwpose = dwpose_refine.refine(dwpose, 1, 1)

#         # 绘制调试视频 (分辨率设大一点方便看字，例如 720x720)
#         rendered_video = draw_debug_pose(dwpose, H=1280, W=1280)
        
#         # 生成输出路径
#         output_filename = os.path.splitext(os.path.basename(dwpose_path))[0] + "_debug.mp4"
#         output_dir = os.path.dirname(dwpose_path)
#         output_path = os.path.join(output_dir, output_filename)
        
#         # 保存
#         save_video_from_frames(rendered_video, output_path, fps=30)
#         print (output_path)
        
#     # except Exception as e:
#     #     print(f"Error: {e}")


    