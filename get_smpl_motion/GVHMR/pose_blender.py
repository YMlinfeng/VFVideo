import os
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import math
import yaml
from easydict import EasyDict as edict
import re
import torch.nn.functional as F
import sys


import numpy as np
import os
import random


def localize_motion(motion: np.ndarray, local_hand_face = False, local_body = True, hw = 1.0, for_train = True, zeroNoConf = True) -> np.ndarray:
    """
    把全局骨骼 (J,3,T) → 局部骨骼 (J,3,T)，并将 hip 行替换为 hip_xy,10。
    """
    if motion.ndim != 3 or motion.shape[1] != 3:
        raise ValueError("motion 必须是 (J,3,T)")

    _, _, T = motion.shape

    if for_train:
        # 让关节随机缺失
        if random.random() < 0.2:
            threshold = random.uniform(5.0, 5.5)
        else:
            threshold = 5.0
        # 补偿宽高比
        motion[:, 1, :] *= hw
    else:
        threshold = 5.0

    if local_body:
        mid_joint = motion[[8, 11]].mean(axis=0, keepdims=True) # 1,3,T
        hip_xy = mid_joint[0, :2, :].copy()          # (2,T)
    
    local = motion.copy()

    if local_body:
        local[:, :2, :] -= hip_xy

    if local_hand_face:
        head_ids = [0, 14, 15, 16, 17]
        body_ids = [1, 2,3,4,5,6,7,8,9,10,11,12,13,18,19]

        local[20:41, :2, :] -= local[7:8, :2, :].copy() #减左手
        local[41:62, :2, :] -= local[4:5, :2, :].copy() #减右手
        local[head_ids, :2, :] -= local[1:2, :2, :].copy() #减脖子

        local[body_ids, :2, :] = (local[body_ids, :2, :] - np.array([[0.5], [0.885]])) * 3.0
        local[head_ids, :2, :] = local[head_ids, :2, :] * 5.0
        local[20:62, :2, :] = local[20:62, :2, :] * 8.0
    else:
        local[:, :2, :] = (local[:, :2, :] - np.array([[0.5], [0.885]])) * 3.0

    # --- 置信度处理 ---
    left_bad   = local[7, 2, :]  <= threshold            # (T,)
    right_bad  = local[4, 2, :]  <= threshold            # (T,)
    neck_bad   = local[1, 2, :]  <= threshold            # (T,)

    # 帧级广播：None 在第 0 维展开，得到 (n_joint, T)
    local[20:41,            2, :] *= (~left_bad)[None, :]
    local[41:62,            2, :] *= (~right_bad)[None, :]
    local[[0,14,15,16,17],  2, :] *= (~neck_bad)[None, :]
    
    if local_body:
        translate = np.concatenate(
            [hip_xy, np.ones((1, T), dtype=motion.dtype) * 10],
            axis=0
        ).reshape(1, 3, T)                                # (1,3,T)
        translate[:, :2, :] -= 0.5                 # 仅改 x、y，不改 z
        local = np.concatenate([local, translate], axis=0)

    conf = (local[:, 2:3] > threshold).astype(np.float32)   # (J,1,F)
    
    if zeroNoConf:
        local[:, :2] *= conf                    # 零填充
    local[:, 2:3] = conf

    return local


def globalize_motion(local: np.ndarray, local_hand_face = False, local_body = True, hw = 1.0) -> np.ndarray:
    if local.ndim != 3:
        raise ValueError("local 必须是 (J,3,T)")
    J_local, _, T = local.shape

    if local_body:
        translate = local[-1, :2, :].copy()                   # (2,T)，取最后一个是translate
        translate[:, :] += 0.5
        joints   = local[:-1, :2, :].copy()           # 只取 xy，形状 (J-1,2,T)
        joints += translate
        joints = np.concatenate([joints[:], translate.reshape(1, 2, T)], axis=0)
    else:
        joints   = local[:, :2, :].copy()           # 只取 xy，形状 (J-1,2,T)

    # 3) 其它关节局部 xy → 全局
    if local_hand_face:
        head_ids = [0, 14, 15, 16, 17]
        body_ids = [1, 2,3,4,5,6,7,8,9,10,11,12,13,18,19]

        joints[body_ids, :2, :] = joints[body_ids, :2, :] / 3.0 + np.array([[0.5], [0.885]])
        joints[head_ids, :2, :] = joints[head_ids, :2, :] / 5.0
        joints[20:62, :2, :] = joints[20:62, :2, :] / 8.0

        joints[20:41, :2, :] += joints[7, :2, :]
        joints[41:62, :2, :] += joints[4, :2, :]
        joints[head_ids, :2, :] += joints[1, :2, :]
    else:
        joints[:, :2, :] = joints[:, :2, :] / 3.0 + np.array([[0.5], [0.885]])

    # global_xyz = np.concatenate([joints, local[:, 2:3, :] * 10.0 + 5.0], axis=1)  # (J, 3, F)
    
    global_xyz = np.concatenate([joints, local[:, 2:3, :] * 10.0], axis=1)  # (J, 3, F)
    global_xyz[:, 1, :] /= hw
    return global_xyz

def conf_check(keypoints_info, cut_down_offset = 0, hw = 1.0):
    margin = 0.2
    xs, ys = keypoints_info[:, 0, :], keypoints_info[:, 1, :]         # (J, F)
    mask = (
        (xs < -margin) | (xs > 1 + margin) |
        (ys < -margin * hw) | (ys > (1 - cut_down_offset + margin) * hw)
    )
    keypoints_info[:, 2, :][mask] = 0.0
    
    
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """Sinusoidal positions with **dynamic length expansion**.

    The table grows automatically if the incoming sequence is longer than
    the current buffer, preventing size‑mismatch errors when token length
    (F·J) exceeds the initial `max_len`.
    """
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.register_buffer('pe', self._build_pe(max_len), persistent=False)

    # ---------------------------------------------------------------------
    def _build_pe(self, length: int) -> torch.Tensor:
        pe = torch.zeros(length, self.d_model)
        pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) *
                        (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe  # (L,D)

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x:(B,L,D)
        L = x.size(1)
        if L > self.pe.size(0):
            # extend the positional table
            new_pe = self._build_pe(L).to(x.device, dtype=x.dtype)
            self.pe = new_pe  # replace buffer (non‑persistent)
        return x + self.pe[:L].unsqueeze(0)

# ---------------------------------------------------------------------------
# 2. Joint‑ID Embedding (learnable)
# ---------------------------------------------------------------------------
class JointPosEnc(nn.Module):
    def __init__(self, njoints: int, d_model: int):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(njoints, d_model) * 0.02)

    def forward(self, B: int, F: int) -> torch.Tensor:  # (B,F*J,D)
        j = self.emb.unsqueeze(0).unsqueeze(0)          # (1,1,J,D)
        j = j.expand(B, F, -1, -1)                     # (B,F,J,D)
        return j.reshape(B, F * self.emb.size(0), -1)

# ---------------------------------------------------------------------------
# 3. Building Blocks
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.SiLU(),
            nn.Linear(dim_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)

class AdaLayerNorm(nn.Module):
    """Adaptive LayerNorm (γ, β from cond vec)."""
    def __init__(self, d_model: int, cond_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias   = nn.Parameter(torch.zeros(d_model))
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * d_model)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        B, L, D = x.shape
        mean = x.mean(-1, keepdim=True)
        var  = (x - mean).pow(2).mean(-1, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        gamma, beta = self.mlp(cond).view(B, 1, D, 2).unbind(-1)
        y = (1 + gamma) * x_hat + beta
        return y * self.weight + self.bias

class AttnBlock(nn.Module):
    """Self‑attention + LN + FF (no conditioning)."""
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.ln1  = nn.LayerNorm(d_model)
        self.ff   = FeedForward(d_model, dim_ff)
        self.ln2  = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(x, x, x, need_weights=False)[0]
        x = self.ln1(x)
        x = x + self.ff(x)
        x = self.ln2(x)
        return x

class CrossAttnBlockAdaLN(nn.Module):
    """Self‑Attn ➜ Cross‑Attn (with *cond* k/v) + AdaLN‑wrapped FF."""
    def __init__(self, d_model: int, nhead: int, dim_ff: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.adaln1 = AdaLayerNorm(d_model, cond_dim)
        self.adaln2 = AdaLayerNorm(d_model, cond_dim)
        self.adaln3 = AdaLayerNorm(d_model, cond_dim)
        self.ff = FeedForward(d_model, dim_ff)

    def forward(self, x: torch.Tensor, mem: torch.Tensor, cond_vec: torch.Tensor):
        # 1. Self‑attn
        x = x + self.self_attn(x, x, x, need_weights=False)[0]
        x = self.adaln1(x, cond_vec)
        # 2. Cross‑attn (query = x, key/value = mem)
        x = x + self.cross_attn(x, mem, mem, need_weights=False)[0]
        x = self.adaln2(x, cond_vec)
        # 3. FFN
        x = x + self.ff(x)
        x = self.adaln3(x, cond_vec)
        return x

# ---------------------------------------------------------------------------
# 4. VelocityTransformer (spatial → full‑cross)
# ---------------------------------------------------------------------------
class VelocityTransformer(nn.Module):
    """Two‑stage attention network (cond via cross‑attention)."""
    def __init__(self, config):
        super().__init__()

        J  = config.njoints
        inner = config.d_model
        heads = config.nhead
        self.J = J

        # linear projections
        self.in_proj   = nn.Linear(3, inner)
        self.cond_proj = nn.Linear(3, inner)

        # embeddings
        self.joint_pos = JointPosEnc(J, inner)
        self.time_pos  = PositionalEncoding(inner)

        # ── Stage 1: spatial (per‑frame)
        self.spatial_blocks = nn.ModuleList([
            AttnBlock(inner, heads, dim_ff=4*inner) for _ in range(config.n_spatial_layers)
        ])

        # ── Stage 2: full spatio‑temporal with cross‑attention
        cond_dim = config.time_dim + config.mode_dim
        self.full_blocks = nn.ModuleList([
            CrossAttnBlockAdaLN(inner, heads, dim_ff=4*inner, cond_dim=cond_dim)
            for _ in range(config.nlayers)
        ])

        # output projection
        self.out_proj = nn.Linear(inner, 3)

        # time / mode embeddings
        self.time_mlp = nn.Sequential(nn.Linear(1, config.time_dim), nn.SiLU(), nn.Linear(config.time_dim, config.time_dim))
        self.hw_proj  = nn.Sequential(nn.Linear(1, config.mode_dim), nn.SiLU(), nn.Linear(config.mode_dim, config.mode_dim))

        print("VelocityTransformer (spatial→full with cross‑attn has conf) initialised")

    # -----------------------------------------------------------------
    def _spatial_stage(self, tokens: torch.Tensor) -> torch.Tensor:
        B, F, J, D = tokens.shape
        x = tokens.reshape(B*F, J, D)
        for blk in self.spatial_blocks:
            x = blk(x)
        return x.reshape(B, F, J, D)

    # -----------------------------------------------------------------
    def forward(self, x_t_flat: torch.Tensor, cond_flat: torch.Tensor, t: torch.Tensor, hw: torch.Tensor):
        B, F, _ = x_t_flat.shape
        _, CF, _ = cond_flat.shape
        J = self.J

        # 1. Tokenise & spatial stage
        x = self.in_proj(x_t_flat.view(B, F, J, 3))
        c = self.cond_proj(cond_flat.view(B, CF, J, 3))
        x = self._spatial_stage(x)
        c = self._spatial_stage(c)

        # 2. Flatten & add positional encodings --------------------
        x_tok = x.reshape(B, F*J, -1)
        c_tok = c.reshape(B, CF*J, -1)

        # (a) Joint‑ID embedding
        x_tok = x_tok + self.joint_pos(B, F)
        c_tok = c_tok + self.joint_pos(B, CF)

        # (b) **Time positional encoding – only along frame axis**
        #     We repeat each frame's sinusoid for all joints, avoiding the
        #     misconception of a unique position per (frame,joint) pair.

        t_gt  = torch.arange(1, F+1,  device=x.device)             # (25,)
        t_cf  = torch.cat([torch.tensor([0], device=x.device), torch.arange(F+1, F+CF, device=x.device)])  # (1+75,)
        time_ids_x = t_gt.repeat_interleave(J)   # (F·J,)
        time_ids_c = t_cf.repeat_interleave(J)   # (CF·J,)

        # time_ids_x = torch.arange(F, device=x.device).repeat_interleave(J)   # len = F·J
        # time_ids_c = torch.arange(CF, device=x.device).repeat_interleave(J)  # len = (F+1)·J
        x_tok = x_tok + self.time_pos.pe[time_ids_x][None]   # broadcast to batch
        c_tok = c_tok + self.time_pos.pe[time_ids_c][None]

        # 3. AdaLN conditioning vector -------------------------------------- AdaLN conditioning vector
        cond_vec = torch.cat([
            self.time_mlp(t.unsqueeze(-1)),
            self.hw_proj(hw.unsqueeze(-1))
        ], dim=-1)

        # 4. Full stage with cross-attention
        for blk in self.full_blocks:
            x_tok = blk(x_tok, c_tok, cond_vec)

        # 5. Project to velocities
        vel = self.out_proj(x_tok.reshape(B, F, J, -1))   # (B,F,J,2)
        return vel.reshape(B, F, J*3)




def bodypose_24to20(candidate, subset, threshhold=0.3):

    candidate[:,18] = np.mean(candidate[:, [18, 19]], axis=1)
    candidate[:,19] = np.mean(candidate[:, [21, 22]], axis=1)

    subset[:,18] = 10 * np.logical_and(
        subset[:, 18] > threshhold,
        subset[:, 19] > threshhold).astype(int)
    subset[:,19] = 10 * np.logical_and(
        subset[:, 21] > threshhold,
        subset[:, 22] > threshhold).astype(int)

    return candidate, subset


def get_config(config_path):
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)
    config = edict(config)
    _, config_filename = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_filename)
    config.name = config_name
    return config

    
def align_image_pose(video_dwpose,
                     image_dwpose,
                     hw, 
                     threshold: float = 5.0,
                     body_indices = (1, 2, 5, 8, 11)):
    # ---------- 1. 置信度筛选 ----------
    score_v = video_dwpose[0, body_indices, 2, 0]  # (N,)
    score_i = image_dwpose[0, body_indices, 2, 0]  # (N,)
    mask    = (score_v > threshold) & (score_i > threshold)
    if not mask.any():                 # 没有共同高置信度 → 原样返回
        return image_dwpose.copy()
    idx = np.asarray(body_indices)[mask]            # (M,)

    # ---------- 2. 取首帧 xy ----------
    pts_v = video_dwpose[0, idx, :2, 0]             # (M,2)
    pts_i = image_dwpose[0, idx, :2, 0]             # (M,2)

    # ---------- 3. uniform scale ----------
    bbox_v = np.ptp(pts_v, axis=0)                  # Δx, Δy
    bbox_i = np.ptp(pts_i, axis=0)
    sx = bbox_i[0] / bbox_v[0] if bbox_v[0] > 0 else 1.0
    sy = bbox_i[1] / bbox_v[1] if bbox_v[1] > 0 else 1.0
    s  = sy if bbox_v[1] > bbox_v[0] else sx        # 取较大轴比例保持纵横比

    # ---------- 4. 锚点（joint 1, frame 0） ----------
    anchor_v = video_dwpose[:, [1], :2, 0:1]        # (1,1,2,1)
    anchor_i = image_dwpose[:, [1], :2, 0:1]        # (1,1,2,1)

    # ---------- 5. 仿射变换 ----------
    # aligned = video_dwpose.copy()
    # aligned[:, :, :2, :] = (aligned[:, :, :2, :] - anchor_v) * s + anchor_i

    aligned = image_dwpose.copy()
    aligned[:, :, :2, :] = (aligned[:, :, :2, :] - anchor_i) / s + anchor_v

    margin = 0.2
    xs, ys = aligned[0, :, 0, :], aligned[0, :, 1, :]         # (J, F)
    mask = (
        (xs < -margin) | (xs > 1 + margin) |
        (ys < -margin * hw) | (ys > (1 + margin) * hw)
    )
    conf = aligned[0, :, 2, :]       # (J, F)
    conf[mask] = 0.0
    aligned[0, :, 2, :] = conf
    
    return aligned


# -------- 2)  打包成 DWpose 原生布局  ---------------------------------------
def pred_to_video_dwpose(frame, pred_local, image_hw, local_hand_face, local_body, save_path=None):
    joints_global = globalize_motion(pred_local, local_hand_face, local_body, image_hw[0, 0]/image_hw[0, 1])          # (62,3,F)
    
    F = joints_global.shape[2]
    packed = np.zeros((F, 1, 134, 3), dtype=joints_global.dtype)
    keep_idx = np.r_[0:20, 92:134]
    packed[:, 0, keep_idx, :] = joints_global.transpose(2, 0, 1)

    if save_path:
        np.save(save_path, packed, allow_pickle=False)
    return packed[:frame]

def filter_ckpt(checkpoint):
    keys_list = list(checkpoint.keys())
    for key in keys_list:
        if 'orig_mod.' in key:
            deal_key = key.replace('_orig_mod.', '')
            checkpoint[deal_key] = checkpoint[key]
            del checkpoint[key]
    return checkpoint

class retargeter:
    def __init__(self, config_path, ckpt, device):
        self.device = device

        self.config = get_config(config_path)

        state = torch.load(ckpt, map_location='cpu')
        self.model  = VelocityTransformer(self.config)
        self.model.load_state_dict(filter_ckpt(state['model'])); 
        self.model.to(device)
        self.model.eval()
    
    def sample(self, seqA, firstB, seqB_frame, steps=5, hw = 1):
        """
        seqA : (B,J,3,F) , firstB : (B,J,3)
        返回预测 seqB_hat (B,J,3,F)
        """
        hw = torch.tensor([hw], device=seqA.device, dtype=torch.float32)

        b, j, c, f = seqA.shape

        # cond 预处理
        seqA_flat = seqA.permute(0, 3, 1, 2).contiguous().view(b, f, -1)
        
        firstB_flat = firstB.view(b, 1, -1)
        cond_flat = torch.cat([firstB_flat, seqA_flat], dim=1)

        # 初始 z1 ~ N(0,1)
        x = torch.randn(b, seqB_frame, j*3, device=seqA.device)

        # ODE  solver
        t_steps = torch.linspace(1., 0., steps+1, device=seqA.device)
        for s in range(steps):
            t = t_steps[s].expand(b)
            dt = t_steps[s+1] - t_steps[s]
            v1 = self.model(x, cond_flat, t, hw)            # ûθ
            v_conf = v1.view(b, seqB_frame, j, 3)[...,2]
            x = x + dt * v1                     # Euler
        
        # second sample
        x0 = x.clone()
        x = torch.randn(b, seqB_frame, j*3, device=seqA.device)
        start_t = 0.3
        x = x*start_t + x0*(1-start_t)

        # ODE  solver
        t_steps = torch.linspace(start_t, 0., steps+1, device=seqA.device)
        for s in range(steps):
            t = t_steps[s].expand(b)
            dt = t_steps[s+1] - t_steps[s]
            v1 = self.model(x, cond_flat, t, hw)            # ûθ
            x = x + dt * v1                     # Euler

        seqB_hat = x.view(b, seqB_frame, j, 3).permute(0, 2, 3, 1).contiguous()

        return seqB_hat                                   # (B,J,3,F)
        
    def pad_frame(self, video_pose, frame_num):
        T = video_pose.shape[0]
        pad_len   = frame_num - T
        if pad_len > 0:
            if T == 1:                               # 只有 1 帧时只能重复
                idx = np.zeros(pad_len, dtype=int)
            else:
                #   cycle = [T-2, T-3, …, 0, 1, …, T-2]   长度 2*(T-1)-1 = 2T-3
                cycle = np.concatenate([np.arange(T-2, -1, -1),   # 倒序部分
                                        np.arange(1, T-1)])       # 正序部分（去掉 0 和 T-1）
                reps = (pad_len + cycle.size - 1) // cycle.size
                idx  = np.tile(cycle, reps)[:pad_len]

            pad_frames = video_pose[idx, ...] 
            video_pose = np.concatenate([video_pose, pad_frames], axis=0).copy()
            return video_pose
            
    def inf(self, video_pose, image_pose, video_hw, image_hw, c24to20 = False):

        torch.manual_seed(666)
        total_frame = 75  # 根据模型来，不能改变
        video_pose_cp = video_pose.copy()
        video_pose = video_pose[25:, :, :, :]

        T = video_pose.shape[0]
        if T < total_frame:
            video_pose = self.pad_frame(video_pose, total_frame)
        else:
            video_pose = video_pose[0:total_frame, ...].copy()

        video_pose = video_pose.transpose(1, 2, 3, 0)
        image_pose = image_pose.transpose(1, 2, 3, 0)

        keep_idx = np.r_[0:20, 92:134]

        keypoints, scores = video_pose[:,:, :2, :], video_pose[:,:, 2,:]
        if c24to20:
            keypoints, scores = bodypose_24to20(keypoints, scores, 5)
        video_pose = np.concatenate((keypoints, scores[:,:, None,:]), axis=-2) # bj3f
        video_pose = video_pose[:, keep_idx, :, :]

        keypoints, scores = image_pose[:,:, :2, :], image_pose[:,:, 2,:]
        if c24to20:
            keypoints, scores = bodypose_24to20(keypoints, scores, 5)
        image_pose = np.concatenate((keypoints, scores[:,:, None,:]), axis=-2) # bj3f
        image_pose = image_pose[:, keep_idx, :, :]
        
        video_pose[:,:, 1, :] *= video_hw[0, 0]/video_hw[0, 1]
        image_pose[:,:, 1, :] *= image_hw[0, 0]/image_hw[0, 1]
        image_pose2 = align_image_pose(video_pose.copy(), image_pose.copy(), video_hw[0, 0]/video_hw[0, 1])
        conf_check(video_pose[0], 0, video_hw[0, 0]/video_hw[0, 1])
        conf_check(image_pose2[0], 0, video_hw[0, 0]/video_hw[0, 1])

        video_pose = torch.tensor(localize_motion(video_pose[0], self.config.data.local_hand_face, self.config.data.local_body, 1.0, False, True), dtype=torch.float32, device=self.device)[None]
        image_pose2 = torch.tensor(localize_motion(image_pose2[0], self.config.data.local_hand_face, self.config.data.local_body, 1.0, False, True), dtype=torch.float32, device=self.device)[None]

        with torch.no_grad():
            pred = self.sample(video_pose.clone(), image_pose2[..., 0].clone(), 25, 5, video_hw[0, 0]/video_hw[0, 1])
            
            if True:
                out = pred_to_video_dwpose(T, pred.clone().cpu()[0].numpy(), video_hw, self.config.data.local_hand_face, self.config.data.local_body,None)
                out = np.concatenate([out, video_pose_cp[25:,...]], axis=0)
                return out
            else:
                output = "/ytech_milm_ssd/lxq/transmomo2/new"
                if not os.path.exists(output):
                    os.makedirs(output)
                
                from lib.util.visualization import vis_merge3_3

                out = pred_to_video_dwpose(T, pred.clone().cpu()[0].numpy(), video_hw, self.config.data.local_hand_face, self.config.data.local_body,None)
                out = np.concatenate([out, video_pose_cp[25:,...]], axis=0)

                vis_merge3_4(video_pose.cpu()[0,:,:,:], image_pose2.cpu()[0,:,:,:], out, out, video_hw[0, 0]/video_hw[0, 1], self.config.data.local_hand_face, self.config.data.local_body, self.config.noblend, output + '/result_{0}.mp4'.format(index))



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default = "/ytech_milm_ssd/lxq/transmomo2/out/l12_h1f1_94w_top10_blend_single_100_fullattn_dmodel512_conf/config.yaml")
    parser.add_argument("--ckpt", type=str, default = "/ytech_milm_ssd/lxq/transmomo2/out/l12_h1f1_94w_top10_blend_single_100_fullattn_dmodel512_conf/checkpoints/epoch_16_step_60000.pt")
    
    # parser.add_argument("-c", "--config", type=str, default = "/ytech_milm_ssd/lxq/transmomo2/out/l12_h1f1_94w_80w_closs1_conf1_ghf_gb_top10_noblend_pair_adaln_slayer2_addmode/config.yaml")
    # parser.add_argument("--ckpt", type=str, default = "/ytech_milm_ssd/lxq/transmomo2/out/l12_h1f1_94w_80w_closs1_conf1_ghf_gb_top10_noblend_pair_adaln_slayer2_addmode/checkpoints/epoch_258_step_1920000.pt")
    opts = parser.parse_args()
    return opts

if __name__ == "__main__":

    opts = parse_args()

    torch.manual_seed(666)
    
    retarget = retargeter(opts.config, opts.ckpt, 'cpu')

    video_pose = np.load('/vh_data/lxq/video_pose_0618_id14_tsl.npy')
    image_pose = np.load('/vh_data/lxq/image_pose_0618_id14_tsl.npy')
    video_hw = np.load('/vh_data/lxq/video_hw_0618_id14_tsl.npy')
    image_hw = np.load('/vh_data/lxq/image_hw_0618_id14_tsl.npy')
    
    # video_pose = np.load('/vh_data/lxq/video_pose.npy') # (121, 1, 134, 3) F B J 3
    # video_hw = np.load('/vh_data/lxq/video_hw.npy') # (1, 2)
    # image_pose = np.load('/vh_data/lxq/image_pose.npy') # (1, 1, 134, 3)
    # image_hw = np.load('/vh_data/lxq/image_hw.npy') # (1, 2)

    retarget.inf(video_pose, image_pose, image_hw, image_hw, False)
    exit(0)

    root_path = '/ytech_milm/tangsonglin05/Keling_HumanMotion/exps/tmp/m2v-diffusers/outputs/0505_transition_mask_pose_debug_v4-12000-test_pose_model'
    items = [os.path.join(root_path, fname) for fname in os.listdir(root_path)]

    for i, item in enumerate(items):
        if os.path.exists(item + '/video_dwpose_array.npy') is False:
            continue

        match = re.search(r'/(\d{4})--', item)
        number = match.group(1)
        if number != '0025':
            continue

        video_pose = np.load(item + '/video_dwpose_array.npy') # (121, 1, 134, 3) F B J 3
        video_hw = np.load(item + '/video_h_w_array.npy') # (1, 2)
        image_pose = np.load(item + '/image_dwpose_array.npy') # (1, 1, 134, 3)
        image_hw = np.load(item + '/image_h_w_array.npy') # (1, 2)


        retarget.inf(video_pose, image_pose, video_hw, image_hw)
        print('done' + str(i))
        # break
        # if i > 10:
        #     break
        # vis_merge3_2_inf(image_pose, video_pose2, video_pose2, './test.mp4')