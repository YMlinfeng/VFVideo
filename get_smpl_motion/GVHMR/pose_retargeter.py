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


# ---------------------------------------------------------------------------
# 1. Positional Encoding (sinusoidal – same as Vaswani et al.)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1,L,D)

    def forward(self, x):           # x: (B,L,D)
        return x + self.pe[:, : x.size(1)]

# ---- 2. 关节级 Spatial Encoder ---------------------------------
class JointPosEnc(nn.Module):
    """可学习的关节 ID 向量 (J,d)。"""
    def __init__(self, njoints, d_model):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(njoints, d_model) * 0.02)

    def forward(self, x):           # (B,F,J,D)
        return x + self.emb[None, None]

class SpatialEncoder(nn.Module):
    def __init__(self, d_model=32, nhead=4, nlayers=2, njoints=62, out_c = 512):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4*d_model,
            batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.pos     = JointPosEnc(njoints, d_model)
        self.out_proj  = nn.Linear(d_model * njoints, out_c)

    def forward(self, x):           # x: (B,F,J,D)
        B,F,J,D = x.shape
        x = self.pos(x)
        
        x = x.reshape(-1, J, D)
        x = self.encoder(x)                         # (B·F,J,D)
        
        x = x.reshape(B, F, J * D)           # (B, F, J*D)
        x = self.out_proj(x) # JD -> 512
        return x

class AdaLayerNorm(nn.Module):
    """Adaptive LayerNorm a la StyleGAN / Stable‑Diffusion.

    γ, β are produced from a conditioning vector and applied as:
        out = (1 + γ) * norm(x) + β
    """

    def __init__(self, d_model: int, cond_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # base affine (can be disabled)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        # tiny MLP → (γ, β)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * d_model),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        gamma_beta = self.mlp(cond).view(B, 1, 2 * D)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        out = (1.0 + gamma) * x_hat + beta
        return out * self.weight + self.bias


class DecoderLayerAdaLN(nn.Module):
    """Transformer decoder layer with AdaLN conditioning."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        cond_dim: int,
        dim_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True, dropout=dropout
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=True, dropout=dropout
        )

        self.adaln1 = AdaLayerNorm(d_model, cond_dim)
        self.adaln2 = AdaLayerNorm(d_model, cond_dim)
        self.adaln3 = AdaLayerNorm(d_model, cond_dim)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )

    def forward(
        self,
        tgt: torch.Tensor,  # (B, L, D)
        memory: torch.Tensor,  # (B, S, D)
        cond: torch.Tensor,  # (B, C)
    ) -> torch.Tensor:
        # 1. self‑attention
        x = self.self_attn(tgt, tgt, tgt, need_weights=False)[0]
        tgt = tgt + x
        tgt = self.adaln1(tgt, cond)

        # 2. cross‑attention
        x = self.cross_attn(tgt, memory, memory, need_weights=False)[0]
        tgt = tgt + x
        tgt = self.adaln2(tgt, cond)

        # 3. feed‑forward
        x = self.ff(tgt)
        tgt = tgt + x
        tgt = self.adaln3(tgt, cond)

        return tgt


# ---------------------------------------------------------------------------
# 4. Full VelocityTransformer
# ---------------------------------------------------------------------------

class VelocityTransformer(nn.Module):
    """Velocity prediction network with AdaLN‑conditioned decoder.

    Args:
        x_t_flat  (B, F, J*3) – current keypoint tensor (xyc)
        cond_flat (B, F+1, J*3) – conditioning sequence (xyc)
        t         (B,) – diffusion / flow time step in [0, 1]
    Returns:
        u_hat_flat (B, F, J*3) – predicted velocity field
    """

    def __init__(
        self,
        config
    ) -> None:
        super().__init__()

        d_model = config.d_model
        nhead = config.nhead
        num_layers = config.nlayers
        njoints = config.njoints
        n_spatial_layers = config.n_spatial_layers
        spatial_d = config.spatial_d

        self.nj = njoints
        self.D = d_model

        # per‑joint linear projection to low‑dim spatial tokens
        self.in_proj = nn.Linear(2, spatial_d)
        self.cond_proj = nn.Linear(3, spatial_d)

        # joint‑spatial encoder → (B, F, d_model)
        self.spatial = SpatialEncoder(
            d_model=spatial_d,
            nhead=4,
            nlayers=n_spatial_layers,
            njoints=njoints,
            out_c=d_model,
        )

        self.mode_emb = nn.Embedding(2, config.mode_dim)
        # time embedding MLP (→ cond_dim = d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.hw_proj = nn.Sequential(
            nn.Linear(1, config.mode_dim),
            nn.SiLU(),
            nn.Linear(config.mode_dim, config.mode_dim),
        )

        # positional encoding (temporal axis)
        self.pos_time = PositionalEncoding(d_model)

        # stack of decoder layers with AdaLN
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayerAdaLN(
                    d_model=d_model,
                    nhead=nhead,
                    cond_dim=d_model + config.mode_dim * 2,
                    dim_ff=4 * d_model,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_proj = nn.Linear(d_model, njoints * 2)
        print('adaln model no conf')

    # ---------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------
    def forward(
        self,
        x_t_flat: torch.Tensor,
        cond_flat: torch.Tensor,
        t: torch.Tensor,
        mode_tag: torch.Tensor,
        hw: torch.Tensor,
        # forConf: bool,
    ) -> torch.Tensor:
        B, F, _ = x_t_flat.shape
        _, CF, _ = cond_flat.shape
        J = self.nj

        # 1. spatial encoding ------------------------------------------------
        x_j = x_t_flat.view(B, F, J, 2)
        c_j = cond_flat.view(B, CF, J, 3)

        q = self.in_proj(x_j)           # (B, F, J, spatial_d)
        k = self.cond_proj(c_j)         # (B, F+1, J, spatial_d)

        q = self.spatial(q)             # (B, F, D)
        k = self.spatial(k)             # (B, F+1, D)

        # 2. temporal + time embedding -------------------------------------
        q = self.pos_time(q)
        k = self.pos_time(k)

        # AdaLN cond vector
        t_emb = self.time_mlp(t.unsqueeze(-1))  # (B, D)
        m_emb   = self.mode_emb(mode_tag)                   # (B, mode_dim)
        hw_emb = self.hw_proj(hw.unsqueeze(-1)) # (B, D)
        cond_vec = torch.cat([t_emb, m_emb, hw_emb], dim=-1)        # (B, cond_dim)

        # 3. decoder stack ---------------------------------------------------
        x = q
        for layer in self.decoder_layers:
            x = layer(x, k, cond_vec)

        u_hat_flat = self.out_proj(x)           # (B, F, J*2)
        return u_hat_flat


class ConfTransformer(nn.Module):
    """Velocity prediction network with AdaLN‑conditioned decoder.

    Args:
        x_t_flat  (B, F, J*3) – current keypoint tensor (xyc)
        cond_flat (B, F+1, J*3) – conditioning sequence (xyc)
        t         (B,) – diffusion / flow time step in [0, 1]
    Returns:
        u_hat_flat (B, F, J*3) – predicted velocity field
    """

    def __init__(
        self,
        config
    ) -> None:
        super().__init__()

        d_model = config.d_model
        nhead = config.nhead
        num_layers = config.nlayers
        njoints = config.njoints
        n_spatial_layers = config.n_spatial_layers
        spatial_d = config.spatial_d

        self.nj = njoints
        self.D = d_model

        # per‑joint linear projection to low‑dim spatial tokens
        self.in_proj = nn.Linear(2, spatial_d)
        self.cond_proj = nn.Linear(3, spatial_d)

        # joint‑spatial encoder → (B, F, d_model)
        self.spatial = SpatialEncoder(
            d_model=spatial_d,
            nhead=4,
            nlayers=n_spatial_layers,
            njoints=njoints,
            out_c=d_model,
        )

        self.mode_emb = nn.Embedding(2, config.mode_dim)
        self.hw_proj = nn.Sequential(
            nn.Linear(1, config.mode_dim),
            nn.SiLU(),
            nn.Linear(config.mode_dim, config.mode_dim),
        )

        # positional encoding (temporal axis)
        self.pos_time = PositionalEncoding(d_model)

        # stack of decoder layers with AdaLN
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayerAdaLN(
                    d_model=d_model,
                    nhead=nhead,
                    cond_dim=config.mode_dim * 2,
                    dim_ff=4 * d_model,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_proj = nn.Linear(d_model, njoints)

        print('adaln model for conf')

    # ---------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------
    def forward(
        self,
        x_t_flat: torch.Tensor,
        cond_flat: torch.Tensor,
        t: torch.Tensor,
        mode_tag: torch.Tensor,
        hw: torch.Tensor,
    ) -> torch.Tensor:
        B, F, _ = x_t_flat.shape
        _, CF, _ = cond_flat.shape
        J = self.nj

        # 1. spatial encoding ------------------------------------------------
        x_j = x_t_flat.view(B, F, J, 2)
        c_j = cond_flat.view(B, CF, J, 3)

        q = self.in_proj(x_j)           # (B, F, J, spatial_d)
        k = self.cond_proj(c_j)         # (B, F+1, J, spatial_d)

        q = self.spatial(q)             # (B, F, D)
        k = self.spatial(k)             # (B, F+1, D)

        # 2. temporal + time embedding -------------------------------------
        q = self.pos_time(q)
        k = self.pos_time(k)

        # AdaLN cond vector
        m_emb   = self.mode_emb(mode_tag)                   # (B, mode_dim)
        hw_emb = self.hw_proj(hw.unsqueeze(-1)) # (B, D)
        cond_vec = torch.cat([m_emb, hw_emb], dim=-1)        # (B, cond_dim)

        # 3. decoder stack ---------------------------------------------------
        x = q
        for layer in self.decoder_layers:
            x = layer(x, k, cond_vec)

        # 4. output projection ----------------------------------------------
        u_hat_flat = self.out_proj(x)           # (B, F, J*1)
        return u_hat_flat



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

    
def align_video_pose(video_dwpose,
                     image_dwpose,
                     hw, 
                     threshold: float = 5.0,
                     body_indices = (1, 2, 5, 8, 11)):
    # ---------- 1. 置信度筛选 ----------
    score_v = video_dwpose[0, body_indices, 2, 0]  # (N,)
    score_i = image_dwpose[0, body_indices, 2, 0]  # (N,)
    mask    = (score_v > threshold) & (score_i > threshold)
    if not mask.any():                 # 没有共同高置信度 → 原样返回
        return video_dwpose.copy()
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
    aligned = video_dwpose.copy()
    aligned[:, :, :2, :] = (aligned[:, :, :2, :] - anchor_v) * s + anchor_i

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
    def __init__(self, config_path, ckpt, config_path2, ckpt2, device):
        self.device = device

        self.config = get_config(config_path)
        state = torch.load(ckpt, map_location='cpu')
        self.model  = VelocityTransformer(self.config)
        self.model.load_state_dict(filter_ckpt(state['model'])); 
        self.model.to(device)
        self.model.eval()

        self.config_conf = get_config(config_path2)
        state2 = torch.load(ckpt2, map_location='cpu')
        self.model_conf  = ConfTransformer(self.config_conf)
        self.model_conf.load_state_dict(filter_ckpt(state2['model'])); 
        self.model_conf.to(device)
        self.model_conf.eval()
    
    def sample(self, seqA, firstB, seqB_frame, steps=5, fix_first = True, mode = 1, hw = 1):
        """
        seqA : (B,J,3,F) , firstB : (B,J,3)
        返回预测 seqB_hat (B,J,3,F)
        """
        hw = torch.tensor([hw], device=seqA.device, dtype=torch.float32)
        tag_mode = torch.tensor([mode], dtype=torch.int32, device=seqA.device)

        b, j, c, f = seqA.shape

        # cond 预处理
        seqA_flat = seqA.permute(0, 3, 1, 2).contiguous().view(b, f, -1)
        if fix_first:
            cond_flat = seqA_flat
        else:
            firstB_flat = firstB.view(b, 1, -1)
            cond_flat = torch.cat([firstB_flat, seqA_flat], dim=1)

        cond_flat_cp = cond_flat.clone()
        # 初始 z1 ~ N(0,1)
        x = torch.randn(b, seqB_frame, j*2, device='cpu').to(seqA.device)

        if fix_first:
            firstB_xy_flat = firstB[:, :, :3].reshape(b, -1)   # (B,J*3)  保留 conf 也无妨
            x[:, 0, :] = firstB_xy_flat

        # ODE  solver
        t_steps = torch.linspace(1., 0., steps+1, device=seqA.device)
        for s in range(steps):
            t = t_steps[s].expand(b)
            dt = t_steps[s+1] - t_steps[s]
            v1 = self.model(x, cond_flat, t, tag_mode, hw)            # ûθ
            x = x + dt * v1                     # Euler
            if fix_first:
                x[:, 0, :] = firstB_xy_flat      # 每步 clamp 一次
        
        # second sample
        x0 = x.clone()
        x = torch.randn(b, seqB_frame, j*2, device='cpu').to(seqA.device)
        start_t = 0.3
        x = x*start_t + x0*(1-start_t)

        if fix_first:
            firstB_xy_flat = firstB[:, :, :3].reshape(b, -1)   # (B,J*3)  保留 conf 也无妨
            x[:, 0, :] = firstB_xy_flat

        # ODE  solver
        t_steps = torch.linspace(start_t, 0., steps+1, device=seqA.device)
        for s in range(steps):
            t = t_steps[s].expand(b)
            dt = t_steps[s+1] - t_steps[s]
            v1 = self.model(x, cond_flat, t, tag_mode, hw)            # ûθ
            x = x + dt * v1                     # Euler
            if fix_first:
                x[:, 0, :] = firstB_xy_flat      # 每步 clamp 一次

        
        # out conf
        conf = self.model_conf(x, cond_flat_cp, 0, tag_mode, hw)            # ûθ
        conf = torch.sigmoid(conf)
        conf_hat = conf.view(b, f, j, 1).permute(0, 2, 3, 1).contiguous()

        seqB_hat = x.view(b, seqB_frame, j, 2).permute(0, 2, 3, 1).contiguous()
        seqB_hat = torch.cat([seqB_hat, conf_hat], dim=2)

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
            
    def inf(self, video_pose, image_pose, video_hw, image_hw):
        # np.save('/vh_data/lxq/video_pose.npy', video_pose)
        # np.save('/vh_data/lxq/image_pose.npy', image_pose)
        # np.save('/vh_data/lxq/video_hw.npy', video_hw)
        # np.save('/vh_data/lxq/image_hw.npy', image_hw)
        
        torch.manual_seed(666) # 固定随机种子
        total_frame = 330-25  # 根据模型来，不能改变
        video_pose_blend = video_pose[:25, :, :, :].copy()
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
        # keypoints, scores = bodypose_24to20(keypoints, scores, 5)
        video_pose = np.concatenate((keypoints, scores[:,:, None,:]), axis=-2) # bj3f
        video_pose = video_pose[:, keep_idx, :, :]

        keypoints, scores = image_pose[:,:, :2, :], image_pose[:,:, 2,:]
        # keypoints, scores = bodypose_24to20(keypoints, scores, 5)
        image_pose = np.concatenate((keypoints, scores[:,:, None,:]), axis=-2) # bj3f
        image_pose = image_pose[:, keep_idx, :, :]
        
        video_pose[:,:, 1, :] *= video_hw[0, 0]/video_hw[0, 1]
        image_pose[:,:, 1, :] *= image_hw[0, 0]/image_hw[0, 1]
        video_pose2 = align_video_pose(video_pose.copy(), image_pose, image_hw[0, 0]/image_hw[0, 1])
        conf_check(video_pose2[0], 0, image_hw[0, 0]/image_hw[0, 1])
        conf_check(image_pose[0], 0, image_hw[0, 0]/image_hw[0, 1])

        video_pose2 = torch.tensor(localize_motion(video_pose2[0], self.config.data.local_hand_face, self.config.data.local_body, 1.0, False, True), dtype=torch.float32, device=self.device)[None]
        image_pose = torch.tensor(localize_motion(image_pose[0], self.config.data.local_hand_face, self.config.data.local_body, 1.0, False, True), dtype=torch.float32, device=self.device)[None]

        with torch.no_grad():
            pred = self.sample(video_pose2.clone(), image_pose[..., 0].clone(), total_frame, 5, self.config.fix_first, 1, image_hw[0, 0]/image_hw[0, 1])
            # pred2 = self.sample(video_pose2.clone(), image_pose[..., 0].clone(), total_frame, 5, self.config.fix_first, 0)
            
            if True:
                out = pred_to_video_dwpose(T, pred.clone().cpu()[0].numpy(), image_hw, self.config.data.local_hand_face, self.config.data.local_body,None)
                out = np.concatenate([video_pose_blend, out], axis=0)
                return out
            else:
                output = "/ytech_milm_ssd/lxq/transmomo2/new"
                if not os.path.exists(output):
                    os.makedirs(output)
                
                from lib.util.visualization import vis_merge3_2_inf
                vis_merge3_2_inf(T, image_pose.cpu()[0,:,:,:], video_pose2.cpu()[0,:,:,:], pred.cpu()[0,:,:,:], image_hw[0, 0]/image_hw[0, 1], self.config.data.local_hand_face, self.config.data.local_body, self.config.noblend, output + '/result_{0}.mp4'.format(0))



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default = "/ytech_milm_ssd/lxq/transmomo2/out/l12_h1f1_94w_160w_conf1_ghf_gb_top10_noblend_pair_adaln_slayer2_mode32_noconf_330_hw_saug/config.yaml")
    parser.add_argument("--ckpt", type=str, default = "/ytech_milm_ssd/lxq/transmomo2/out/l12_h1f1_94w_160w_conf1_ghf_gb_top10_noblend_pair_adaln_slayer2_mode32_noconf_330_hw_saug/checkpoints/epoch_237_step_1760000.pt")
    
    parser.add_argument("--config2", type=str, default = "/ytech_milm_ssd/lxq/transmomo2/out/l12_h3f1_94w_160w_conf1_ghf_gb_top10_noblend_pair_adaln_slayer2_mode32_forconf_330/config.yaml")
    parser.add_argument("--ckpt2", type=str, default = "/ytech_milm_ssd/lxq/transmomo2/out/l12_h3f1_94w_160w_conf1_ghf_gb_top10_noblend_pair_adaln_slayer2_mode32_forconf_330/checkpoints/epoch_2_step_20000.pt")
    
    # parser.add_argument("--config", type=str, default = "/ytech_milm_ssd/lxq/transmomo2/out/l12_h1f1_94w_160w_conf1_ghf_gb_top10_noblend_pair_adaln_slayer2_mode32_noconf_330_hw_saug/config.yaml")
    # parser.add_argument("--ckpt", type=str, default = "/ytech_milm_ssd/lxq/transmomo2/out/l12_h3f1_94w_160w_conf1_ghf_gb_top10_noblend_pair_adaln_slayer2_mode32_forconf_330/checkpoints/epoch_32_step_240000.pt")
    
    opts = parser.parse_args()
    return opts

if __name__ == "__main__":

    opts = parse_args()

    torch.manual_seed(666)
    
    retarget = retargeter(opts.config, opts.ckpt, opts.config2, opts.ckpt2, 'cpu')

    video_pose = np.load('/vh_data/lxq/video_pose.npy') # (121, 1, 134, 3) F B J 3
    video_hw = np.load('/vh_data/lxq/video_hw.npy') # (1, 2)
    image_pose = np.load('/vh_data/lxq/image_pose.npy') # (1, 1, 134, 3)
    image_hw = np.load('/vh_data/lxq/image_hw.npy') # (1, 2)

    retarget.inf(video_pose, image_pose, video_hw, image_hw)
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