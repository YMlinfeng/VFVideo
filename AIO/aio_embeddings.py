import math

import torch
import torch.nn as nn
from diffusers.models.activations import FP32SiLU
from diffusers.models.embeddings import TimestepEmbedding, Timesteps


import numpy as np
import os

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class BaseRoPE:
    @torch.compile
    def apply_rope(self, x, freqs_cos, freqs_sin, shift_freqs_cos=None, shift_freqs_sin=None, num_wins=None):
        batch, num_heads, num_patches, embed_dim = x.shape
        inputs = x
        x = x.reshape(batch, num_heads, num_patches, -1, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        x = x.reshape(batch, num_heads, num_patches, embed_dim)

        freqs_cos = freqs_cos.reshape(-1, embed_dim)
        freqs_sin = freqs_sin.reshape(-1, embed_dim)

        if shift_freqs_cos is not None and shift_freqs_sin is not None and num_wins is not None:
            # print('shift_rope')
            # 处理cos部分
            shift_freqs_cos = shift_freqs_cos.reshape(-1, embed_dim)
            normal_freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(0)  # [1, 1, num_patches, embed_dim]
            shift_freqs_cos = shift_freqs_cos.unsqueeze(0).unsqueeze(0)  # [1, 1, num_patches, embed_dim]

            # 在第二个维度重复num_heads次
            normal_freqs_cos = normal_freqs_cos.expand(-1, num_heads, -1, -1)  # [1, num_heads, num_patches, embed_dim]
            shift_freqs_cos = shift_freqs_cos.expand(-1, num_heads, -1, -1)  # [1, num_heads, num_patches, embed_dim]

            # 创建完整的freqs_cos序列
            freqs_cos_list = [shift_freqs_cos] + [normal_freqs_cos] * (num_wins - 1)
            freqs_cos = torch.cat(freqs_cos_list, dim=0)  # [windows, num_heads, num_patches, embed_dim]

            # 处理sin部分
            shift_freqs_sin = shift_freqs_sin.reshape(-1, embed_dim)
            normal_freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(0)  # [1, 1, num_patches, embed_dim]
            shift_freqs_sin = shift_freqs_sin.unsqueeze(0).unsqueeze(0)  # [1, 1, num_patches, embed_dim]

            # 在第二个维度重复num_heads次
            normal_freqs_sin = normal_freqs_sin.expand(-1, num_heads, -1, -1)  # [1, num_heads, num_patches, embed_dim]
            shift_freqs_sin = shift_freqs_sin.expand(-1, num_heads, -1, -1)  # [1, num_heads, num_patches, embed_dim]

            # 创建完整的freqs_sin序列
            freqs_sin_list = [shift_freqs_sin] + [normal_freqs_sin] * (num_wins - 1)
            freqs_sin = torch.cat(freqs_sin_list, dim=0)  # [windows, num_heads, num_patches, embed_dim]

            # 将freqs_cos和freqs_sin在第一个维度重复batch // windows次
            freqs_cos = freqs_cos.repeat(batch // num_wins, 1, 1, 1)  # [batch, num_heads, num_patches, embed_dim]
            freqs_sin = freqs_sin.repeat(batch // num_wins, 1, 1, 1)  # [batch, num_heads, num_patches, embed_dim]

        return inputs * freqs_cos + x * freqs_sin

    def _forward(self, x, patch_resolution, num_wins=None):
        if self.freqs_cos.device != x.device or self.freqs_cos.dtype != x.dtype:
            self.freqs_cos = self.freqs_cos.to(x.device, x.dtype)
        if self.freqs_sin.device != x.device or self.freqs_sin.dtype != x.dtype:
            self.freqs_sin = self.freqs_sin.to(x.device, x.dtype)
        freqs_cos, freqs_sin = self.freqs_cos, self.freqs_sin

        for dim_idx, length in enumerate(patch_resolution):
            freqs_cos = torch.narrow(freqs_cos, dim_idx, 0, length)
            freqs_sin = torch.narrow(freqs_sin, dim_idx, 0, length)

        if num_wins is not None:
            length = freqs_sin.shape[0]
            shift_freqs_cos = freqs_cos[: length // 2]
            shift_freqs_sin = freqs_sin[: length // 2]
            shift_freqs_cos = torch.cat([shift_freqs_cos] * 2)
            shift_freqs_sin = torch.cat([shift_freqs_sin] * 2)
        else:
            shift_freqs_cos = None
            shift_freqs_sin = None

        return self.apply_rope(x, freqs_cos, freqs_sin, shift_freqs_cos, shift_freqs_sin, num_wins)


class RoPE3D(BaseRoPE):
    def __init__(self, embed_dim, max_patch_resolution=(40, 160, 160), interpolation_scale=1, theta_time=10000.0, theta_space=10000.0, embed_dim_time=None, embed_dim_space=None, repeat_interleave=True, multi_id=False):
        super().__init__()

        self.embed_dim = embed_dim
        if embed_dim_time is not None and embed_dim_space is not None:
            if (embed_dim_time + embed_dim_space * 2) != embed_dim:
                raise ValueError("embed_dim_time + embed_dim_space * 2 must be equal to embed_dim.")
            self.embed_dim_time = embed_dim_time
            self.embed_dim_space = embed_dim_space
        else:
            self.embed_dim_time = embed_dim // 3
            self.embed_dim_space = embed_dim // 3

        self.theta_time = theta_time
        self.theta_space = theta_space
        self.max_patch_resolution = max_patch_resolution
        self.interpolation_scale = interpolation_scale
        self.repeat_interleave = repeat_interleave
        self.multi_id = multi_id

        self.frequency_time = 1.0 / (self.theta_time ** (torch.arange(0, self.embed_dim_time, 2).float() / self.embed_dim_time))
        self.frequency_space = 1.0 / (self.theta_space ** (torch.arange(0, self.embed_dim_space, 2).float() / self.embed_dim_space))

        if not multi_id:
            freqs_cos, freqs_sin = self.compute_position_embedding(max_patch_resolution)
            self.freqs_cos = freqs_cos
            self.freqs_sin = freqs_sin
        else:
            # multi_id 模式：不预计算，标记为 None
            self.freqs_cos = None
            self.freqs_sin = None

    def compute_position_embedding_two_id(self): 
        '''
        RoPE(x, pos) = x * cos(θ * pos) + rotate(x) * sin(θ * pos)
        '''
        frequency_time = 1.0 / (self.theta_time ** (torch.arange(0, self.embed_dim_time, 2).float() / self.embed_dim_time))
        frequency_space = 1.0 / (self.theta_space ** (torch.arange(0, self.embed_dim_space, 2).float() / self.embed_dim_space))

        t, h, w = self.max_patch_resolution
        if self.multi_id:
            ##position_t = torch.arange(t)[:, None].float() @ frequency_time[None, :]
            ##position_t = torch.cat([position_t[[39,49]], position_t[:-2]])
            ##position_t = torch.cat([position_t[[33,40]], position_t[:-2]])
            past_frames = 2
            position_t = torch.arange(-past_frames, t-past_frames)[:, None].float() @ frequency_time[None, :] # (64, 85)
        else:
            position_t = torch.arange(t)[:, None].float() @ frequency_time[None, :] # @是外积（矩阵乘法），每个位置 × 每个频率 = 角度(每一行是该帧在各个频率下的「角度」，后续会用 cos/sin 转换)； postion t = θ⋅pos
        print ('-------------------------self.multi_id:', self.multi_id)

        position_h = torch.arange(h)[:, None].float() / self.interpolation_scale @ frequency_space[None, :] # ([256, 85])
        position_w = torch.arange(w)[:, None].float() / self.interpolation_scale @ frequency_space[None, :]

        # 原始向量: [x0, x1, x2, x3, ...]
        # 配对:     [(x0,x1), (x2,x3), ...]  ← 每对共用同一个角度
                
        # 所以角度也要翻倍匹配:
        # repeat_interleave: [θ0, θ0, θ1, θ1, ...]  ← 交错式
        # repeat:            [θ0, θ1, θ0, θ1, ...]  ← 平铺式（较少用）
        if self.repeat_interleave:
            position_t = position_t.repeat_interleave(2, dim=1) # (64, 170)
            position_h = position_h.repeat_interleave(2, dim=1) # ([256, 170])
            position_w = position_w.repeat_interleave(2, dim=1)
        else:
            position_t = position_t.repeat(1, 2)
            position_h = position_h.repeat(1, 2)
            position_w = position_w.repeat(1, 2)

        frame = position_t[:, None, None, :].expand(t, h, w, self.embed_dim_time) # frame 只关心"第几帧"，不管空间位置 → 在 H, W 维度广播（所有空间位置共享同一帧编码）
        #print ('frame_shape', frame.shape) #frame_shape torch.Size([64, 160, 160, 32])
        height = position_h[None, :, None, :].expand(t, h, w, self.embed_dim_space) # height 只关心"第几行" → 在 T, W 维度广播
        width = position_w[None, None, :, :].expand(t, h, w, self.embed_dim_space)
        position = torch.cat((frame, height, width), dim=-1) # 结果：每个 (t, h, w) 位置都有独立的时空编码

        freqs_cos = position.cos() # [t, h, w, embed_dim]
        freqs_sin = position.sin()

        if self.embed_dim > self.embed_dim_time + self.embed_dim_space * 2: # 当嵌入维度无法被3整除时，多余维度使用特殊补齐策略：余弦编码用1填充（保持原向量），正弦编码用0填充（消除旋转效果）即：剩余维度不应用位置编码
            res_embed_dim = self.embed_dim - (self.embed_dim_time + self.embed_dim_space * 2)
            cos_shape = freqs_cos.shape[:-1] + (res_embed_dim,)
            sin_shape = freqs_sin.shape[:-1] + (res_embed_dim,)
            freqs_cos = torch.cat((freqs_cos, torch.ones(cos_shape)), dim=-1)
            freqs_sin = torch.cat((freqs_sin, torch.zeros(sin_shape)), dim=-1)

        return freqs_cos, freqs_sin
    def compute_position_embedding(self, patch_resolution):
        t, h, w = patch_resolution
        
        frequency_time = self.frequency_time
        frequency_space = self.frequency_space
        
        if self.multi_id:
            num_ref_frames = 2
            grid_rows, grid_cols = 3, 3
            frames_per_grid = 2
            num_grids = grid_rows * grid_cols  # 9
            total_ref_time = num_grids * frames_per_grid  # 18
            
            def split_indices(total, num_splits):
                base = total // num_splits
                remainder = total % num_splits
                splits = [0]
                for i in range(num_splits):
                    splits.append(splits[-1] + base + (1 if i < remainder else 0))
                return splits
            
            h_splits = split_indices(h, grid_rows)
            w_splits = split_indices(w, grid_cols)
            
            time_positions = torch.zeros(t, h, w, device=frequency_time.device)
            
            for frame_idx in range(num_ref_frames):
                for grid_row in range(grid_rows):
                    for grid_col in range(grid_cols):
                        grid_idx = grid_row * grid_cols + grid_col  # 行优先: 0-8
                        real_time_pos = grid_idx * frames_per_grid + frame_idx - total_ref_time
                        
                        h_start, h_end = h_splits[grid_row], h_splits[grid_row + 1]
                        w_start, w_end = w_splits[grid_col], w_splits[grid_col + 1]
                        time_positions[frame_idx, h_start:h_end, w_start:w_end] = real_time_pos
            
            for gen_frame_idx in range(num_ref_frames, t):
                time_positions[gen_frame_idx, :, :] = gen_frame_idx - num_ref_frames
            
            # (t, h, w, embed_dim_time/2)
            frame = time_positions[:, :, :, None] @ frequency_time[None, :]
        else:
            position_t = torch.arange(t, device=frequency_time.device)[:, None].float() @ frequency_time[None, :]

        position_h = torch.arange(h, device=frequency_space.device)[:, None].float() / self.interpolation_scale @ frequency_space[None, :]
        position_w = torch.arange(w, device=frequency_space.device)[:, None].float() / self.interpolation_scale @ frequency_space[None, :]

        if self.repeat_interleave:
            if self.multi_id:
                frame = frame.repeat_interleave(2, dim=-1)
            else:
                position_t = position_t.repeat_interleave(2, dim=1)
            position_h = position_h.repeat_interleave(2, dim=1)
            position_w = position_w.repeat_interleave(2, dim=1)
        else:
            if self.multi_id:
                frame = frame.repeat(1, 1, 1, 2)
            else:
                position_t = position_t.repeat(1, 2)
            position_h = position_h.repeat(1, 2)
            position_w = position_w.repeat(1, 2)

        if not self.multi_id:
            frame = position_t[:, None, None, :].expand(t, h, w, self.embed_dim_time)
        
        height = position_h[None, :, None, :].expand(t, h, w, self.embed_dim_space)
        width = position_w[None, None, :, :].expand(t, h, w, self.embed_dim_space)

        position = torch.cat((frame, height, width), dim=-1)
        freqs_cos = position.cos()
        freqs_sin = position.sin()

        if self.embed_dim > self.embed_dim_time + self.embed_dim_space * 2:
            res_embed_dim = self.embed_dim - (self.embed_dim_time + self.embed_dim_space * 2)
            cos_shape = freqs_cos.shape[:-1] + (res_embed_dim,)
            sin_shape = freqs_sin.shape[:-1] + (res_embed_dim,)
            freqs_cos = torch.cat((freqs_cos, torch.ones(cos_shape, device=freqs_cos.device)), dim=-1)
            freqs_sin = torch.cat((freqs_sin, torch.zeros(sin_shape, device=freqs_sin.device)), dim=-1)

        return freqs_cos, freqs_sin

    def _forward(self, x, patch_resolution, num_wins=None):
        if self.multi_id:
            if self.frequency_time.device != x.device:
                self.frequency_time = self.frequency_time.to(x.device)
            if self.frequency_space.device != x.device:
                self.frequency_space = self.frequency_space.to(x.device)
            
            freqs_cos, freqs_sin = self.compute_position_embedding(patch_resolution)
            freqs_cos = freqs_cos.to(x.dtype)
            freqs_sin = freqs_sin.to(x.dtype)
        else:
            if self.freqs_cos.device != x.device or self.freqs_cos.dtype != x.dtype:
                self.freqs_cos = self.freqs_cos.to(x.device, x.dtype)
            if self.freqs_sin.device != x.device or self.freqs_sin.dtype != x.dtype:
                self.freqs_sin = self.freqs_sin.to(x.device, x.dtype)
            freqs_cos, freqs_sin = self.freqs_cos, self.freqs_sin

            for dim_idx, length in enumerate(patch_resolution):
                freqs_cos = torch.narrow(freqs_cos, dim_idx, 0, length)
                freqs_sin = torch.narrow(freqs_sin, dim_idx, 0, length)

        if num_wins is not None:
            length = freqs_sin.shape[0]
            shift_freqs_cos = freqs_cos[: length // 2]
            shift_freqs_sin = freqs_sin[: length // 2]
            shift_freqs_cos = torch.cat([shift_freqs_cos] * 2)
            shift_freqs_sin = torch.cat([shift_freqs_sin] * 2)
        else:
            shift_freqs_cos = None
            shift_freqs_sin = None

        return self.apply_rope(x, freqs_cos, freqs_sin, shift_freqs_cos, shift_freqs_sin, num_wins)

    def forward(self, x, patch_resolution, num_wins=None):
        if isinstance(patch_resolution, list) and all(isinstance(resolution, tuple) for resolution in patch_resolution):
            output = torch.zeros_like(x)
            for i, resolution in enumerate(patch_resolution):
                valid_sequence_length = math.prod(resolution)
                sub_x = x[i : i + 1, :, :valid_sequence_length]
                sub_output = self._forward(sub_x, resolution, num_wins)
                output[i : i + 1, :, :valid_sequence_length] = sub_output
            return output
        elif isinstance(patch_resolution, tuple):
            return self._forward(x, patch_resolution, num_wins)
        else:
            raise TypeError("patch_resolution must be a list of tuple or a tuple.")


class CombinedTimestepConditionEmbeddings(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        use_text_condition: bool = False,
        use_frames_resolution_condition: bool = False,
        use_unpadded_resolution_condition: bool = False,
        use_resolution_condition: bool = False,
        use_frames_condition: bool = False,
        use_noise_aug_condition: bool = False,
        split_conditions: bool = False,
        sample_proj_bias: bool = True,
    ):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim, sample_proj_bias=sample_proj_bias)

        self.use_additional_conditions = (
            use_frames_resolution_condition or use_unpadded_resolution_condition or use_resolution_condition or use_frames_condition
        )
        self.use_text_condition = use_text_condition
        self.use_frames_resolution_condition = use_frames_resolution_condition
        self.use_unpadded_resolution_condition = use_unpadded_resolution_condition
        self.use_resolution_condition = use_resolution_condition
        self.use_frames_condition = use_frames_condition
        self.use_noise_aug_condition = use_noise_aug_condition
        self.split_conditions = split_conditions

        if self.use_additional_conditions:
            self.additional_condition_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        if use_text_condition:
            self.text_embedder = TimestepEmbedding(in_channels=2048, time_embed_dim=embedding_dim, sample_proj_bias=sample_proj_bias)
        if use_frames_resolution_condition:
            self.frames_resolution_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim // 3, sample_proj_bias=sample_proj_bias)
        if use_unpadded_resolution_condition:
            self.unpadded_resolution_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim // 2, sample_proj_bias=sample_proj_bias)
        if use_resolution_condition:
            self.resolution_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim // 2, sample_proj_bias=sample_proj_bias)
        if use_frames_condition:
            self.frames_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim, sample_proj_bias=sample_proj_bias)
        ## add for vsr
        if use_noise_aug_condition:
            self.additional_noise_aug_proj = zero_module(nn.Linear(in_features=embedding_dim,
                                                    out_features=embedding_dim, bias=False))

        # zero init
        for name, module in self.named_children():
            if isinstance(module, TimestepEmbedding) and not name.endswith("timestep_embedder"):
                module.linear_2.weight.data.zero_()
                if hasattr(module.linear_2, "bias") and module.linear_2.bias is not None:
                    module.linear_2.bias.data.zero_()

    def forward(
        self, timestep, batch_size, hidden_dtype, frames=None, resolution=None, unpadded_resolution=None,
        frames_resolution=None, prompt_embeds_pooled=None, noise_aug_val=None
    ):
        timesteps_proj = self.time_proj(timestep.view(-1))
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        conditioning = timesteps_emb
        spatial_conditioning = 0
        temporal_conditioning = 0

        if self.use_text_condition and prompt_embeds_pooled is not None:
            text_emb = self.text_embedder(prompt_embeds_pooled)
            conditioning = conditioning + text_emb
        if self.use_frames_resolution_condition and frames_resolution is not None:
            frames_resolution_emb = self.additional_condition_proj(frames_resolution.flatten()).to(hidden_dtype)
            frames_resolution_emb = self.frames_resolution_embedder(frames_resolution_emb).reshape(batch_size, -1)
            spatial_conditioning = spatial_conditioning + frames_resolution_emb
        if self.use_unpadded_resolution_condition and unpadded_resolution is not None:
            unpadded_resolution_emb = self.additional_condition_proj(unpadded_resolution.flatten()).to(hidden_dtype)
            unpadded_resolution_emb = self.unpadded_resolution_embedder(unpadded_resolution_emb).reshape(batch_size, -1)
            spatial_conditioning = spatial_conditioning + unpadded_resolution_emb
        if self.use_resolution_condition and resolution is not None:
            resolution_emb = self.additional_condition_proj(resolution.flatten()).to(hidden_dtype)
            resolution_emb = self.resolution_embedder(resolution_emb).reshape(batch_size, -1)
            spatial_conditioning = spatial_conditioning + resolution_emb
        if self.use_frames_condition and frames is not None:
            frames_emb = self.additional_condition_proj(frames.flatten()).to(hidden_dtype)
            frames_emb = self.frames_embedder(frames_emb).reshape(batch_size, -1)
            temporal_conditioning = temporal_conditioning + frames_emb

        ## add for vsr
        if self.use_noise_aug_condition and noise_aug_val is not None:
            noise_timesteps_proj = self.time_proj(noise_aug_val.view(-1))
            noise_timesteps_emb = self.timestep_embedder(noise_timesteps_proj.to(dtype=hidden_dtype)) # (N, D)
            noise_timesteps_emb = self.additional_noise_aug_proj(noise_timesteps_emb)
            '''
            print(f"-----noise_timesteps_emb: {noise_aug_val}")
            torch.save(noise_timesteps_emb, "/home/xiamenghan/DiffusionGen/m2v-diffusers-vsr/output/cache_dir/aug_emb.pt")
            '''
            if conditioning.shape[0] != noise_timesteps_emb.shape[0]:
                repeat_times = int(conditioning.shape[0] // noise_timesteps_emb.shape[0])
                noise_timesteps_emb = noise_timesteps_emb.repeat(repeat_times,1)
            conditioning = conditioning + noise_timesteps_emb

        if self.split_conditions:
            return conditioning, spatial_conditioning, temporal_conditioning
        else:
            return conditioning + spatial_conditioning + temporal_conditioning



def main():
    import torch
    embed_dim = 512
    T = 10
    H = 12 
    W = 15  
    patch_resolution = (T, H, W)
    rope = RoPE3D(
        embed_dim=embed_dim,
        max_patch_resolution=patch_resolution,
        multi_id=True,
    )
    old_freqs_cos, old_freqs_sin = rope.compute_position_embedding_two_id() # ([10, 12, 15, 512])
    new_freqs_cos, new_freqs_sin = rope.compute_position_embedding(patch_resolution) # ([10, 12, 15, 512])
    print(f"\n形状: old={old_freqs_cos.shape}, new={new_freqs_cos.shape}")
    gen_old_cos = old_freqs_cos[2:, :, :, :]
    gen_new_cos = new_freqs_cos[2:, :, :, :]
    gen_old_sin = old_freqs_sin[2:, :, :, :]
    gen_new_sin = new_freqs_sin[2:, :, :, :]
    
    cos_match = torch.allclose(gen_old_cos, gen_new_cos, atol=1e-6)
    sin_match = torch.allclose(gen_old_sin, gen_new_sin, atol=1e-6)
    print(f"  freqs_cos 匹配: {cos_match}")
    print(f"  freqs_sin 匹配: {sin_match}")
    
    ref_old_cos = old_freqs_cos[:2, :, :, :]
    ref_new_cos = new_freqs_cos[:2, :, :, :]
    
    ref_same = torch.allclose(ref_old_cos, ref_new_cos, atol=1e-6)
    print(f"  参考帧 freqs_cos 相同: {ref_same}")
    
    # 计算预期的 time_positions
    grid_rows, grid_cols = 3, 3
    frames_per_grid = 2
    total_ref_time = 18
    
    def split_indices(total, num_splits):
        base = total // num_splits
        remainder = total % num_splits
        splits = [0]
        for i in range(num_splits):
            splits.append(splits[-1] + base + (1 if i < remainder else 0))
        return splits
    
    h_splits = split_indices(H, grid_rows)
    w_splits = split_indices(W, grid_cols)
    
    print(f"  H={H}, h_splits={h_splits}") # H=12, h_splits=[0, 4, 8, 12]
    print(f"  W={W}, w_splits={w_splits}") # W=15, w_splits=[0, 5, 10, 15]
    
    time_positions = torch.zeros(T, H, W)
    for frame_idx in range(2):
        for grid_row in range(grid_rows):
            for grid_col in range(grid_cols):
                grid_idx = grid_row * grid_cols + grid_col
                real_time_pos = grid_idx * frames_per_grid + frame_idx - total_ref_time
                h_start, h_end = h_splits[grid_row], h_splits[grid_row + 1]
                w_start, w_end = w_splits[grid_col], w_splits[grid_col + 1]
                time_positions[frame_idx, h_start:h_end, w_start:w_end] = real_time_pos
    
    for gen_idx in range(2, T):
        time_positions[gen_idx, :, :] = gen_idx - 2
    
    print(f"\n  帧0 时间位置（九宫格）:")
    print("  预期: 格子0=-18, 格子1=-16, ..., 格子8=-2")
    for grid_row in range(3):
        row_values = []
        for grid_col in range(3):
            h_idx = h_splits[grid_row]
            w_idx = w_splits[grid_col]
            row_values.append(int(time_positions[0, h_idx, w_idx].item()))
        print(f"    行{grid_row}: {row_values}")
    
    print(f"\n  帧1 时间位置（九宫格）:")
    print("  预期: 格子0=-17, 格子1=-15, ..., 格子8=-1")
    for grid_row in range(3):
        row_values = []
        for grid_col in range(3):
            h_idx = h_splits[grid_row]
            w_idx = w_splits[grid_col]
            row_values.append(int(time_positions[1, h_idx, w_idx].item()))
        print(f"    行{grid_row}: {row_values}")
    
    print(f"\n  帧2 时间位置（生成帧）:  {int(time_positions[2, 0, 0].item())}")
    print(f"  帧3 时间位置（生成帧）:  {int(time_positions[3, 0, 0].item())}")
    
    #-----------------#
    batch = 1
    num_heads = 8
    num_patches = T * H * W
    torch.manual_seed(42)
    x = torch.randn(batch, num_heads, num_patches, embed_dim)
    output = rope.forward(x, patch_resolution)
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {output.shape}")
    assert x.shape == output.shape
    assert not torch.allclose(x, output)
    

if __name__ == "__main__":
    main()
