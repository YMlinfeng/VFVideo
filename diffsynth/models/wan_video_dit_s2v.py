import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .wan_video_dit import rearrange, precompute_freqs_cis_3d, DiTBlock, Head, CrossAttention, modulate, sinusoidal_embedding_1d


def torch_dfs(model: nn.Module, parent_name='root'):
    module_names, modules = [], []
    current_name = parent_name if parent_name else 'root'
    module_names.append(current_name)
    modules.append(model)

    for name, child in model.named_children():
        if parent_name:
            child_name = f'{parent_name}.{name}'
        else:
            child_name = name
        child_modules, child_names = torch_dfs(child, child_name)
        module_names += child_names
        modules += child_modules
    return modules, module_names


def rope_precompute(x, grid_sizes, freqs, start=None): # freqs:(1024,64)
    b, s, n, c = x.size(0), x.size(1), x.size(2), x.size(3) // 2 # # b=1, s=21000, n=40, c=64（复数频率数量（128/2=64，因为两个实数组成一个复数））

    # split freqs
    if type(freqs) is list:
        trainable_freqs = freqs[1]
        freqs = freqs[0]
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1) # # 分割为: [64 - 42, 21, 21] = [22, 21, 21]

    # 为什么用复数？ 因为复数乘法自动实现旋转！
    output = torch.view_as_complex(x.detach().reshape(b, s, n, -1, 2).to(torch.float64)) # [1, 21000, 40, 128] → [1, 21000, 40, 64, 2] → [1, 21000, 40, 64] (复数)
    seq_bucket = [0]
    if not type(grid_sizes) is list:
        grid_sizes = [grid_sizes]
    for g in grid_sizes:
        if not type(g) is list:
            g = [torch.zeros_like(g), g]
        batch_size = g[0].shape[0]
        for i in range(batch_size):
            if start is None:
                f_o, h_o, w_o = g[0][i]
            else:
                f_o, h_o, w_o = start[i]

            f, h, w = g[1][i]
            t_f, t_h, t_w = g[2][i]
            seq_f, seq_h, seq_w = f - f_o, h - h_o, w - w_o
            seq_len = int(seq_f * seq_h * seq_w)
            if seq_len > 0:
                if t_f > 0:
                    factor_f, factor_h, factor_w = (t_f / seq_f).item(), (t_h / seq_h).item(), (t_w / seq_w).item()
                    # Generate a list of seq_f integers starting from f_o and ending at math.ceil(factor_f * seq_f.item() + f_o.item())
                    if f_o >= 0:
                        f_sam = np.linspace(f_o.item(), (t_f + f_o).item() - 1, seq_f).astype(int).tolist()
                    else:
                        f_sam = np.linspace(-f_o.item(), (-t_f - f_o).item() + 1, seq_f).astype(int).tolist()
                    h_sam = np.linspace(h_o.item(), (t_h + h_o).item() - 1, seq_h).astype(int).tolist()
                    w_sam = np.linspace(w_o.item(), (t_w + w_o).item() - 1, seq_w).astype(int).tolist()

                    assert f_o * f >= 0 and h_o * h >= 0 and w_o * w >= 0
                    freqs_0 = freqs[0][f_sam] if f_o >= 0 else freqs[0][f_sam].conj() # # 查表！从预计算的频率表中取出对应位置
                    freqs_0 = freqs_0.view(seq_f, 1, 1, -1) # # 形状: [14, 1, 1, 22]

                    freqs_i = torch.cat(
                        [
                            freqs_0.expand(seq_f, seq_h, seq_w, -1),
                            freqs[1][h_sam].view(1, seq_h, 1, -1).expand(seq_f, seq_h, seq_w, -1),
                            freqs[2][w_sam].view(1, 1, seq_w, -1).expand(seq_f, seq_h, seq_w, -1),
                        ],
                        dim=-1
                    ).reshape(seq_len, 1, -1) # # 最终: [14, 40, 35, 64]->[19600, 1, 64]
                elif t_f < 0:
                    freqs_i = trainable_freqs.unsqueeze(1)
                # apply rotary embedding
                output[i, seq_bucket[-1]:seq_bucket[-1] + seq_len] = freqs_i
        seq_bucket.append(seq_bucket[-1] + seq_len)
    return output


class CausalConv1d(nn.Module):
    '''
    •对应论文：这就是所谓的“Causal”。
    •数学解释：假设卷积核大小 K=3。计算时刻t的输出时，卷积窗口覆盖的是［t—2，t—1,t。如果没有这个 Padding 而是用默认的
    padding=1，固口会覆盖t- 1,t,t+1，那就泄露了 t+1（未来）的信息。
    代码确认：这段代码强制模型只能看”过去”。
    
    '''
    def __init__(self, chan_in, chan_out, kernel_size=3, stride=1, dilation=1, pad_mode='replicate', **kwargs):
        super().__init__()

        self.pad_mode = pad_mode
        padding = (kernel_size - 1, 0)  # T，在左边（过去）补 K-1 个 0，在右边（未来）补 0 个 0
        self.time_causal_padding = padding

        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        # # 先 Pad 再 Conv，确保卷积核永远摸不到右边的数据
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)

# 步骤 B: 时序压缩 (Temporal Downsampling)
class MotionEncoder_tc(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int, num_heads=int, need_global=False, dtype=None, device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()

        self.num_heads = num_heads
        self.need_global = need_global
        self.conv1_local = CausalConv1d(in_dim, hidden_dim // 4 * num_heads, 3, stride=1) # # 不压缩
        if need_global:
            self.conv1_global = CausalConv1d(in_dim, hidden_dim // 4, 3, stride=1)
        self.norm1 = nn.LayerNorm(hidden_dim // 4, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.act = nn.SiLU()
        self.conv2 = CausalConv1d(hidden_dim // 4, hidden_dim // 2, 3, stride=2) # 2*
        self.conv3 = CausalConv1d(hidden_dim // 2, hidden_dim, 3, stride=2) # 2*

        if need_global:
            self.final_linear = nn.Linear(hidden_dim, hidden_dim, **factory_kwargs)

        self.norm1 = nn.LayerNorm(hidden_dim // 4, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.norm2 = nn.LayerNorm(hidden_dim // 2, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def forward(self, x):
        x = rearrange(x, 'b t c -> b c t')
        x_ori = x.clone()
        b, c, t = x.shape
        x = self.conv1_local(x)
        x = rearrange(x, 'b (n c) t -> (b n) t c', n=self.num_heads)
        x = self.norm1(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv2(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm2(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv3(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm3(x)
        x = self.act(x)
        x = rearrange(x, '(b n) t c -> b t n c', b=b)
        padding = self.padding_tokens.repeat(b, x.shape[1], 1, 1).to(device=x.device, dtype=x.dtype)
        x = torch.cat([x, padding], dim=-2)
        x_local = x.clone()

        if not self.need_global:
            return x_local

        x = self.conv1_global(x_ori)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm1(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv2(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm2(x)
        x = self.act(x)
        x = rearrange(x, 'b t c -> b c t')
        x = self.conv3(x)
        x = rearrange(x, 'b c t -> b t c')
        x = self.norm3(x)
        x = self.act(x)
        x = self.final_linear(x)
        x = rearrange(x, '(b n) t c -> b t n c', b=b)

        return x, x_local


class FramePackMotioner(nn.Module):

    def __init__(self, inner_dim=1024, num_heads=16, zip_frame_buckets=[1, 2, 16], drop_mode="drop", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
        self.zip_frame_buckets = torch.tensor(zip_frame_buckets, dtype=torch.long)

        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.freqs = torch.cat(precompute_freqs_cis_3d(inner_dim // num_heads), dim=1)
        self.drop_mode = drop_mode

    def forward(self, motion_latents, add_last_motion=2):
        motion_frames = motion_latents[0].shape[1]
        mot = []
        mot_remb = []
        for m in motion_latents:
            lat_height, lat_width = m.shape[2], m.shape[3]
            # 1. 创建 padding tensor [16通道, 19帧, H, W]
            padd_lat = torch.zeros(16, self.zip_frame_buckets.sum(), lat_height, lat_width).to(device=m.device, dtype=m.dtype)
            overlap_frame = min(padd_lat.shape[1], m.shape[1])
            if overlap_frame > 0:
                # 2. 填入历史帧（右对齐，最新的在最右边）
                padd_lat[:, -overlap_frame:] = m[:, -overlap_frame:]

            if add_last_motion < 2 and self.drop_mode != "drop":
                zero_end_frame = self.zip_frame_buckets[:self.zip_frame_buckets.__len__() - add_last_motion - 1].sum()
                padd_lat[:, -zero_end_frame:] = 0

            padd_lat = padd_lat.unsqueeze(0)
            # 3. 按桶分割（逆序：最远→最近）
            clean_latents_4x, clean_latents_2x, clean_latents_post = padd_lat[:, :, -self.zip_frame_buckets.sum():, :, :].split(
                list(self.zip_frame_buckets)[::-1], dim=2
            )  # 16, 2 ,1

            # patchfy # 4. 分别投影
            clean_latents_post = self.proj(clean_latents_post).flatten(2).transpose(1, 2)
            clean_latents_2x = self.proj_2x(clean_latents_2x).flatten(2).transpose(1, 2)
            clean_latents_4x = self.proj_4x(clean_latents_4x).flatten(2).transpose(1, 2)

            if add_last_motion < 2 and self.drop_mode == "drop":
                clean_latents_post = clean_latents_post[:, :0] if add_last_motion < 2 else clean_latents_post
                clean_latents_2x = clean_latents_2x[:, :0] if add_last_motion < 1 else clean_latents_2x

            # 5. 拼接成 motion token 序列
            motion_lat = torch.cat([clean_latents_post, clean_latents_2x, clean_latents_4x], dim=1)

            # rope
            # 6. 计算对应的 RoPE 位置编码（使用负时间索引表示"过去"）
            # start_time_id 为负数，表示这些帧在当前帧之前
            start_time_id = -(self.zip_frame_buckets[:1].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[0]
            grid_sizes = [] if add_last_motion < 2 and self.drop_mode == "drop" else \
                        [
                            [torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                            torch.tensor([end_time_id, lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1),
                            torch.tensor([self.zip_frame_buckets[0], lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1), ]
                        ]

            start_time_id = -(self.zip_frame_buckets[:2].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[1] // 2
            grid_sizes_2x = [] if add_last_motion < 1 and self.drop_mode == "drop" else \
            [
                [torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                torch.tensor([end_time_id, lat_height // 4, lat_width // 4]).unsqueeze(0).repeat(1, 1),
                torch.tensor([self.zip_frame_buckets[1], lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1), ]
            ]

            start_time_id = -(self.zip_frame_buckets[:3].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[2] // 4
            grid_sizes_4x = [
                [
                    torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([end_time_id, lat_height // 8, lat_width // 8]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([self.zip_frame_buckets[2], lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1),
                ]
            ]

            grid_sizes = grid_sizes + grid_sizes_2x + grid_sizes_4x

            motion_rope_emb = rope_precompute(
                motion_lat.detach().view(1, motion_lat.shape[1], self.num_heads, self.inner_dim // self.num_heads),
                grid_sizes,
                self.freqs,
                start=None
            )

            mot.append(motion_lat)
            mot_remb.append(motion_rope_emb)
        return mot, mot_remb


class AdaLayerNorm(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim) #52M*12=624M=0.6B参数
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, elementwise_affine=False)

    def forward(self, x, temb):
        temb = self.linear(F.silu(temb)) #（14，10240）
        shift, scale = temb.chunk(2, dim=1) #（14，5120）
        shift = shift[:, None, :] # shift.unsqueeze(1) 
        scale = scale[:, None, :] 
        x = self.norm(x) * (1 + scale) + shift # 保持默认缩放为1
        return x


class AudioInjector_WAN(nn.Module):

    def __init__(
        self,
        all_modules,
        all_modules_names,
        dim=2048,
        num_heads=32,
        inject_layer=[0, 27], #?
        enable_adain=False,
        adain_dim=2048,
    ):
        super().__init__()
        self.injected_block_id = {}
        audio_injector_id = 0
        for mod_name, mod in zip(all_modules_names, all_modules):
            if isinstance(mod, DiTBlock):
                for inject_id in inject_layer:
                    if f'transformer_blocks.{inject_id}' in mod_name:
                        self.injected_block_id[inject_id] = audio_injector_id
                        audio_injector_id += 1

        self.injector = nn.ModuleList([CrossAttention(
            dim=dim,
            num_heads=num_heads,
        ) for _ in range(audio_injector_id)])
        self.injector_pre_norm_feat = nn.ModuleList([nn.LayerNorm(
            dim,
            elementwise_affine=False,
            eps=1e-6,
        ) for _ in range(audio_injector_id)])
        self.injector_pre_norm_vec = nn.ModuleList([nn.LayerNorm(
            dim,
            elementwise_affine=False,
            eps=1e-6,
        ) for _ in range(audio_injector_id)])
        if enable_adain: # 直接通过修改视频特征的均值和方差来融入音频信息
            self.injector_adain_layers = nn.ModuleList([AdaLayerNorm(output_dim=dim * 2, embedding_dim=adain_dim) for _ in range(audio_injector_id)])

# 步骤 A: 加权平均
class CausalAudioEncoder(nn.Module):

    def __init__(self, dim=5120, num_layers=25, out_dim=2048, num_token=4, need_global=False):
        super().__init__()
        self.encoder = MotionEncoder_tc(in_dim=dim, hidden_dim=out_dim, num_heads=num_token, need_global=need_global)
        weight = torch.ones((1, num_layers, 1, 1)) * 0.01

        self.weights = torch.nn.Parameter(weight)
        self.act = torch.nn.SiLU()

    def forward(self, features):
        # features B * num_layers * dim * video_length
        weights = self.act(self.weights.to(device=features.device, dtype=features.dtype))
        weights_sum = weights.sum(dim=1, keepdims=True)
        # Weighted Average
        weighted_feat = ((features * weights) / weights_sum).sum(dim=1)  # b dim f
        weighted_feat = weighted_feat.permute(0, 2, 1)  # b f dim
        res = self.encoder(weighted_feat)  # b f n dim
        return res  # b f n dim


class WanS2VDiTBlock(DiTBlock):

    def forward(self, x, context, t_mod, seq_len_x, freqs):
        # 1. 计算调节参数
        # self.modulation 是可学习的参数
        # t_mod 是输入的时间嵌入= (1, 6, 2, 5120)
        # 这一步把时间信息变成了 6 份控制参数 (shift, scale, gate 各两份)
        t_mod = (self.modulation.unsqueeze(2).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        # t_mod[:, :, 0] for x, t_mod[:, :, 1] for other like ref, motion, etc.
        t_mod = [
            torch.cat([element[:, :, 0].expand(1, seq_len_x, x.shape[-1]), element[:, :, 1].expand(1, x.shape[1] - seq_len_x, x.shape[-1])], dim=1)
            for element in t_mod
        ]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = t_mod
        
        # 1. Modulate (AdaLN): 根据时间 t 调整数据的分布 (均值和方差) # ← 动态调制
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        # 2. Self-Attention: 视频里的像素和参考图的像素互相“看”一眼
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs)) #x:(1,18000,5120),freqs:(18000,40,64)
        
        x = x + self.cross_attn(self.norm3(x), context)
        
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp) # ← 动态调制
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        
        return x #(1, 18000, 5120)


class WanS2VModel(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        cond_dim: int,
        audio_dim: int,
        num_audio_token: int,
        # enable_adain: bool = True,
        enable_adain: bool = False,
        audio_inject_layers: list = [0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39],
        # audio_inject_layers: list = [24, 27, 30, 33, 36, 39],
        zero_timestep: bool = True,
        add_last_motion: bool = True,
        framepack_drop_mode: str = "padd",
        fuse_vae_embedding_in_latents: bool = True,
        require_vae_embedding: bool = False,
        seperated_timestep: bool = False,
        require_clip_embedding: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.enbale_adain = enable_adain
        self.add_last_motion = add_last_motion
        self.zero_timestep = zero_timestep
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.require_vae_embedding = require_vae_embedding
        self.seperated_timestep = seperated_timestep
        self.require_clip_embedding = require_clip_embedding
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size) # 将输入的视频 Latents（通常是 VAE 的输出）切分成 Patch（图块），类似于 Vision Transformer 的第一步
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.blocks = nn.ModuleList([WanS2VDiTBlock(False, dim, num_heads, ffn_dim, eps) for _ in range(num_layers)]) # 这是模型的主体，由多个 WanS2VDiTBlock 堆叠而成。每一层包含自注意力（Self-Attention）和前馈网络（FFN）。
        self.head = Head(dim, out_dim, patch_size, eps) # 最后一层，将 Transformer 的输出还原回 Latent 空间的大小;最终预测噪声的输出层
        self.freqs = torch.cat(precompute_freqs_cis_3d(dim // num_heads), dim=1)

        # self.cond_encoder = nn.Conv3d(cond_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.casual_audio_encoder = CausalAudioEncoder(dim=audio_dim, out_dim=dim, num_token=num_audio_token, need_global=enable_adain)
        all_modules, all_modules_names = torch_dfs(self.blocks, parent_name="root.transformer_blocks")
        self.audio_injector = AudioInjector_WAN(
            all_modules,
            all_modules_names,
            dim=dim,
            num_heads=num_heads,
            inject_layer=audio_inject_layers,
            enable_adain=enable_adain,
            adain_dim=dim,
        )
        self.trainable_cond_mask = nn.Embedding(3, dim)
        # self.frame_packer = FramePackMotioner(inner_dim=dim, num_heads=num_heads, zip_frame_buckets=[1, 2, 16], drop_mode=framepack_drop_mode)

    def patchify(self, x: torch.Tensor):
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x,
            'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2]
        )

    def process_motion_frame_pack(self, motion_latents, drop_motion_frames=False, add_last_motion=2):
        flattern_mot, mot_remb = self.frame_packer(motion_latents, add_last_motion)
        if drop_motion_frames:
            return [m[:, :0] for m in flattern_mot], [m[:, :0] for m in mot_remb]
        else:
            return flattern_mot, mot_remb

    def inject_motion(self, x, rope_embs, mask_input, motion_latents, drop_motion_frames=True, add_last_motion=2):
        # inject the motion frames token to the hidden states
        mot, mot_remb = self.process_motion_frame_pack(motion_latents, drop_motion_frames=drop_motion_frames, add_last_motion=add_last_motion)
        if len(mot) > 0:
            x = torch.cat([x, mot[0]], dim=1)
            rope_embs = torch.cat([rope_embs, mot_remb[0]], dim=1)
            mask_input = torch.cat(
                [mask_input, 2 * torch.ones([1, x.shape[1] - mask_input.shape[1]], device=mask_input.device, dtype=mask_input.dtype)], dim=1
            )
        return x, rope_embs, mask_input

    def after_transformer_block(self, block_idx, hidden_states, audio_emb_global, audio_emb, original_seq_len, use_unified_sequence_parallel=False):
        # return 
        '''
        hidden_states:[Batch, (Frames * H * W), Channels]
        audio_emb:[Batch, Frames, n_audio_tokens(论文中的t), Channels]
        '''
        if block_idx in self.audio_injector.injected_block_id.keys():
            audio_attn_id = self.audio_injector.injected_block_id[block_idx]
            num_frames = audio_emb.shape[1] # torch.Size([1, 14, 5, 5120])
            if use_unified_sequence_parallel:
                from xfuser.core.distributed import get_sp_group
                hidden_states = get_sp_group().all_gather(hidden_states, dim=1)

            # 1. 取出视频部分（去掉 Reference Image 等 Condition）
            input_hidden_states = hidden_states[:, :original_seq_len].clone()  # [b (f=14 h=40 w=30) c] = (1,16800,5120)
            
            # 2. 合并维度，强制对齐 (The Rearrange Trick)；n: 这里是 num_audio_token (比如 1200)
            # 原本模型认为它在处理 1 个包含 81 帧的视频。现在通过 Reshape，模型误以为它在处理 81 个只有 1 帧的短视频
            # 结果就是，在做 Attention (self.audio_injector) 时，第 5 帧的画面（作为 Query）只能去查询第 5 帧的音频（作为 Key/Value）。
            # 它物理上无法“看到”第 6 帧的音频，因为第 6 帧被隔离在 Batch 维度的其他位置了（Attention 不跨 Batch 计算）。
            # 效果: 在接下来的 Cross-Attention 中，第 5 帧的视频 Token (Query) 只能看到第 5 帧的音频 Token (Key/Value)，它看不见第 4 帧或第 6 帧的音频。这就实现了精准的逐帧音画同步。
            input_hidden_states = rearrange(input_hidden_states, "b (t n) c -> (b t) n c", t=num_frames) # (14,1200,5120)
            # audio_emb_global = rearrange(audio_emb_global, "b t n c -> (b t) n c") # ([1, 14, 1, 5120])->(14,1,5120)
            
            # AdaIN 注入 (使用全局音频特征),比如音量、音调,来调整视频特征的整体风格
            # adain_hidden_states = self.audio_injector.injector_adain_layers[audio_attn_id]( # 0.6B参数 
            #     input_hidden_states, 
            #     temb=audio_emb_global[:, 0] # 取出了那个唯一的 Token，形状 [14, 5120]
            # )
            # attn_hidden_states = adain_hidden_states
            attn_hidden_states = input_hidden_states

            audio_emb = rearrange(audio_emb, "b t n c -> (b t) n c", t=num_frames) #(14,5,5120)
            attn_audio_emb = audio_emb
            
            #3. Cross-Attention 注入 (使用序列音频特征（比如音素变化）来引导视频特征的细节（比如嘴型）。)
            # self.audio_injector.injector 是一个 ModuleList，包含可训练参数
            residual_out = self.audio_injector.injector[audio_attn_id](attn_hidden_states, attn_audio_emb) #1.2B参数
            residual_out = rearrange(residual_out, "(b t) n c -> b (t n) c", t=num_frames)
            # 残差连接 (Residual Connection): 原特征 + 音频带来的变化
            # 修改前：
            # hidden_states[:, :original_seq_len] = hidden_states[:, :original_seq_len] + residual_out # 将hidden_states的前original_seq_len=16800个token加上residual_out，然后保持其余部分不变

            # 修改后（使用非原地拼接,否则报in-place错误）：
            video_part = hidden_states[:, :original_seq_len] + residual_out # #(1,16800,5120)
            other_part = hidden_states[:, original_seq_len:] #(1,1200,5120)
            hidden_states = torch.cat([video_part, other_part], dim=1)
            
            if use_unified_sequence_parallel:
                from xfuser.core.distributed import get_sequence_parallel_world_size, get_sequence_parallel_rank
                hidden_states = torch.chunk(hidden_states, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
        return hidden_states

    def cal_audio_emb(self, audio_input, motion_frames=[73, 19]):
        audio_input = torch.cat([audio_input[..., 0:1].repeat(1, 1, 1, motion_frames[0]), audio_input], dim=-1)
        if self.enbale_adain == True:
            audio_emb_global, audio_emb = self.casual_audio_encoder(audio_input)
            audio_emb_global = audio_emb_global[:, motion_frames[1]:].clone()
            merged_audio_emb = audio_emb[:, motion_frames[1]:, :]
            return audio_emb_global, merged_audio_emb
        else:
            audio_emb = self.casual_audio_encoder(audio_input)
            merged_audio_emb = audio_emb[:, motion_frames[1]:, :]
            return None, merged_audio_emb


    def get_grid_sizes(self, grid_size_x, grid_size_ref):
        f, h, w = grid_size_x
        rf, rh, rw = grid_size_ref
        grid_sizes_x = torch.tensor([f, h, w], dtype=torch.long).unsqueeze(0)
        grid_sizes_x = [[torch.zeros_like(grid_sizes_x), grid_sizes_x, grid_sizes_x]]
        grid_sizes_ref = [[
            torch.tensor([30, 0, 0]).unsqueeze(0),
            torch.tensor([31, rh, rw]).unsqueeze(0),
            torch.tensor([1, rh, rw]).unsqueeze(0),
        ]]
        return grid_sizes_x + grid_sizes_ref

    def forward(
        self,
        latents,
        timestep,
        context,
        audio_input,
        motion_latents,
        pose_cond,
        use_gradient_checkpointing_offload=False,
        use_gradient_checkpointing=False
    ):
        origin_ref_latents = latents[:, :, 0:1] # 取出参考图 (16通道)
        x = latents[:, :, 1:] # 取出噪声 (16通道)

        # context embedding
        context = self.text_embedding(context)

        # audio encode
        audio_emb_global, merged_audio_emb = self.cal_audio_emb(audio_input)

        # x and pose_cond
        # pose_cond = torch.zeros_like(x) if pose_cond is None else pose_cond
        # 吃噪声 x (16通道)
        x, (f, h, w) = self.patchify(self.patch_embedding(x)) # + self.cond_encoder(pose_cond))  # torch.Size([1, 29120, 5120])
        seq_len_x = x.shape[1]

        # reference image
        # 吃参考图 origin_ref_latents (16通道)
        ref_latents, (rf, rh, rw) = self.patchify(self.patch_embedding(origin_ref_latents))  # torch.Size([1, 1456, 5120])
        grid_sizes = self.get_grid_sizes((f, h, w), (rf, rh, rw))
        # 在 Token 层面拼接，而不是通道层面
        x = torch.cat([x, ref_latents], dim=1)
        # mask
        mask = torch.cat([torch.zeros([1, seq_len_x]), torch.ones([1, ref_latents.shape[1]])], dim=1).to(torch.long).to(x.device)
        # freqs
        pre_compute_freqs = rope_precompute(
            x.detach().view(1, x.size(1), self.num_heads, self.dim // self.num_heads), grid_sizes, self.freqs, start=None
        )
        # motion
        # x, pre_compute_freqs, mask = self.inject_motion(x, pre_compute_freqs, mask, motion_latents, add_last_motion=2)

        x = x + self.trainable_cond_mask(mask).to(x.dtype)

        # t_mod
        timestep = torch.cat([timestep, torch.zeros([1], dtype=timestep.dtype, device=timestep.device)])
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim)).unsqueeze(2).transpose(0, 2)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block_id, block in enumerate(self.blocks):
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        context,
                        t_mod,
                        seq_len_x,
                        pre_compute_freqs[0],
                        use_reentrant=False,
                    )
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(lambda x: self.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                        x,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    context,
                    t_mod,
                    seq_len_x,
                    pre_compute_freqs[0],
                    use_reentrant=False,
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(lambda x: self.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                    x,
                    use_reentrant=False,
                )
            else:
                # 标准 DiT 计算
                x = block(x, context, t_mod, seq_len_x, pre_compute_freqs[0])
                # 音频注入 
                x = self.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)

        x = x[:, :seq_len_x]
        x = self.head(x, t[:-1]) # Head 还原到 latent 空间
        x = self.unpatchify(x, (f, h, w))
        # make compatible with wan video
        x = torch.cat([origin_ref_latents, x], dim=2)
        return x # 去噪后的视频 latents
