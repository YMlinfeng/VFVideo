"""
FreeCus Attention Handler for Wan Video Models

This module implements the FreeCus (Free Lunch Subject-driven Customization) mechanism
adapted for Wan video generation models. It enables zero-shot subject-driven video
generation by sharing attention features from a reference image.

Key components:
1. Pivotal Attention Sharing (PAS) - Share K/V from reference image in vital layers
2. Adjustment of Noise Shifting (ANS) - Use negative shift for finer detail extraction
3. Semantic Features Compensation (SFC) - Augment with MLLM-derived descriptions

Reference: FreeCus: Free Lunch Subject-driven Customization in Diffusion Transformers
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, List, Dict, Tuple
from einops import rearrange


class WanFreeCusState:
    """Global state for FreeCus attention sharing."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.stored_kv: Dict[str, List[torch.Tensor]] = {'key': [], 'value': []}
        self.mask_indices: Optional[torch.Tensor] = None
        self.global_step: int = 0
        self.num_blocks: int = 0
        self.num_inference_steps: int = 50
        self.mode: str = 'disabled'  # 'disabled', 'save', 'use'
        self.vital_layers: List[int] = []
        self.scale: float = 1.1


# Global state instance
_freecus_state = WanFreeCusState()


def get_freecus_state() -> WanFreeCusState:
    """Get the global FreeCus state."""
    return _freecus_state


class WanFreeCusSelfAttention(nn.Module):
    """
    Modified SelfAttention module that supports FreeCus attention sharing.
    
    This wraps the original SelfAttention and adds:
    - K/V storage during 'save' mode
    - K/V injection during 'use' mode
    """
    
    def __init__(self, original_self_attn: nn.Module, layer_idx: int):
        super().__init__()
        self.original_attn = original_self_attn
        self.layer_idx = layer_idx
        
        # Copy attributes from original attention
        self.dim = original_self_attn.dim
        self.num_heads = original_self_attn.num_heads
        self.head_dim = original_self_attn.head_dim
        
        # Reference to original modules
        self.q = original_self_attn.q
        self.k = original_self_attn.k
        self.v = original_self_attn.v
        self.o = original_self_attn.o
        self.norm_q = original_self_attn.norm_q
        self.norm_k = original_self_attn.norm_k
        self.attn = original_self_attn.attn
    
    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        state = get_freecus_state()
        
        # Track block execution for step counting
        state.num_blocks += 1
        total_blocks = len(state.vital_layers) if state.vital_layers else 30  # Default Wan block count
        if state.num_blocks >= total_blocks:
            state.global_step += 1
            state.global_step %= state.num_inference_steps
            state.num_blocks = 0
        
        # Compute Q, K, V
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        
        # Apply RoPE
        from .wan_video_dit import rope_apply
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        
        if state.mode == 'save':
            # Save K/V for later use
            # Apply mask to extract only subject region
            if state.mask_indices is not None:
                # Extract subject region K/V
                k_subject = k[:, state.mask_indices, :]
                v_subject = v[:, state.mask_indices, :]
                state.stored_kv['key'].append(state.scale * k_subject.detach().clone())
                state.stored_kv['value'].append(v_subject.detach().clone())
            else:
                state.stored_kv['key'].append(state.scale * k.detach().clone())
                state.stored_kv['value'].append(v.detach().clone())
            
            # Normal attention
            x = self.attn(q, k, v)
            
        elif state.mode == 'use' and self.layer_idx in state.vital_layers:
            # Inject saved K/V
            # Calculate index into stored K/V
            vital_layer_idx = state.vital_layers.index(self.layer_idx)
            kv_index = state.global_step * len(state.vital_layers) + vital_layer_idx
            
            if kv_index < len(state.stored_kv['key']):
                ref_k = state.stored_kv['key'][kv_index].to(k.device, k.dtype)
                ref_v = state.stored_kv['value'][kv_index].to(v.device, v.dtype)
                
                # Expand reference K/V to match batch size if needed
                if ref_k.shape[0] != k.shape[0]:
                    ref_k = ref_k.expand(k.shape[0], -1, -1)
                    ref_v = ref_v.expand(v.shape[0], -1, -1)
                
                # Concatenate reference K/V with current K/V
                k_combined = torch.cat([ref_k, k], dim=1)
                v_combined = torch.cat([ref_v, v], dim=1)
                
                # Attention with combined K/V
                x = self.attn(q, k_combined, v_combined)
            else:
                # Fallback to normal attention if index out of range
                x = self.attn(q, k, v)
        else:
            # Normal attention
            x = self.attn(q, k, v)
        
        return self.o(x)


class WanFreeCusHandler:
    """
    Handler for FreeCus attention sharing in Wan video models.
    
    Usage:
        handler = WanFreeCusHandler()
        
        # Phase 1: Extract attention from reference image
        handler.register(dit, mode='save', mask_indices=subject_mask_indices,
                        num_inference_steps=50, vital_layers=[0,1,2,10,12,14,25,27,29])
        # Run inference on reference image...
        
        # Phase 2: Generate video with shared attention
        handler.register(dit, mode='use', scale=1.1)
        # Run video generation...
        
        # Cleanup
        handler.clear()
    """
    
    def __init__(self):
        self._original_attns: Dict[int, nn.Module] = {}
        self._registered_dit = None
    
    def register(
        self,
        dit: nn.Module,
        mode: str = 'save',
        num_inference_steps: int = 50,
        scale: float = 1.1,
        mask_indices: Optional[torch.Tensor] = None,
        vital_layers: Optional[List[int]] = None,
    ):
        """
        Register FreeCus attention sharing on a DiT model.
        
        Args:
            dit: The WanModel DiT to modify
            mode: 'save' to extract attention, 'use' to inject attention
            num_inference_steps: Number of denoising steps
            scale: Scale factor for reference K (λr in paper)
            mask_indices: Indices of subject region in flattened spatial dimension
            vital_layers: List of layer indices for attention sharing
        """
        state = get_freecus_state()
        
        # Default vital layers for Wan models (30 blocks)
        # Selected based on FreeCus paper's layer selection strategy
        if vital_layers is None:
            vital_layers = [0, 1, 2, 10, 12, 14, 25, 27, 29]
        
        state.mode = mode
        state.num_inference_steps = num_inference_steps
        state.scale = scale
        state.mask_indices = mask_indices
        state.vital_layers = vital_layers
        state.global_step = 0
        state.num_blocks = 0
        
        if mode == 'save':
            state.stored_kv = {'key': [], 'value': []}
        
        # Replace self_attn in vital layers with FreeCus version
        if self._registered_dit is not dit:
            # Restore previous modifications if any
            self._restore_original_attns()
            
            # Store original attentions and replace with FreeCus versions
            for idx in vital_layers:
                if idx < len(dit.blocks):
                    block = dit.blocks[idx]
                    self._original_attns[idx] = block.self_attn
                    block.self_attn = WanFreeCusSelfAttention(block.self_attn, idx)
            
            self._registered_dit = dit
    
    def _restore_original_attns(self):
        """Restore original attention modules."""
        if self._registered_dit is not None:
            for idx, original_attn in self._original_attns.items():
                if idx < len(self._registered_dit.blocks):
                    self._registered_dit.blocks[idx].self_attn = original_attn
            self._original_attns.clear()
            self._registered_dit = None
    
    def clear(self):
        """Clear stored K/V and restore original attention modules."""
        state = get_freecus_state()
        state.reset()
        self._restore_original_attns()
    
    def get_stored_kv_count(self) -> int:
        """Get the number of stored K/V pairs."""
        state = get_freecus_state()
        return len(state.stored_kv['key'])


def create_freecus_diffusion_trajectory(
    latents: torch.Tensor,
    noise: torch.Tensor,
    scheduler,
    num_inference_steps: int,
    shift_type: str = 'negative',
) -> List[torch.Tensor]:
    """
    Create diffusion trajectory for reference image with adjusted noise shifting.
    
    This implements the ANS (Adjustment of Noise Shifting) from FreeCus paper.
    Using negative shift allows extracting finer details from the reference image.
    
    Args:
        latents: Clean latents of reference image [B, C, T, H, W]
        noise: Random noise tensor with same shape as latents
        scheduler: FlowMatchScheduler instance
        num_inference_steps: Number of denoising steps
        shift_type: 'negative' for detail preservation, 'normal' for standard
    
    Returns:
        List of noisy latents at each timestep
    """
    trajectory = []
    
    # Get timesteps from scheduler
    timesteps = scheduler.timesteps
    
    for t in timesteps:
        # Calculate sigma based on shift type
        # For FlowMatch: z_t = (1 - sigma) * z_0 + sigma * noise
        sigma = t.item() / 1000.0  # Normalize to [0, 1]
        
        if shift_type == 'negative':
            # Reduce noise level to preserve more details
            # This is analogous to using -μ in Flux's dynamic shifting
            sigma = sigma * 0.7  # Reduce noise strength
        elif shift_type == 'half_negative':
            sigma = sigma * 0.85
        
        # Create noisy sample
        noisy_latent = (1 - sigma) * latents + sigma * noise
        trajectory.append(noisy_latent)
    
    return trajectory


def prepare_reference_mask(
    mask: torch.Tensor,
    target_height: int,
    target_width: int,
    num_frames: int = 1,
    patch_size: Tuple[int, int, int] = (1, 2, 2),
) -> torch.Tensor:
    """
    Prepare subject mask for attention sharing.
    
    Args:
        mask: Binary mask [H, W] or [1, 1, H, W]
        target_height: Target latent height
        target_width: Target latent width
        num_frames: Number of frames (1 for image)
        patch_size: DiT patch size (T, H, W)
    
    Returns:
        Indices of subject region in flattened spatial dimension
    """
    # Ensure mask is 4D
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
    
    # Resize mask to latent resolution
    mask = F.interpolate(
        mask.float(),
        size=(target_height, target_width),
        mode='bilinear',
        align_corners=False
    )
    
    # Account for patch size
    patch_h, patch_w = patch_size[1], patch_size[2]
    patched_h = target_height // patch_h
    patched_w = target_width // patch_w
    
    # Downsample to patch resolution
    mask = F.interpolate(
        mask,
        size=(patched_h, patched_w),
        mode='bilinear',
        align_corners=False
    )
    
    # Flatten and find non-zero indices
    mask_flat = mask.view(-1)
    indices = (mask_flat > 0.5).nonzero(as_tuple=True)[0]
    
    return indices


class FreeCusVideoProcessor:
    """
    High-level processor for FreeCus video generation.
    
    This class orchestrates the full FreeCus pipeline:
    1. Segment subject from reference image
    2. Extract attention features
    3. Generate video with attention sharing
    """
    
    def __init__(
        self,
        segmentation_model=None,
        mllm_model=None,
    ):
        """
        Initialize FreeCus processor.
        
        Args:
            segmentation_model: Model for subject segmentation (e.g., BiRefNet)
            mllm_model: Model for subject description (e.g., Qwen2-VL)
        """
        self.segmentation_model = segmentation_model
        self.mllm_model = mllm_model
        self.handler = WanFreeCusHandler()
    
    def extract_subject_mask(
        self,
        image,
        target_height: int,
        target_width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract subject mask from reference image.
        
        Args:
            image: PIL Image or path to image
            target_height: Target height for mask
            target_width: Target width for mask
        
        Returns:
            Tuple of (masked_image, mask_indices)
        """
        if self.segmentation_model is None:
            # Return full image mask if no segmentation model
            mask = torch.ones(1, 1, target_height, target_width)
            indices = torch.arange(target_height * target_width)
            return image, indices
        
        # Use segmentation model
        masked_image, mask = self.segmentation_model.extract(image)
        
        # Convert mask to tensor and prepare indices
        mask_tensor = torch.from_numpy(mask).float() / 255.0
        indices = prepare_reference_mask(
            mask_tensor,
            target_height // 8,  # VAE downsampling
            target_width // 8,
        )
        
        return masked_image, indices
    
    def generate_subject_caption(
        self,
        image_path: str,
        subject_word: str = 'subject',
        original_prompt: str = '',
    ) -> str:
        """
        Generate subject-specific caption using MLLM.
        
        Args:
            image_path: Path to reference image
            subject_word: Word describing the subject type
            original_prompt: Original generation prompt
        
        Returns:
            Enhanced prompt with subject description
        """
        if self.mllm_model is None:
            return original_prompt
        
        # Get description from MLLM
        description = self.mllm_model.get_filtered_description(
            original_prompt, subject_word
        )
        
        # Combine with original prompt
        enhanced_prompt = f"{original_prompt}. {description}"
        return enhanced_prompt
    
    def prepare_reference_attention(
        self,
        dit: nn.Module,
        vae: nn.Module,
        reference_image,
        scheduler,
        num_inference_steps: int = 50,
        mask_indices: Optional[torch.Tensor] = None,
        vital_layers: Optional[List[int]] = None,
        shift_type: str = 'negative',
        device: str = 'cuda',
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Extract attention features from reference image.
        
        This runs the reference image through the denoising process
        to extract K/V features at each timestep.
        
        Args:
            dit: WanModel DiT
            vae: WanVideoVAE
            reference_image: PIL Image or tensor
            scheduler: FlowMatchScheduler
            num_inference_steps: Number of denoising steps
            mask_indices: Subject mask indices
            vital_layers: Layers for attention sharing
            shift_type: Noise shift type ('negative' recommended)
            device: Computation device
            dtype: Computation dtype
        """
        from PIL import Image
        import numpy as np
        
        # Encode reference image
        if isinstance(reference_image, Image.Image):
            # Preprocess image
            image = reference_image.convert('RGB')
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            image = image.unsqueeze(2)  # Add time dimension [B, C, T, H, W]
        else:
            image = reference_image
        
        image = image.to(device=device, dtype=dtype)
        
        # Encode to latent space
        ref_latents = vae.encode(image, device=device)
        
        # Generate noise
        noise = torch.randn_like(ref_latents)
        
        # Create diffusion trajectory with adjusted shifting
        trajectory = create_freecus_diffusion_trajectory(
            ref_latents, noise, scheduler, num_inference_steps, shift_type
        )
        
        # Register handler in save mode
        self.handler.register(
            dit,
            mode='save',
            num_inference_steps=num_inference_steps,
            mask_indices=mask_indices,
            vital_layers=vital_layers,
        )
        
        # Run through denoising to extract attention
        # Note: This is a simplified version. In practice, you'd run
        # the full denoising loop with the trajectory
        
        return trajectory
    
    def enable_attention_sharing(
        self,
        dit: nn.Module,
        scale: float = 1.1,
        vital_layers: Optional[List[int]] = None,
    ):
        """
        Enable attention sharing for video generation.
        
        Call this after prepare_reference_attention and before video generation.
        
        Args:
            dit: WanModel DiT
            scale: Scale factor for reference attention
            vital_layers: Layers for attention sharing
        """
        state = get_freecus_state()
        self.handler.register(
            dit,
            mode='use',
            num_inference_steps=state.num_inference_steps,
            scale=scale,
            vital_layers=vital_layers or state.vital_layers,
        )
    
    def cleanup(self):
        """Clean up and restore original model state."""
        self.handler.clear()
