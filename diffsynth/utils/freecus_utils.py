"""
FreeCus Utility Functions for Wan Video Models

This module provides utility functions for FreeCus integration with Wan video models:
1. Noise shifting calculations
2. Diffusion trajectory generation
3. Subject mask processing
4. MLLM integration helpers

Reference: FreeCus: Free Lunch Subject-driven Customization in Diffusion Transformers
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Union
from PIL import Image


def calculate_flow_match_sigma(
    timestep: float,
    shift_type: str = 'normal',
    shift_factor: float = 1.0,
) -> float:
    """
    Calculate sigma for FlowMatch scheduler with optional shifting.
    
    In FlowMatch: z_t = (1 - sigma) * z_0 + sigma * noise
    
    Args:
        timestep: Current timestep (0-1000 scale)
        shift_type: 'normal', 'negative', 'half_negative', 'twice_negative'
        shift_factor: Additional scaling factor
    
    Returns:
        Adjusted sigma value
    """
    # Normalize timestep to [0, 1]
    sigma = timestep / 1000.0
    
    if shift_type == 'negative':
        # Reduce noise level significantly for detail preservation
        sigma = sigma * 0.7 * shift_factor
    elif shift_type == 'half_negative':
        sigma = sigma * 0.85 * shift_factor
    elif shift_type == 'twice_negative':
        sigma = sigma * 0.5 * shift_factor
    else:  # 'normal'
        sigma = sigma * shift_factor
    
    return sigma


def create_noisy_sample(
    clean_latents: torch.Tensor,
    noise: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """
    Create noisy sample using FlowMatch forward process.
    
    z_t = (1 - sigma) * z_0 + sigma * noise
    
    Args:
        clean_latents: Clean latent representation
        noise: Random noise tensor
        sigma: Noise level (0 = clean, 1 = pure noise)
    
    Returns:
        Noisy latent tensor
    """
    return (1 - sigma) * clean_latents + sigma * noise


def generate_diffusion_trajectory(
    clean_latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    shift_type: str = 'negative',
) -> List[torch.Tensor]:
    """
    Generate full diffusion trajectory for reference image.
    
    This creates noisy versions of the reference at each timestep,
    which are used to extract attention features.
    
    Args:
        clean_latents: Clean latent representation [B, C, T, H, W]
        noise: Random noise tensor with same shape
        timesteps: Tensor of timesteps from scheduler
        shift_type: Type of noise shifting
    
    Returns:
        List of noisy latents at each timestep
    """
    trajectory = []
    
    for t in timesteps:
        sigma = calculate_flow_match_sigma(t.item(), shift_type)
        noisy_latent = create_noisy_sample(clean_latents, noise, sigma)
        trajectory.append(noisy_latent)
    
    return trajectory


def preprocess_image_for_vae(
    image: Union[Image.Image, np.ndarray, torch.Tensor],
    target_height: int,
    target_width: int,
    device: str = 'cuda',
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Preprocess image for VAE encoding.
    
    Args:
        image: Input image (PIL, numpy, or tensor)
        target_height: Target height
        target_width: Target width
        device: Target device
        dtype: Target dtype
    
    Returns:
        Preprocessed tensor [B, C, T, H, W]
    """
    if isinstance(image, Image.Image):
        # Resize and convert to tensor
        image = image.convert('RGB').resize((target_width, target_height))
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # [C, H, W]
    elif isinstance(image, np.ndarray):
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        if image.dim() == 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
    
    # Ensure correct shape [B, C, T, H, W]
    if image.dim() == 3:
        image = image.unsqueeze(0).unsqueeze(2)  # [C, H, W] -> [B, C, T, H, W]
    elif image.dim() == 4:
        image = image.unsqueeze(2)  # [B, C, H, W] -> [B, C, T, H, W]
    
    return image.to(device=device, dtype=dtype)


def prepare_subject_mask(
    mask: Union[Image.Image, np.ndarray, torch.Tensor],
    latent_height: int,
    latent_width: int,
    patch_size: Tuple[int, int, int] = (1, 2, 2),
    threshold: float = 0.5,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare subject mask for attention sharing.
    
    Args:
        mask: Binary mask (PIL grayscale, numpy, or tensor)
        latent_height: Height in latent space
        latent_width: Width in latent space
        patch_size: DiT patch size (T, H, W)
        threshold: Binarization threshold
        device: Target device
    
    Returns:
        Tuple of (mask_tensor, mask_indices)
        - mask_tensor: Resized mask [1, 1, H, W]
        - mask_indices: Indices of subject region in flattened dimension
    """
    # Convert to tensor
    if isinstance(mask, Image.Image):
        mask = np.array(mask.convert('L')).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)
    elif isinstance(mask, np.ndarray):
        if mask.max() > 1.0:
            mask = mask.astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)
    
    # Ensure 4D shape [B, C, H, W]
    while mask.dim() < 4:
        mask = mask.unsqueeze(0)
    
    mask = mask.float().to(device)
    
    # Resize to latent resolution
    mask = F.interpolate(
        mask,
        size=(latent_height, latent_width),
        mode='bilinear',
        align_corners=False
    )
    
    # Account for patch size
    patch_h, patch_w = patch_size[1], patch_size[2]
    patched_h = latent_height // patch_h
    patched_w = latent_width // patch_w
    
    # Downsample to patch resolution
    mask_patched = F.interpolate(
        mask,
        size=(patched_h, patched_w),
        mode='bilinear',
        align_corners=False
    )
    
    # Flatten and find non-zero indices
    mask_flat = mask_patched.view(-1)
    indices = (mask_flat > threshold).nonzero(as_tuple=True)[0]
    
    return mask, indices


def expand_mask_to_video(
    mask_indices: torch.Tensor,
    num_frames: int,
    spatial_size: int,
) -> torch.Tensor:
    """
    Expand image mask indices to video dimensions.
    
    For video, we need to repeat the mask for each frame.
    
    Args:
        mask_indices: Indices for single frame
        num_frames: Number of video frames
        spatial_size: Spatial dimension size (H * W after patching)
    
    Returns:
        Expanded indices for all frames
    """
    all_indices = []
    for f in range(num_frames):
        frame_offset = f * spatial_size
        frame_indices = mask_indices + frame_offset
        all_indices.append(frame_indices)
    
    return torch.cat(all_indices)


class SubjectDescriptionGenerator:
    """
    Generate subject descriptions using Multimodal LLMs.
    
    This implements the SFC (Semantic Features Compensation) from FreeCus.
    """
    
    def __init__(
        self,
        vl_model_path: Optional[str] = None,
        llm_model_path: Optional[str] = None,
        device: str = 'cuda',
    ):
        """
        Initialize the description generator.
        
        Args:
            vl_model_path: Path to vision-language model (e.g., Qwen2-VL)
            llm_model_path: Path to LLM for filtering (e.g., Qwen2.5)
            device: Computation device
        """
        self.device = device
        self.vl_model = None
        self.llm_model = None
        self.vl_processor = None
        self.llm_tokenizer = None
        
        if vl_model_path is not None:
            self._load_vl_model(vl_model_path)
        if llm_model_path is not None:
            self._load_llm_model(llm_model_path)
    
    def _load_vl_model(self, model_path: str):
        """Load vision-language model."""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            self.vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype="auto", device_map=self.device
            )
            self.vl_processor = AutoProcessor.from_pretrained(model_path)
        except Exception as e:
            print(f"Warning: Could not load VL model: {e}")
    
    def _load_llm_model(self, model_path: str):
        """Load LLM for filtering."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype="auto", device_map=self.device
            )
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            print(f"Warning: Could not load LLM model: {e}")
    
    def generate_description(
        self,
        image_path: str,
        subject_word: str = 'subject',
        max_tokens: int = 20,
    ) -> str:
        """
        Generate brief description of subject in image.
        
        Args:
            image_path: Path to image
            subject_word: Word describing the subject type
            max_tokens: Maximum tokens in description
        
        Returns:
            Subject description string
        """
        if self.vl_model is None:
            return ""
        
        prompt = f"Describe this {subject_word} briefly and precisely in max {max_tokens} words, focusing on its overall appearance and key distinguishing features."
        
        try:
            from qwen_vl_utils import process_vision_info
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            text = self.vl_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.vl_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            
            generated_ids = self.vl_model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.vl_processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            return output_text
        except Exception as e:
            print(f"Warning: Description generation failed: {e}")
            return ""
    
    def filter_description(
        self,
        description: str,
        subject_word: str = 'subject',
    ) -> str:
        """
        Filter description to remove irrelevant information.
        
        Args:
            description: Raw description from VL model
            subject_word: Word describing the subject type
        
        Returns:
            Filtered description focusing on physical characteristics
        """
        if self.llm_model is None or not description:
            return description
        
        filter_prompt = (
            f"Please extract only the physical characteristics and features of the main {subject_word} "
            f"from this description, removing any information about actions, environment, background, "
            f"other subjects, or surrounding elements. Return only the extracted description without "
            f"any additional commentary. The description is: '{description}'"
        )
        
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": filter_prompt}
            ]
            
            text = self.llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.device)
            
            generated_ids = self.llm_model.generate(**model_inputs, max_new_tokens=512)
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.llm_tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            
            return response
        except Exception as e:
            print(f"Warning: Description filtering failed: {e}")
            return description
    
    def get_enhanced_prompt(
        self,
        image_path: str,
        original_prompt: str,
        subject_word: str = 'subject',
    ) -> str:
        """
        Get enhanced prompt with subject description.
        
        Args:
            image_path: Path to reference image
            original_prompt: Original generation prompt
            subject_word: Word describing the subject type
        
        Returns:
            Enhanced prompt combining original and subject description
        """
        description = self.generate_description(image_path, subject_word)
        filtered_description = self.filter_description(description, subject_word)
        
        if filtered_description:
            return f"{original_prompt}. {filtered_description}"
        return original_prompt


def compute_attention_similarity(
    query: torch.Tensor,
    key: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    """
    Compute attention similarity scores.
    
    Args:
        query: Query tensor [B, S, D]
        key: Key tensor [B, S, D]
        num_heads: Number of attention heads
    
    Returns:
        Attention scores [B, H, S, S]
    """
    from einops import rearrange
    
    head_dim = query.shape[-1] // num_heads
    
    q = rearrange(query, 'b s (h d) -> b h s d', h=num_heads)
    k = rearrange(key, 'b s (h d) -> b h s d', h=num_heads)
    
    scale = head_dim ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    return attn


def visualize_attention_map(
    attention: torch.Tensor,
    height: int,
    width: int,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Visualize attention map as heatmap.
    
    Args:
        attention: Attention tensor [B, H, S, S] or [S, S]
        height: Spatial height
        width: Spatial width
        save_path: Optional path to save visualization
    
    Returns:
        Visualization as numpy array
    """
    import matplotlib.pyplot as plt
    
    # Average over batch and heads if needed
    if attention.dim() == 4:
        attention = attention.mean(dim=(0, 1))
    elif attention.dim() == 3:
        attention = attention.mean(dim=0)
    
    # Take mean attention to each position
    attn_map = attention.mean(dim=0)
    
    # Reshape to spatial dimensions
    attn_map = attn_map[:height * width].view(height, width)
    attn_map = attn_map.cpu().numpy()
    
    # Normalize
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    
    if save_path:
        plt.figure(figsize=(8, 8))
        plt.imshow(attn_map, cmap='hot')
        plt.colorbar()
        plt.savefig(save_path)
        plt.close()
    
    return attn_map
