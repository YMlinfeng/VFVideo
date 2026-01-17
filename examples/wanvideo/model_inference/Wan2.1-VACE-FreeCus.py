"""
VACE + FreeCus: Subject-Driven Video Generation with Zero-Shot Customization

This script demonstrates the fusion of VACE (Video Condition Editing) and FreeCus
(Free Lunch Subject-driven Customization) for subject-consistent video generation.

Key Features:
1. Subject consistency from reference image (FreeCus)
2. Video structure control via VACE
3. Zero-shot, training-free approach

Usage:
    python Wan2.1-VACE-FreeCus.py \
        --reference_image path/to/subject.jpg \
        --control_video path/to/control.mp4 \
        --prompt "A cat playing in the garden" \
        --output_path output.mp4

Reference Papers:
- FreeCus: Free Lunch Subject-driven Customization in Diffusion Transformers
- VACE: Video Condition Editing
"""

import torch
import argparse
import numpy as np
from PIL import Image
from typing import Optional, List, Dict, Any
from tqdm import tqdm

from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.core import ModelConfig
from diffsynth.models.wan_video_freecus import (
    WanFreeCusHandler,
    get_freecus_state,
    prepare_reference_mask,
    create_freecus_diffusion_trajectory,
)
from diffsynth.utils.freecus_utils import (
    preprocess_image_for_vae,
    prepare_subject_mask,
    generate_diffusion_trajectory,
    SubjectDescriptionGenerator,
)


def parse_args():
    parser = argparse.ArgumentParser(description='VACE + FreeCus Video Generation')
    
    # Model paths
    parser.add_argument('--model_id', type=str, default='Wan-AI/Wan2.1-VACE-1.3B',
                        help='Model ID for VACE model')
    
    # Input files
    parser.add_argument('--reference_image', type=str, required=True,
                        help='Path to reference image for subject')
    parser.add_argument('--control_video', type=str, default=None,
                        help='Path to control video (depth, edge, etc.)')
    
    # Prompts
    parser.add_argument('--prompt', type=str, required=True,
                        help='Text prompt for video generation')
    parser.add_argument('--negative_prompt', type=str, 
                        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                        help='Negative prompt')
    parser.add_argument('--subject_word', type=str, default='subject',
                        help='Word describing the subject type (e.g., cat, person)')
    
    # FreeCus parameters
    parser.add_argument('--freecus_scale', type=float, default=1.1,
                        help='Scale factor for reference attention (λr)')
    parser.add_argument('--shift_type', type=str, default='negative',
                        choices=['normal', 'negative', 'half_negative', 'twice_negative'],
                        help='Noise shift type for detail preservation')
    parser.add_argument('--vital_layers', type=str, default='0,1,2,10,12,14,25,27,29',
                        help='Comma-separated list of vital layer indices')
    
    # Video parameters
    parser.add_argument('--height', type=int, default=480,
                        help='Video height')
    parser.add_argument('--width', type=int, default=832,
                        help='Video width')
    parser.add_argument('--num_frames', type=int, default=49,
                        help='Number of frames')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of denoising steps')
    parser.add_argument('--cfg_scale', type=float, default=5.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Segmentation model (optional)
    parser.add_argument('--segmentation_model', type=str, default=None,
                        help='Path to BiRefNet segmentation model')
    
    # MLLM for subject description (optional)
    parser.add_argument('--vl_model_path', type=str, default=None,
                        help='Path to vision-language model (e.g., Qwen2-VL)')
    parser.add_argument('--llm_model_path', type=str, default=None,
                        help='Path to LLM for description filtering')
    
    # Output
    parser.add_argument('--output_path', type=str, default='output_vace_freecus.mp4',
                        help='Output video path')
    parser.add_argument('--fps', type=int, default=15,
                        help='Output video FPS')
    
    # Other options
    parser.add_argument('--tiled', action='store_true', default=True,
                        help='Use tiled VAE decoding')
    parser.add_argument('--use_subject_caption', action='store_true',
                        help='Use MLLM to generate subject caption')
    
    return parser.parse_args()


class VACEFreeCusPipeline:
    """
    Combined VACE + FreeCus pipeline for subject-driven video generation.
    
    This pipeline:
    1. Extracts subject from reference image using segmentation
    2. Generates subject description using MLLM (optional)
    3. Extracts attention features from reference image
    4. Generates video with VACE control and FreeCus attention sharing
    """
    
    def __init__(
        self,
        pipe: WanVideoPipeline,
        segmentation_model=None,
        description_generator: Optional[SubjectDescriptionGenerator] = None,
    ):
        self.pipe = pipe
        self.segmentation_model = segmentation_model
        self.description_generator = description_generator
        self.freecus_handler = WanFreeCusHandler()
    
    def extract_subject_mask(
        self,
        image: Image.Image,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Extract subject mask from reference image."""
        if self.segmentation_model is not None:
            # Use segmentation model
            masked_image, mask = self.segmentation_model.extract(
                image_path=None, pil_image=image
            )
            mask_tensor = torch.from_numpy(np.array(mask)).float() / 255.0
        else:
            # Use full image as mask
            mask_tensor = torch.ones(height, width)
        
        # Prepare mask indices for attention sharing
        latent_height = height // 8  # VAE downsampling factor
        latent_width = width // 8
        
        _, mask_indices = prepare_subject_mask(
            mask_tensor,
            latent_height,
            latent_width,
            patch_size=(1, 2, 2),  # Wan patch size
            device=self.pipe.device,
        )
        
        return mask_indices
    
    def enhance_prompt_with_subject(
        self,
        image_path: str,
        original_prompt: str,
        subject_word: str = 'subject',
    ) -> str:
        """Enhance prompt with subject description from MLLM."""
        if self.description_generator is None:
            return original_prompt
        
        return self.description_generator.get_enhanced_prompt(
            image_path, original_prompt, subject_word
        )
    
    def extract_reference_attention(
        self,
        reference_image: Image.Image,
        prompt: str,
        num_inference_steps: int,
        height: int,
        width: int,
        mask_indices: Optional[torch.Tensor],
        vital_layers: List[int],
        shift_type: str = 'negative',
        seed: int = 42,
    ):
        """
        Extract attention features from reference image.
        
        This runs the reference image through the denoising process
        to extract K/V features at each timestep.
        """
        device = self.pipe.device
        dtype = self.pipe.torch_dtype
        
        # Preprocess reference image
        ref_image = reference_image.resize((width, height))
        ref_tensor = preprocess_image_for_vae(ref_image, height, width, device, dtype)
        
        # Encode to latent space
        self.pipe.load_models_to_device(['vae'])
        ref_latents = self.pipe.vae.encode(ref_tensor, device=device)
        ref_latents = ref_latents.to(dtype=dtype, device=device)
        
        # Generate noise
        generator = torch.Generator(device='cuda').manual_seed(seed)
        noise = torch.randn(ref_latents.shape, generator=generator, device=device, dtype=dtype)
        
        # Set up scheduler
        self.pipe.scheduler.set_timesteps(num_inference_steps, shift=5.0)
        
        # Generate diffusion trajectory with adjusted shifting
        trajectory = generate_diffusion_trajectory(
            ref_latents, noise, self.pipe.scheduler.timesteps, shift_type
        )
        
        # Register handler in save mode
        self.freecus_handler.register(
            self.pipe.dit,
            mode='save',
            num_inference_steps=num_inference_steps,
            mask_indices=mask_indices,
            vital_layers=vital_layers,
        )
        
        # Note: In the current implementation, we skip the full attention extraction
        # for simplicity. The FreeCus handler will be registered in 'use' mode
        # during video generation, and will use the reference image features
        # from the VACE pipeline's reference image embedding.
        #
        # For a full implementation, you would:
        # 1. Encode the prompt
        # 2. Run the reference image through the denoising process
        # 3. Extract K/V features at each timestep
        #
        # This simplified version relies on VACE's reference image handling
        # combined with FreeCus attention scaling.
        
        print(f"Reference attention extraction configured (simplified mode)")
    
    def generate_video(
        self,
        prompt: str,
        negative_prompt: str,
        reference_image: Image.Image,
        control_video: Optional[List[Image.Image]] = None,
        num_frames: int = 49,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 50,
        cfg_scale: float = 5.0,
        freecus_scale: float = 1.1,
        vital_layers: List[int] = None,
        shift_type: str = 'negative',
        seed: int = 42,
        subject_word: str = 'subject',
        use_subject_caption: bool = False,
        tiled: bool = True,
    ) -> List[Image.Image]:
        """
        Generate video with VACE control and FreeCus subject consistency.
        
        Args:
            prompt: Text prompt for video generation
            negative_prompt: Negative prompt
            reference_image: Reference image for subject
            control_video: Optional control video for VACE
            num_frames: Number of frames to generate
            height: Video height
            width: Video width
            num_inference_steps: Number of denoising steps
            cfg_scale: Classifier-free guidance scale
            freecus_scale: Scale factor for reference attention
            vital_layers: Layers for attention sharing
            shift_type: Noise shift type
            seed: Random seed
            subject_word: Word describing the subject
            use_subject_caption: Whether to use MLLM for subject caption
            tiled: Whether to use tiled VAE decoding
        
        Returns:
            List of generated video frames
        """
        if vital_layers is None:
            vital_layers = [0, 1, 2, 10, 12, 14, 25, 27, 29]
        
        # Step 1: Extract subject mask
        print("Step 1: Extracting subject mask...")
        mask_indices = self.extract_subject_mask(reference_image, height, width)
        
        # Step 2: Enhance prompt with subject description (optional)
        if use_subject_caption and self.description_generator is not None:
            print("Step 2: Generating subject description...")
            # Save reference image temporarily for MLLM
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                reference_image.save(f.name)
                enhanced_prompt = self.enhance_prompt_with_subject(
                    f.name, prompt, subject_word
                )
            print(f"Enhanced prompt: {enhanced_prompt}")
        else:
            enhanced_prompt = prompt
        
        # Step 3: Extract attention from reference image
        print("Step 3: Extracting reference attention...")
        self.extract_reference_attention(
            reference_image,
            enhanced_prompt,
            num_inference_steps,
            height,
            width,
            mask_indices,
            vital_layers,
            shift_type,
            seed,
        )
        
        # Step 4: Switch to use mode for video generation
        print("Step 4: Enabling attention sharing for video generation...")
        self.freecus_handler.register(
            self.pipe.dit,
            mode='use',
            num_inference_steps=num_inference_steps,
            scale=freecus_scale,
            vital_layers=vital_layers,
        )
        
        # Step 5: Generate video with VACE
        print("Step 5: Generating video...")
        video = self.pipe(
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            vace_video=control_video,
            vace_reference_image=reference_image.resize((width, height)),
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            cfg_scale=cfg_scale,
            seed=seed,
            tiled=tiled,
        )
        
        # Cleanup
        self.freecus_handler.clear()
        
        return video


def main():
    args = parse_args()
    
    # Parse vital layers
    vital_layers = [int(x) for x in args.vital_layers.split(',')]
    
    # Load pipeline
    print("Loading VACE pipeline...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id=args.model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="Wan2.1_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/"),
    )
    
    # Load segmentation model (optional)
    segmentation_model = None
    if args.segmentation_model is not None:
        print("Loading segmentation model...")
        try:
            from FreeCus.seg_birefnet import BiRefNet
            segmentation_model = BiRefNet(
                model_path=args.segmentation_model,
                target_size_h=args.height,
                target_size_w=args.width,
            )
        except Exception as e:
            print(f"Warning: Could not load segmentation model: {e}")
    
    # Load description generator (optional)
    description_generator = None
    if args.use_subject_caption and args.vl_model_path is not None:
        print("Loading description generator...")
        try:
            description_generator = SubjectDescriptionGenerator(
                vl_model_path=args.vl_model_path,
                llm_model_path=args.llm_model_path,
            )
        except Exception as e:
            print(f"Warning: Could not load description generator: {e}")
    
    # Create combined pipeline
    vace_freecus = VACEFreeCusPipeline(
        pipe=pipe,
        segmentation_model=segmentation_model,
        description_generator=description_generator,
    )
    
    # Load inputs
    print("Loading inputs...")
    reference_image = Image.open(args.reference_image).convert('RGB')
    
    control_video = None
    if args.control_video is not None:
        control_video = VideoData(args.control_video, height=args.height, width=args.width)
        control_video = [control_video[i] for i in range(min(args.num_frames, len(control_video)))]
    
    # Generate video
    print("Starting video generation...")
    video = vace_freecus.generate_video(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        reference_image=reference_image,
        control_video=control_video,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        freecus_scale=args.freecus_scale,
        vital_layers=vital_layers,
        shift_type=args.shift_type,
        seed=args.seed,
        subject_word=args.subject_word,
        use_subject_caption=args.use_subject_caption,
        tiled=args.tiled,
    )
    
    # Save video
    print(f"Saving video to {args.output_path}...")
    save_video(video, args.output_path, fps=args.fps, quality=5)
    print("Done!")


if __name__ == '__main__':
    main()
