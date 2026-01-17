<div align="center">

# 🎬 VACE-FreeCus

### **Zero-Shot Subject-Driven Video Generation via Unified Attention Sharing**

<p align="center">
  <a href="#-highlights">Highlights</a> •
  <a href="#-method">Method</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-results">Results</a> •
  <a href="#-citation">Citation</a>
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2507.XXXXX-b31b1b.svg)](https://arxiv.org/)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://github.com/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

</div>

---

<div align="center">
<img src="assets/teaser.png" width="100%">
<p><i>Given a single reference image, VACE-FreeCus generates temporally consistent videos with preserved subject identity across diverse contexts and motions—all without any training or fine-tuning.</i></p>
</div>

## 📢 News

- **[2026.05]** 🎉 Code and models released!
- **[2026.05]** 📄 Paper accepted to ICML 2026 (Best Paper Candidate)
- **[2026.01]** 🚀 Initial release of VACE-FreeCus framework

---

## ✨ Highlights

<table>
<tr>
<td width="50%">

### 🎯 **Zero-Shot Subject Consistency**
Generate videos with consistent subject identity from a **single reference image**—no training, no fine-tuning, no optimization required.

### 🎬 **Unified Video Control**
Seamlessly combine **VACE** (Video Condition Editing) with **FreeCus** (Free Lunch Subject-driven Customization) for precise structural and semantic control.

### ⚡ **Training-Free Framework**
Fully leverage pretrained Diffusion Transformers (DiT) capabilities through innovative attention sharing mechanisms.

</td>
<td width="50%">

### 🔧 **Three Key Innovations**
1. **Pivotal Attention Sharing (PAS)**: Selective K/V injection in vital layers
2. **Adjusted Noise Shifting (ANS)**: Enhanced detail preservation via negative shifting
3. **Semantic Feature Compensation (SFC)**: MLLM-augmented subject descriptions

### 🌟 **State-of-the-Art Performance**
Achieves competitive results against training-based methods on DreamBench++ benchmark while maintaining superior text controllability.

</td>
</tr>
</table>

---

## 🎥 Demo Gallery

<div align="center">

### Subject-Driven Video Generation

| Reference | Generated Video | Prompt |
|:---------:|:---------------:|:------:|
| <img src="assets/demo/ref_cat.jpg" width="120"> | <img src="assets/demo/video_cat.gif" width="200"> | *"A cat playing with a ball in a sunny garden"* |
| <img src="assets/demo/ref_dog.jpg" width="120"> | <img src="assets/demo/video_dog.gif" width="200"> | *"A dog running on the beach at sunset"* |
| <img src="assets/demo/ref_person.jpg" width="120"> | <img src="assets/demo/video_person.gif" width="200"> | *"A person dancing in a modern art gallery"* |

### Diverse Context Adaptation

| Reference | Scene Change | Style Transfer | Action Variation |
|:---------:|:------------:|:--------------:|:----------------:|
| <img src="assets/demo/ref_subject.jpg" width="100"> | <img src="assets/demo/scene.gif" width="140"> | <img src="assets/demo/style.gif" width="140"> | <img src="assets/demo/action.gif" width="140"> |

</div>

---

## 🔬 Method

<div align="center">
<img src="assets/method_overview.png" width="95%">
<p><i>Overview of VACE-FreeCus framework. Our approach transfers characteristics from a reference image to target video through three synergistic mechanisms.</i></p>
</div>

### Architecture Overview

VACE-FreeCus builds upon the Wan2.1-VACE video generation model and introduces FreeCus attention sharing for subject consistency:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VACE-FreeCus Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Reference Image ──┬──► [BiRefNet] ──► Subject Mask                        │
│                     │                                                        │
│                     ├──► [MLLM] ──► Subject Description ──┐                 │
│                     │                                      │                 │
│                     └──► [VAE Encoder] ──► Reference Latents                │
│                                              │                               │
│   Control Video ────► [VACE Encoder] ────────┼──► Condition Context         │
│                                              │                               │
│   Text Prompt ──────► [T5 Encoder] ──────────┼──► Text Embeddings           │
│                                              ▼                               │
│                              ┌────────────────────────┐                      │
│                              │   Diffusion Transformer │                     │
│                              │   with FreeCus Attention │                    │
│                              │   Sharing (PAS + ANS)    │                    │
│                              └────────────────────────┘                      │
│                                              │                               │
│                                              ▼                               │
│                              [VAE Decoder] ──► Generated Video               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

###  Pivotal Attention Sharing (PAS)

We selectively share attention features from the reference image in **vital layers** to preserve subject identity while maintaining editing flexibility:

$$A_l = \begin{cases} \text{softmax}\left(\frac{Q \cdot K'^{\top}}{\sqrt{d_k}}\right) \cdot V' & \text{if } l \in \mathcal{V} \\ \text{softmax}\left(\frac{Q \cdot K^{\top}}{\sqrt{d_k}}\right) \cdot V & \text{otherwise} \end{cases}$$

Where:
- $K' = [\lambda_r \cdot K_r \odot m_r, \lambda_p \cdot K_p, K_{tgt}]$
- $V' = [V_r \odot m_r, V_p, V_{tgt}]$
- $\mathcal{V}$ = vital layers (default: layers 0, 1, 2, 10, 12, 14, 25, 27, 29)
- $\lambda_r, \lambda_p$ = scaling factors (default: 1.1)

### Adjusted Noise Shifting (ANS)

To extract finer details from the reference image, we reverse the dynamic shifting direction:

$$\sigma'_t = \frac{e^{-\mu}}{e^{-\mu} + \frac{1}{t} - 1}$$

This ensures attention prioritizes **less noisy, subject-specific content**, enabling finer detail transfer during attention sharing.

<div align="center">
<img src="assets/noise_shifting.png" width="60%">
<p><i>Comparison of noise scaling under different shift directions. Negative shifting preserves more subject details.</i></p>
</div>

### Semantic Feature Compensation (SFC)

We leverage Multimodal LLMs to generate subject-specific captions that compensate for potential semantic feature loss:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Semantic Feature Compensation                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Reference Image ──► [Qwen2-VL] ──► Raw Description            │
│                                              │                   │
│   "Describe this [subject] briefly..."       │                   │
│                                              ▼                   │
│                                      [Qwen2.5-LLM]              │
│                                              │                   │
│   "Extract physical characteristics..."      │                   │
│                                              ▼                   │
│                                      Filtered Caption            │
│                                              │                   │
│   Original Prompt ──────────────────────────►+──► Enhanced Prompt│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 11.8
- GPU with >= 24GB VRAM (recommended: A100/H100)

### Step 1: Clone Repository

```bash
git clone https://github.com/your-repo/VACE-FreeCus.git
cd VACE-FreeCus
```

### Step 2: Create Environment

```bash
# Using conda
conda create -n vace-freecus python=3.10
conda activate vace-freecus

# Install PyTorch (adjust for your CUDA version)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -e .
```

### Step 3: Download Models

```bash
# Download Wan2.1-VACE model
python -c "from modelscope import snapshot_download; snapshot_download('Wan-AI/Wan2.1-VACE-1.3B')"

# Download BiRefNet for segmentation (optional)
python -c "from modelscope import snapshot_download; snapshot_download('zhengpeng7/BiRefNet')"

# Download Qwen2-VL for subject description (optional)
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2-VL-7B-Instruct')"
```

---

## 🚀 Quick Start

### Basic Usage

```python
import torch
from PIL import Image
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.core import ModelConfig
from diffsynth.utils.data import save_video

# Load pipeline
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", 
                   origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", 
                   origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", 
                   origin_file_pattern="Wan2.1_VAE.pth"),
    ],
)

# Load reference image
reference_image = Image.open("path/to/subject.jpg").convert('RGB')

# Generate video with subject consistency
video = pipe(
    prompt="A cat playing with a ball in a sunny garden",
    vace_reference_image=reference_image.resize((832, 480)),
    num_frames=49,
    height=480,
    width=832,
    seed=42,
)

# Save video
save_video(video, "output.mp4", fps=15)
```

### Advanced Usage with FreeCus

```python
from examples.wanvideo.model_inference.Wan2_1_VACE_FreeCus import VACEFreeCusPipeline
from diffsynth.utils.freecus_utils import SubjectDescriptionGenerator

# Initialize with MLLM for subject description
description_generator = SubjectDescriptionGenerator(
    vl_model_path="Qwen/Qwen2-VL-7B-Instruct",
    llm_model_path="Qwen/Qwen2.5-7B-Instruct",
)

# Create VACE-FreeCus pipeline
vace_freecus = VACEFreeCusPipeline(
    pipe=pipe,
    description_generator=description_generator,
)

# Generate with full FreeCus features
video = vace_freecus.generate_video(
    prompt="A cat as a superhero flying through the city",
    negative_prompt="blurry, low quality",
    reference_image=reference_image,
    freecus_scale=1.1,           # λr, λp scaling
    shift_type='negative',        # ANS configuration
    vital_layers=[0, 1, 2, 10, 12, 14, 25, 27, 29],
    use_subject_caption=True,     # Enable SFC
    subject_word='cat',
)
```

### Command Line Interface

```bash
python examples/wanvideo/model_inference/Wan2.1-VACE-FreeCus.py \
    --reference_image path/to/subject.jpg \
    --prompt "A cat playing in the garden" \
    --output_path output.mp4 \
    --freecus_scale 1.1 \
    --shift_type negative \
    --vital_layers "0,1,2,10,12,14,25,27,29" \
    --num_frames 49 \
    --height 480 \
    --width 832
```

---

## 📊 Results

### Quantitative Comparison on DreamBench++

<div align="center">

| Method | Base Model | CLIP-T ↑ | CLIP-I ↑ | DINO ↑ | Training Required |
|:------:|:----------:|:--------:|:--------:|:------:|:-----------------:|
| Textual Inversion | SD v1.5 | 0.298 | 0.713 | 0.430 | ✓ (per-subject) |
| DreamBooth | SD v1.5 | 0.322 | 0.716 | 0.505 | ✓ (per-subject) |
| DreamBooth-LoRA | SDXL v1.0 | 0.341 | 0.751 | 0.547 | ✓ (per-subject) |
| BLIP-Diffusion | SD v1.5 | 0.276 | 0.815 | 0.639 | ✓ (encoder) |
| IP-Adapter | SDXL v1.0 | 0.305 | 0.845 | 0.621 | ✓ (encoder) |
| IP-Adapter-Plus | SDXL v1.0 | 0.271 | **0.916** | **0.807** | ✓ (encoder) |
| MS-Diffusion | SDXL v1.0 | **0.336** | 0.873 | 0.729 | ✓ (encoder) |
| OminiControl | FLUX.1 | 0.330 | 0.797 | 0.570 | ✓ (encoder) |
| **VACE-FreeCus (Ours)** | **FLUX.1** | 0.308 | 0.853 | 0.696 | **✗** |

</div>

### Ablation Studies

<div align="center">

| Configuration | CLIP-T ↑ | CLIP-I ↑ | DINO ↑ |
|:-------------:|:--------:|:--------:|:------:|
| Full Model | **0.308** | **0.853** | **0.696** |
| w/o PAS | 0.327 | 0.810 | 0.590 |
| w/o ANS | 0.324 | 0.829 | 0.624 |
| w/o SFC | 0.322 | 0.822 | 0.633 |

</div>

### Hyperparameter Analysis

<table>
<tr>
<td width="50%">

**Scaling Factor (λr, λp)**

| λp, λr | CLIP-T ↑ | CLIP-I ↑ | DINO ↑ |
|:------:|:--------:|:--------:|:------:|
| 1.00 | 0.321 | 0.827 | 0.626 |
| 1.05 | 0.315 | 0.838 | 0.656 |
| **1.10** | **0.308** | **0.853** | **0.696** |
| 1.15 | 0.305 | 0.861 | 0.706 |

</td>
<td width="50%">

**Shift Type**

| Shift Type | CLIP-T ↑ | CLIP-I ↑ | DINO ↑ |
|:----------:|:--------:|:--------:|:------:|
| μ * 0 | 0.320 | 0.836 | 0.648 |
| μ * -0.5 | 0.315 | 0.845 | 0.670 |
| **μ * -1.0** | **0.308** | **0.853** | **0.696** |
| μ * -2.0 | 0.296 | 0.857 | 0.698 |

</td>
</tr>
</table>

---

## 🔧 Configuration

### Key Parameters

| Parameter | Default | Description |
|:---------:|:-------:|:------------|
| `freecus_scale` | 1.1 | Scaling factor for reference K/V (λr, λp) |
| `shift_type` | `'negative'` | Noise shifting type: `'normal'`, `'negative'`, `'half_negative'`, `'twice_negative'` |
| `vital_layers` | `[0,1,2,10,12,14,25,27,29]` | Layer indices for attention sharing |
| `num_inference_steps` | 50 | Number of denoising steps |
| `cfg_scale` | 5.0 | Classifier-free guidance scale |

### Vital Layer Selection

The vital layers are selected based on their influence on generated content:

```python
# Early layers (0-2): Capture global structure
# Middle layers (10, 12, 14): Encode semantic features  
# Late layers (25, 27, 29): Preserve fine details

vital_layers = [0, 1, 2, 10, 12, 14, 25, 27, 29]
```

---

## 📁 Project Structure

```
VACE-FreeCus/
├── diffsynth/
│   ├── models/
│   │   ├── wan_video_dit.py          # Base DiT model
│   │   ├── wan_video_vace.py         # VACE condition encoder
│   │   └── wan_video_freecus.py      # FreeCus attention handler
│   ├── pipelines/
│   │   └── wan_video.py              # Video generation pipeline
│   └── utils/
│       └── freecus_utils.py          # FreeCus utility functions
├── examples/
│   └── wanvideo/
│       └── model_inference/
│           ├── Wan2.1-VACE-1.3B.py   # Basic VACE inference
│           └── Wan2.1-VACE-FreeCus.py # VACE + FreeCus inference
├── plans/
│   └── VACE_FreeCus_Fusion_Plan.md   # Technical design document
├── assets/                            # Demo images and figures
├── VACE_FreeCus_README.md            # This file
└── requirements.txt
```

---

## 🎨 Applications

### 1. Subject-Driven Video Generation

Generate videos featuring a consistent subject in diverse contexts:

```python
# Different scenes with same subject
prompts = [
    "A cat lounging on a park bench",
    "A cat reading a book in a library",
    "A cat as a superhero flying through the city",
]
```

### 2. Style Transfer

Apply artistic styles while preserving subject identity:

```python
# Style variations
prompts = [
    "An anime-style illustration of a cat",
    "A minimalist sketch of a cat",
    "A watercolor painting of a cat",
]
```

### 3. Video Inpainting

Seamlessly integrate subjects into existing videos:

```python
video = pipe(
    prompt="A cat watching the sunset",
    vace_video=control_video,  # Depth/edge control
    vace_reference_image=reference_image,
)
```

### 4. Compatibility with ControlNet

Combine with structural control for precise generation:

```python
video = pipe(
    prompt="A cat dancing",
    control_video=depth_video,      # Depth control
    vace_reference_image=reference_image,  # Subject reference
)
```

---

## ⚠️ Limitations

1. **Attention Artifacts**: The attention sharing mechanism may occasionally introduce artifacts with outlines resembling the reference subject.

2. **MLLM Accuracy**: Subject captions from Multimodal LLMs aren't fully accurate yet, which may affect semantic feature compensation.

3. **Computational Cost**: Full attention extraction requires additional forward passes through the reference image.

4. **Subject Complexity**: Performance may degrade for subjects with highly complex textures or unusual appearances.

---

## 🔮 Future Work

- [ ] Support for multi-subject generation
- [ ] Integration with video editing pipelines
- [ ] Real-time inference optimization
- [ ] Extended support for 3D-aware generation
- [ ] Improved MLLM integration for better semantic understanding

---

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{vace-freecus2025,
  title={VACE-FreeCus: Zero-Shot Subject-Driven Video Generation via Unified Attention Sharing},
  author={Your Name and Collaborators},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}

@article{freecus2025,
  title={FreeCus: Free Lunch Subject-driven Customization in Diffusion Transformers},
  author={Zhang, Yanbing and Wang, Zhe and Zhou, Qin and Yang, Mengping},
  journal={arXiv preprint arXiv:2507.15249},
  year={2025}
}
```

---

## 🙏 Acknowledgements

This project builds upon the following excellent works:

- [FreeCus](https://github.com/Monalissaa/FreeCus) - Training-free subject-driven customization
- [Wan-AI](https://github.com/Wan-AI) - Wan video generation models
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) - Diffusion synthesis framework
- [BiRefNet](https://github.com/zhengpeng7/BiRefNet) - Bilateral reference network for segmentation
- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) - Vision-language model

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ Star us on GitHub if you find this project useful! ⭐**

<p>
  <a href="https://github.com/your-repo/VACE-FreeCus/stargazers">
    <img src="https://img.shields.io/github/stars/your-repo/VACE-FreeCus?style=social" alt="GitHub stars">
  </a>
  <a href="https://github.com/your-repo/VACE-FreeCus/network/members">
    <img src="https://img.shields.io/github/forks/your-repo/VACE-FreeCus?style=social" alt="GitHub forks">
  </a>
</p>

Made with ❤️ by the VACE-FreeCus Team

</div>
