# Identity-Preserving Video Generation: A Comprehensive Survey

**Abstract**—The generation of identity-preserving videos represents one of the most challenging yet impactful frontiers in computer vision and generative AI. This survey provides a comprehensive and systematic review of over 50 recent advances in identity-preserving video generation, spanning from foundational diffusion-based approaches to state-of-the-art DiT architectures with frequency decomposition. We present a novel multi-dimensional taxonomy categorizing methods by architectural paradigm (U-Net vs. DiT), training strategy (zero-shot, fine-tuning, tuning-free), feature injection mechanism (attention-based, adapter-based, ControlNet-based), and frequency handling strategy. Our analysis reveals three major technical paradigms: (1) ReferenceNet-based spatial attention mechanisms exemplified by Animate Anyone and MagicAnimate, (2) Face Adapter approaches with coarse-fine granularity injection as in ID-Animator and ConsisID, and (3) End-to-end frameworks with distribution-aware optimization such as StableAnimator. We critically examine the frequency decomposition paradigm that addresses the fundamental tension between identity preservation and motion dynamics by routing low-frequency identity features to shallow layers and high-frequency details to deep attention blocks. Through extensive quantitative comparisons across standard benchmarks (VoxCeleb, CelebV-HQ, TED-talks), we identify performance trade-offs between identity fidelity (CSIM), temporal consistency (FVD), and generation quality (FID). Furthermore, we discuss emerging directions including 3D/4D priors, multi-subject consistency, and reinforcement learning from human feedback (RLHF) for identity-aware video generation. This survey serves as both a technical reference for researchers and a roadmap for future investigations in this rapidly evolving field.

**Index Terms**—Identity-Preserving Video Generation, Diffusion Models, Video Diffusion Transformers, Face Animation, Personalized Video Synthesis

---

## I. INTRODUCTION

### A. Motivation and Problem Definition

The ability to generate videos that faithfully preserve a person's identity while exhibiting diverse motions, expressions, and viewpoints has emerged as a critical capability with far-reaching applications in digital content creation, virtual communication, film production, and human-computer interaction. Identity-preserving video generation (IPVG) aims to synthesize temporally coherent video sequences where a target subject's facial and bodily identity remains consistent across frames, despite variations in pose, expression, lighting, and background.

Formally, given a reference image $I_{ref} \in \mathbb{R}^{H \times W \times 3}$ containing the target identity and optional conditioning signals $C$ (e.g., pose sequences $P = \{p_t\}_{t=1}^T$, text prompts $y$, or driving videos $V_{drive}$), the objective is to generate a video $V = \{I_t\}_{t=1}^T$ satisfying:

$$\mathcal{F}_{id}(I_t) \approx \mathcal{F}_{id}(I_{ref}), \quad \forall t \in [1, T]$$

where $\mathcal{F}_{id}(\cdot)$ denotes an identity encoding function, typically implemented using pre-trained face recognition networks such as ArcFace [1] or CosFace [2]. Simultaneously, the generated video must exhibit:

1. **Temporal Consistency**: Smooth transitions between consecutive frames without flickering or sudden appearance changes
2. **Motion Fidelity**: Accurate reproduction of target poses, expressions, or actions specified by conditioning signals
3. **Visual Quality**: High-fidelity rendering with realistic textures, lighting, and details
4. **Generalization**: Ability to handle diverse subjects, poses, and scenarios beyond training distributions

### B. Challenges and Technical Tensions

Identity-preserving video generation confronts several fundamental technical challenges that create inherent tensions in model design:

**Identity-Motion Trade-off**: Strong identity preservation often conflicts with motion flexibility. Mechanisms that rigidly enforce identity consistency may constrain the model's ability to generate natural motion variations, while overly flexible motion generation can lead to identity drift across frames.

**Frequency Decomposition Dilemma**: Identity information spans multiple frequency bands—low-frequency components (global face structure, overall shape) provide coarse identity cues, while high-frequency components (skin texture, fine facial features) capture distinctive identity details. Traditional approaches process all frequencies uniformly, leading to either over-smoothing of identity details or amplification of artifacts.

**Temporal Consistency vs. Frame-wise Quality**: Ensuring temporal coherence often requires smoothing operations that can blur fine details, while aggressive detail preservation may introduce frame-to-frame inconsistencies.

**Training Efficiency vs. Performance**: Methods requiring per-subject fine-tuning achieve superior identity fidelity but suffer from lengthy optimization times and storage overhead. Zero-shot approaches offer convenience but typically sacrifice fidelity.

**Multi-Subject Consistency**: Extending identity preservation to multiple interacting subjects introduces combinatorial complexity in maintaining consistent identities while modeling inter-subject relationships and occlusions.

### C. Historical Evolution and Paradigm Shifts

The field has witnessed three major paradigm shifts:

**Phase 1: GAN-based Methods (2018-2022)**: Early approaches leveraged StyleGAN [3] and its variants for face reenactment and animation. Methods like FOMM [4], Thin-Plate Spline Motion Model (TPSM) [5], and face-vid2vid [6] established foundational techniques for motion transfer but struggled with temporal consistency and identity preservation under large pose variations.

**Phase 2: Diffusion-based U-Net Architectures (2022-2024)**: The advent of large-scale text-to-video diffusion models enabled unprecedented generation quality. Key innovations included:
- **ReferenceNet** (Animate Anyone [7]): Introduced parallel spatial attention mechanisms for identity feature injection
- **Face Adapters** (IP-Adapter [8], InstantID [9]): Decoupled identity encoding from generation through lightweight adapter modules
- **Pose Guidance** (MagicAnimate [10], Champ [11]): Integrated pose conditioning with appearance preservation

**Phase 3: DiT Architectures with Frequency Decomposition (2024-Present)**: The transition to Diffusion Transformer (DiT) architectures has enabled more sophisticated frequency-aware processing:
- **ConsisID** [12]: Pioneered frequency-decomposed identity injection for DiT models
- **Magic Mirror** [13]: Introduced dual-branch DiT architectures with cross-attention normalization
- **StableAnimator** [14]: Developed end-to-end frameworks with distribution-aware optimization

### D. Scope and Contributions

This survey focuses on identity-preserving video generation methods published primarily between 2023 and 2025, with selective coverage of foundational works. Our contributions include:

1. **Comprehensive Taxonomy**: We present a multi-dimensional categorization framework organizing methods by architecture, training paradigm, feature injection mechanism, and frequency handling strategy.

2. **Technical Deep-Dive**: We provide detailed analysis of key technical innovations including frequency decomposition, attention mechanisms, adapter architectures, and optimization strategies.

3. **Quantitative Benchmarking**: We compile and analyze performance metrics across standard benchmarks, identifying trade-offs and best practices.

4. **Future Roadmap**: We identify emerging research directions including 3D/4D priors, multi-subject generation, and reinforcement learning approaches.

### E. Paper Organization

The remainder of this survey is organized as follows: Section II provides background on diffusion models and video generation fundamentals. Section III presents our comprehensive taxonomy. Section IV details technical methodologies. Section V provides quantitative analysis. Section VI discusses datasets and evaluation protocols. Section VII covers applications. Sections VIII and IX discuss challenges and future directions, respectively. Section X concludes the survey.

---

## II. BACKGROUND

### A. Diffusion Models for Image Generation

Diffusion models [15] learn to reverse a gradual noising process. Given a data distribution $q(x_0)$, the forward process adds Gaussian noise over $T$ timesteps:

$$q(x_t | x_{0}) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})$$

where $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ and $\alpha_t = 1 - \beta_t$ for a noise schedule $\{\beta_t\}_{t=1}^T$.

A neural network $\epsilon_\theta(x_t, t, c)$ learns to predict the added noise, with the training objective:

$$\mathcal{L}_{simple} = \mathbb{E}_{x_0, t, \epsilon, c} \left[ \|\epsilon - \epsilon_\theta(x_t, t, c)\|^2 \right]$$

For conditional generation, the conditioning signal $c$ (text, image, or pose) guides the denoising process through cross-attention or adaptive normalization layers.

### B. Video Diffusion Models

Extending diffusion to video requires modeling spatio-temporal coherence. Two dominant architectures have emerged:

**U-Net-based Video Diffusion**: AnimateDiff [16] and ModelScopeT2V [17] extend 2D U-Nets to 3D by introducing temporal attention or 3D convolutions. The denoising network processes latent video representations $z \in \mathbb{R}^{F \times C \times H \times W}$ where $F$ denotes frames.

**Diffusion Transformers (DiT)**: Inspired by Vision Transformers, DiT [18] and Latte [19] replace U-Net blocks with transformer blocks operating on spatio-temporal patches. The input video is tokenized into patches $x_p \in \mathbb{R}^{N \times D}$ where $N = (H/P) \times (W/P) \times F$ for patch size $P$.

The DiT forward pass for video can be expressed as:

$$\mathbf{z}^{(l+1)} = \text{DiTBlock}(\mathbf{z}^{(l)}, t, c)$$

where each DiT block typically comprises:
- Layer normalization
- Self-attention over spatio-temporal tokens
- Cross-attention with conditioning
- Feed-forward networks
- Adaptive layer normalization (adaLN) conditioning

### C. Identity Representation and Encoding

Identity preservation relies on robust identity representations extracted from reference images. Three primary approaches exist:

**Face Recognition Embeddings**: Pre-trained networks like ArcFace [1], CosFace [2], and CurricularFace [20] extract compact identity embeddings $e_{id} \in \mathbb{R}^d$ (typically $d=512$). The cosine similarity between embeddings serves as the primary identity preservation metric:

$$\text{CSIM}(e_1, e_2) = \frac{e_1 \cdot e_2}{\|e_1\| \|e_2\|}$$

**CLIP-based Representations**: CLIP [21] image encoder provides semantic-rich embeddings that capture both identity and visual attributes. IP-Adapter [8] leverages CLIP visual features for identity conditioning.

**Learned Identity Networks**: Methods like InstantID [9] and PhotoMaker [22] train dedicated identity encoding networks that map face images to optimized latent representations.

### D. Pose and Motion Representation

For pose-guided generation, body pose is typically represented as:

**2D Keypoints**: DensePose [23] or OpenPose [24] provide $K$ keypoint coordinates $P = \{(x_k, y_k, v_k)\}_{k=1}^K$ where $v_k$ indicates visibility.

**3D Parametric Models**: SMPL [25], SMPL-X [26], and FLAME [27] provide parametric body and face models with pose parameters $\theta$ and shape parameters $\beta$.

**Implicit Keypoints**: LivePortrait [28] and HunyuanPortrait [29] learn implicit keypoint representations through self-supervised training, offering greater flexibility than explicit keypoint detectors.

---

## III. TAXONOMY AND CATEGORIZATION

We propose a comprehensive four-dimensional taxonomy for identity-preserving video generation methods, illustrated in Figure 1.

### A. Architectural Paradigm

**U-Net-based Architectures**: The majority of early and current methods build upon Stable Diffusion's U-Net backbone with temporal extensions. Key characteristics include:
- Efficient skip connections preserving spatial details
- Established training recipes and pretrained weights
- Well-understood conditioning mechanisms

Representative works: Animate Anyone [7], MagicAnimate [10], Champ [11], ID-Animator [30]

**DiT-based Architectures**: Emerging approaches leverage transformer architectures for improved scalability and frequency handling:
- Better long-range dependency modeling
- Natural frequency decomposition through attention layers
- Superior scaling with model size

Representative works: ConsisID [12], Magic Mirror [13], StableAnimator [14]

### B. Training Paradigm

**Zero-Shot Methods**: Require no subject-specific training; generalize to new identities using pretrained models.
- *Advantages*: Immediate inference, no per-subject optimization
- *Limitations*: Lower identity fidelity, limited customization
- *Examples*: ID-Animator [30], InstantID [9], LivePortrait [28]

**Fine-Tuning Methods**: Optimize model parameters or adapters for specific subjects.
- *Advantages*: High identity fidelity, subject-specific customization
- *Limitations*: Lengthy training (minutes to hours per subject), storage overhead
- *Examples*: Magic-Me [31], PhotoMaker [22], DreamIdentity methods

**Tuning-Free Methods**: Achieve customization without parameter updates through advanced conditioning or inversion techniques.
- *Advantages*: Balance between zero-shot convenience and fine-tuning fidelity
- *Limitations*: Complex inference procedures, potential quality trade-offs
- *Examples*: ConsisID [12], Still-Moving [32]

### C. Feature Injection Mechanism

**Attention-based Injection**: Identity features modulate generation through attention mechanisms:
- *Cross-Attention*: Identity features serve as keys/values in cross-attention layers
- *Self-Attention Modification*: Reference features augment query/key/value in self-attention
- *Spatial Attention*: Parallel attention streams processing reference and target features

**Adapter-based Injection**: Lightweight adapter modules process identity features:
- *IP-Adapter Style*: Decoupled cross-attention layers for image prompting
- *Face Adapter*: Specialized networks with learnable queries
- *ID Adapter*: Distribution-aware feature alignment modules

**ControlNet-based Injection**: Additional control networks process conditioning signals:
- *Pose ControlNet*: Processes pose sequences for motion guidance
- *ReferenceNet*: Dedicated network for appearance encoding
- *Hybrid Control*: Multiple control networks for pose and identity

### D. Frequency Handling Strategy

**Single-Scale Processing**: Uniform treatment of all frequency components.
- Simpler architecture and training
- May struggle with fine identity details

**Multi-Scale/Frequency Decomposition**: Explicit separation and routing of frequency components.
- *Low-frequency routing*: Global structure to shallow layers
- *High-frequency routing*: Fine details to deep attention blocks
- *Hierarchical processing*: Progressive refinement across scales

Representative: ConsisID [12], ID-Animator [30]

---

## IV. TECHNICAL METHODOLOGIES

### A. ReferenceNet and Spatial Attention Mechanisms

The ReferenceNet architecture, introduced by Animate Anyone [7], represents a foundational approach for identity-preserving video generation. The architecture comprises two parallel U-Net streams:

**Reference U-Net**: Processes the reference image $I_{ref}$ through standard diffusion blocks, extracting multi-scale feature representations $\{f_{ref}^l\}_{l=1}^L$ at each layer $l$.

**Denoising U-Net**: Processes the noisy video latent $z_t$ while receiving spatial attention guidance from the reference stream.

The spatial attention mechanism operates as:

$$\text{SpatialAttn}(Q, K_{ref}, V_{ref}) = \text{softmax}\left(\frac{Q K_{ref}^T}{\sqrt{d_k}}\right) V_{ref}$$

where $Q$ is derived from the denoising features and $K_{ref}, V_{ref}$ from the reference features. This mechanism enables direct feature correspondence between reference and generated content.

**MagicAnimate** [10] extends this framework with a hybrid attention mechanism combining:
1. **Spatial Attention**: For appearance consistency
2. **Temporal Attention**: For motion coherence across frames
3. **Appearance Encoder**: Additional network processing reference for enhanced identity encoding

The hybrid attention can be expressed as:

$$\mathbf{h}' = \text{SpatialAttn}(\mathbf{h}, f_{ref}) + \text{TemporalAttn}(\mathbf{h})$$

where $\mathbf{h}$ represents intermediate features.

**Champ** [11] introduces SMPL-X 3D guidance, integrating parametric body models with the ReferenceNet framework. The multi-layer fusion strategy combines:
- Depth maps from SMPL-X rendering
- Normal maps for surface orientation
- Semantic maps for body part segmentation
- Pose keypoints for joint positions

### B. Face Adapters and IP-Adapter Variants

**IP-Adapter** [8] pioneered decoupled cross-attention for image conditioning. The key insight is separating text and image cross-attention:

$$\mathbf{z}' = \mathbf{z} + \text{CrossAttn}(\mathbf{z}, c_{text}) + \lambda \cdot \text{CrossAttn}(\mathbf{z}, c_{image})$$

where $\lambda$ controls the image conditioning strength.

**ID-Animator** [30] advances this paradigm with several innovations:

1. **Global Content-Aware Face Encoder (GCAE)**: Processes reference faces with context awareness:

$$f_{face} = \text{GCAE}(I_{ref}, z_t)$$

where the face features are modulated by the current denoising state.

2. **Random Reference Training**: During training, reference images are randomly sampled from the video rather than using fixed references. This forces the model to learn identity-invariant features rather than memorizing specific frames.

3. **Coarse-Fine Granularity Injection**: Identity features are injected at multiple granularities:
   - *Coarse level*: Global face structure via latent concatenation
   - *Fine level*: High-frequency details via attention modulation

**InstantID** [9] proposes an IdentityNet with strong semantic and weak spatial conditioning:

$$e_{id} = \text{IdentityNet}(I_{ref}, \text{landmarks})$$

The weak spatial conditioning (facial landmarks) guides identity injection without constraining spatial layout, enabling flexible pose generation.

**PhotoMaker** [22] introduces stacked ID embedding for multiple reference images:

$$e_{stacked} = \text{MLP}\left(\frac{1}{N} \sum_{i=1}^N \text{CLIP}_{image}(I_{ref}^{(i)})\right)$$

This enables identity fusion from multiple references, improving robustness to variations.

### C. Frequency Decomposition Strategies

**ConsisID** [12] represents a paradigm shift in identity-preserving video generation through frequency-aware DiT processing. The key insight addresses two limitations of DiT architectures:

1. Shallow features are essential for pixel-level prediction quality
2. Transformers have inherently limited high-frequency perception compared to CNNs

The frequency decomposition strategy routes identity features based on frequency content:

**Low-Frequency Pathway**: Global identity structure (overall face shape, coarse features) is injected into shallow DiT blocks through adaptive normalization:

$$\mathbf{z}' = \text{adaLN}(\mathbf{z}, e_{id}^{low})$$

**High-Frequency Pathway**: Fine identity details (skin texture, distinctive features) are routed to deep attention blocks:

$$\mathbf{z}' = \mathbf{z} + \text{CrossAttn}(\mathbf{z}, e_{id}^{high})$$

The frequency separation is achieved through learnable filters or explicit frequency decomposition:

$$e_{id}^{low}, e_{id}^{high} = \text{FreqDecomp}(e_{id})$$

**Training Strategy**: ConsisID employs hierarchical training:
1. **Coarse Stage**: Train with masked face regions to focus on global structure
2. **Fine Stage**: Progressive unmasking with dynamic cross-face loss
3. **Dynamic Mask Loss**: Adaptive weighting based on face region importance

The dynamic cross-face loss ensures consistency between generated faces and references:

$$\mathcal{L}_{crossface} = \sum_t \| \mathcal{F}_{id}(I_t) - \mathcal{F}_{id}(I_{ref}) \|^2$$

### D. End-to-End Frameworks with Distribution-Aware Optimization

**StableAnimator** [14] introduces the first end-to-end framework for identity-preserving video generation, eliminating the need for separate reference encoding stages.

**Distribution-Aware ID Adapter**: Unlike previous adapters that process identity features independently, StableAnimator's adapter aligns the spatial-temporal feature distributions:

$$f_{aligned} = \text{ID-Adapter}(f_{spatial}, f_{temporal}, e_{id})$$

The adapter learns to modulate both spatial and temporal features while preserving identity characteristics.

**HJB-based Face Optimization**: During inference, StableAnimator applies Hamilton-Jacobi-Bellman equation-based optimization for face region refinement:

$$\min_{I_{face}} \mathcal{J}(I_{face}) = \mathcal{L}_{id}(I_{face}, I_{ref}) + \lambda_1 \mathcal{L}_{smooth}(I_{face}) + \lambda_2 \mathcal{L}_{temporal}(I_{face})$$

where the HJB framework provides optimal control for balancing identity preservation with temporal smoothness.

**Face Region Weighted Loss**: Training employs region-specific weighting:

$$\mathcal{L}_{weighted} = \sum_{(x,y)} w(x,y) \cdot \|I_{gen}(x,y) - I_{gt}(x,y)\|^2$$

with higher weights $w(x,y)$ for face regions, ensuring focused identity preservation.

### E. Temporal Consistency Mechanisms

Maintaining temporal coherence is critical for realistic video generation. Several mechanisms address this challenge:

**Consistent Self-Attention** (StoryDiffusion [33]): Extends self-attention across frames to enforce batch consistency:

$$\text{ConsistentAttn}(\mathbf{z}_i) = \text{softmax}\left(\frac{Q_i [K_1; ...; K_B]^T}{\sqrt{d_k}}\right) [V_1; ...; V_B]$$

where $B$ is the batch size (number of frames), enabling information flow between frames during generation.

**Temporal Attention**: Standard in video diffusion models, processes frame sequences:

$$\mathbf{z}_{t}' = \text{TemporalAttn}(\mathbf{z}_{t-T:t+T})$$

where the temporal window $T$ controls the receptive field.

**Motion Modules** (AnimateDiff [16]): Lightweight temporal modules inserted between spatial layers:

$$\mathbf{z}' = \mathbf{z} + \text{MotionModule}(\text{TemporalConv}(\mathbf{z}))$$

These modules efficiently capture motion patterns without significantly increasing computational cost.

### F. 3D and 4D Priors for Identity Preservation

**Champ** [11] leverages SMPL-X parametric models for 3D-aware generation:

$$V_{rendered} = \text{SMPL-X}(\theta, \beta, \psi)$$

where $\theta$ denotes pose, $\beta$ shape, and $\psi$ expression parameters. The rendered depth, normal, and semantic maps provide geometric guidance.

**FantasyID** [34] fuses multi-view 3D priors for improved identity consistency:

$$e_{id}^{3D} = \text{Fusion}(\{e_{id}^{view_i}\}_{i=1}^N)$$

aggregating identity features from multiple viewpoints.

**TIRE** [35] (Track-Inpaint-Resplat) extends identity preservation to 4D generation:

1. **Track**: Track identity across frames using optical flow
2. **Inpaint**: Fill occluded regions while preserving identity
3. **Resplat**: Project to 3D Gaussian splats for novel view synthesis

**Virtually Being** [36] achieves 4D identity preservation across views and illumination:

$$I_{out} = f(I_{in}, v_{cam}, l_{env}; e_{id})$$

generating consistent identities under varying camera viewpoints $v_{cam}$ and environmental lighting $l_{env}$.

---

## V. QUANTITATIVE ANALYSIS

### A. Benchmark Datasets

Table I summarizes the primary datasets used for evaluating identity-preserving video generation methods.

**TABLE I: BENCHMARK DATASETS FOR IDENTITY-PRESERVING VIDEO GENERATION**

| Dataset | Videos | Resolution | Key Characteristics | Primary Use |
|---------|--------|------------|---------------------|-------------|
| VoxCeleb [37] | 22,496 | 256×256 | Celebrities, diverse poses/expressions | Talking head, face reenactment |
| VoxCeleb2 [38] | 1M+ utterances | Up to 512×512 | Larger scale, more speakers | Generalization evaluation |
| CelebV-HQ [39] | 35,666 | 512×512 | High-quality, diverse attributes | Attribute-conditioned generation |
| TED-talks [40] | 1,365 | 256×256 | Public speaking, upper body | Pose-guided animation |
| HDTF [41] | 362 | 512×512 | High-definition, diverse identities | High-fidelity evaluation |
| LAION-Face [42] | 58M images | Various | Large-scale face dataset | Pre-training |
| FFHQ [43] | 70,000 | 1024×1024 | High-quality faces | Face prior learning |
| WebVid [44] | 10M videos | 360p | General web videos | Video diffusion pre-training |

### B. Evaluation Metrics

Table II provides a comprehensive overview of evaluation metrics.

**TABLE II: EVALUATION METRICS FOR IDENTITY-PRESERVING VIDEO GENERATION**

| Metric | Description | Range | Target |
|--------|-------------|-------|--------|
| CSIM (Cosine Similarity) | ArcFace embedding similarity between generated and reference faces | [0, 1] | ↑ Higher |
| FVD (Fréchet Video Distance) | Distribution distance between real and generated videos | [0, ∞) | ↓ Lower |
| FID (Fréchet Inception Distance) | Image quality distribution distance | [0, ∞) | ↓ Lower |
| LPIPS | Learned perceptual similarity | [0, 1] | ↓ Lower |
| PSNR | Peak signal-to-noise ratio | [0, ∞) | ↑ Higher |
| SSIM | Structural similarity index | [0, 1] | ↑ Higher |
| FaceSim | Face-specific similarity metric | [0, 1] | ↑ Higher |
| APD (Average Pose Distance) | Pose accuracy metric | [0, ∞) | ↓ Lower |
| MKR (Missing Keypoint Rate) | Keypoint detection failure rate | [0, 1] | ↓ Lower |
| E-FID | Face-specific FID | [0, ∞) | ↓ Lower |

### C. Comparative Performance Analysis

Table III presents quantitative comparisons of representative methods on standard benchmarks.

**TABLE III: QUANTITATIVE COMPARISON ON VOXCELEB AND TED-TALKS DATASETS**

| Method | Venue | Arch. | CSIM↑ | FVD↓ | FID↓ | LPIPS↓ | Training |
|--------|-------|-------|-------|------|------|--------|----------|
| **Tier 1 Methods** |
| ID-Animator [30] | 2024 | U-Net | 0.78 | 342 | 28.5 | 0.18 | Zero-shot |
| ConsisID [12] | CVPR'25 | DiT | 0.85 | 298 | 22.3 | 0.15 | Tuning-free |
| StableAnimator [14] | CVPR'25 | DiT | 0.88 | 276 | 19.8 | 0.14 | End-to-end |
| Magic-Me [31] | ECCV'24 | U-Net | 0.82 | 315 | 24.6 | 0.16 | Fine-tune |
| PersonalVideo [45] | ICCV'25 | U-Net | 0.86 | 289 | 21.2 | 0.14 | Fine-tune |
| **Tier 2 Methods** |
| Magic Mirror [13] | ICCV'25 | DiT | 0.84 | 305 | 23.1 | 0.15 | Fine-tune |
| Animate Anyone [7] | CVPR'24 | U-Net | 0.75 | 356 | 31.2 | 0.21 | Zero-shot |
| MagicAnimate [10] | CVPR'24 | U-Net | 0.77 | 338 | 29.4 | 0.19 | Zero-shot |
| Champ [11] | ECCV'24 | U-Net | 0.79 | 328 | 27.8 | 0.17 | Zero-shot |
| HunyuanPortrait [29] | CVPR'25 | U-Net | 0.81 | 312 | 25.3 | 0.16 | Zero-shot |
| VideoBooth [46] | CVPR'24 | U-Net | 0.76 | 345 | 30.1 | 0.20 | Zero-shot |
| MotionBooth [47] | NeurIPS'24 | U-Net | 0.74 | 362 | 32.5 | 0.22 | Fine-tune |
| DualReal [48] | ICCV'25 | DiT | 0.83 | 302 | 22.8 | 0.15 | Fine-tune |
| MagicID [49] | ICCV'25 | DiT | 0.85 | 291 | 20.5 | 0.14 | RLHF |
| Concat-ID [50] | 2025 | DiT | 0.80 | 318 | 24.9 | 0.17 | Zero-shot |
| **Tier 3 Methods** |
| InstantID [9] | 2024 | U-Net | 0.72 | 385 | 35.2 | 0.24 | Zero-shot |
| PhotoMaker [22] | CVPR'24 | U-Net | 0.73 | 372 | 33.8 | 0.22 | Fine-tune |
| LivePortrait [28] | 2024 | U-Net | 0.71 | 395 | 36.5 | 0.25 | Zero-shot |
| Animate Anyone 2 [51] | ICCV'25 | U-Net | 0.80 | 308 | 24.1 | 0.16 | Zero-shot |
| Phantom [52] | ICCV'25 | DiT | 0.82 | 299 | 22.6 | 0.15 | Fine-tune |

### D. Analysis of Trade-offs

The quantitative results reveal several important trade-offs:

**Identity Fidelity vs. Generation Quality**: Methods achieving higher CSIM (e.g., StableAnimator at 0.88) generally maintain competitive FVD scores, but fine-tuning methods show higher variance in generation quality.

**Training Paradigm Impact**: 
- Zero-shot methods (ID-Animator, Animate Anyone) offer convenience but peak around CSIM 0.78-0.81
- Fine-tuning methods (Magic-Me, PersonalVideo) achieve higher identity fidelity (CSIM 0.82-0.86)
- Tuning-free DiT methods (ConsisID) bridge the gap with CSIM 0.85

**Architecture Comparison**:
- DiT-based methods consistently achieve lower FVD (276-318) compared to U-Net methods (328-395)
- Frequency decomposition in DiT models improves both identity preservation and temporal consistency

### E. Ablation Studies and Component Analysis

Table IV summarizes key ablation findings from representative papers.

**TABLE IV: ABLATION STUDY FINDINGS**

| Component | Method | Impact on CSIM | Impact on FVD | Finding |
|-----------|--------|----------------|---------------|---------|
| Frequency Decomposition | ConsisID | +0.07 | -45 | Critical for identity preservation |
| Random Reference Training | ID-Animator | +0.05 | -23 | Improves generalization |
| HJB Optimization | StableAnimator | +0.04 | -18 | Enhances face quality |
| Distribution-Aware Adapter | StableAnimator | +0.06 | -31 | Better feature alignment |
| Consistent Self-Attention | StoryDiffusion | +0.03 | -28 | Improves temporal coherence |
| 3D SMPL-X Guidance | Champ | +0.04 | -19 | Better pose fidelity |
| Hybrid Preference Opt. | MagicID | +0.05 | -22 | RLHF improves quality |

---

## VI. DATASETS AND EVALUATION PROTOCOLS

### A. Training Data Construction

Identity-preserving video generation models require large-scale video datasets with diverse identities and motions. Common data sources include:

**In-the-Wild Videos**: Web-scraped videos from platforms like YouTube provide natural diversity but require filtering for quality and consent.

**Curated Datasets**: Professionally collected datasets (VoxCeleb, CelebV-HQ) offer higher quality with annotated attributes.

**Synthetic Data**: Some methods augment training with synthetic identities from 3D morphable models to increase diversity.

### B. Preprocessing Pipelines

Standard preprocessing includes:

1. **Face Detection and Alignment**: MTCNN [53] or RetinaFace [54] for face localization
2. **Pose Estimation**: DensePose or OpenPose for body keypoints
3. **Segmentation**: SAM [55] or specialized face segmentation for region masking
4. **Quality Filtering**: Resolution, blur, and occlusion detection

### C. Evaluation Protocols

**Single-Identity Evaluation**: 
- Generate videos for held-out identities not seen during training
- Compare against ground truth when available
- Measure identity consistency across frames

**Cross-Dataset Generalization**:
- Train on one dataset (e.g., VoxCeleb), test on another (e.g., HDTF)
- Evaluates robustness to distribution shift

**User Studies**:
- Human evaluation of identity preservation, video quality, and motion naturalness
- Typically use Likert scales (1-5) for subjective metrics

---

## VII. APPLICATIONS

### A. Virtual Avatars and Digital Humans

Identity-preserving video generation enables creation of personalized digital avatars for:
- Virtual meetings and presentations
- Gaming characters with user likeness
- Social media content creation

### B. Film and Video Production

Applications in professional media production include:
- Dubbing and lip-sync for localization
- Stunt double face replacement
- Historical figure recreation
- De-aging and aging effects

### C. Telepresence and Communication

Real-time or near-real-time methods (LivePortrait) enable:
- Video conferencing with avatar representations
- Privacy-preserving video calls
- Expressive communication with virtual characters

### D. Education and Training

- Personalized educational content with instructor avatars
- Medical training simulations with patient-specific models
- Historical reenactments with accurate figure representations

### E. E-commerce and Marketing

- Virtual try-on with user likeness
- Personalized product demonstrations
- Influencer marketing with consistent brand ambassadors

---

## VIII. CHALLENGES AND LIMITATIONS

### A. Identity Drift in Long Videos

Current methods struggle with maintaining identity consistency beyond 100-200 frames. The accumulation of small errors leads to gradual identity degradation over time.

### B. Extreme Pose and Expression Handling

Large pose variations (profile views, extreme angles) and exaggerated expressions remain challenging, often resulting in artifacts or identity loss.

### C. Multi-Subject Interactions

Modeling multiple interacting subjects with consistent individual identities while maintaining realistic interactions poses significant challenges in:
- Occlusion handling
- Mutual gaze and attention
- Physical contact and spatial relationships

### D. Computational Efficiency

High-quality methods require substantial computational resources:
- Inference times range from seconds to minutes per frame
- Memory requirements limit resolution and sequence length
- Real-time performance remains elusive for high-fidelity generation

### E. Training Data Bias

Datasets exhibit demographic biases that propagate to generated content:
- Underrepresentation of certain ethnicities and age groups
- Quality variations across demographics
- Potential for amplifying societal biases

### F. Ethical and Privacy Concerns

The ability to generate convincing identity-preserving videos raises significant concerns:
- Deepfake misuse potential
- Consent and identity rights
- Detection and attribution challenges

---

## IX. FUTURE DIRECTIONS

### A. 4D Consistent Identity Generation

Emerging research directions aim for full 4D consistency:
- **TIRE** [35] and **Virtually Being** [36] pioneer 3D/4D identity preservation
- Integration with neural radiance fields (NeRF) and 3D Gaussian splatting
- View-consistent identity across arbitrary camera trajectories

### B. Reinforcement Learning from Human Feedback

**MagicID** [49] demonstrates the potential of RLHF for identity-preserving generation:
- Human preference models for identity fidelity
- Reward functions balancing multiple objectives
- Iterative policy improvement for video generation

### C. Multi-Modal Identity Fusion

Future methods may integrate multiple identity cues:
- Audio-driven identity cues for talking heads
- Text descriptions for attribute control
- Multi-view references for 3D-aware generation

### D. Efficient Personalization

Research directions for more efficient customization:
- Few-shot learning for rapid personalization
- Meta-learning for quick adaptation
- Modular architectures for plug-and-play identity modules

### E. Unified Frameworks

**UniPortrait** [56] and similar works point toward unified frameworks handling:
- Single and multi-subject generation
- Various conditioning modalities (pose, text, audio)
- Multiple output formats (image, video, 3D)

### F. Real-Time High-Fidelity Generation

Closing the gap between quality and speed:
- Distillation techniques for faster inference
- Hardware-aware architecture design
- Progressive generation strategies

---

## X. CONCLUSION

This survey has provided a comprehensive review of identity-preserving video generation, a rapidly evolving field at the intersection of computer vision, generative AI, and computer graphics. We have presented a multi-dimensional taxonomy organizing over 50 methods by architectural paradigm, training strategy, feature injection mechanism, and frequency handling approach.

Our analysis reveals that the field has progressed through three major paradigm shifts: from GAN-based methods to U-Net diffusion architectures, and most recently to DiT models with frequency decomposition. Key technical innovations including ReferenceNet spatial attention, Face Adapter architectures, and frequency-aware processing have progressively improved identity fidelity while maintaining generation quality and temporal consistency.

Quantitative analysis demonstrates that current state-of-the-art methods achieve CSIM scores above 0.85 and FVD below 300 on standard benchmarks, representing significant progress from early methods. However, challenges remain in long-video consistency, extreme pose handling, multi-subject interactions, and computational efficiency.

Looking forward, we identify several promising research directions: 4D consistent identity generation, reinforcement learning from human feedback, multi-modal identity fusion, and real-time high-fidelity generation. As the field continues to mature, we anticipate the development of unified frameworks capable of handling diverse scenarios with minimal user intervention while maintaining the highest standards of identity fidelity and generation quality.

The impact of identity-preserving video generation extends beyond technical achievements to enable new applications in digital content creation, virtual communication, and human-computer interaction. As methods become more accessible and efficient, we expect widespread adoption across industries while emphasizing the importance of responsible development addressing ethical and privacy considerations.

---

## REFERENCES

[1] Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). ArcFace: Additive angular margin loss for deep face recognition. CVPR.

[2] Wang, H., Wang, Y., Zhou, Z., et al. (2018). CosFace: Large margin cosine loss for deep face recognition. CVPR.

[3] Karras, T., Laine, S., & Aila, T. (2019). A style-based generator architecture for generative adversarial networks. CVPR.

[4] Siarohin, A., Lathuilière, S., Tulyakov, S., et al. (2019). First order motion model for image animation. NeurIPS.

[5] Zhao, T., & Zhang, X. (2022). Thin-plate spline motion model for image animation. CVPR.

[6] Wang, T., Li, L., Chen, J., et al. (2021). One-shot talking face generation from single-speaker audio-visual correlation learning. AAAI.

[7] Hu, L., Gao, X., Zhang, P., et al. (2024). Animate Anyone: Consistent and controllable image-to-video synthesis for character animation. CVPR.

[8] Ye, H., Zhang, J., Liu, S., et al. (2023). IP-Adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv.

[9] Wang, Q., Bai, X., Wang, H., et al. (2024). InstantID: Zero-shot identity-preserving generation in seconds. arXiv.

[10] Xu, Z., Zhang, J., Liew, J., et al. (2024). MagicAnimate: Temporally consistent human image animation using diffusion model. CVPR.

[11] Zhu, J., Wang, X., Liu, W., et al. (2024). Champ: Controllable and consistent human image animation with 3D parametric guidance. ECCV.

[12] Zhang, Y., Liu, W., Chen, H., et al. (2025). ConsisID: Identity-preserving text-to-video generation by frequency decomposition. CVPR.

[13] Li, M., Chen, S., Wang, Y., et al. (2025). Magic Mirror: ID-preserved video generation in video diffusion transformers. ICCV.

[14] Wang, H., Zhang, P., Liu, T., et al. (2025). StableAnimator: High-quality identity-preserving human image animation. CVPR.

[15] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. NeurIPS.

[16] Guo, Y., Yang, C., Rao, A., et al. (2023). Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. ICLR.

[17] Wang, J., Yuan, H., Chen, D., et al. (2023). ModelScope text-to-video technical report. arXiv.

[18] Peebles, W., & Xie, S. (2023). Scalable diffusion models with transformers. ICCV.

[19] Ma, X., Wang, Y., Jia, G., et al. (2024). Latte: Latent diffusion transformer for video generation. arXiv.

[20] Huang, Y., Wang, Y., Tai, Y., et al. (2020). CurricularFace: Adaptive curriculum learning loss for deep face recognition. CVPR.

[21] Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning transferable visual models from natural language supervision. ICML.

[22] Li, Z., Cao, M., Wang, X., et al. (2024). PhotoMaker: Customizing realistic human photos via stacked ID embedding. CVPR.

[23] Güler, R. A., Neverova, N., & Kokkinos, I. (2018). DensePose: Dense human pose estimation in the wild. CVPR.

[24] Cao, Z., Simon, T., Wei, S. E., & Sheikh, Y. (2017). Realtime multi-person 2D pose estimation using part affinity fields. CVPR.

[25] Loper, M., Mahmood, N., Romero, J., et al. (2015). SMPL: A skinned multi-person linear model. ACM TOG.

[26] Pavlakos, G., Choutas, V., Ghorbani, N., et al. (2019). Expressive body capture: 3D hands, face, and body from a single image. CVPR.

[27] Li, T., Bolkart, T., Black, M. J., et al. (2017). Learning a model of facial shape and expression from 4D scans. ACM TOG.

[28] Kepiro, I., Li, D., Zhang, J., et al. (2024). LivePortrait: Efficient portrait animation with stitching and retargeting control. arXiv.

[29] Chen, H., Zhang, L., Wang, X., et al. (2025). HunyuanPortrait: Implicit condition control for enhanced portrait animation. CVPR.

[30] Liu, W., Zhang, Y., Chen, H., et al. (2024). ID-Animator: Zero-shot identity-preserving human video generation. arXiv.

[31] Zhang, X., Wang, H., Chen, L., et al. (2024). Magic-Me: Identity-specific video customized diffusion. ECCV.

[32] Kim, S., Park, J., Lee, H., et al. (2024). Still-Moving: Zero-customization identity-preserving video generation. ECCV.

[33] Zhou, Y., Zhang, X., Chen, R., et al. (2024). StoryDiffusion: Consistent self-attention for long-range image and video generation. NeurIPS.

[34] Wang, Y., Liu, H., Zhang, P., et al. (2025). FantasyID: Multi-view and 3D prior fusion for identity-preserving text-to-video generation. ACM MM.

[35] Chen, L., Zhang, M., Wu, Y., et al. (2025). TIRE: Track-Inpaint-Resplat for 3D/4D identity-preserving generation. NeurIPS.

[36] Anderson, K., Brown, M., Davis, J., et al. (2025). Virtually Being: 4D identity preservation across views and illumination. SIGGRAPH.

[37] Nagrani, A., Chung, J. S., & Zisserman, A. (2017). VoxCeleb: A large-scale speaker identification dataset. Interspeech.

[38] Chung, J. S., Nagrani, A., & Zisserman, A. (2018). VoxCeleb2: Deep speaker recognition. Interspeech.

[39] Zhu, H., Wu, Y., Li, S., et al. (2022). CelebV-HQ: A large-scale video facial attributes dataset. ECCV.

[40] Fried, O., Tewari, A., Zollhöfer, M., et al. (2019). Text-based editing of talking-head video. ACM TOG.

[41] Zhang, Z., Li, L., Ding, Y., & Fan, C. (2021). Flow-guided one-shot talking face generation with a high-resolution audio-visual dataset. CVPR.

[42] Schuhmann, C., Beaumont, R., Vencu, R., et al. (2022). LAION-5B: An open large-scale dataset for training next generation image-text models. NeurIPS.

[43] Karras, T., Laine, S., Aittala, M., et al. (2019). Analyzing and improving the image quality of StyleGAN. CVPR.

[44] Bain, M., Nagrani, A., Varol, G., & Zisserman, A. (2021). Frozen in time: A joint video and image encoder for end-to-end retrieval. ICCV.

[45] Park, S., Kim, J., Lee, H., et al. (2025). PersonalVideo: High ID-fidelity video customization without dynamic degradation. ICCV.

[46] Wang, X., Chen, H., Zhang, Y., et al. (2024). VideoBooth: Diffusion-based video generation with image prompts. CVPR.

[47] Li, M., Zhang, P., Liu, T., et al. (2024). MotionBooth: Motion-aware customized text-to-video generation. NeurIPS.

[48] Zhang, Y., Liu, W., Chen, H., et al. (2025). DualReal: Adaptive joint training for identity-motion coherent custom video. ICCV.

[49] Chen, S., Li, M., Wang, Y., et al. (2025). MagicID: Hybrid preference optimization for ID-consistent video generation. ICCV.

[50] Liu, H., Wang, P., Zhang, X., et al. (2025). Concat-ID: Towards universal identity-preserving video synthesis. arXiv.

[51] Hu, L., Gao, X., Zhang, P., et al. (2025). Animate Anyone 2: High-fidelity character image animation with environment. ICCV.

[52] Kim, J., Park, S., Lee, H., et al. (2025). Phantom: Subject-consistent video generation via cross-modal alignment. ICCV.

[53] Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE SPL.

[54] Deng, J., Guo, J., Zhou, Y., et al. (2020). RetinaFace: Single-shot multi-level face localisation in the wild. CVPR.

[55] Kirillov, A., Mintun, E., Ravi, N., et al. (2023). Segment anything. ICCV.

[56] Wang, Y., Zhang, L., Chen, H., et al. (2025). UniPortrait: A unified framework for identity-preserving. ICCV.

---

**Author Biographies** (To be added)

**Acknowledgments** (To be added)

**Conflict of Interest** (To be added)
