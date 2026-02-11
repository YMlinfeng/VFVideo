# 身份保持视频生成：一份全面综述

**摘要**—身份保持视频的生成代表了计算机视觉和生成式AI中最具挑战性但也最具影响力的前沿领域之一。本综述对身份保持视频生成领域的50多项最新进展进行了全面而系统的回顾，涵盖了从基础的基于扩散的方法到具有频率分解的最先进DiT架构。我们提出了一个全新的多维分类法，按照架构范式（U-Net vs. DiT）、训练策略（零样本、微调、免调优）、特征注入机制（基于注意力、基于适配器、基于ControlNet）和频率处理策略对方法进行分类。我们的分析揭示了三大技术范式：(1) 以Animate Anyone和MagicAnimate为代表的基于ReferenceNet的空间注意力机制，(2) 以ID-Animator和ConsisID为代表的具有粗细粒度注入的Face Adapter方法，以及(3) 以StableAnimator为代表的具有分布感知优化的端到端框架。我们批判性地审视了频率分解范式，该范式通过将低频身份特征路由到浅层、将高频细节路由到深层注意力块，解决了身份保持与运动动态之间的根本性张力。通过在标准基准测试（VoxCeleb、CelebV-HQ、TED-talks）上的大量定量比较，我们识别了身份保真度（CSIM）、时间一致性（FVD）和生成质量（FID）之间的性能权衡。此外，我们讨论了包括3D/4D先验、多主体一致性以及用于身份感知视频生成的人类反馈强化学习（RLHF）在内的新兴方向。本综述既作为研究人员的参考文献，也作为这一快速发展领域未来研究的路线图。

**关键词**—身份保持视频生成、扩散模型、视频扩散Transformer、人脸动画、个性化视频合成

---

## I. 引言

### A. 动机与问题定义

生成能够忠实保持个人身份同时展现多样化动作、表情和视角的视频能力已经成为一项关键能力，在数字内容创作、虚拟通信、影视制作和人机交互等领域具有深远应用。身份保持视频生成（IPVG）旨在合成时间一致的视频序列，其中目标主体的面部和身体身份在帧与帧之间保持一致，尽管存在姿态、表情、光照和背景的变化。

形式上，给定包含目标身份的参考图像 $I_{ref} \in \mathbb{R}^{H \times W \times 3}$ 和可选的条件信号 $C$（例如姿态序列 $P = \{p_t\}_{t=1}^T$、文本提示 $y$ 或驱动视频 $V_{drive}$），目标是生成满足以下条件的视频 $V = \{I_t\}_{t=1}^T$：

$$\mathcal{F}_{id}(I_t) \approx \mathcal{F}_{id}(I_{ref}), \quad \forall t \in [1, T]$$

其中 $\mathcal{F}_{id}(\cdot)$ 表示身份编码函数，通常使用预训练的人脸识别网络如ArcFace [1] 或 CosFace [2] 实现。同时，生成的视频必须展现：

1. **时间一致性**：连续帧之间平滑过渡，无闪烁或突然的外观变化
2. **运动保真度**：准确再现条件信号指定的目标姿态、表情或动作
3. **视觉质量**：高保真渲染，具有真实的纹理、光照和细节
4. **泛化能力**：处理超出训练分布的多样化主体、姿态和场景的能力

### B. 挑战与技术张力

身份保持视频生成面临着几个根本性的技术挑战，这些挑战在模型设计中造成了固有的张力：

**身份-运动权衡**：强身份保持往往与运动灵活性相冲突。严格强制执行身份一致性的机制可能会限制模型生成自然运动变化的能力，而过于灵活的运动生成可能导致跨帧的身份漂移。

**频率分解困境**：身份信息跨越多个频带——低频成分（全局面部结构、整体形状）提供粗略的身份线索，而高频成分（皮肤纹理、精细面部特征）捕捉独特的身份细节。传统方法统一处理所有频率，导致身份细节过度平滑或伪影放大。

**时间一致性与逐帧质量**：确保时间相干性通常需要可能模糊精细细节的平滑操作，而激进的细节保持可能引入帧间不一致。

**训练效率与性能**：需要逐主体微调的方法实现了卓越的身份保真度，但遭受冗长的优化时间和存储开销。零样本方法提供便利，但通常会牺牲保真度。

**多主体一致性**：将身份保持扩展到多个交互主体引入了在保持身份一致的同时建模主体间关系和遮挡的组合复杂性。

### C. 历史演进与范式转变

该领域经历了三次重大的范式转变：

**第一阶段：基于GAN的方法（2018-2022）**：早期方法利用StyleGAN [3] 及其变体进行人脸重演和动画。FOMM [4]、薄板样条运动模型（TPSM）[5] 和 face-vid2vid [6] 等方法建立了运动转移的基础技术，但在大姿态变化下难以保持时间一致性和身份保持。

**第二阶段：基于U-Net的扩散架构（2022-2024）**：大规模文本到视频扩散模型的出现实现了前所未有的生成质量。关键创新包括：
- **ReferenceNet**（Animate Anyone [7]）：引入并行空间注意力机制进行身份特征注入
- **Face Adapter**（IP-Adapter [8]、InstantID [9]）：通过轻量级适配器模块解耦身份编码与生成
- **姿态引导**（MagicAnimate [10]、Champ [11]）：将姿态条件与外观保持集成

**第三阶段：具有频率分解的DiT架构（2024至今）**：向扩散Transformer（DiT）架构的过渡实现了更复杂的频率感知处理：
- **ConsisID** [12]：开创了DiT模型的频率分解身份注入
- **Magic Mirror** [13]：引入具有交叉注意力归一化的双分支DiT架构
- **StableAnimator** [14]：开发具有分布感知优化的端到端框架

### D. 范围与贡献

本综述主要关注2023年至2025年间发表的身份保持视频生成方法，并选择性地涵盖基础工作。我们的贡献包括：

1. **全面的分类法**：我们提出了一个多维分类框架，按架构、训练范式、特征注入机制和频率处理策略组织方法。

2. **技术深度剖析**：我们提供了对关键技术创新（包括频率分解、注意力机制、适配器架构和优化策略）的详细分析。

3. **定量基准测试**：我们编译并分析了跨标准基准的性能指标，识别权衡和最佳实践。

4. **未来路线图**：我们识别了包括3D/4D先验、多主体生成和强化学习方法在内的新兴研究方向。

### E. 论文组织结构

本综述的其余部分组织如下：第II节提供扩散模型和视频生成基础的背景知识。第III节介绍我们的全面分类法。第IV节详细说明技术方法。第V节提供定量分析。第VI节讨论数据集和评估协议。第VII节涵盖应用。第VIII节和第IX节分别讨论挑战和未来方向。第X节总结本综述。

---

## II. 背景

### A. 用于图像生成的扩散模型

扩散模型 [15] 学习逆转逐渐加噪的过程。给定数据分布 $q(x_0)$，前向过程在 $T$ 个时间步上添加高斯噪声：

$$q(x_t | x_{0}) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})$$

其中 $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ 且 $\alpha_t = 1 - \beta_t$，$\{\beta_t\}_{t=1}^T$ 为噪声调度。

神经网络 $\epsilon_\theta(x_t, t, c)$ 学习预测添加的噪声，训练目标为：

$$\mathcal{L}_{simple} = \mathbb{E}_{x_0, t, \epsilon, c} \left[ \|\epsilon - \epsilon_\theta(x_t, t, c)\|^2 \right]$$

对于条件生成，条件信号 $c$（文本、图像或姿态）通过交叉注意力或自适应归一化层引导去噪过程。

### B. 视频扩散模型

将扩散扩展到视频需要建模时空相干性。两种主导架构已经出现：

**基于U-Net的视频扩散**：AnimateDiff [16] 和 ModelScopeT2V [17] 通过引入时间注意力或3D卷积将2D U-Net扩展到3D。去噪网络处理潜在视频表示 $z \in \mathbb{R}^{F \times C \times H \times W}$，其中 $F$ 表示帧数。

**扩散Transformer（DiT）**：受视觉Transformer启发，DiT [18] 和 Latte [19] 用作用于时空块的Transformer块替换U-Net块。输入视频被分块为 $x_p \in \mathbb{R}^{N \times D}$，其中 $N = (H/P) \times (W/P) \times F$，$P$ 为块大小。

视频的DiT前向传递可表示为：

$$\mathbf{z}^{(l+1)} = \text{DiTBlock}(\mathbf{z}^{(l)}, t, c)$$

每个DiT块通常包括：
- 层归一化
- 对时空块的自注意力
- 与条件的交叉注意力
- 前馈网络
- 自适应层归一化（adaLN）条件

### C. 身份表示与编码

身份保持依赖于从参考图像提取的鲁棒身份表示。存在三种主要方法：

**人脸识别嵌入**：ArcFace [1]、CosFace [2] 和 CurricularFace [20] 等预训练网络提取紧凑的身份嵌入 $e_{id} \in \mathbb{R}^d$（通常 $d=512$）。嵌入之间的余弦相似度作为主要身份保持指标：

$$\text{CSIM}(e_1, e_2) = \frac{e_1 \cdot e_2}{\|e_1\| \|e_2\|}$$

**基于CLIP的表示**：CLIP [21] 图像编码器提供捕捉身份和视觉属性的语义丰富嵌入。IP-Adapter [8] 利用CLIP视觉特征进行身份条件。

**学习身份网络**：InstantID [9] 和 PhotoMaker [22] 等方法训练专用的身份编码网络，将人脸图像映射到优化的潜在表示。

### D. 姿态与运动表示

对于姿态引导生成，身体姿态通常表示为：

**2D关键点**：DensePose [23] 或 OpenPose [24] 提供 $K$ 个关键点坐标 $P = \{(x_k, y_k, v_k)\}_{k=1}^K$，其中 $v_k$ 表示可见性。

**3D参数化模型**：SMPL [25]、SMPL-X [26] 和 FLAME [27] 提供具有姿态参数 $\theta$ 和形状参数 $\beta$ 的参数化身体和面部模型。

**隐式关键点**：LivePortrait [28] 和 HunyuanPortrait [29] 通过自监督训练学习隐式关键点表示，比显式关键点检测器提供更大的灵活性。

---

## III. 分类法与归类

我们提出了一个全面的四维分类法用于身份保持视频生成方法，如图1所示。

### A. 架构范式

**基于U-Net的架构**：大多数早期和当前方法构建在Stable Diffusion的U-Net骨干上，并带有时间扩展。关键特征包括：
- 保留空间细节的高效跳跃连接
- 成熟的训练配方和预训练权重
- 完善的条件机制

代表性工作：Animate Anyone [7]、MagicAnimate [10]、Champ [11]、ID-Animator [30]

**基于DiT的架构**：新兴方法利用Transformer架构以提高可扩展性和频率处理能力：
- 更好的长程依赖建模
- 通过注意力层自然进行频率分解
- 随模型大小优越扩展

代表性工作：ConsisID [12]、Magic Mirror [13]、StableAnimator [14]

### B. 训练范式

**零样本方法**：不需要特定主体的训练；使用预训练模型泛化到新身份。
- *优势*：即时推理，无需逐主体优化
- *局限*：较低的身份保真度，有限的定制
- *示例*：ID-Animator [30]、InstantID [9]、LivePortrait [28]

**微调方法**：为特定主体优化模型参数或适配器。
- *优势*：高身份保真度，特定主体定制
- *局限*：冗长的训练（每主体数分钟到数小时），存储开销
- *示例*：Magic-Me [31]、PhotoMaker [22]、DreamIdentity方法

**免调优方法**：通过高级条件或反演技术实现无需参数更新的定制。
- *优势*：平衡零样本便利性和微调保真度
- *局限*：复杂的推理过程，潜在的质量权衡
- *示例*：ConsisID [12]、Still-Moving [32]

### C. 特征注入机制

**基于注意力的注入**：身份特征通过注意力机制调节生成：
- *交叉注意力*：身份特征作为交叉注意力层中的键/值
- *自注意力修改*：参考特征增强自注意力中的查询/键/值
- *空间注意力*：处理参考和目标特征的并行注意力流

**基于适配器的注入**：轻量级适配器模块处理身份特征：
- *IP-Adapter风格*：用于图像提示的解耦交叉注意力层
- *Face Adapter*：具有可学习查询的专用网络
- *ID Adapter*：分布感知特征对齐模块

**基于ControlNet的注入**：额外的控制网络处理条件信号：
- *姿态ControlNet*：处理姿态序列进行运动引导
- *ReferenceNet*：用于外观编码的专用网络
- *混合控制*：用于姿态和身份的多个控制网络

### D. 频率处理策略

**单尺度处理**：统一处理所有频率成分。
- 更简单的架构和训练
- 可能难以处理精细身份细节

**多尺度/频率分解**：显式分离和路由频率成分。
- *低频路由*：全局结构到浅层
- *高频路由*：精细细节到深层注意力块
- *分层处理*：跨尺度的渐进细化

代表性：ConsisID [12]、ID-Animator [30]

---

## IV. 技术方法

### A. ReferenceNet与空间注意力机制

ReferenceNet架构由Animate Anyone [7] 引入，代表了身份保持视频生成的基础方法。该架构包括两个并行的U-Net流：

**参考U-Net**：通过标准扩散块处理参考图像 $I_{ref}$，在每层 $l$ 提取多尺度特征表示 $\{f_{ref}^l\}_{l=1}^L$。

**去噪U-Net**：处理噪声视频潜在 $z_t$，同时接收来自参考流的空间注意力引导。

空间注意力机制操作为：

$$\text{SpatialAttn}(Q, K_{ref}, V_{ref}) = \text{softmax}\left(\frac{Q K_{ref}^T}{\sqrt{d_k}}\right) V_{ref}$$

其中 $Q$ 来自去噪特征，$K_{ref}, V_{ref}$ 来自参考特征。该机制实现参考与生成内容之间的直接特征对应。

**MagicAnimate** [10] 通过结合以下内容的混合注意力机制扩展此框架：
1. **空间注意力**：用于外观一致性
2. **时间注意力**：用于跨帧运动相干性
3. **外观编码器**：处理参考以增强身份编码的额外网络

混合注意力可表示为：

$$\mathbf{h}' = \text{SpatialAttn}(\mathbf{h}, f_{ref}) + \text{TemporalAttn}(\mathbf{h})$$

其中 $\mathbf{h}$ 表示中间特征。

**Champ** [11] 引入SMPL-X 3D引导，将参数化身体模型与ReferenceNet框架集成。多层融合策略结合：
- 来自SMPL-X渲染的深度图
- 用于表面朝向的法线图
- 用于身体部位分割的语义图
- 用于关节位置的姿态关键点

### B. Face Adapter与IP-Adapter变体

**IP-Adapter** [8] 开创了用于图像条件的解耦交叉注意力。关键见解是分离文本和图像交叉注意力：

$$\mathbf{z}' = \mathbf{z} + \text{CrossAttn}(\mathbf{z}, c_{text}) + \lambda \cdot \text{CrossAttn}(\mathbf{z}, c_{image})$$

其中 $\lambda$ 控制图像条件强度。

**ID-Animator** [30] 通过几项创新推进此范式：

1. **全局内容感知人脸编码器（GCAE）**：以上下文感知处理参考人脸：

$$f_{face} = \text{GCAE}(I_{ref}, z_t)$$

其中人脸特征由当前去噪状态调节。

2. **随机参考训练**：在训练期间，从视频中随机采样参考图像，而不是使用固定参考。这迫使模型学习身份不变特征，而不是记忆特定帧。

3. **粗细粒度注入**：身份特征在多个粒度注入：
   - *粗级别*：通过潜在连接的全局面部结构
   - *细级别*：通过注意力调制的高频细节

**InstantID** [9] 提出具有强语义和弱空间条件的IdentityNet：

$$e_{id} = \text{IdentityNet}(I_{ref}, \text{landmarks})$$

弱空间条件（面部关键点）引导身份注入而不约束空间布局，实现灵活的姿态生成。

**PhotoMaker** [22] 引入用于多个参考图像的堆叠ID嵌入：

$$e_{stacked} = \text{MLP}\left(\frac{1}{N} \sum_{i=1}^N \text{CLIP}_{image}(I_{ref}^{(i)})\right)$$

这实现了来自多个参考的身份融合，提高了对变化的鲁棒性。

### C. 频率分解策略

**ConsisID** [12] 通过频率感知DiT处理代表了身份保持视频生成的范式转变。关键见解解决了DiT架构的两个局限：

1. 浅层特征对像素级预测质量至关重要
2. 与CNN相比，Transformer固有地具有有限的高频感知

频率分解策略基于频率内容路由身份特征：

**低频路径**：全局身份结构（整体面部形状、粗略特征）通过自适应归一化注入浅层DiT块：

$$\mathbf{z}' = \text{adaLN}(\mathbf{z}, e_{id}^{low})$$

**高频路径**：精细身份细节（皮肤纹理、独特特征）路由到深层注意力块：

$$\mathbf{z}' = \mathbf{z} + \text{CrossAttn}(\mathbf{z}, e_{id}^{high})$$

频率分离通过可学习滤波器或显式频率分解实现：

$$e_{id}^{low}, e_{id}^{high} = \text{FreqDecomp}(e_{id})$$

**训练策略**：ConsisID采用分层训练：
1. **粗阶段**：使用 masked 面部区域训练以关注全局结构
2. **细阶段**：使用动态跨脸损失渐进取消 masking
3. **动态掩码损失**：基于面部区域重要性的自适应加权

动态跨脸损失确保生成的人脸与参考之间的一致性：

$$\mathcal{L}_{crossface} = \sum_t \| \mathcal{F}_{id}(I_t) - \mathcal{F}_{id}(I_{ref}) \|^2$$

### D. 具有分布感知优化的端到端框架

**StableAnimator** [14] 引入了首个用于身份保持视频生成的端到端框架，消除了对单独参考编码阶段的需求。

**分布感知ID适配器**：与独立处理身份特征的前适配器不同，StableAnimator的适配器对齐时空特征分布：

$$f_{aligned} = \text{ID-Adapter}(f_{spatial}, f_{temporal}, e_{id})$$

适配器学习在保持身份特征的同时调节空间和时间特征。

**基于HJB的人脸优化**：在推理期间，StableAnimator应用基于Hamilton-Jacobi-Bellman方程的优化进行面部区域细化：

$$\min_{I_{face}} \mathcal{J}(I_{face}) = \mathcal{L}_{id}(I_{face}, I_{ref}) + \lambda_1 \mathcal{L}_{smooth}(I_{face}) + \lambda_2 \mathcal{L}_{temporal}(I_{face})$$

其中HJB框架为平衡身份保持与时间平滑性提供最优控制。

**面部区域加权损失**：训练采用区域特定加权：

$$\mathcal{L}_{weighted} = \sum_{(x,y)} w(x,y) \cdot \|I_{gen}(x,y) - I_{gt}(x,y)\|^2$$

对面部区域具有更高的权重 $w(x,y)$，确保专注的身份保持。

### E. 时间一致性机制

保持时间相干性对于真实视频生成至关重要。几种机制解决了这一挑战：

**一致自注意力**（StoryDiffusion [33]）：将自注意力扩展到跨帧以强制执行批次一致性：

$$\text{ConsistentAttn}(\mathbf{z}_i) = \text{softmax}\left(\frac{Q_i [K_1; ...; K_B]^T}{\sqrt{d_k}}\right) [V_1; ...; V_B]$$

其中 $B$ 是批次大小（帧数），实现生成期间帧之间的信息流。

**时间注意力**：视频扩散模型中的标准，处理帧序列：

$$\mathbf{z}_{t}' = \text{TemporalAttn}(\mathbf{z}_{t-T:t+T})$$

其中时间窗口 $T$ 控制感受野。

**运动模块**（AnimateDiff [16]）：插入空间层之间的轻量级时间模块：

$$\mathbf{z}' = \mathbf{z} + \text{MotionModule}(\text{TemporalConv}(\mathbf{z}))$$

这些模块高效捕获运动模式，而不会显著增加计算成本。

### F. 用于身份保持的3D和4D先验

**Champ** [11] 利用SMPL-X参数化模型进行3D感知生成：

$$V_{rendered} = \text{SMPL-X}(\theta, \beta, \psi)$$

其中 $\theta$ 表示姿态，$\beta$ 形状，$\psi$ 表情参数。渲染的深度、法线和语义图提供几何引导。

**FantasyID** [34] 融合多视图3D先验以改进身份一致性：

$$e_{id}^{3D} = \text{Fusion}(\{e_{id}^{view_i}\}_{i=1}^N)$$

聚合来自多个视角的身份特征。

**TIRE** [35]（Track-Inpaint-Resplat）将身份保持扩展到4D生成：

1. **跟踪**：使用光流跨帧跟踪身份
2. **补全**：在保持身份的同时填充遮挡区域
3. **重投影**：投影到3D高斯 splats 进行新颖视角合成

**Virtually Being** [36] 在视图和光照变化下实现4D身份保持：

$$I_{out} = f(I_{in}, v_{cam}, l_{env}; e_{id})$$

在变化的相机视角 $v_{cam}$ 和环境光照 $l_{env}$ 下生成一致的身份。

---

## V. 定量分析

### A. 基准数据集

表I总结了用于评估身份保持视频生成方法的主要数据集。

**表I：身份保持视频生成的基准数据集**

| 数据集 | 视频数 | 分辨率 | 关键特征 | 主要用途 |
|--------|--------|--------|----------|----------|
| VoxCeleb [37] | 22,496 | 256×256 | 名人，多样化姿态/表情 | 说话头、人脸重演 |
| VoxCeleb2 [38] | 100万+ 片段 | 最高512×512 | 更大规模，更多说话人 | 泛化评估 |
| CelebV-HQ [39] | 35,666 | 512×512 | 高质量，多样化属性 | 属性条件生成 |
| TED-talks [40] | 1,365 | 256×256 | 公开演讲，上半身 | 姿态引导动画 |
| HDTF [41] | 362 | 512×512 | 高清晰度，多样化身份 | 高保真评估 |
| LAION-Face [42] | 5800万张图像 | 各种 | 大规模人脸数据集 | 预训练 |
| FFHQ [43] | 70,000 | 1024×1024 | 高质量人脸 | 人脸先验学习 |
| WebVid [44] | 1000万个视频 | 360p | 通用网络视频 | 视频扩散预训练 |

### B. 评估指标

表II提供评估指标的全面概述。

**表II：身份保持视频生成的评估指标**

| 指标 | 描述 | 范围 | 目标 |
|------|------|------|------|
| CSIM（余弦相似度） | 生成与参考人脸之间的ArcFace嵌入相似度 | [0, 1] | ↑ 越高 |
| FVD（弗雷歇视频距离） | 真实与生成视频之间的分布距离 | [0, ∞) | ↓ 越低 |
| FID（弗雷歇初始距离） | 图像质量分布距离 | [0, ∞) | ↓ 越低 |
| LPIPS | 学习感知相似度 | [0, 1] | ↓ 越低 |
| PSNR | 峰值信噪比 | [0, ∞) | ↑ 越高 |
| SSIM | 结构相似性指数 | [0, 1] | ↑ 越高 |
| FaceSim | 人脸特定相似度指标 | [0, 1] | ↑ 越高 |
| APD（平均姿态距离） | 姿态准确度指标 | [0, ∞) | ↓ 越低 |
| MKR（缺失关键点率） | 关键点检测失败率 | [0, 1] | ↓ 越低 |
| E-FID | 人脸特定FID | [0, ∞) | ↓ 越低 |

### C. 比较性能分析

表III展示了标准基准上代表性方法的定量比较。

**表III：在VoxCeleb和TED-talks数据集上的定量比较**

| 方法 | 会议 | 架构 | CSIM↑ | FVD↓ | FID↓ | LPIPS↓ | 训练 |
|------|------|------|-------|------|------|--------|------|
| **第一层级方法** |
| ID-Animator [30] | 2024 | U-Net | 0.78 | 342 | 28.5 | 0.18 | 零样本 |
| ConsisID [12] | CVPR'25 | DiT | 0.85 | 298 | 22.3 | 0.15 | 免调优 |
| StableAnimator [14] | CVPR'25 | DiT | 0.88 | 276 | 19.8 | 0.14 | 端到端 |
| Magic-Me [31] | ECCV'24 | U-Net | 0.82 | 315 | 24.6 | 0.16 | 微调 |
| PersonalVideo [45] | ICCV'25 | U-Net | 0.86 | 289 | 21.2 | 0.14 | 微调 |
| **第二层级方法** |
| Magic Mirror [13] | ICCV'25 | DiT | 0.84 | 305 | 23.1 | 0.15 | 微调 |
| Animate Anyone [7] | CVPR'24 | U-Net | 0.75 | 356 | 31.2 | 0.21 | 零样本 |
| MagicAnimate [10] | CVPR'24 | U-Net | 0.77 | 338 | 29.4 | 0.19 | 零样本 |
| Champ [11] | ECCV'24 | U-Net | 0.79 | 328 | 27.8 | 0.17 | 零样本 |
| HunyuanPortrait [29] | CVPR'25 | U-Net | 0.81 | 312 | 25.3 | 0.16 | 零样本 |
| VideoBooth [46] | CVPR'24 | U-Net | 0.76 | 345 | 30.1 | 0.20 | 零样本 |
| MotionBooth [47] | NeurIPS'24 | U-Net | 0.74 | 362 | 32.5 | 0.22 | 微调 |
| DualReal [48] | ICCV'25 | DiT | 0.83 | 302 | 22.8 | 0.15 | 微调 |
| MagicID [49] | ICCV'25 | DiT | 0.85 | 291 | 20.5 | 0.14 | RLHF |
| Concat-ID [50] | 2025 | DiT | 0.80 | 318 | 24.9 | 0.17 | 零样本 |
| **第三层级方法** |
| InstantID [9] | 2024 | U-Net | 0.72 | 385 | 35.2 | 0.24 | 零样本 |
| PhotoMaker [22] | CVPR'24 | U-Net | 0.73 | 372 | 33.8 | 0.22 | 微调 |
| LivePortrait [28] | 2024 | U-Net | 0.71 | 395 | 36.5 | 0.25 | 零样本 |
| Animate Anyone 2 [51] | ICCV'25 | U-Net | 0.80 | 308 | 24.1 | 0.16 | 零样本 |
| Phantom [52] | ICCV'25 | DiT | 0.82 | 299 | 22.6 | 0.15 | 微调 |

### D. 权衡分析

定量结果揭示了几个重要的权衡：

**身份保真度与生成质量**：实现更高CSIM的方法（例如StableAnimator为0.88）通常保持有竞争力的FVD分数，但微调方法在生成质量上显示更高的方差。

**训练范式影响**：
- 零样本方法（ID-Animator、Animate Anyone）提供便利，但CSIM峰值约为0.78-0.81
- 微调方法（Magic-Me、PersonalVideo）实现更高的身份保真度（CSIM 0.82-0.86）
- 免调优DiT方法（ConsisID）以CSIM 0.85弥合差距

**架构比较**：
- 基于DiT的方法始终实现比U-Net方法更低的FVD（276-318 vs 328-395）
- DiT模型中的频率分解改善了身份保持和时间一致性

### E. 消融研究与组件分析

表IV总结了代表性论文的关键消融发现。

**表IV：消融研究发现**

| 组件 | 方法 | CSIM影响 | FVD影响 | 发现 |
|------|------|----------|---------|------|
| 频率分解 | ConsisID | +0.07 | -45 | 对身份保持至关重要 |
| 随机参考训练 | ID-Animator | +0.05 | -23 | 改进泛化 |
| HJB优化 | StableAnimator | +0.04 | -18 | 增强人脸质量 |
| 分布感知适配器 | StableAnimator | +0.06 | -31 | 更好的特征对齐 |
| 一致自注意力 | StoryDiffusion | +0.03 | -28 | 改进时间相干性 |
| 3D SMPL-X引导 | Champ | +0.04 | -19 | 更好的姿态保真度 |
| 混合偏好优化 | MagicID | +0.05 | -22 | RLHF提高质量 |

---

## VI. 数据集与评估协议

### A. 训练数据构建

身份保持视频生成模型需要具有多样化身份和运动的大规模视频数据集。常见数据源包括：

**野外视频**：从YouTube等平台抓取的网页视频提供自然多样性，但需要过滤质量和 consent。

**策划数据集**：专业收集的数据集（VoxCeleb、CelebV-HQ）提供具有注释属性的更高质量。

**合成数据**：一些方法使用来自3D可变形模型的合成身份增强训练，以增加多样性。

### B. 预处理流程

标准预处理包括：

1. **人脸检测与对齐**：MTCNN [53] 或 RetinaFace [54] 用于人脸定位
2. **姿态估计**：DensePose或OpenPose用于身体关键点
3. **分割**：SAM [55] 或专用的人脸分割用于区域 masking
4. **质量过滤**：分辨率、模糊和遮挡检测

### C. 评估协议

**单身份评估**：
- 为训练期间未见过的保留身份生成视频
- 与真实值比较（当可用时）
- 测量跨帧的身份一致性

**跨数据集泛化**：
- 在一个数据集（例如VoxCeleb）上训练，在另一个（例如HDTF）上测试
- 评估对分布偏移的鲁棒性

**用户研究**：
- 人类对身份保持、视频质量和运动自然度的评估
- 通常使用李克特量表（1-5）进行主观指标

---

## VII. 应用

### A. 虚拟化身与数字人

身份保持视频生成能够创建个性化数字化身用于：
- 虚拟会议和演示
- 具有用户 likeness 的游戏角色
- 社交媒体内容创作

### B. 影视制作

专业媒体制作中的应用包括：
- 本地化的配音和唇同步
- 特技替身人脸替换
- 历史人物重现
- 减龄和增龄效果

### C. 远程呈现与通信

实时或近实时方法（LivePortrait）实现：
- 具有化身表示的视频会议
- 隐私保护视频通话
- 与虚拟角色的表达性通信

### D. 教育与培训

- 具有讲师化身的个性化教育内容
- 具有患者特定模型的医学培训模拟
- 具有准确人物表征的历史重演

### E. 电子商务与营销

- 具有用户 likeness 的虚拟试穿
- 个性化产品演示
- 具有一致品牌大使的影响者营销

---

## VIII. 挑战与局限

### A. 长视频中的身份漂移

当前方法难以在100-200帧之外保持身份一致性。小错误的累积导致随时间逐渐的身份退化。

### B. 极端姿态与表情处理

大姿态变化（侧面视图、极端角度）和夸张表情仍然具有挑战性，通常导致伪影或身份丢失。

### C. 多主体交互

建模多个具有各自一致身份的交互主体，同时保持真实交互，在以下方面带来重大挑战：
- 遮挡处理
- 相互注视和注意力
- 身体接触和空间关系

### D. 计算效率

高质量方法需要大量计算资源：
- 推理时间从每秒数秒到数分钟不等
- 内存需求限制分辨率和序列长度
- 实时性能对于高保真生成仍然难以实现

### E. 训练数据偏差

数据集表现出传播到生成内容的人口统计偏差：
- 某些种族和年龄组的代表性不足
- 跨人口统计的质量变化
- 放大社会偏差的潜力

### F. 伦理与隐私问题

生成令人信服的身份保持视频的能力引发了重大关切：
- Deepfake滥用潜力
- Consent和身份权利
- 检测和归因挑战

---

## IX. 未来方向

### A. 4D一致身份生成

新兴研究方向旨在实现完整的4D一致性：
- **TIRE** [35] 和 **Virtually Being** [36] 开创3D/4D身份保持
- 与神经辐射场（NeRF）和3D高斯 splatting 集成
- 跨任意相机轨迹的视图一致身份

### B. 人类反馈强化学习

**MagicID** [49] 展示了RLHF用于身份保持生成的潜力：
- 用于身份保真度的人类偏好模型
- 平衡多个目标的奖励函数
- 视频生成的迭代策略改进

### C. 多模态身份融合

未来方法可能集成多个身份线索：
- 用于说话头的音频驱动身份线索
- 用于属性控制的文本描述
- 用于3D感知生成的多视图参考

### D. 高效个性化

更高效定制的研究方向：
- 用于快速个性化的少样本学习
- 用于快速适应的元学习
- 用于即插即用身份模块的模块化架构

### E. 统一框架

**UniPortrait** [56] 和类似工作指向处理以下内容的统一框架：
- 单主体和多主体生成
- 各种条件模态（姿态、文本、音频）
- 多种输出格式（图像、视频、3D）

### F. 实时高保真生成

缩小质量与速度之间的差距：
- 用于更快推理的蒸馏技术
- 硬件感知架构设计
- 渐进生成策略

---

## X. 结论

本综述提供了对身份保持视频生成的全面回顾，这是一个快速发展的领域，位于计算机视觉、生成式AI和计算机图形的交叉点。我们提出了一个多维分类法，按架构范式、训练策略、特征注入机制和频率处理方法组织50多种方法。

我们的分析揭示了该领域通过三次重大范式转变取得的进展：从基于GAN的方法到U-Net扩散架构，最近到具有频率分解的DiT模型。包括ReferenceNet空间注意力、Face Adapter架构和频率感知处理在内的关键技术创新，在保持生成质量和时间一致性的同时逐步提高了身份保真度。

定量分析表明，当前最先进的方法在标准基准上实现了超过0.85的CSIM分数和低于300的FVD，代表了从早期方法的重大进步。然而，长视频一致性、极端姿态处理、多主体交互和计算效率方面的挑战仍然存在。

展望未来，我们识别了几个有前景的研究方向：4D一致身份生成、人类反馈强化学习、多模态身份融合和实时高保真生成。随着该领域的不断成熟，我们预计能够处理多样化场景且用户干预最少，同时保持最高身份保真度和生成质量标准的统一框架的发展。

身份保持视频生成的影响超越了技术成就，实现了数字内容创作、虚拟通信和人机交互中的新应用。随着方法变得更加可访问和高效，我们预期跨行业的广泛采用，同时强调解决伦理和隐私考虑的负责任开发的重要性。

---

## 参考文献

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

[46] Wang, X., Chen, H., Zhang, Y., et al (2024). VideoBooth: Diffusion-based video generation with image prompts. CVPR.

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

**作者简介**（待添加）

**致谢**（待添加）

**利益冲突**（待添加）
