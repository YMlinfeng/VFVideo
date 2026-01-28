from .base_pipeline import BasePipeline
import torch


def FlowMatchSFTLoss_cat(pipe: BasePipeline, **inputs):
    # 1. 采样时间步（在调度器定义的范围内随机选一个时间步，代表"加多少噪声"）timestep 越大 → 噪声越多；为什么要随机采样？我们希望模型学会处理任意噪声程度
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps)) #todo
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    # 2. 生成随机噪声 [1, 16, 15, 80, 60]
    noise = torch.randn_like(inputs["input_latents"]) # [1, 16, 15, 80, 60]
    # 3. 加噪：x_t = interpolate(x_0, noise, t)
    ref_latent = inputs["input_latents"][:, :, 0:1]  # 干净的首帧
    gen_latents = inputs["input_latents"][:, :, 1:]  # 后续帧
    noise_gen = noise[:, :, 1:]  # 对应的噪声
    noisy_gen_latents = pipe.scheduler.add_noise(gen_latents, noise_gen, timestep)
    inputs["latents"] = torch.cat([ref_latent, noisy_gen_latents], dim=2)
    # inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    # 4. 计算训练目标：v = noise - x_0 (速度场)
    target_gen = pipe.scheduler.training_target(gen_latents, noise_gen, timestep)
    # training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    # 5. 模型预测速度场
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep) # torch.Size([1, 16, 15, 80, 60])
    # 6. MSE Loss + 时间步加权
    noise_pred_gen = noise_pred[:, :, 1:]  # 只取后续帧的预测
    loss = torch.nn.functional.mse_loss(noise_pred_gen.float(), target_gen.float())
    loss = loss * pipe.scheduler.training_weight(timestep) # 不同时间步的难度不同，可能需要不同的权重
    return loss

def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    # 1. 采样时间步
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
 
    latents = inputs["input_latents"]  # [B, C, F, H, W]
    B, C, F, H, W = latents.shape
    noise = torch.randn_like(latents)  # [B, C, F, H, W]
    
    # 3. 构建 per-frame 的时间步 t: [B, 1, F, 1, 1]
    #    首帧 t=0（保持干净），其他帧 t=sigma
    num_train_timesteps = getattr(pipe.scheduler.config, 'num_train_timesteps', 1000)
    sigma = timestep.float() / num_train_timesteps  # 归一化到 [0, 1]
    
    t = torch.zeros(B, 1, F, 1, 1, device=latents.device, dtype=latents.dtype)
    t[:, :, 1:, :, :] = sigma.view(-1, 1, 1, 1, 1)  # 只有后续帧有噪声
    
    # 4. Flow Matching 加噪: x_t = (1-t) * x_0 + t * z
    #    首帧 (t=0): x_t = x_0 (自然保持干净，无需拼接！)
    #    其他帧:     x_t = (1-sigma) * x_0 + sigma * z
    noisy_latents = (1 - t) * latents + t * noise
    inputs["latents"] = noisy_latents
    
    # 5. 训练目标: v = z - x_0 (velocity field)
    target = noise - latents
    
    # 6. 模型预测
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    # 7. 构建 loss mask: [B, 1, F, 1, 1]
    #    首帧为0（不参与loss），其他帧为1
    loss_mask = torch.zeros(B, 1, F, 1, 1, device=latents.device, dtype=latents.dtype)
    loss_mask[:, :, 1:, :, :] = 1.0
    
    # 8. 计算 Masked MSE Loss
    diff_sq = (pred.float() - target.float()) ** 2  # [B, C, F, H, W]
    masked_diff = diff_sq * loss_mask  # broadcast: [B, C, F, H, W]
    
    # 归一化：只除以有效元素数量（排除首帧）
    num_valid_frames = F - 1
    loss = masked_diff.sum() / (B * C * num_valid_frames * H * W)
    
    # 9. 时间步加权
    loss = loss * pipe.scheduler.training_weight(timestep)
    
    return loss

def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips 
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            target = (latents_ - inputs_shared["latents"]) / (sigma_ - sigma)
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
