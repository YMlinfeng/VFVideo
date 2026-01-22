import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
import torch
from tqdm import tqdm
import time
import time
from contextlib import contextmanager
from collections import defaultdict

import time
from collections import defaultdict
from contextlib import contextmanager
import statistics
import os
import torch
from datetime import datetime

import time
import torch
import torch.distributed as dist
from contextlib import contextmanager
from collections import defaultdict
from datetime import datetime
import statistics


class DetailedStepTimer:
    """
    针对多机多卡训练的详细计时器
    
    功能：
    1. 使用 CUDA Events 精确测量 GPU 时间（避免 CPU-GPU 异步误差）
    2. 分离 backward 纯计算 vs 梯度同步时间
    3. 测量进程间等待/负载不均衡
    4. 支持每个 rank 独立记录，最后汇总比较
    """
    
    def __init__(self, log_file="training_perf_detailed.log", warmup_steps=2):
        self.times = defaultdict(list)
        self.step_keys = []
        self.log_file = log_file
        self.warmup_steps = warmup_steps  # 前几步不计入统计（预热）
        self.current_step = 0
        
        self.use_cuda = torch.cuda.is_available()
        
        # 每个 rank 的独立时间记录（用于分析负载不均衡）
        self.per_rank_times = defaultdict(list)
    
    @contextmanager
    def time_step(self, name):
        """
        使用 CUDA Events 的精确 GPU 计时
        - CUDA Events 直接在 GPU 上记录时间戳，避免 CPU-GPU 异步带来的误差
        """
        if name not in self.step_keys:
            self.step_keys.append(name)
        
        if self.use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # 同步确保之前的操作完成
            torch.cuda.synchronize()
            start_event.record()
            
            yield
            
            end_event.record()
            torch.cuda.synchronize()
            
            # elapsed_time 返回毫秒，转为秒
            elapsed = start_event.elapsed_time(end_event) / 1000.0
        else:
            start = time.perf_counter()
            yield
            elapsed = time.perf_counter() - start
        
        self.times[name].append(elapsed)
    
    @contextmanager
    def time_step_no_sync(self, name):
        """
        不带 GPU 同步的计时（用于测量 CPU 端操作或故意不同步的场景）
        """
        if name not in self.step_keys:
            self.step_keys.append(name)
        
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self.times[name].append(elapsed)
    
    def record(self, name, elapsed):
        """手动记录时间"""
        if name not in self.step_keys:
            self.step_keys.append(name)
        self.times[name].append(elapsed)
    
    def measure_load_imbalance(self, accelerator, name="load_imbalance"):
        """
        测量进程间的负载不均衡（barrier 等待时间）
        
        原理：每个进程到达 barrier 的时间不同，先到的要等后到的
        这个等待时间反映了各 GPU 计算速度的差异
        """
        if name not in self.step_keys:
            self.step_keys.append(name)
        
        if self.use_cuda:
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        
        if dist.is_initialized():
            dist.barrier()
        
        elapsed = time.perf_counter() - start
        self.times[name].append(elapsed)
        return elapsed
    
    def time_backward_separated(self, accelerator, model, loss):
        """
        分离测量 backward 的各个阶段：
        1. backward_compute: 纯反向传播计算
        2. gradient_sync: 梯度 AllReduce 同步
        
        关键技巧：使用 accelerator.no_sync() 来阻止自动梯度同步
        """
        for name in ["backward_compute", "gradient_sync", "backward_total"]:
            if name not in self.step_keys:
                self.step_keys.append(name)
        
        # 检查是否应该同步梯度（梯度累积的最后一步）
        should_sync = accelerator.sync_gradients
        
        if self.use_cuda:
            torch.cuda.synchronize()
        
        total_start = time.perf_counter()
        
        if should_sync:
            # === 方法1：如果是同步步，尝试分离计算和通信 ===
            
            # 先用 no_sync 执行 backward（只计算，不同步）
            compute_start = time.perf_counter()
            
            # 使用 no_sync 上下文
            with accelerator.no_sync(model):
                accelerator.backward(loss)
            
            if self.use_cuda:
                torch.cuda.synchronize()
            compute_time = time.perf_counter() - compute_start
            
            # 手动触发梯度同步
            sync_start = time.perf_counter()
            
            # 对于 DDP，手动执行 allreduce
            if hasattr(accelerator, 'reducer') or hasattr(model, 'reducer'):
                # 某些情况下需要手动触发
                pass  # DDP 会在下次 forward 时自动同步，或使用下面的方法
            
            # 使用 distributed 原语手动同步（更精确但需要小心处理）
            if dist.is_initialized() and accelerator.num_processes > 1:
                # 获取所有参数的梯度并执行 allreduce
                for param in model.parameters():
                    if param.grad is not None:
                        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            
            if self.use_cuda:
                torch.cuda.synchronize()
            sync_time = time.perf_counter() - sync_start
            
        else:
            # === 梯度累积中间步，不需要同步 ===
            compute_start = time.perf_counter()
            accelerator.backward(loss)
            
            if self.use_cuda:
                torch.cuda.synchronize()
            
            compute_time = time.perf_counter() - compute_start
            sync_time = 0.0
        
        total_time = time.perf_counter() - total_start
        
        self.times["backward_compute"].append(compute_time)
        self.times["gradient_sync"].append(sync_time)
        self.times["backward_total"].append(total_time)
        
        return compute_time, sync_time
    
    def time_backward_with_profiler(self, accelerator, loss, profile_memory=False):
        """
        使用 PyTorch Profiler 来分析 backward
        这是最精确的方法，可以看到 NCCL 通信的详细时间
        """
        if "backward_profiled" not in self.step_keys:
            self.step_keys.append("backward_profiled")
        
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.use_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=profile_memory,
            with_stack=True
        ) as prof:
            if self.use_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            
            accelerator.backward(loss)
            
            if self.use_cuda:
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        
        self.times["backward_profiled"].append(elapsed)
        
        return prof  # 返回 profiler 对象，可以进一步分析
    
    def collect_all_ranks(self, accelerator):
        """
        收集所有 rank 的计时数据到主进程
        用于分析各 GPU 的负载差异
        """
        if not dist.is_initialized():
            return
        
        world_size = accelerator.num_processes
        rank = accelerator.process_index
        
        # 将本进程的数据转为 tensor
        all_rank_data = {}
        
        for key in self.step_keys:
            local_times = torch.tensor(self.times[key], dtype=torch.float32)
            if self.use_cuda:
                local_times = local_times.cuda()
            
            # 收集所有 rank 的数据
            gathered = [torch.zeros_like(local_times) for _ in range(world_size)]
            dist.all_gather(gathered, local_times)
            
            all_rank_data[key] = [t.cpu().numpy() for t in gathered]
        
        self.all_rank_data = all_rank_data
    
    def print_summary(self, accelerator, show_per_rank=True):
        """打印详细的性能摘要"""
        if not accelerator.is_main_process:
            return
        
        if not self.step_keys:
            return
        
        # 跳过预热步
        effective_times = {
            k: v[self.warmup_steps:] if len(v) > self.warmup_steps else v
            for k, v in self.times.items()
        }
        
        num_steps = max(len(effective_times[k]) for k in self.step_keys) if self.step_keys else 0
        
        output = []
        output.append("\n" + "=" * 140)
        output.append(f"{'详细性能分析报告 - ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^140s}")
        output.append(f"{'(跳过前 ' + str(self.warmup_steps) + ' 步预热)':^140s}")
        output.append("=" * 140)
        
        # === 1. 每步详情 ===
        output.append("\n【每步耗时详情 (ms)】")
        headers = ["Step"] + self.step_keys + ["Total"]
        col_width = max(15, max(len(h) for h in headers) + 2)
        header_str = "".join([f"{h:>{col_width}s}" for h in headers])
        output.append(header_str)
        output.append("-" * len(header_str))
        
        for i in range(num_steps):
            row_vals = []
            step_total = 0.0
            for key in self.step_keys:
                vals = effective_times.get(key, [])
                if i < len(vals):
                    val = vals[i] * 1000  # 转为 ms
                else:
                    val = 0.0
                step_total += val
                row_vals.append(f"{val:{col_width}.2f}")
            
            row_str = f"{i:>{col_width}d}" + "".join(row_vals) + f"{step_total:{col_width}.2f}"
            output.append(row_str)
        
        # === 2. 统计摘要 ===
        output.append("\n" + "=" * 140)
        output.append(f"{'统计摘要':^140s}")
        output.append("=" * 140)
        
        total_time_all = sum(sum(v) for v in effective_times.values())
        
        stats_header = f"{'阶段':<25s} | {'平均(ms)':>12s} | {'标准差(ms)':>12s} | {'最小(ms)':>12s} | {'最大(ms)':>12s} | {'总计(s)':>12s} | {'占比':>8s}"
        output.append(stats_header)
        output.append("-" * len(stats_header))
        
        for name in self.step_keys:
            values = effective_times.get(name, [])
            if values and any(v > 0 for v in values):
                # 过滤掉 0 值进行统计（针对梯度累积场景）
                non_zero_values = [v for v in values if v > 0]
                if non_zero_values:
                    avg = statistics.mean(non_zero_values) * 1000
                    std = statistics.stdev(non_zero_values) * 1000 if len(non_zero_values) > 1 else 0
                    min_val = min(non_zero_values) * 1000
                    max_val = max(non_zero_values) * 1000
                    total = sum(values)
                    ratio = (total / total_time_all) * 100 if total_time_all > 0 else 0
                    
                    output.append(
                        f"{name:<25s} | {avg:>12.2f} | {std:>12.2f} | {min_val:>12.2f} | "
                        f"{max_val:>12.2f} | {total:>12.2f} | {ratio:>7.1f}%"
                    )
        
        # === 3. 性能瓶颈分析 ===
        output.append("\n" + "=" * 140)
        output.append(f"{'瓶颈分析':^140s}")
        output.append("=" * 140)
        
        # 找出占比最高的阶段
        ratios = {}
        for name in self.step_keys:
            values = effective_times.get(name, [])
            if values:
                total = sum(values)
                ratios[name] = total / total_time_all if total_time_all > 0 else 0
        
        sorted_ratios = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
        
        output.append("耗时排名 (从高到低):")
        for i, (name, ratio) in enumerate(sorted_ratios[:5], 1):
            bar_len = int(ratio * 50)
            bar = "█" * bar_len + "░" * (50 - bar_len)
            output.append(f"  {i}. {name:<20s} [{bar}] {ratio*100:.1f}%")
        
        # === 4. 建议 ===
        output.append("\n【优化建议】")
        if "data_loading" in ratios and ratios.get("data_loading", 0) > 0.2:
            output.append("  ⚠️  数据加载占比较高 (>20%)，建议：增加 num_workers、使用 pin_memory、预取数据")
        
        if "gradient_sync" in ratios and ratios.get("gradient_sync", 0) > 0.3:
            output.append("  ⚠️  梯度同步占比较高 (>30%)，建议：增大 batch size、使用梯度累积、检查网络带宽")
        
        if "load_imbalance" in ratios and ratios.get("load_imbalance", 0) > 0.1:
            output.append("  ⚠️  负载不均衡严重 (>10%)，建议：检查各 GPU 的数据分布是否均匀")
        
        forward_time = ratios.get("forward", 0)
        backward_time = ratios.get("backward_compute", ratios.get("backward", 0))
        if backward_time > 0 and forward_time > 0:
            ratio_fb = backward_time / forward_time
            output.append(f"  📊 Backward/Forward 比值: {ratio_fb:.2f} (正常范围: 2.0-3.0)")
        
        final_log = "\n".join(output)
        
        print(final_log)
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(final_log + "\n")
        print("completed!")



class StepTimer:
    def __init__(self, log_file="training_perf.log"):
        self.times = defaultdict(list)
        self.step_keys = []
        self.log_file = log_file

    @contextmanager
    def time_step(self, name):
        if name not in self.step_keys:
            self.step_keys.append(name)
        # 确保 GPU 同步，否则时间统计不准
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        self.times[name].append(elapsed)

    def record(self, name, elapsed):
        if name not in self.step_keys:
            self.step_keys.append(name)
        self.times[name].append(elapsed)

    def print_summary(self, accelerator):
        # 仅在主进程中执行打印和写文件
        if not accelerator.is_main_process:
            return

        if not self.step_keys:
            return
            
        # 以记录最多的 key 为准（通常是 data_loading）
        num_steps = max(len(self.times[k]) for k in self.step_keys)
        
        output = []
        output.append("\n" + "="*120)
        output.append(f"{'Step 耗时详情 (单位: ms) - ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^120s}")
        output.append("="*120)
        
        headers = ["Step"] + self.step_keys + ["Total"]
        col_width = 15 
        header_str = "".join([f"{h:>{col_width}s}" for h in headers])
        output.append(header_str)
        output.append("-" * len(header_str))

        for i in range(num_steps):
            row_vals = []
            step_total = 0.0
            for key in self.step_keys:
                # 健壮性处理：如果某项没有记录（比如梯度累积跳过了），记为 0
                if i < len(self.times[key]):
                    val = self.times[key][i] * 1000
                else:
                    val = 0.0
                step_total += val
                row_vals.append(f"{val:{col_width}.2f}")
            
            row_str = f"{i:>{col_width}d}" + "".join(row_vals) + f"{step_total:{col_width}.2f}"
            output.append(row_str)

        # 统计摘要
        output.append("\n" + "="*120)
        output.append(f"{'统计摘要 (平均值)':^120s}")
        output.append("="*120)
        total_time_all_steps = sum(sum(v) for v in self.times.values())
        
        for name in self.step_keys:
            values = self.times[name]
            if values:
                avg = statistics.mean(values) * 1000
                total = sum(values)
                ratio = (total / total_time_all_steps) * 100 if total_time_all_steps > 0 else 0
                output.append(f"{name:<25s} | 平均: {avg:10.2f} ms | 总计: {total:10.2f} s | 占比: {ratio:7.1f}%")
        
        final_log = "\n".join(output)
        
        # 1. 打印到终端
        print(final_log)
        
        # 2. 保存到文件
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(final_log + "\n")

def diagnose_default_training_status(model):
    """
    诊断模型当前的默认训练状态（在人工修改 requires_grad 之前），debug模式下才输出
    """
    print("\n" + "="*50)
    print("🕵️ [诊断模式] 检查模型默认训练状态...")
    print("="*50)
    
    trainable_params = []
    frozen_params = []
    
    trainable_numel = 0
    frozen_numel = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
            trainable_numel += param.numel()
        else:
            frozen_params.append(name)
            frozen_numel += param.numel()
            
    # 统计数据
    total_layers = len(trainable_params) + len(frozen_params)
    total_params = trainable_numel + frozen_numel
    
    print(f"📊 统计结果:")
    print(f"   - 总层数 (Keys): {total_layers}")
    print(f"   - 总参数量 (Elements): {total_params / 1e9:.2f} B (十亿)")
    print(f"   -------------------------------------------")
    print(f"   🔓 可训练层数 (Trainable): {len(trainable_params)}")
    print(f"      - 参数量: {trainable_numel / 1e9:.2f} B")
    print(f"      - 占比: {trainable_numel / total_params * 100:.2f}%")
    print(f"   🔒 不可训练层数 (Frozen): {len(frozen_params)}")
    print(f"      - 参数量: {frozen_numel / 1e9:.2f} B")
    print(f"   -------------------------------------------")
    
    # 打印具体名字（为了防止刷屏，每种只打印前5个和后5个）
    if len(trainable_params) > 0:
        print(f"\n📝 可训练参数示例 (Top 5):")
        for p in trainable_params[:10]:
            print(f"   - [√] {p}")
        if len(trainable_params) > 10: print("   ... (中间省略) ...")
        # 打印最后几个，看看音频部分在不在
        for p in trainable_params[-10:]:
            print(f"   - [√] {p}")
            
    if len(frozen_params) > 0:
        print(f"\n🧊 不可训练参数示例 (Top 5):")
        for p in frozen_params[:10]:
            print(f"   - [x] {p}")
            
    print("="*50 + "\n")


def prepare_model_and_optimizer_groups(model, base_lr=1e-5, target_lr=1e-4):
    # 1. 定义高学习率（且需要置零）的目标模块前缀
    target_prefixes = (
        "audio_injector", 
        # "trainable_cond_mask", 
        # "frame_packer"
    )
    
    # 2. 容器初始化
    high_lr_params = []
    low_lr_params = []
    
    # 统计用变量
    stats = {
        "high_lr_count": 0,    # 高学习率参数个数
        "low_lr_count": 0,     # 低学习率参数个数 (Backbone中原本可训练的)
        "frozen_skipped": 0,   # 被跳过的冻结参数 (如 TextEncoder)
        "zero_value_count": 0, # 实际值为0的参数个数
        "total_params": 0
    }

    # 3. 遍历模型所有参数
    for name, param in model.named_parameters():
        stats["total_params"] += 1
        
        # 判断是否属于目标模块 (Audio/Mask/Packer)
        is_target_module = any(prefix in name for prefix in target_prefixes)
        
        if is_target_module:
            # ============================================
            # A. 目标模块：强制训练 + 强制置零 + 高学习率
            # ============================================
            param.requires_grad = True # 确保开启
            
            # 执行全量置零
            # with torch.no_grad():
            #     param.zero_()
            
            high_lr_params.append(param)
            stats["high_lr_count"] += 1
            
            # 验证置零
            # if param.sum() == 0: #! bug:这个条件对于浮点数不够稳定
            if torch.allclose(param, torch.zeros_like(param)):
                stats["zero_value_count"] += 1
                
        else:
            # ============================================
            # B. 非目标模块：尊重原状态 (只收录本来就开了梯度的)
            # ============================================
            if param.requires_grad:
                # 原本就是可训练的 (比如 Backbone 的 Attention) -> 低学习率
                low_lr_params.append(param)
                stats["low_lr_count"] += 1
            else:
                # 原本就是冻结的 (比如 Text Encoder) -> 跳过，不进优化器
                stats["frozen_skipped"] += 1

    # # 4. 打印详细统计报告
    # print(f"\n📊 参数统计报告:")
    # print(f"   -------------------------------------------")
    # print(f"   [Total] 模型总参数层数: {stats['total_params']}")
    # print(f"   -------------------------------------------")
    # print(f"   🔥 [High LR Group] (Target Modules, lr={target_lr})")
    # print(f"       - 包含: {target_prefixes}")
    # print(f"       - 数量: {stats['high_lr_count']}")
    # print(f"       - 置零验证: {stats['zero_value_count']} / {stats['high_lr_count']} (应相等)")
    
    # print(f"   ❄️ [Low LR Group] (Backbone SFT, lr={base_lr})")
    # print(f"       - 数量: {stats['low_lr_count']}")
    # print(f"       - 说明: 这些是SFT权重中原本开启梯度的部分")
    
    # print(f"   🧊 [Skipped/Frozen] (Not Training)")
    # print(f"       - 数量: {stats['frozen_skipped']}")
    # print(f"       - 说明: 这些参数保持冻结，不消耗显存存梯度 (如TextEncoder)")
    # print(f"   -------------------------------------------")

    # 5. 构建优化器所需的参数组列表
    optimizer_grouped_parameters = [
        {
            "params": low_lr_params, 
            "lr": base_lr,
            "name": "backbone_low_lr"
        },
        {
            "params": high_lr_params, 
            "lr": target_lr,
            "name": "audio_new_high_lr"
        }
    ]
    
    return optimizer_grouped_parameters

def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        # small_lr_rate = 1e-5
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        debug = args.debug
    
    if debug:
        diagnose_default_training_status(model)
    # optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_grouped_parameters = prepare_model_and_optimizer_groups(
        model, 
        base_lr=1e-5, 
        target_lr=learning_rate
    )
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers) if debug else torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    if debug:
        model_logger.on_training_start(accelerator, model)
        log_name = f"perf_debug_{accelerator.process_index}_{datetime.now().strftime('%m%d_%H%M')}.log"
        timer = DetailedStepTimer(log_file=log_name, warmup_steps=2)
        
        for epoch_id in range(num_epochs):
            end_time = time.perf_counter()
            
            for step_index, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch_id}", 
                                                    disable=not accelerator.is_main_process)):
                if step_index > 14:
                    break
                # for i, group in enumerate(optimizer.param_groups):
                #     print(f"Group {i} ({group.get('name', 'unnamed')}): lr = {group['lr']}")
                
                timer.current_step = step_index
                
                # 1. 数据加载时间（CPU 端）
                data_load_time = time.perf_counter() - end_time
                timer.record("data_loading", data_load_time)
                
                # 2. 数据传输到 GPU 的时间（如果数据还在 CPU）
                with timer.time_step("data_to_gpu"):
                    # 如果你的 data 需要手动移动到 GPU
                    # data = {k: v.to(accelerator.device) for k, v in data.items()}
                    pass  # Accelerate 通常已经处理了这个
                
                with accelerator.accumulate(model):
                    
                    with timer.time_step("zero_grad"):
                        optimizer.zero_grad()
                    
                    with timer.time_step("forward"):
                        loss = model(data)
                    
                    # === 关键：分离 backward 计算和梯度同步 ===
                    if accelerator.sync_gradients:
                        # 这是梯度累积的最后一步，会触发 AllReduce
                        
                        # 方法 A：简单测量总时间（推荐先用这个）
                        with timer.time_step("backward_total"):
                            accelerator.backward(loss)
                        
                        # 方法 B：尝试分离计算和同步（需要 no_sync 支持）
                        # timer.time_backward_separated(accelerator, model, loss)
                        
                        # 测量进程间等待（负载不均衡指标）
                        timer.measure_load_imbalance(accelerator, "pre_step_barrier")
                        
                        with timer.time_step("optimizer_step"):
                            optimizer.step()
                        
                        with timer.time_step("model_logger"):
                            model_logger.on_step_end(accelerator, model, save_steps)
                        
                        with timer.time_step("scheduler_step"):
                            scheduler.step()
                            
                    else:
                        # 梯度累积中间步，不同步
                        with timer.time_step("backward_no_sync"):
                            accelerator.backward(loss)
                        
                        timer.record("pre_step_barrier", 0)
                        timer.record("optimizer_step", 0)
                        timer.record("model_logger", 0)
                        timer.record("scheduler_step", 0)
                
                end_time = time.perf_counter()
        
        accelerator.wait_for_everyone()
        timer.print_summary(accelerator)
        model_logger.on_training_end(accelerator, model, save_steps)
    else:
        model_logger.on_training_start(accelerator, model)
    
        for epoch_id in range(num_epochs):
            for data in tqdm(dataloader):
                with accelerator.accumulate(model):
                    optimizer.zero_grad() # PyTorch 默认会累积梯度，所以每次迭代开始要手动清零
                    loss = model(data) 
                    accelerator.backward(loss) # 计算 loss 对所有可训练参数的梯度,DL的核心——链式法则求导
                    optimizer.step() # 根据梯度更新模型权重, 新权重 = 旧权重 - 学习率 × 梯度
                    model_logger.on_step_end(accelerator, model, save_steps)
                    scheduler.step() # 随着训练进行，调整学习率（通常是逐渐减小）
        accelerator.wait_for_everyone() # 确保所有进程完成
        model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
