import os, torch
from accelerate import Accelerator


class ModelLogger:
    def __init__(
        self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x,
        use_swanlab=False, swanlab_mode="local", swanlab_project="wan_video", swanlab_run_name=None, args_dict=None
    ):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        
        self.use_swanlab = use_swanlab
        self.swanlab_mode = swanlab_mode
        self.swanlab_project = swanlab_project
        self.swanlab_run_name = swanlab_run_name
        self.args_dict = args_dict
        
        if self.use_swanlab:
            try:
                import swanlab
                self.swanlab = swanlab
            except ImportError:
                print("[Warning] SwanLab is not installed. Disabling SwanLab tracking. Install via `pip install swanlab`.")
                self.use_swanlab = False

    def on_training_start(self, accelerator: Accelerator, model: torch.nn.Module):
        """在训练开始前保存初始模型，方便后续对比，并初始化SwanLab"""
        if self.use_swanlab and accelerator.is_main_process:
            self.swanlab.init(
                project=self.swanlab_project,
                name=self.swanlab_run_name,
                mode=self.swanlab_mode,  # 'local' for offline clusters, 'cloud' for online
                config=self.args_dict
            )
            print(f"🚀 [SwanLab] Initialized tracking in '{self.swanlab_mode}' mode.")
            
        self.save_model(accelerator, model, "initial.safetensors")
        # print("save initial.safetensors...")

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, loss=None, lr=None):
        self.num_steps += 1
        
        # 实时记录监控指标
        if self.use_swanlab and accelerator.is_main_process:
            log_dict = {}
            if loss is not None:
                log_dict["train/loss"] = loss
            if lr is not None:
                log_dict["train/lr"] = lr
            if log_dict:
                self.swanlab.log(log_dict, step=self.num_steps)
                
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")
            
        if self.use_swanlab and accelerator.is_main_process:
            print(f"🏁 [SwanLab] Training completed. Finalizing log...")
            self.swanlab.finish()

    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)

