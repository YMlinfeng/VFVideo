from ..vram.initialization import skip_model_initialization
from ..vram.disk_map import DiskMap
from ..vram.layers import enable_vram_management
from .file import load_state_dict
import torch

def zero_init_controlnet_style(model):    
    target_prefixes = (
        # "cond_encoder",
        # "casual_audio_encoder", 
        "audio_injector.injector",
        # "trainable_cond_mask",
        # "frame_packer"
    )
    
    zero_count = 0
    kept_count = 0
    
    for name, param in model.named_parameters():
        # 检查参数名是否以前缀开头
        if any(name.startswith(prefix) for prefix in target_prefixes):
            with torch.no_grad():
                param.zero_()  # 核心操作：全部填 0
                zero_count += 1
                # print(f"   [置零] {name}") # 想要看详细列表可以取消注释
        else:
            kept_count += 1

    print(f"zero procession completed")
    print(f"   - zero count: {zero_count} ")
    print(f"   - kept_count: {kept_count} ")


def load_model(model_class, path, config=None, torch_dtype=torch.bfloat16, device="cpu", state_dict_converter=None, use_disk_map=False, module_map=None, vram_config=None, vram_limit=None):
    config = {} if config is None else config
    # Why do we use `skip_model_initialization`?
    # It skips the random initialization of model parameters,
    # thereby speeding up model loading and avoiding excessive memory usage.
    with skip_model_initialization():
        model = model_class(**config)
    # What is `module_map`?
    # This is a module mapping table for VRAM management.
    if module_map is not None:
        devices = [vram_config["offload_device"], vram_config["onload_device"], vram_config["preparing_device"], vram_config["computation_device"]]
        device = [d for d in devices if d != "disk"][0]
        dtypes = [vram_config["offload_dtype"], vram_config["onload_dtype"], vram_config["preparing_dtype"], vram_config["computation_dtype"]]
        dtype = [d for d in dtypes if d != "disk"][0]
        if vram_config["offload_device"] != "disk":
            state_dict = DiskMap(path, device, torch_dtype=dtype)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            else:
                state_dict = {i: state_dict[i] for i in state_dict}
            model.load_state_dict(state_dict, assign=True)
            model = enable_vram_management(model, module_map, vram_config=vram_config, disk_map=None, vram_limit=vram_limit)
        else:
            disk_map = DiskMap(path, device, state_dict_converter=state_dict_converter)
            model = enable_vram_management(model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=vram_limit)
    else:
        # Why do we use `DiskMap`?
        # Sometimes a model file contains multiple models,
        # and DiskMap can load only the parameters of a single model,
        # avoiding the need to load all parameters in the file.
        if use_disk_map:
            state_dict = DiskMap(path, device, torch_dtype=torch_dtype)
        else:
            state_dict = load_state_dict(path, torch_dtype, device)
        # Why do we use `state_dict_converter`?
        # Some models are saved in complex formats,
        # and we need to convert the state dict into the appropriate format.
        if state_dict_converter is not None:
            state_dict = state_dict_converter(state_dict)
        else:
            state_dict = {i: state_dict[i] for i in state_dict} # 这一步很慢，显存在这里升高到34G

        # 拦截并修改 state_dict
        # 1. 解决尺寸不匹配 (Size Mismatch)
        # 检查是否试图把 I2V (36通道) 的权重塞进 S2V (16通道) 的模型
        # if "patch_embedding.weight" in state_dict:
        #     weight = state_dict["patch_embedding.weight"]
        #     # 判断条件: 当前加载的权重是 36 通道，但模型类名包含 S2V
        #     if weight.shape[1] == 36 and "S2V" in model.__class__.__name__:
        #         print(f"检测到 I2V 权重 (36通道) 正在注入 S2V 模型。执行裁剪操作...")
        #         # 只保留前 16 个通道，丢弃参考图和Mask通道
        #         state_dict["patch_embedding.weight"] = weight[:, :16, :, :, :].contiguous() # !!坑点
        #         '''
        #         这个切分操作会造成内存不连续，因为torch的底层是用指针而非copy来找到数据
        #         如果在cpu上做切分操作，accelerate打包的时候会自动copy这16本书到GPU上，contiguous（但推理时没用到acc框架，仍然报错）
        #         但如果在gpu上做切分操作，就会报错【GPU 显存里的这个 Parameter 是不连续的（镂空的）】
        #         '''
        #         '''
        #         DeepSpeed 广播：DeepSpeed 启动分布式训练，第一件事是做 Broadcast（把 Rank 0 的权重同步给所有人）。
        #             分布式通信（NCCL）为了追求极致速度，要求直接传输一块完整的内存块。
        #             它一检查：“哎？这块内存怎么全是窟窿？中间这些数据发不发？发了是错的，不发我不知道怎么读。”
        #             结果：直接抛出 ValueError: Tensors must be contiguous。
        #         所以，在做任何张量切片（Slicing）、转置（Transpose/Permute）操作后，如果后续涉及到底层通信或特定算子，加上 .contiguous() 是一个良好的编程习惯。
        #         '''
        
        # 2. 解决键缺失 (Missing Keys)
        # 尝试加载，如果因为 Missing Keys 报错，则启用 strict=False
        # try:
        #     # 尝试标准加载
        #     model.load_state_dict(state_dict, assign=True)
        # except RuntimeError as e:
        #     error_msg = str(e)
        #     # 如果是 Missing key 错误，且我们是在搞 S2V，说明是音频层缺失，允许跳过
        #     if "Missing key(s)" in error_msg and "S2V" in model.__class__.__name__:
        #         print(f"检测到音频层缺失 (符合预期)。启用 strict=False 加载...")
        #         # 注意：assign=True 和 strict=False 同时使用取决于 PyTorch 版本
        #         # 如果报错不支持 assign 参数，请去掉 assign=True
        #         model.load_state_dict(state_dict, strict=False, assign=True)
        #     else:
        #         # 如果是其他错误 (比如 shape 依然不对)，则抛出
        #         raise e
        # print("检查并初始化剩余的Meta参数...")
        # for name, _ in model.named_parameters():
        #     # 必须逐级查找参数，因为 .named_parameters() 返回的是副本或视图
        #     # 我们需要修改原始模型中的属性
        #     parts = name.split('.')
        #     module = model
        #     for part in parts[:-1]:
        #         module = getattr(module, part)
        #     attr_name = parts[-1]
        #     param = getattr(module, attr_name)
            
        #     # 如果参数依然在 'meta' 设备上，说明它没被加载到，需要随机初始化
        #     if str(param.device) == 'meta':
        #         # print(f"   -> 初始化: {name}") # 太多了可以不打印
        #         # 创建一个新的实体 Tensor
        #         new_param = torch.empty(param.size(), dtype=torch_dtype, device=device)
        #         # 使用正态分布初始化 (mean=0, std=0.02 是 DiT 的常用初始化)
        #         torch.nn.init.normal_(new_param, mean=0.0, std=0.02)
                
        #         # 包装回 Parameter
        #         if isinstance(param, torch.nn.Parameter):
        #             new_param = torch.nn.Parameter(new_param)
                
        #         # 替换掉原本的 Meta Tensor
        #         setattr(module, attr_name, new_param)
        
        # print("所有参数实体化完成！")
        #--- 以上为修改 ---#



        # 1. 先加载 S2V checkpoint
        model.load_state_dict(state_dict, strict=False, assign=True) # assign,控制内存填充的方式

        # 2. 选择一种方式初始化音频模块
        # 方式一（推荐）： 
        # zero_init_controlnet_style(model)

        # 方式二：
        # zero_init_all(model)

        # model.load_state_dict(state_dict, assign=True)  # old version

        # Why do we call `to()`?
        # Because some models override the behavior of `to()`,
        # especially those from libraries like Transformers.

        model = model.to(dtype=torch_dtype, device=device)
    if hasattr(model, "eval"):
        model = model.eval()
    return model # nonzero_keys = [k for k, v in model.state_dict().items() if not v.eq(0).all()]


def load_model_with_disk_offload(model_class, path, config=None, torch_dtype=torch.bfloat16, device="cpu", state_dict_converter=None, module_map=None):
    if isinstance(path, str):
        path = [path]
    config = {} if config is None else config
    with skip_model_initialization():
        model = model_class(**config)
    if hasattr(model, "eval"):
        model = model.eval()
    disk_map = DiskMap(path, device, state_dict_converter=state_dict_converter)
    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": "disk",
        "onload_device": "disk",
        "preparing_dtype": torch.float8_e4m3fn,
        "preparing_device": device,
        "computation_dtype": torch_dtype,
        "computation_device": device,
    }
    enable_vram_management(model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=80)
    return model
