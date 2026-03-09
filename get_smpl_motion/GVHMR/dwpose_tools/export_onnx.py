from mmpose.apis import init_model as init_pose_estimator
import torch
import onnxruntime
import numpy as np
import onnx
# 配置文件和权重路径
pose_config = "./models/rtmw-x_8xb320-270e_cocktail14-384x288.py"
pose_ckpt = "./models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth"

# 初始化模型并设置 eval 模式
model = init_pose_estimator(
    pose_config,
    pose_ckpt,
    device="cuda:0"
)
model.eval()

# 冻结 BatchNorm 层，确保精度一致性
for module in model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = False
        module.eval()

# 定义输入张量并移动到 GPU
input_tensor = torch.rand((1, 3, 384, 288), dtype=torch.float32).to("cuda:0")

# 导出 ONNX 模型
onnx_path = "./models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.onnx"
torch.onnx.export(
    model,
    (input_tensor,),
    onnx_path,
    export_params=True,
    opset_version=15,
    do_constant_folding=False,  # 禁止常量折叠以减少数值误差
    # verbose=True,       # 启用详细输出，查看算子转换情况
    input_names=['input'],
    output_names=['output']
)
print(f"ONNX 模型已导出至: {onnx_path}")
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)  # 检查模型是否符合 ONNX 标准
print("ONNX 模型检查通过")