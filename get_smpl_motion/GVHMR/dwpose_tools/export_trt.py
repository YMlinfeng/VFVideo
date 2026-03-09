import torch
import time
from trt_utilities import Engine


def export_trt(trt_path: str, onnx_path: str, use_fp16: bool):
    engine = Engine(trt_path)

    torch.cuda.empty_cache()

    s = time.time()
    ret = engine.build(
        onnx_path,
        use_fp16,
        enable_preview=True,
    )
    e = time.time()
    print(f"Time taken to build: {(e-s)} seconds")

    return ret


export_trt(trt_path="./models/yolox_l_fp32_trt10.4.0.engine",
           onnx_path="./models/yolox_l.onnx", use_fp16=False)
export_trt(trt_path="./models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122_fp32_trt10.4.0.engine",
           onnx_path="./models/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.onnx", use_fp16=False)
