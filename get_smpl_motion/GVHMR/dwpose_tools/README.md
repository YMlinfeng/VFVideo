<div align="center">

# Dwpose Tensorrt ⚡

## 环境
开发机：zy-V100
anaconda环境：movie-asd

## 代码
```
python infer_image_lists.py # 单个图像目录

python infer_json.py #通过json文件批量跑

```













[![python](https://img.shields.io/badge/python-3.10.12-green)](https://www.python.org/downloads/release/python-31012/)
[![cuda](https://img.shields.io/badge/cuda-12.4-green)](https://developer.nvidia.com/cuda-downloads)
[![trt](https://img.shields.io/badge/TRT-10.2-green)](https://developer.nvidia.com/tensorrt)
[![mit](https://img.shields.io/github/license/saltstack/salt)](https://www.apache.org/licenses/LICENSE-2.0)

</div>

<p align="center">
  <img height=250 src="assets/demo.png" />
</p>

This repo provides a Tensorrt implementation of [Dwpose](https://github.com/IDEA-Research/DWPose) for ultra fast pose detection on images and videos

## ⏱️ Performance

| Device | FPS |
| :----: | :-: |
|  L40S  | 22  |

## 🚀 Installation

```bash
git clone https://github.com/yuvraj108c/Dwpose-Tensorrt
cd ./Dwpose-Tensorrt
pip install -r requirements.txt
```

## ☀️ Usage

1. Download the following onnx models into `models` folder:
   - [dw-ll_ucoco_384.onnx](https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx)
   - [yolox_l.onnx](https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx)
2. Build tensorrt engines for both of these models by running:

   - `python export_trt.py`

3. Run the inference scripts to perform pose detection on images or videos
   - `python inference_image.py`
   - `python inference_video.py`

## 🤖 Environment tested

- Ubuntu 22.04 LTS, Cuda 12.4, Tensorrt 10.2.0.post1, Python 3.10, L40S GPU
- Windows (Not tested, but should work)

## 👏 Credits

- https://github.com/IDEA-Research/DWPose
- https://github.com/legraphista/dwpose-video

## License

Apache 2.0
