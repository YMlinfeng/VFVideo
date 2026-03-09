import cv2
import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
from dwpose import DWposeDetector
from tqdm import tqdm  # 导入 tqdm 库
# Set configs
det_config = './dwpose/yolox_config/yolox_l_8xb8-300e_coco.py'
det_ckpt = './models/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
pose_config = './dwpose/dwpose_config/dwpose-l_384x288.py'
pose_ckpt = './models/dw-ll_ucoco_384.pth'

det_onnx = './models/yolox_l.onnx'
pose_onnx = './models/dw-ll_ucoco_384.onnx'

det_trt = './models/yolox_l.engine'
pose_trt = './models/dw-ll_ucoco_384.engine'

out_dir = "outputs_temp"
os.makedirs(out_dir, exist_ok=True)

# Set device
device = "cuda:0"

def process_frame(dwpose, frame, filename, plot):
    time1 = time.time()
    candidate_133, subset_133, bbox, output_img = dwpose(image_np_hwc=frame, show_body=True,
                        show_face=True, show_hands=True, plot=plot)
    time2 = time.time()

    if plot:
        cv2.imwrite(os.path.join(out_dir, filename), output_img)

    return filename, candidate_133, subset_133, bbox, output_img, time2 - time1

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Choose the type of program to run.")
    parser.add_argument('type', nargs='?', default='pt', choices=['trt', 'onnx', 'pt', 'pth'], help="Type of the program to run.")
    parser.add_argument('--plot', action='store_true', help="Whether to show the plot.")
    parser.add_argument('--folder', required=True, help="Folder containing images to process.")
    parser.add_argument('--threads', type=int, default=4, help="Number of threads to use.")
    args = parser.parse_args()

    print("Run model type: ", args.type)

    # Init DWposeDetector
    dwpose = DWposeDetector(det_config, det_ckpt, pose_config, pose_ckpt,
             det_onnx=det_onnx, pose_onnx=pose_onnx,
             det_trt=det_trt, pose_trt=pose_trt,
             device=device, type=args.type)

    # Get all image files in the specified folder, sorted by filename
    image_files = sorted([f for f in os.listdir(args.folder) if f.endswith(('.jpg', '.jpeg', '.png'))], key=lambda x: int(os.path.splitext(x)[0]))

    time_s = time.time()
    t2 = 0
    results = []

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = []
        for filename in image_files:
            file_path = os.path.join(args.folder, filename)
            frame = cv2.imread(file_path)
            futures.append(executor.submit(process_frame, dwpose, frame, filename, args.plot))

        # 使用 tqdm 显示进度条
        for future in tqdm(futures, desc="Processing frames", unit="frame"):
            filename, candidate_133, subset_133, bbox, output_img, duration = future.result()
            results.append((filename, candidate_133, subset_133, bbox, output_img))
            t2 += duration

    time_e = time.time()

    print("time infer,", t2 / len(futures))
    print("time all,", time_e - time_s)
    

    # 处理汇总结果，按需要存储或使用
    # 例如，将结果保存到文件或进一步处理

if __name__ == "__main__":
    #command
    # CUDA_VISIBLE_DEVICES=0 python inference_image_ori_oneGPU_multiProcess.py pt --threads 8 --plot --folder /home/zhouyan03/.jupyter/temp_videos/1/                              
    main()
