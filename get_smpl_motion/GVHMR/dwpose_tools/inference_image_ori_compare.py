import cv2
import os
from dwpose import DWposeDetector
import time
import argparse
import glob
from tqdm import tqdm

time0 = time.time()

# set configs
det_config = './dwpose/yolox_config/yolox_l_8xb8-300e_coco.py'
det_ckpt = './models/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
pose_config = './dwpose/dwpose_config/dwpose-l_384x288.py'
pose_ckpt = './models/dw-ll_ucoco_384.pth'

det_onnx = './models/yolox_l.onnx'
pose_onnx = './models/dw-ll_ucoco_384.onnx'

det_trt = './models/yolox_l.engine'
pose_trt = './models/dw-ll_ucoco_384.engine'

folder = '/home/zhouyan03/.jupyter/temp_videos/a_speaker1_20s'
pattern = folder + '/*.jpg'

# 使用 glob 查找所有 .jpg 文件
jpg_files = glob.glob(pattern)
all_num = len(jpg_files)
print("all_num", all_num)

out_dir = "outputs_11_21"
os.makedirs(out_dir, exist_ok=True)

# set device
device = "cuda:0"

# 叠加关键点到图像
def draw_keypoints_on_image(image_input, keypoints_list, keypoint_scores_list, bboxs, H, W, threshold=0.3, normalized=False):
    image = image_input.copy()
    if not normalized:
        H = 1
        W = 1
    for kk in range(len(keypoints_list)):
        keypoints = keypoints_list[kk]
        keypoint_scores = keypoint_scores_list[kk]
        for i, (point, score) in enumerate(zip(keypoints, keypoint_scores)):
            if score > threshold:
                x, y = int(point[0] * W), int(point[1] * H)
                cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    for bbox in bboxs:
        # 画框
        cv2.rectangle(image, (int(bbox[0] * W), int(bbox[1] * H)), (int(bbox[2] * W), int(bbox[3] * H)), (0, 0, 255), 2)
    return image

def main():
    time_s = time.time()
    # 创建解析器
    parser = argparse.ArgumentParser(description="Choose the type of program to run.")
    
    # 添加参数
    parser.add_argument('type', nargs='?', default='pt', choices=['trt', 'onnx', 'pt', 'pth'], help="Type of the program to run.")
    parser.add_argument('--plot', default=True, help="Whether to show the plot.")
    
    # 解析命令行参数
    args = parser.parse_args()

    print("Run model type: ", args.type)
    print("args.plot", args.plot)
    # init
    dwpose = DWposeDetector(det_config, det_ckpt, pose_config, pose_ckpt, 
                            det_onnx=det_onnx, pose_onnx=pose_onnx, 
                            det_trt=det_trt, pose_trt=pose_trt, 
                            device=device, type=args.type)

    
    t2 = 0
    threshold = 0.5
    COL = 3
    # 如果 --plot 被启用，创建一个 VideoWriter 对象来保存视频

    VideoWriter_init = False
    # for i in range(1, all_num + 1):  # 从 1 开始，直到 all_num
    for i in tqdm(range(1, all_num + 1), desc="Processing", unit="image"):
        filename = f"{i:08d}.jpg"
        name = os.path.join(folder, filename)
        
        # 读取图像，添加检查
        frame = cv2.imread(name)
        if frame is None:
            print(f"Warning: Unable to read {name}. Skipping.")
            continue
        
        if not VideoWriter_init:
            VideoWriter_init = True
            if args.plot:
                # 假设原图和输出图像的大小是相同的，这里你可以根据实际情况调整
                frame_height, frame_width = frame.shape[:2]
                
                # 设置缩小的比例
                scale_factor = 0.5  # 缩小为 50%
                new_width = int(frame_width * scale_factor)
                new_height = int(frame_height * scale_factor)

                # 使用 mp4v 编码格式来生成 MP4 文件
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                # 创建视频写入对象，使用缩小后的尺寸
                out_video = cv2.VideoWriter(os.path.join(out_dir, f'output_video_{args.type}.mp4'), fourcc, 25, (new_width * COL, new_height))  # 帧率设置为 25fps

        time1 = time.time()
        candidate_133, subset_133, bbox, output_img = dwpose(image_np_hwc=frame, show_body=True,
                        show_face=True, show_hands=True, threshold=threshold, plot=args.plot)

        time2 = time.time()
        t2 += (time2 - time1) 
        
        if args.plot:
            candidate_133 = [candidate_133[i] for i in range(candidate_133.shape[0])]
            subset_133 = [subset_133[i] for i in range(subset_133.shape[0])]
            bboxs = [bbox[i] for i in range(bbox.shape[0])]

            frame_draw = draw_keypoints_on_image(frame, candidate_133, subset_133, bboxs, \
                            frame_height, frame_width, threshold=threshold, normalized=False)
            
            # 将原图和输出图像并排拼接
            combined_frame = cv2.hconcat([frame, frame_draw, output_img])  # 将左右两张图拼接在一起

            # 缩小图像
            resized_combined_frame = cv2.resize(combined_frame, (new_width * COL, new_height))  # 按照缩放比例调整图像尺寸

            # 写入视频
            out_video.write(resized_combined_frame)

    # 完成视频保存
    if args.plot:
        out_video.release()

    time_e = time.time()

    print("time infer,", t2 / all_num)  # 使用 all_num 计算每张图的平均推理时间
    print("time all,", time_e - time_s)


if __name__ == "__main__":
    main()
