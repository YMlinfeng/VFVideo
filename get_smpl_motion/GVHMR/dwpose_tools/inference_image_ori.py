import cv2
import os
from dwpose import DWposeDetector
import time
import argparse

time0 = time.time()
frame = cv2.imread("./assets/usain-bolt.jpg")


#

# set configs
det_config = './dwpose/yolox_config/yolox_l_8xb8-300e_coco.py'
det_ckpt = './models/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
pose_config = './dwpose/dwpose_config/dwpose-l_384x288.py'
pose_ckpt = './models/dw-ll_ucoco_384.pth'

det_onnx = './models/yolox_l.onnx'
pose_onnx = './models/dw-ll_ucoco_384.onnx'

det_trt = './models/yolox_l.engine'
pose_trt = './models/dw-ll_ucoco_384.engine'

#folder = '/mmu_audio_hdd/MILM_data/SpeakingVideosImage/oneSpeaker/vhuman_project_formal/koubo-xiaoxuan1/'
#folder = '/mmu_audio_hdd/MILM_data/SpeakingVideosImage/oneSpeaker/vhuman_project_formal/koubo-zile3'

out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)
# dwpose = DWposeDetector()
# set device
device = "cuda:0"


def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description="Choose the type of program to run.")
    
    # 添加参数
    parser.add_argument('type', nargs='?', default='pt', choices=['trt', 'onnx', 'pt', 'pth'], help="Type of the program to run.")
    parser.add_argument('--plot', action='store_true', help="Whether to show the plot.")
    
    # 解析命令行参数
    args = parser.parse_args()

    print("Run model type: ", args.type)
    # init
    dwpose = DWposeDetector(det_config, det_ckpt, pose_config, pose_ckpt,\
             det_onnx = det_onnx, pose_onnx=pose_onnx, \
             det_trt = det_trt, pose_trt=pose_trt, \
             device = device, type=args.type)

    time_s = time.time()
    t2 = 0
    for i in range(1, 2):
        filename = f"{i:08d}.jpg"
        #name = os.path.join(folder, filename)
        #frame = cv2.imread(name)
        time1 = time.time()
        candidate_133, subset_133, bbox, output_img = dwpose(image_np_hwc=frame, show_body=True,
                        show_face=True, show_hands=True, plot=args.plot)

        time2 = time.time()
        t2 += (time2 - time1) 
        
        if args.plot:
            cv2.imwrite(os.path.join(out_dir, filename), output_img)

    time_e = time.time()

    print("time infer,", t2/250)
    print("time all,", time_e - time_s)


if __name__ == "__main__":
    #Readme
    #command：
    #python inference_image_ori.py  --plot
    #python inference_image_ori.py pt --plot
    #python inference_image_ori.py onnx --plot
    #python inference_image_ori.py trt --plot
    
    main()