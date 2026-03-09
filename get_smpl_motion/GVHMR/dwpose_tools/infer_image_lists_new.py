import os
import json
import re
import cv2
from dwpose import DWposeDetector
import time
from meta_info import meta_info
from tqdm import tqdm
import numpy as np

def process_image(image_path, dwpose_model, plot=False):
    """
    运行dwpose推理，返回当前帧inference结果

    :param image_path: 图像文件的路径
    :return: 每帧的JSON数据
    """
    frame = cv2.imread(image_path)
    height, width = frame.shape[:2]
    pose, scores, bbox, output_img = dwpose_model(image_np_hwc=frame, show_body=True,
                    show_face=True, show_hands=True, plot=plot)

    bbox_score = 1.0
    num_max = max(len(pose), len(bbox), 1)
    if len(pose)>0:
        # normalized the pose axis
        pose[:, :, 0] = pose[:, :, 0]/float(width)
        pose[:, :, 1] = pose[:, :, 1]/float(height)
        pose_out = np.concatenate((pose, np.expand_dims(scores, -1)), axis=-1)
    else :
        pose_out = np.zeros((num_max, 133, 3))
    
    if len(bbox)>0:
        # normalized the bbox axis
        bbox[:, 0] = bbox[:, 0]/float(width)
        bbox[:, 2] = bbox[:, 2]/float(width)
        bbox[:, 1] = bbox[:, 1]/float(height)
        bbox[:, 3] = bbox[:, 3]/float(height)
    else:
        bbox = np.zeros((num_max, 4))
        bbox_score = 0

    # read result and write json
    n, num, loc = pose.shape
    n = min(n, len(bbox)) # 以防没有bbox，但还是预测了pose
    instances = []
    for i in range(n): # 多个人体pose
        instance = {"keypoints":[], "keypoint_scores":[], "bbox":[], "bbox_score":bbox_score}
        instance["bbox"].append(bbox[i].tolist())
        for j in range(num):
            instance["keypoints"].append([float(pose[i,j,0]), float(pose[i,j,1])])
            instance["keypoint_scores"].append(float(scores[i, j]))
        instances.append(instance)
    
    frame_id = int(re.search(r'(\d+)\.jpg$', os.path.basename(image_path)).group(1))

    if plot:
        cv2.imwrite(os.path.join('out_dir', os.path.basename(image_path)), output_img)

    return {
        "frame_id": frame_id,
        "height": height,
        "width": width,
        "instances": instances
    }, pose_out, bbox

def natural_sort_key(s):
    """
    用于自然排序的键函数，可以正确排序像 1.jpg, 2.jpg, 10.jpg 这样的文件名。
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def process_images_in_folder(folder_path, output_json_path, type='pth', plot=False, device='cuda:0'):
    """
    处理文件夹内所有图像并生成最终的JSON结果。

    :param folder_path: 包含图像的文件夹路径
    :param output_json_path: 输出JSON文件的路径
    """
    # init model
    det_config = './dwpose/yolox_config/yolox_l_8xb8-300e_coco.py'
    det_ckpt = './models/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
    pose_config = './dwpose/dwpose_config/dwpose-l_384x288.py'
    pose_ckpt = './models/dw-ll_ucoco_384.pth'

    det_onnx = './models/yolox_l.onnx'
    pose_onnx = './models/dw-ll_ucoco_384.onnx'

    det_trt = './models/yolox_l.engine'
    pose_trt = './models/dw-ll_ucoco_384.engine'

    dwpose = DWposeDetector(det_config, det_ckpt, pose_config, pose_ckpt,\
             det_onnx = det_onnx, pose_onnx=pose_onnx, \
             det_trt = det_trt, pose_trt=pose_trt, \
             device = device, type=type)

    # meta json 
    result_json = {}

    # 获取文件夹内所有图像文件，并按自然顺序排序
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')], key=natural_sort_key)

    # inference every frame
    print(" image files : ", len(image_files))
    pose_vec, bbox_vec, instance_info = [], [], []
    for image_file in tqdm(image_files):
        image_path = os.path.join(folder_path, image_file)
        frame_data, pose, bbox = process_image(image_path, dwpose, plot)
        if len(pose)>0 and len(bbox)>0:
            pose_vec.append(pose)
            bbox_vec.append(bbox)
        instance_info.append(frame_data)
    
    # 裸数据写入npz文件
    output = os.path.dirname(output_json_path)
    if len(output)>0:
        file = os.path.join(output, 'pose.npz')
        pose_out = np.array(pose_vec)
        bbox_out = np.array(bbox_vec)
        np.savez(file, pose=pose_out, bbox=bbox_out)

        data = np.load(file)
        pose_input = data['pose']
        bbox_input = data['bbox']

    # 写入最终的JSON文件
    result_json["instance_info"]=instance_info
    result_json["meta_info"] = meta_info
    result_json["meta_info"]["normalized"] = True
    with open(output_json_path, 'w') as json_file:
        json.dump(result_json, json_file, indent=4)


if __name__ == '__main__':
    # 示例使用
    folder_path = 'images'
    output_json_path = './output.json'
    type = 'pth' # choices=['trt', 'onnx', 'pt', 'pth']
    plot = False
    # folder_path = '/mmu_audio_hdd/MILM_data/SpeakingVideosImage/oneSpeaker/vhuman_project_formal/koubo-xiaoxuan1'
    # output_json_path = '/mmu_audio_hdd/MILM_data/SpeakingVideosDWPose/oneSpeaker/vhuman_project_formal/koubo-xiaoxuan1/results_koubo-xiaoxuan1.json'
    # folder_path = '/mmu_audio_hdd/MILM_data/SpeakingVideosImage/oneSpeaker/vhuman_project_formal/koubo-aikun'
    # output_json_path = '/mmu_audio_hdd/MILM_data/SpeakingVideosDWPose/oneSpeaker/vhuman_project_formal/koubo-aikun'
    folder_path = '/mmu_audio_hdd/MILM_data/SpeakingVideosImage/twoSpeaker/record/conversation_wr_lxy/a/a_speaker1'
    process_images_in_folder(folder_path, output_json_path, type, plot)
