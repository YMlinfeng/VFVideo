import os
import json
import re
import cv2
from dwpose import DWposeDetector
import time
from meta_info import meta_info
from tqdm import tqdm
import numpy as np
import logging

class Pose2dInfer:
    def __init__(self):
        # init model
        current_file_path = os.path.abspath(__file__) 
        current_dir_path = os.path.dirname(current_file_path) 
        self.det_config = current_dir_path+'/dwpose/yolox_config/yolox_l_8xb8-300e_coco.py'
        self.det_ckpt = current_dir_path+'/models/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
        self.pose_config = current_dir_path+'/dwpose/dwpose_config/dwpose-l_384x288.py'
        self.pose_ckpt = current_dir_path+'/models/dw-ll_ucoco_384.pth'

        self.det_onnx = current_dir_path+'/models/yolox_l.onnx'
        self.pose_onnx = current_dir_path+'/models/dw-ll_ucoco_384.onnx'

        self.det_trt = current_dir_path+'/models/yolox_l.engine'
        self.pose_trt = current_dir_path+'/models/dw-ll_ucoco_384.engine'

        self.dwpose = None
    
    def process_image(self, frame, frame_id, plot=False):
        """
        运行dwpose推理，返回当前帧inference结果

        :param image: 图像
        :return: 每帧的JSON数据
        """
        height, width = frame.shape[:2]
        pose, scores, bbox, output_img = self.dwpose(image_np_hwc=frame, show_body=True,
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

        return {
            "frame_id": frame_id,
            "height": height,
            "width": width,
            "instances": instances
        }, pose_out, bbox, output_img

    def write_result(self, output_path, pose_vec, bbox_vec, instance_info):
        # 裸数据写入npz文件
        output_npz_path = os.path.join(output_path, 'pose.npz')
        pose_out = np.array(pose_vec)
        bbox_out = np.array(bbox_vec)
        np.savez(output_npz_path, pose=pose_out, bbox=bbox_out)

        # 写入最终的JSON文件
        result_json = {}
        output_json_path = os.path.join(output_path, 'pose.json')
        result_json["instance_info"]=instance_info
        result_json["meta_info"] = meta_info
        result_json["meta_info"]["normalized"] = True
        with open(output_json_path, 'w') as json_file:
            json.dump(result_json, json_file, indent=4)

        return {
            'pose2d':output_json_path,
            'pose2d_npz':output_npz_path
        }
    
    def process(self, input, input_type, output_path, type='pth', plot=False, device='cuda:0'):
        try:
            # init model
            if self.dwpose == None:
                self.dwpose = DWposeDetector(self.det_config, self.det_ckpt, self.pose_config, self.pose_ckpt,\
                det_onnx = self.det_onnx, pose_onnx=self.pose_onnx, \
                det_trt = self.det_trt, pose_trt=self.pose_trt, \
                device = device, type=type)
            
            if input_type == 'video_path':
                pose_vec, bbox_vec, instance_info = self.process_video(input, output_path, type='pth', plot=plot, device='cuda:0')
            elif input_type == 'frames':
                pose_vec, bbox_vec, instance_info = self.process_frames(input, output_path, type='pth', plot=plot, device='cuda:0')
            elif input_type == 'images_path':
                pose_vec, bbox_vec, instance_info = self.process_images_in_folder(input, output_path, type='pth', plot=plot, device='cuda:0')
            else:
                logging.error("input_type is not video_path, frames, images_path, return error !")
                return {}
            
            result = self.write_result(output_path, pose_vec, bbox_vec, instance_info)
            return result

        except Exception as e:
            logging.error(e)
        

    def process_images_in_folder(self, folder_path, output_path, type='pth', plot=False, device='cuda:0'):
        """
        处理文件夹内所有图像并生成最终的JSON结果。

        :param folder_path: 包含图像的文件夹路径,请注意，这里的实现是按照文件夹内图像名均为数字
        :param output_path: 输出json文件和结果视频(plot=true)的地址
        :return: 输出JSON字符串
        """
        # 获取文件夹内所有图像文件，并按自然顺序排序
        def natural_sort_key(s):
            """
            用于自然排序的键函数，可以正确排序像 1.jpg, 2.jpg, 10.jpg 这样的文件名。
            """
            return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'))], key=natural_sort_key)

        # inference every frame
        print(" image files : ", len(image_files))
        pose_vec, bbox_vec, instance_info = [], [], []
        writer = None
        for index, image_file in tqdm(enumerate(image_files)):
            image_path = os.path.join(folder_path, image_file)
            image = cv2.imread(image_path)
            frame_data, pose, bbox, output_img = self.process_image(image, index, plot)
            if len(pose)>0 and len(bbox)>0:
                pose_vec.append(pose)
                bbox_vec.append(bbox)
            instance_info.append(frame_data)

            if plot:
                if writer == None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    output_video_path = os.path.join(output_path, "result.mp4")
                    height, width = output_img.shape[:2]
                    fps = 25
                    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                
                writer.write(output_img)
        if writer:
            writer.release()
        
        return pose_vec, bbox_vec, instance_info

    def process_frames(self, frames, output_path, type='pth', plot=False, device='cuda:0'):
        """
        处理图像lists并生成最终的JSON结果。

        :param image_list: 图像数据list
        :param output_path: 输出json文件和结果视频(plot=true)的地址
        :return: 输出JSON字符串
        """
        # inference every frame
        print(" image files : ", len(frames))
        pose_vec, bbox_vec, instance_info = [], [], []
        writer = None
        for index, image in tqdm(enumerate(frames)):
            frame_data, pose, bbox, output_img = self.process_image(image, index, plot)
            if len(pose)>0 and len(bbox)>0:
                pose_vec.append(pose)
                bbox_vec.append(bbox)
            instance_info.append(frame_data)

            if plot:
                if writer == None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    output_video_path = os.path.join(output_path, "result.mp4")
                    height, width = output_img.shape[:2]
                    fps = 25
                    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                
                writer.write(output_img)
        if writer:
            writer.release()
        
        return pose_vec, bbox_vec, instance_info

    def process_video(self, video_path, output_path, type='pth', plot=False, device='cuda:0'):
        """
        处理文件夹内所有图像并生成最终的JSON结果。

        :param folder_path: 包含图像的文件夹路径,请注意，这里的实现是按照文件夹内图像名均为数字
        :param output_path: 输出json文件和结果视频(plot=true)的地址
        :return: 输出JSON字符串
        """
        # inference every frame
        pose_vec, bbox_vec, instance_info = [], [], []
        cap = cv2.VideoCapture(video_path)
        index = 0
        writer = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            index+=1

            frame_data, pose, bbox, output_img = self.process_image(frame, index, plot)
            if len(pose)>0 and len(bbox)>0:
                pose_vec.append(pose)
                bbox_vec.append(bbox)
            instance_info.append(frame_data)

            if plot:
                if writer == None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    output_video_path = os.path.join(output_path, "result.mp4")
                    height, width = output_img.shape[:2]
                    print("height, witdh : ", height, width)
                    # cv2.imread(os.path.join(output_path, "result.png"), output_img)
                    fps = 25
                    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                
                writer.write(output_img)
        
        if writer:
            writer.release()
        
        return pose_vec, bbox_vec, instance_info

if __name__ == '__main__':
    # 示例使用
    folder_path = 'images'
    folder_path = '8.mp4'
    output_json_path = './output_json/'
    type = 'pth' # choices=['trt', 'onnx', 'pt', 'pth']
    plot = True
    # folder_path = '/mmu_audio_hdd/MILM_data/SpeakingVideosImage/oneSpeaker/vhuman_project_formal/koubo-xiaoxuan1'
    # output_json_path = '/mmu_audio_hdd/MILM_data/SpeakingVideosDWPose/oneSpeaker/vhuman_project_formal/koubo-xiaoxuan1/results_koubo-xiaoxuan1.json'
    # folder_path = '/mmu_audio_hdd/MILM_data/SpeakingVideosImage/oneSpeaker/vhuman_project_formal/koubo-aikun'
    # output_json_path = '/mmu_audio_hdd/MILM_data/SpeakingVideosDWPose/oneSpeaker/vhuman_project_formal/koubo-aikun'
    # folder_path = '/mmu_audio_hdd/MILM_data/SpeakingVideosImage/twoSpeaker/record/conversation_wr_lxy/a/a_speaker1'

    object = Pose2dInfer()
    result = object.process(folder_path, 'video_path', output_json_path, type, plot)
    print("object result : ", result)
