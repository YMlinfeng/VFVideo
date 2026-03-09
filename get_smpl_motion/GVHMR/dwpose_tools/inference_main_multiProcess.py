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
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import torch

class Pose2dInfer:
    def __init__(self, atype = 'pt', thread_type = False, num_instances = 4):
        # init model
        type = atype # choices=['trt', 'onnx', 'pt', 'pth'], 这里使用pth模型
        self.thread_type = thread_type
        self.num_instances = num_instances
        device = 'cuda:0'

        current_file_path = os.path.abspath(__file__) 
        current_dir_path = os.path.dirname(current_file_path)
        # yolox
        self.det_config = '/share/group_wenziyu/zhouyan/zy_ws/MLLM/share-model/pose2d/yolox_l_8xb8-300e_coco.py'
        self.det_ckpt   = '/share/group_wenziyu/zhouyan/zy_ws/MLLM/share-model/pose2d/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
        # dwpose
        # self.pose_config = current_dir_path+'/dwpose/dwpose_config/dwpose-l_384x288.py'
        # self.pose_ckpt = current_dir_path+'/models/dw-ll_ucoco_384.pth'
        # rtmw-x
        self.pose_config = '/share/group_wenziyu/zhouyan/zy_ws/MLLM/share-model/pose2d/rtmw-x_8xb320-270e_cocktail14-384x288.py'
        self.pose_ckpt   = '/share/group_wenziyu/zhouyan/zy_ws/MLLM/share-model/pose2d/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth'

        # 废弃
        self.det_onnx  = current_dir_path+'/models/yolox_l.onnx'
        self.pose_onnx = current_dir_path+'/models/dw-ll_ucoco_384.onnx'

        # trt
        self.det_trt  = '/share/group_wenziyu/zhouyan/zy_ws/MLLM/share-model/pose2d/yolox_l_fp32_trt10.4.0.engine'
        self.pose_trt = '/share/group_wenziyu/zhouyan/zy_ws/MLLM/share-model/pose2d/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122_fp32_trt10.4.0.engine'
    
        # 定义有界队列，限制任务队列大小
        max_queue_size = 600  # 4K(4096*2160*4)占用32M空间，队列最大占用≈19G A10内存32G
        self.task_queue = Queue(maxsize=max_queue_size)

        # self.dwpose = None
        if self.thread_type:
            self.dwpose_queue = Queue()
            for _ in range(self.num_instances):
                cuda_stream = torch.cuda.Stream()
                xdwpose = DWposeDetector(self.det_config, self.det_ckpt, self.pose_config, self.pose_ckpt,\
                    det_onnx = self.det_onnx, pose_onnx=self.pose_onnx, \
                    det_trt = self.det_trt, pose_trt=self.pose_trt, \
                    device = device, type=type, cuda_stream = cuda_stream)
                # 预热
                frame = np.ones((640, 640, 3), dtype=np.uint8) * 255  # 乘以 255 确保是白色
                xdwpose(image_np_hwc=frame, show_body=True, show_face=True, show_hands=True, plot=False)
                self.dwpose_queue.put(xdwpose)
        else:   
            self.dwpose = DWposeDetector(self.det_config, self.det_ckpt, self.pose_config, self.pose_ckpt,\
                    det_onnx = self.det_onnx, pose_onnx=self.pose_onnx, \
                    det_trt = self.det_trt, pose_trt=self.pose_trt, \
                    device = device, type=type)
    
    def process_image(self, dwpose, frame, frame_id, plot=False):
        """
        运行dwpose推理，返回当前帧inference结果

        :param image: 图像
        :return: 每帧的JSON数据
        """
        time1 = time.time()
        height, width = frame.shape[:2]
        pose, scores, bbox, output_img = dwpose(image_np_hwc=frame, show_body=True,
                        show_face=True, show_hands=True, plot=plot)
        # if plot:
        #     cv2.imwrite('result.png', output_img)

        # print("pose : ", pose)
        # print("score : ", scores)
        # print("bbox : ", bbox)

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

        time2 = time.time()
        # print(f"{frame_id} time infer: {(time2 - time1)*1000:.2f}ms")
        
        # 防止内存溢出
        input_frame = frame
        if not plot:
            output_img = None
            input_frame = None
        return {
            "frame_id": frame_id,
            "height": height,
            "width": width,
            "instances": instances
        }, pose_out, bbox, output_img, time2 - time1, input_frame
    
    def process_frame_with_queue(self, frame, frame_id, plot=False):
        """
        从队列中获取 dwpose 实例，处理帧后再放回队列
        """
        dwpose = self.dwpose_queue.get()  # 从队列中取出一个 dwpose 实例
        try:
            # 使用 dwpose 实例处理帧
            result = self.process_image(dwpose, frame, frame_id, plot)
        finally:
            self.dwpose_queue.put(dwpose)  # 使用完后将 dwpose 实例放回队列
            self.task_queue.get()  # 从队列中移除已完成的任务
        return result
    
    def process_images_in_folder(self, folder_path, output_path, type='pth', plot=False, device='cuda:0'):
        """
        处理文件夹内所有图像并生成最终的JSON结果。

        :param folder_path: 包含图像的文件夹路径,请注意，这里的实现是按照文件夹内图像名均为数字
        :param output_path: 输出json文件和结果视频(plot=true)的地址
        :return: 输出JSON字符串
        """
        time_s = time.time()
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
        times = []
        
        if self.thread_type:
            tr = 0
            ti = 0
            with ThreadPoolExecutor(max_workers=self.num_instances) as executor:
                futures = []
                for index, image_file in enumerate(image_files):
                    time0 = time.time()
                    image_path = os.path.join(folder_path, image_file)
                    image = cv2.imread(image_path)
                    time1 = time.time()
                    # print(f"{index} time read: {(time1 - time0)*1000:.2f}ms")
                    
                    # 如果任务队列已满，等待队列有空闲
                    while self.task_queue.full():
                        time.sleep(0.01)  # 避免忙等待
                
                    # 提交任务到线程池，并将 Future 放入队列
                    future = executor.submit(self.process_frame_with_queue, image, index, plot)
                    futures.append(future)
                    self.task_queue.put(future)
                    
                # 使用 tqdm 显示进度条
                for future in tqdm(futures, desc="Processing frames", unit="frame"):
                    frame_data, pose, bbox, output_img, duration, input_frame = future.result()
                    if len(pose)>0 and len(bbox)>0:
                        pose_vec.append(pose)
                        bbox_vec.append(bbox)
                    instance_info.append(frame_data)              
                    ti += duration

                    if plot:
                        if writer == None:
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            output_video_path = os.path.join(output_path, "result-images-multi.mp4")
                            height, width = output_img.shape[:2]
                            fps = 25
                            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                        
                        writer.write(output_img)
            
            print("avg read,", tr / len(futures))
            print("avg infer,", ti / len(futures))
        else:
            for index, image_file in tqdm(enumerate(image_files)):
                time0 = time.time()
                image_path = os.path.join(folder_path, image_file)
                image = cv2.imread(image_path)
                time1 = time.time()
                
                frame_data, pose, bbox, output_img, duration, input_frame = self.process_image(self.dwpose, image, index, plot)
                time2 = time.time()
                
                if len(pose)>0 and len(bbox)>0:
                    pose_vec.append(pose)
                    bbox_vec.append(bbox)
                instance_info.append(frame_data)

                if plot:
                    if writer == None:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        output_video_path = os.path.join(output_path, "result-images.mp4")
                        height, width = output_img.shape[:2]
                        fps = 25
                        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                    
                    # 融合 frame 与 output，alpha 为 0.5
                    alpha = 0.5
                    blended_frame = cv2.addWeighted(image, alpha, output_img, 1 - alpha, 0)
                    # 写入融合后的帧    
                    writer.write(blended_frame)
                
                time3 = time.time()
                print(f"{index}: time all: {(time3 - time0)*1000:.2f}ms \
                    read: {(time1 - time0)*1000:.2f}ms \
                    infer: {(time2 - time1)*1000:.2f}ms \
                    write: {(time3 - time2)*1000:.2f}ms")
                times.append([time0, time1, time2, time3])

            # 计算耗时分布
            t_all = 0; t_read = 0; t_infer = 0; t_draw = 0; t_write = 0
            for i in range(0, len(times)):
                time0, time1, time2, time3 = times[i]
                t_all += (time3 - time0)
                t_read += (time1 - time0)
                t_infer += (time2 - time1)
                t_write += (time3 - time2)

            print(f"平均耗时: {(t_all/len(times))*1000:.2f}ms \
                平均读取: {(t_read/len(times))*1000:.2f}ms \
                平均推理: {(t_infer/len(times))*1000:.2f}ms \
                平均写入: {(t_write/len(times))*1000:.2f}ms")
            
        if writer:
            writer.release()
        
        time_e = time.time()
        print(f"time all: {time_e - time_s:.2f}s")
        
        return pose_vec, bbox_vec, instance_info
    
    def process_video(self, video_path, output_path, type='pth', plot=False, device='cuda:0'):
        """
        处理文件夹内所有图像并生成最终的JSON结果。

        :param folder_path: 包含图像的文件夹路径,请注意，这里的实现是按照文件夹内图像名均为数字
        :param output_path: 输出json文件和结果视频(plot=true)的地址
        :return: 输出JSON字符串
        """
        # inference every frame
        time_s = time.time()
        pose_vec, bbox_vec, instance_info = [], [], []
        cap = cv2.VideoCapture(video_path)
        time_v = time.time()
        index = 0
        writer = None
        times = []
        
        if self.thread_type:
            tr = 0
            ti = 0
            with ThreadPoolExecutor(max_workers=self.num_instances) as executor:
                futures = []
                while True:
                    time0 = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        break
                    index+=1
                    time1 = time.time()
                    tr += time1 - time0
                    # print(f"{index} time read: {(time1 - time0)*1000:.2f}ms")
                    
                    # 如果任务队列已满，等待队列有空闲
                    while self.task_queue.full():
                        time.sleep(0.01)  # 避免忙等待
                
                    # 提交任务到线程池，并将 Future 放入队列
                    future = executor.submit(self.process_frame_with_queue, frame, index, plot)
                    futures.append(future)
                    self.task_queue.put(future)
                
                # 使用 tqdm 显示进度条
                for future in tqdm(futures, desc="Processing frames", unit="frame"):
                    frame_data, pose, bbox, output_img, duration, input_frame = future.result()
                    if len(pose)>0 and len(bbox)>0:
                        pose_vec.append(pose)
                        bbox_vec.append(bbox)
                    instance_info.append(frame_data)                    
                    ti += duration

                    if plot:
                        if writer == None:
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            output_video_path = os.path.join(output_path, "result-video-multi.mp4")
                            height, width = output_img.shape[:2]
                            print("height, witdh : ", height, width)
                            # cv2.imread(os.path.join(output_path, "result.png"), output_img)
                            fps = 25
                            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                        
                        # 融合 frame 与 output，alpha 为 0.5
                        alpha = 0.5
                        blended_frame = cv2.addWeighted(input_frame, alpha, output_img, 1 - alpha, 0)
                        # 写入融合后的帧    
                        writer.write(blended_frame)
            
            print("avg read,", tr / len(futures), len(futures))
            print("avg infer,", ti / len(futures), len(futures))
        else:
            while True:
                time0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                index+=1
                time1 = time.time()
                
                frame_data, pose, bbox, output_img, duration, input_frame = self.process_image(self.dwpose, frame, index, plot)
                time2 = time.time()
                
                if len(pose)>0 and len(bbox)>0:
                    pose_vec.append(pose)
                    bbox_vec.append(bbox)
                instance_info.append(frame_data)

                if plot:
                    if writer == None:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        output_video_path = os.path.join(output_path, "result-video.mp4")
                        height, width = output_img.shape[:2]
                        print("height, witdh : ", height, width)
                        # cv2.imread(os.path.join(output_path, "result.png"), output_img)
                        fps = 25
                        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                    
                    # 融合 frame 与 output，alpha 为 0.5
                    alpha = 0.5
                    blended_frame = cv2.addWeighted(frame, alpha, output_img, 1 - alpha, 0)
                    # 写入融合后的帧    
                    writer.write(blended_frame)

                time3 = time.time()
                # print(f"{index}: time all: {(time3 - time0)*1000:.2f}ms \
                #     read: {(time1 - time0)*1000:.2f}ms \
                #     infer: {(time2 - time1)*1000:.2f}ms \
                #     write: {(time3 - time2)*1000:.2f}ms")
                times.append([time0, time1, time2, time3])

            # 计算耗时分布
            t_all = 0; t_read = 0; t_infer = 0; t_draw = 0; t_write = 0
            for i in range(0, len(times)):
                time0, time1, time2, time3 = times[i]
                t_all += (time3 - time0)
                t_read += (time1 - time0)
                t_infer += (time2 - time1)
                t_write += (time3 - time2)

            print(f"平均耗时: {(t_all/len(times))*1000:.2f}ms \
                平均读取: {(t_read/len(times))*1000:.2f}ms \
                平均推理: {(t_infer/len(times))*1000:.2f}ms \
                平均写入: {(t_write/len(times))*1000:.2f}ms")
            
        if writer:
            writer.release()
        
        time_e = time.time()
        print(f"time all: {time_e - time_s:.2f}s \
                decode: {time_v - time_s:.2f}s \
                read+infer: {time_e - time_v:.2f}s")
    
        return pose_vec, bbox_vec, instance_info

def check_MAE():
    video_path = "./images/skate.mp4"
    output_path = "outputs"
    os.makedirs(output_path, exist_ok=True)
    
    # 计算 MAE
    print('======================================================================')
    object_pt = Pose2dInfer(atype = 'pt', thread_type = False, num_instances = 4)
    pose_vec_pt, bbox_vec_pt, instance_info_pt = object_pt.process_video(video_path, output_path)
    list_bbox_pt = []
    list_score_pt = []
    list_keypoints_pt = []
    for i in range(len(instance_info_pt)):
        list_bbox_pt.append(instance_info_pt[i]['instances'][0]['bbox'])
        list_score_pt.append(instance_info_pt[i]['instances'][0]['bbox_score'])
        list_keypoints_pt.append(instance_info_pt[i]['instances'][0]['keypoints'])    
    data_bbox_pt = np.stack(list_bbox_pt, axis=0)
    data_score_pt = np.stack(list_score_pt, axis=0)
    data_keypoints_pt = np.stack(list_keypoints_pt, axis=0)
    print(f"pt bbox: {data_bbox_pt.shape}, score: {data_score_pt.shape}, keypoints: {data_keypoints_pt.shape}")

    print('======================================================================')
    object_trt = Pose2dInfer(atype = 'trt', thread_type = True, num_instances = 4)
    pose_vec_trt, bbox_vec_trt, instance_info_trt = object_trt.process_video(video_path, output_path)
    list_bbox_trt = []
    list_score_trt = []
    list_keypoints_trt = []
    for i in range(len(instance_info_trt)):
        list_bbox_trt.append(instance_info_trt[i]['instances'][0]['bbox'])
        list_score_trt.append(instance_info_trt[i]['instances'][0]['bbox_score'])
        list_keypoints_trt.append(instance_info_trt[i]['instances'][0]['keypoints'])    
    data_bbox_trt = np.stack(list_bbox_trt, axis=0)
    data_score_trt = np.stack(list_score_trt, axis=0)
    data_keypoints_trt = np.stack(list_keypoints_trt, axis=0)
    print(f"trt bbox: {data_bbox_trt.shape}, score: {data_score_trt.shape}, keypoints: {data_keypoints_trt.shape}")
    
    print('======================================================================')
    print("pt bbox: ", data_bbox_pt[:5, 0])
    print("trt bbox: ", data_bbox_trt[:5, 0])
    result_bbox = np.abs(data_bbox_trt - data_bbox_pt).mean()
    print(f"MAE of BBox: {result_bbox}")

    print('======================================================================')
    print("pt score: ", data_score_pt[:15])
    print("trt score: ", data_score_trt[:15])
    result_score = np.abs(data_score_trt - data_score_pt).mean()
    print(f"MAE of Score: {result_score}")
    
    print('======================================================================')
    print("pt keypoints: ", data_keypoints_pt[:1, 0])
    print("trt keypoints: ", data_keypoints_trt[:1, 0])
    result_keypoints = np.abs(data_keypoints_trt - data_keypoints_pt).mean()
    print(f"MAE of Keypoints: {result_keypoints}")

video_vec = [  '/ytech_milm/video/SpeakingVideos/oneSpeaker/vhuman_project_formal/tongtong/action-tongtong.mp4', 
    '/ytech_milm/video/SpeakingVideos/oneSpeaker/live_cut_1105/20241108/2qCIYwsevv8/video/2qCIYwsevv8_cut00000003.mp4', 
    '/ytech_milm/video/SpeakingVideos/oneSpeaker/live_cut_1105/20240903.0/64iZSyUxfJ8/video/64iZSyUxfJ8_cut00000000.mp4', 
    '/ytech_milm/video/SpeakingVideos/oneSpeaker/kling/1031/subset_0/84e9a811cfd78ed62ca65d43e6e98ddb.mp4', 
    '/ytech_milm/video/SpeakingVideos/oneSpeaker/kling/1031/subset_0/e277d5b407f04726808da4759d90a821.mp4',
    '/ytech_milm/video/SpeakingVideos/oneSpeaker/emo_v2_partA/sub_3/2558_582092a8-29d7-42cd-855f-1de0d0260df2_record-1132-Scene-001_1_erase_ocr.mp4']

def get_video_duration(video_path):
    import subprocess
    import json
    # 使用 subprocess 调用 ffprobe 命令
    cmd = [
        'ffprobe',
        '-v', 'error',  # 忽略详细的错误日志
        '-show_entries', 'format=duration',  # 只显示时长信息
        '-of', 'json',  # 输出格式为 JSON
        video_path
    ]

    # 执行命令并获取输出
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 解析 JSON 输出
    output = result.stdout
    info = json.loads(output)

    # 获取视频时长（单位：秒）
    duration = float(info['format']['duration'])

    return duration

if __name__ == "__main__":
    output_path = "outputs"
    os.makedirs(output_path, exist_ok=True)
    video_path = "./images/skate.mp4"
    folder_path = "./images"

    # trt
    object = Pose2dInfer(atype = 'trt', thread_type = True, num_instances = 4)
    # 视频
    result_video = object.process_video(video_path, output_path, plot=True)
    print("trt video done")
    # # 图片
    # result_image = object.process_images_in_folder(folder_path, output_path)
    # print("trt images done")
    
    # pt
    object_pt = Pose2dInfer(atype = 'pt', thread_type = False)
    # 视频
    result_video = object_pt.process_video(video_path, output_path, plot=True)
    print("pt video done")
    # # 图片
    # result_image = object_pt.process_images_in_folder(folder_path, output_path)
    # print("pt images done")
    
    duration = get_video_duration(video_path)
    print("video duration is : ", duration)