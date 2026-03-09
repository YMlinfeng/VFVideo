import sys, os
import time
import torch
import pickle
import cv2
from GVHMR.retarget_tools import Retarget

# 确保只有当直接运行此脚本时才执行 main 函数
if __name__ == "__main__":


    '''-------------------------------------外部调用组织输入------------------------------------'''
    #demo 输入
    IMG_PATH = "./output/all_result.pkl"
    VIDEO_PATH = "./output/all_result1.pkl"
    input_image_path = "./human_x4.jpeg"

    with open(IMG_PATH, "rb") as f:
        img_data_binary = f.read()
    with open(VIDEO_PATH, "rb") as f:
        video_data_binary = f.read()
    with open(input_image_path, "rb") as f:
        input_image_binary = f.read()
    
    #设置输入参数
    human_ID_list = [1,0] 
    video_data_binary_list = [video_data_binary, video_data_binary]
    change_direction_list = [False, True] #每对动作重定向是否保持朝向
    duration = 10266 ##单位ms 5133 和 10266 @gaopengran
    target_resolution = 720 #@gaopengran
    '''------------------------------------------------------------------------------------------'''

    #初始化
    retarget = Retarget(smpl_model_path='/ytech_milm/liujiwen/kling_motion_service/smpl_all_checkpoints/checkpoints')

    #重定向推理
    start_time = time.time()
    bvh_binary_list, motion_video_binary, input_image_binary, error_code, error_message = retarget.retarget_with_data_list(input_image_binary, img_data_binary, video_data_binary_list, human_ID_list, change_direction_list, duration=duration, target_resolution=target_resolution)
    ##bvh_binary_list, motion_video_binary, depth_video_binary, input_image_binary, error_code, error_message = retarget.retarget_with_data_list(input_image_binary, img_data_binary, video_data_binary_list, human_ID_list, change_direction_list, duration=duration, target_resolution=target_resolution)
    print ("retarget_time:", time.time()-start_time)

    print ('error_message', error_message)
    print ('len(bvh_binary_list)', len(bvh_binary_list))


    #输出写到本地用来调试:
    with open('./output/motion_video.mp4', 'wb') as f:
        f.write(motion_video_binary)
    # with open('./output/depth_video.mp4', 'wb') as f:
    #     f.write(depth_video_binary)
    with open('./output/input_image.png', 'wb') as f:
        f.write(input_image_binary)
    with open('./output/motion_cap-1.bvh', 'wb') as f:
        f.write(bvh_binary_list[0])
