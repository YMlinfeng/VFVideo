
import sys, os
import time

from GVHMR.smpl_Infer_service_ljw import SmplInfer


'''input demo
img2motion: ./human_x5.png 
video2motion: ./甄子丹.mp4
text2motion: "a person jump."
BVH2motion: /ytech_milm/lixiaohan/code/kelingHumanMotion/get_smpl_motion/bvh2smplx/001_Neutral_0_mirror_x_0_9.bvh
'''

# 确保只有当直接运行此脚本时才执行 main 函数
if __name__ == "__main__":

    #初始化
    smpl_infer = SmplInfer(smpl_checkpoints_path='/ytech_milm/liujiwen/kling_motion_service/smpl_all_checkpoints',is_image=True)

    image_path_list = open('/ytech_milm/liujiwen/kling_motion_service/get_smpl_motion/run_list_demo/image_path.txt').read().strip().split('\n')
    for i, input_file in enumerate(image_path_list):
        task_mode='image2motion'
        if task_mode=='text2motion': 
            input_text=input_file
            input_data_binary=None
        else:
            with open(input_file, "rb") as f:
                input_data_binary = f.read()
            input_text=None
    
        tgt_video_length = 241 #24fps*10

        #推理
        try:
            start_time = time.time()
            error_code, output_binary, json_bbox_with_ID = smpl_infer.infer(input_data_binary, input_text, task_mode=task_mode, tgt_video_length=tgt_video_length)
            print ('time:', time.time()-start_time)
            
            output_path = '/ytech_milm/liujiwen/kling_motion_service/get_smpl_motion/output/image/'+str(i)+'.pkl'
            print (input_file, output_path, 'error_code:', error_code)
            print 
            #输出写到本地用来调试:
            with open(output_path, 'wb') as f:
                f.write(output_binary)
            print (json_bbox_with_ID)
        except:
            pass
        



