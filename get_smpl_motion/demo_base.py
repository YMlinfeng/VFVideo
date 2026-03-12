import inspect
import numpy as np
from collections import namedtuple

# === 1. 修复 inspect 兼容性 (针对 Python 3.11/3.12) ===
if not hasattr(inspect, 'getargspec'):
    ArgSpec = namedtuple('ArgSpec', ['args', 'varargs', 'keywords', 'defaults'])
    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)
    inspect.getargspec = getargspec
    inspect.ArgSpec = ArgSpec

# === 2. 修复 numpy 兼容性 (针对 NumPy 1.24+) ===
# 这里的目的是把 numpy 删掉的那些别名手动塞回去
# 这样 chumpy 执行 'from numpy import bool' 时才不会报错
if not hasattr(np, 'bool'): np.bool = bool
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'complex'): np.complex = complex
if not hasattr(np, 'object'): np.object = object
if not hasattr(np, 'str'): np.str = str
if not hasattr(np, 'unicode'): np.unicode = str
import sys, os
import time
import cv2
from GVHMR.smpl_Infer_service_ljw import SmplInfer


'''input demo
img2motion: ./human_x5.png 
video2motion: ./甄子丹.mp4
text2motion: "a person jump."
BVH2motion: /ytech_milm/lixiaohan/code/kelingHumanMotion/get_smpl_motion/bvh2smplx/001_Neutral_0_mirror_x_0_9.bvh
'''

def load_binary(file_path: str) -> bytes:
    with open(file_path, "rb") as f:
        return f.read()

# 确保只有当直接运行此脚本时才执行 main 函数
if __name__ == "__main__":

    # '''-------------------------------外部调用组织输入------------------------------------'''
    # input_file = sys.argv[1]
    # task_mode=sys.argv[2] #['image2motion', 'video2motion', 'text2motion', 'BVH2motion']
    # print ('input_file', input_file)
    # print ('task_mode', task_mode)
    # is_image = (task_mode=='image2motion')
    # if task_mode=='text2motion': 
    #     input_text=input_file
    #     input_data_binary=None
    # else:
    #     with open(input_file, "rb") as f:
    #         input_data_binary = f.read()
    #     input_text=None
    
    # duration = 10266 ##单位ms 5133 和 10266 @gaopengran
    # '''---------------------------------------------------------------------------------'''

    #初始化
    smpl_infer = SmplInfer(smpl_checkpoints_path='/ytech_milm/liujiwen/kling_motion_service/smpl_all_checkpoints',is_image=True)
    
    input_id_image_path_list = ['/ytech_milm/mengzijie/DiffSynth-Studio/movement/image/多ID图注入/复杂情况/黎明/liming_1.png', '/ytech_milm/mengzijie/DiffSynth-Studio/movement/image/多ID图注入/复杂情况/黎明/liming_2.png', '/ytech_milm/mengzijie/DiffSynth-Studio/movement/image/多ID图注入/复杂情况/黎明/liming_3.png']
    input_id_image_path_list = ['/ytech_milm/mengzijie/DiffSynth-Studio/movement/image/多ID图注入/复杂情况/黎明/liming_1.png']
    

    input_id_image_path_list = ['/ytech_milm/mengzijie/DiffSynth-Studio/movement/image/多ID图注入/复杂情况/周星驰/zhouxingchi_14.png',
    '/ytech_milm/mengzijie/DiffSynth-Studio/movement/image/多ID图注入/复杂情况/周星驰/zhouxingchi_2.png', 
    "/m2v_intern/mengzijie/DiffSynth-Studio/dataset/traindataset/ID_Encoder/周星驰.png"] 

    input_id_image_list_binary = [load_binary(input_id_image_path) for input_id_image_path in input_id_image_path_list]

    video_path = '/ytech_m2v2_hdd/liujiwen/audio_v3/Qwen3-VL/video_id_test_dataset_all/video_id_test_0114/id.mp4'
    imgae_path = '/m2v_intern/mengzijie/DiffSynth-Studio/dataset/traindataset/ID_Encoder/zhouxingchicelian.png'
    with open(video_path, "rb") as f:
        video_data = f.read()
    with open(imgae_path, "rb") as f:
        imgae_data = f.read()

    res = smpl_infer.get_face_grid(imgae_data, id_video_data=None, input_id_image_list_binary=input_id_image_list_binary, output_dir='./debug/', target_size=[720, 1280], save_path_dir='/ytech_m2v2_hdd/liujiwen/ID_Encoder/motion/m2v-diffusers/get_smpl_motion/debug/')
    print (len(res))

    cv2.imwrite('./debug1.png', res[0][:,:,::-1])
    cv2.imwrite('./debug2.png', res[-1][:,:,::-1])

    # #推理
    # start_time = time.time()
    # error_code, error_message, output_binary, json_bbox_with_ID, bvh_binary = smpl_infer.infer(input_data_binary, input_text, task_mode=task_mode, duration=duration)
    # print ('time:', time.time()-start_time)


    # print ('error_message', error_message)
    # # 输出写到本地用来调试:
    # with open('./output/all_result1.pkl', 'wb') as f:
    #     f.write(output_binary)
    # print (json_bbox_with_ID)
    # with open('./output/motion_cap-1.bvh', 'wb') as f:
    #     f.write(bvh_binary)