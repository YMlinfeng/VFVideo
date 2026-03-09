# -*- coding: utf-8 -*-
"""
批量测试脚本
测试execute_input_validation函数功能
从"口型拒识待优化样例"目录读取视频进行批量测试
"""

import os
import sys
import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from GVHMR.input_validation_tester import InputValidationTester, InputErrorCode, TaskType

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def load_binary(file_path: str) -> bytes:
    """读取二进制文件"""
    with open(file_path, "rb") as f:
        return f.read()


def get_video_files(directory: str) -> list:
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    video_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(root, file))
    
    return sorted(video_files)


def save_image_list_as_images(image_list, output_dir):
    """将image_list保存为图片"""
    if image_list is None:
        return
    
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    for i, img in enumerate(image_list):
        img_path = os.path.join(images_dir, f'frame_{i:04d}.png')
        # 转换RGB到BGR用于cv2保存
        cv2.imwrite(img_path, img[:, :, ::-1])


def save_image_list_as_video(image_list, output_path, fps=30):
    """将image_list拼接成视频"""
    if image_list is None or len(image_list) == 0:
        return
    
    h, w = image_list[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for img in image_list:
        # 转换RGB到BGR
        writer.write(img[:, :, ::-1])
    
    writer.release()


def save_log(result, output_dir, video_path):
    log_path = os.path.join(output_dir, 'log.json')
    
    log_data = {
        'video_or_image_path': video_path,
        'timestamp': datetime.now().isoformat(),
        'task_type': result['task_type'],
        'error_code': result['error_code'],
        'error_message': result['error_message'],
        'start_idx': result['start_idx'],
        'end_idx': result['end_idx'],
        'image_list_length': len(result['image_list']) if result['image_list'] is not None else 0,
        'removed_frames': result['removed_frames']
    }
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2, cls=NpEncoder)

def load_binary(file_path: str) -> bytes:
    with open(file_path, "rb") as f:
        return f.read()
    
def main():
    input_id_image_path_list = [
        '/ytech_m2v4_hdd/mengzijie/get_smpl_motion/test/1758355655637050511.jpg',
        '/ytech_m2v4_hdd/mengzijie/get_smpl_motion/test/Google.png',
        '/ytech_m2v4_hdd/mengzijie/get_smpl_motion/test/动画猫.jpg'
    ]
    input_id_image_path_list = []
    # input_id_image_list_binary = [load_binary(input_id_image_path) for input_id_image_path in input_id_image_path_list]

    # input_id_image_path_list = [
    #     '/ytech_milm/mengzijie/DiffSynth-Studio/movement/image/多ID图注入/复杂情况/周星驰/zhouxingchi_14.png',
    #     '/ytech_milm/mengzijie/DiffSynth-Studio/movement/image/多ID图注入/复杂情况/周星驰/zhouxingchi_2.png',
    #     '/ytech_m2v4_hdd/mengzijie/get_smpl_motion/debug/77d95d4f-b5dd-41d8-8ccf-7290556edb15/id_image_2.png'
    # ]
    image_path = '/ytech_m2v4_hdd/mengzijie/get_smpl_motion/debug'
    video_dir = '/ytech_m2v4_hdd/mengzijie/get_smpl_motion/debug/random_samples_50'
    # video_dir = 'a'
    output_base_dir = './debug_img'
    os.makedirs(output_base_dir, exist_ok=True)
    video_files = get_video_files(video_dir)
    print(f"找到 {len(video_files)} 个视频文件")
    tester = InputValidationTester(
        smpl_checkpoints_path='/ytech_milm/liujiwen/kling_motion_service/smpl_all_checkpoints'
    )
    summary = {
        'total': len(video_files),
        'success': 0,
        'failed': 0,
        'results': []
    }
    # --------------------------------------
    folders = sorted([f for f in os.listdir(image_path) if f.isdigit()], key=int)

    for f in folders:
        # print(f)
        img_dir = os.path.join(image_path, f, "images").strip()
        if os.path.isdir(img_dir):
            print(img_dir)
            img_dir_list = os.listdir(img_dir)
            print(img_dir_list)
            imgs = [os.path.join(img_dir,str(i)) for i in img_dir_list if i.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
            input_id_image_path_list.extend(imgs)
                
            # 凑够3个就停止
            if len(input_id_image_path_list) >= 3:
                input_id_image_path_list = input_id_image_path_list[:3]
        else:
            input_id_image_path_list = []
        output_dir = os.path.join(output_base_dir, str(f))
        current_idx = int(f)
        os.makedirs(output_dir, exist_ok=True)        
        
        print(input_id_image_path_list)

    # for i, video_path in enumerate(video_files):
        # video
        # file_name = os.path.basename(video_path)      # 得到 "1.mp4"
        # file_id_str = os.path.splitext(file_name)[0]  # 得到 "1"
        # # current_idx = int(file_id_str)
        # file_id_str = file_id_str.split("_")
        # current_idx = int(file_id_str[-1].replace("idx", "")) 
        # print(f"\n[{i + 1}/{len(video_files)}] 处理: {video_path} (ID: {current_idx})")
        # output_dir = os.path.join(output_base_dir, str(current_idx))
        # os.makedirs(output_dir, exist_ok=True)
        # try:
        result = tester.execute_input_validation(
            image_list=input_id_image_path_list,
            # video_path=video_path
            video_path=None
        )
        save_log(result, output_dir, image_path)
        if result['error_code'] == InputErrorCode.SUCCESS and result['image_list'] is not None:
            save_image_list_as_images(result['image_list'], output_dir)
            summary['success'] += 1; print(f"  成功: start_idx={result['start_idx']}, end_idx={result['end_idx']}, frames={len(result['image_list'])}")
        else:
            summary['failed'] += 1; print(f"  失败: error_code={result['error_code']}, message={result['error_message']}")

        summary['results'].append({
            'idx': current_idx,
            # 'video_path': video_path,
            'error_code': result['error_code'],
            'error_message': result['error_message'],
            'start_idx': result['start_idx'],
            'end_idx': result['end_idx']
        })
            
        # except Exception as e:
        #     print(f"  异常: {e}")
        #     summary['failed'] += 1
        #     summary['results'].append({
        #         'idx': current_idx,
        #         'video_path': video_path,
        #         'error_code': -1,
        #         'error_message': str(e),
        #         'start_idx': None,
        #         'end_idx': None
        #     })
        #     error_log = {
        #         'video_path': video_path,
        #         'timestamp': datetime.now().isoformat(),
        #         'error': str(e)
        #     }
        #     with open(os.path.join(output_dir, 'error.json'), 'w', encoding='utf-8') as f:
        #         json.dump(error_log, f, ensure_ascii=False, indent=2)
    
    # 保存汇总结果
    summary_path = os.path.join(output_base_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n========== 测试完成 ==========")
    print(f"总计: {summary['total']}")
    print(f"成功: {summary['success']}")
    print(f"失败: {summary['failed']}")
    print(f"汇总结果已保存到: {summary_path}")


if __name__ == "__main__":
    main()
