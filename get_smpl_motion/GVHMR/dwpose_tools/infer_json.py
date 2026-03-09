import os 
import json
import argparse
import logging
import shutil
from tqdm import tqdm
from pathlib import Path  
import subprocess
from infer_image_lists import process_images_in_folder

# import os.path as Path

def parse_args():  
    # 创建解析器  
    parser = argparse.ArgumentParser(description='write json result')
    parser.add_argument('--json_path', default='/mmu_audio_hdd/MILM_data/SpeakingVideosJson/oneSpeaker-vhuman.json', type=str, help='输入JSON文件地址')
    
    # 解析命令行参数  
    args = parser.parse_args()  
    print("args : ", args)

    return args

# 针对json中是mp4文件索引写的
def main_video(args):
    # read json 
    json_path = args.json_path
    directories = {}
    paths = []

    result = {}
    input = {}
    if os.path.exists(json_path): # 要读的json文件存在
        # read json
        with open(json_path, 'r') as f:
            try:
                directories = json.load(f)
                for key, value in directories.items():
                    if value['duration'] > 600: # TODO: 有时候是不需要的
                        print(value['img_path'])
                        input[key] = value['img_path']
            except:
                logging.error("can't open the json file : ", json_path)
                return 
        
        for key, value in input.items():
            video = value
            out = video.replace('SpeakingVideosImage', 'SpeakingVideosDWPose')
            os.makedirs(out, exist_ok=True)
            out_json = os.path.join(out, 'results_'+os.path.basename(out)+'.json')
            try:
                print(video)
                print(out_json)
                process_images_in_folder(video, out_json)
                result[key] = {'pose2d':os.path.join(out, 'results_'+os.path.basename(out)+'.json'), 'pose2d_npz':os.path.join(out, 'pose.npz')}
                # print(key, result[key])
            except Exception:
                logging.error("mmpose error", Exception)
                continue

                    # paths.append(value['path'])


    result_vhuman_json = args.json_path[:-5]+'_dwpose_2d.json'
    with open(result_vhuman_json, 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == '__main__':
    args = parse_args()
    main_video(args)