import os
import random
import subprocess
import glob
import cv2  # 用于获取视频精确时长和帧率

# --- 配置参数 ---
src_dir = "/ytech_m2v4_hdd/mengzijie/get_smpl_motion/a"
dst_root = "/ytech_m2v4_hdd/mengzijie/get_smpl_motion/debug"
sub_dir_name = "random_samples_50"
target_dir = os.path.join(dst_root, sub_dir_name)
total_clips = 50

# 创建目标目录
os.makedirs(target_dir, exist_ok=True)

def get_video_info(file_path):
    """使用OpenCV获取视频的时长和帧率"""
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return duration, fps

def main():
    # 1. 查找源视频 (支持 mp4, avi, mov 等)
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        video_files.extend(glob.glob(os.path.join(src_dir, ext)))
    
    if len(video_files) < 2:
        print(f"错误：在 {src_dir} 下没找到足够的视频文件（当前找到 {len(video_files)} 个）。")
        return

    # 只取前两个视频（按要求）
    video_files = video_files[:2]
    print(f"选定的源视频: {[os.path.basename(f) for f in video_files]}")

    # 2. 循环生成 50 个片段
    for i in range(total_clips):
        # 随机选一个视频
        src_video = random.choice(video_files)
        duration, fps = get_video_info(src_video)
        
        if duration is None:
            continue

        # 随机生成时长 (0.5s - 2.5s)
        clip_duration = round(random.uniform(0.5, 2.5), 2)
        
        # 确保随机的开始时间不会导致超出视频总长
        if duration <= clip_duration:
            start_time = 0
            clip_duration = duration - 0.1
        else:
            start_time = round(random.uniform(0, duration - clip_duration), 2)
        
        # 计算帧数
        frame_count = int(clip_duration * fps)
        
        # 构造文件名: {视频秒数}_{帧数}.mp4
        output_filename = f"{clip_duration}_{frame_count}_idx{i}.mp4"
        output_path = os.path.join(target_dir, output_filename)

        # 3. 使用 ffmpeg 进行切分
        # -ss 在 -i 之前可以提高搜索速度
        # -t 指定持续时间
        # -c:v libx264 重新编码以确保剪切位置精准
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-t', str(clip_duration),
            '-i', src_video,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-strict', 'experimental',
            output_path
        ]
        
        print(f"正在生成第 {i+1}/50 个片段: {output_filename} (从 {start_time}s 开始)")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"\n任务完成！50个视频片段保存在: {target_dir}")

if __name__ == "__main__":
    main()