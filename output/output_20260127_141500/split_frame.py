import cv2
import os

# 视频路径
video_path = "/m2v_intern/mengzijie/DiffSynth-Studio/output/output_20260127_141500/复杂情况_何炅_2_英文_口播_稍慢节奏_中年男声_2.mp4"

# 1. 确定保存路径：在原目录下创建一个与视频文件名相同的子文件夹
video_dir = os.path.dirname(video_path)
video_name = os.path.splitext(os.path.basename(video_path))[0]
save_dir = os.path.join(video_dir, video_name + "_frames")

# 2. 创建文件夹
os.makedirs(save_dir, exist_ok=True)
print(f"帧将保存在: {save_dir}")

# 3. 读取视频并逐帧保存
cap = cv2.VideoCapture(video_path)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 保存为 00000.jpg, 00001.jpg ...
    save_path = os.path.join(save_dir, f"{frame_idx:05d}.jpg")
    cv2.imwrite(save_path, frame)
    
    if frame_idx % 50 == 0:
        print(f"已保存第 {frame_idx} 帧...")
    frame_idx += 1

cap.release()
print(f"完成！共保存 {frame_idx} 帧。")