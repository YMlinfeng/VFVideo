import cv2
import os

folder = './outputs_11_21'
# 输入的视频文件路径
video_files = [
    'output_video_pt.mp4', 
    'output_video_onnx.mp4', 
    'output_video_trt.mp4'
]

# 输出的视频文件路径
output_video = os.path.join(folder, 'output_video_combined.mp4')

# 打开第一个视频文件，获取视频信息
cap1 = cv2.VideoCapture(os.path.join(folder,video_files[0]))
cap2 = cv2.VideoCapture(os.path.join(folder,video_files[1]))
cap3 = cv2.VideoCapture(os.path.join(folder,video_files[2]))

# 获取视频的帧率和宽度、高度
fps = cap1.get(cv2.CAP_PROP_FPS)
frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建一个 VideoWriter 对象，用于写入合并后的视频
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码格式
out_video = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height * 3))  # 输出视频的高度是原视频的 3 倍

while True:
    # 读取每个视频的当前帧
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()

    # 如果其中一个视频结束，跳出循环
    if not ret1 or not ret2 or not ret3:
        break

    # 将三张图像竖直拼接
    combined_frame = cv2.vconcat([frame1, frame2, frame3])

    # 将合并的帧写入输出视频
    out_video.write(combined_frame)

# 释放所有资源
cap1.release()
cap2.release()
cap3.release()
out_video.release()

print(f"视频合成完成，保存为 {output_video}")
