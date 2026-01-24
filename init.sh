
#!/bin/bash

cat >> ~/.tmux.conf << 'EOF'
set -g mouse on
set -g mode-keys vi
bind -T copy-mode-vi v send -X begin-selection
bind -T copy-mode-vi y send -X copy-pipe-and-cancel "xclip -sel clip 2>/dev/null || pbcopy"
EOF
tmux source ~/.tmux.conf 2>/dev/null || true
echo "Done. 重启tmux或按 Ctrl+b : 输入 source ~/.tmux.conf"

# pip install -e .

# conda activate /m2v_intern/mengzijie/env/wan2.2/
# pkill -f check_gpu_big
# bash /m2v_intern/mengzijie/DiffSynth-Studio/gpu.sh
apt update
apt install unzip
export PATH="/m2v_intern/mengzijie/env/wan2.2/bin:$PATH"

which python
cd /m2v_intern/mengzijie/DiffSynth-Studio/


##--------------------------##
# cd /ytech_milm_disk2/tangsonglin05/exp/online_service/human-motion-pe/
VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python i2v_v4_infer_motion_image_direction_set_seed.py --csv_path /ytech_milm/mengzijie/DiffSynth-Studio/movement/MzjTest_Fixed_Path.csv




##--------------------------##

# 接下来需要你在这个csv文件后面再给我添加一列，名字叫IDvideo，然后下面每一列是一个路径，包括一个mp4文件，找到路径的方法如下：
# 首先找到这个csv中的image表头字段，然后提取这个字段路径中倒数第二个目录的名字，然后在这个路径中：/m2v_intern/mengzijie/DiffSynth-Studio/movement/image/新增image
# ，找到以这个名字命名的目录，目录中有几张图片和唯一的一个视频，我们需要的就是这个视频的路径，拿到路径后，把路径中的/m2v_intern/换成/ytech_milm/，然后把这个路径填入csv文件的IDvideo字段中即可。
# 比如，image中第一行数据的完整路径是：/ytech_milm/mengzijie/DiffSynth-Studio/movement/image/多ID图注入/常规情况/赵本山/首帧_赵本山.jpg，你找到倒数第二个子目录：赵本山，然后在/m2v_intern/mengzijie/DiffSynth-Studio/movement/image/新增image
# 路径下找名为“赵本山”的子目录，然后找到这个目录下的唯一一个mp4文件即可

# 注意，只会遇到两种情况，第一种是找到了视频，第二种是这个路径下：/m2v_intern/mengzijie/DiffSynth-Studio/movement/image/新增image没有这个以人名命名的子目录，如果遇到第二种情况，需要你把没有的这个人民记录到一个txt文件中，
# 其余的所有情况都是异常情况（bug），请你编程时写足够鲁棒的逻辑为我筛查bug，最后，在代码的最开始要写注释，写明白这个脚本的作用