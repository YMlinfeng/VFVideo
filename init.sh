
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
# apt update
# apt install unzip
export PATH="/m2v_intern/mengzijie/env/wan2.2/bin:$PATH"

which python
cd /m2v_intern/mengzijie/DiffSynth-Studio/


##--------------------------##
# cd /ytech_milm_disk2/tangsonglin05/exp/online_service/human-motion-pe/
# VLLM_WORKER_MULTIPROC_METHOD=spawn CUDA_VISIBLE_DEVICES=2,3  python i2v_v4_infer_motion_image_direction_set_seed.py --csv_path /ytech_milm/mengzijie/DiffSynth-Studio/movement/Mzj54.csv




##--------------------------##

# git branch -M main
# git push -u origin main

# git remote -v

# # 如果没有 origin，添加它
# git remote add origin https://github.com/YMlinfeng/VFVideo.git

# # 然后推送
# git push -u origin master
# git config user.name "YMlinfeng"
# git config user.email "xiao102851@163.com"
# git config --global user.name "YMlinfeng"
# git config --global user.email "xiao102851@163.com"

# # 可选：设置其他常用配置
# git config --global core.editor vim  # 设置默认编辑器为 vim
# git config --global color.ui auto    # 启用彩色输出
# git config --global init.defaultBranch main  # 设置默认分支名为 main