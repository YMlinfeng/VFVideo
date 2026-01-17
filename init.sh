
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

export PATH="/m2v_intern/mengzijie/env/wan2.2/bin:$PATH"

which python
cd /m2v_intern/mengzijie/DiffSynth-Studio/


