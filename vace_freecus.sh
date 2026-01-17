# python examples/wanvideo/model_inference/Wan2.1-VACE-FreeCus.py \
#   --reference_image "data/examples/wan/cat_fightning.jpg" \
#   --prompt "两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。"\
#   --control_video ""data/examples/wan/depth_video.mp4"" \


python examples/wanvideo/model_inference/Wan2.1-VACE-FreeCus.py \
  --reference_image "data/examples/wan/cat_fightning.jpg" \
  --prompt "两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。" \
  --control_video "data/examples/wan/depth_video.mp4" \
  --model_id "Wan-AI/Wan2.1-VACE-1.3B" \
  --negative_prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
  --subject_word "subject" \
  --freecus_scale 1.1 \
  --shift_type "negative" \
  --vital_layers "0,1,2,10,12,14,25,27,29" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --num_inference_steps 50 \
  --cfg_scale 5.0 \
  --seed 42 \
  --segmentation_model "" \
  --vl_model_path "" \
  --llm_model_path "" \
  --output_path "output_vace_freecus.mp4" \
  --fps 15 \
  --tiled \
  --use_subject_caption