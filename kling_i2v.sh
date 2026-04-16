export http_proxy=http://10.66.16.238:11080 https_proxy=http://10.66.16.238:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
export PATH="/m2v_intern/mengzijie/env/wan2.2/bin:$PATH"
export PYTHONWARNINGS="ignore::FutureWarning"

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path "" \
  --dataset_metadata_path "/ytech_m2v2_hdd/liujiwen/audio_v3/m2v-diffusers/id_data_480_720_1080_with_pose/720_filter_avs_7d6w_30fps_split_121f_with_md5_vae_caption_te_fix_pose_new.csv,/ytech_m2v2_hdd/liujiwen/audio_v3/m2v-diffusers/id_data_480_720_1080_with_pose/720_filter_emo_40w_30fps_split_121f_with_md5_vae_caption_te_0925_40w_fix_pose.csv,/ytech_m2v2_hdd/liujiwen/audio_v3/m2v-diffusers/id_data_480_720_1080_with_pose/movie_0812_all_normal_filtered_with_md5_caption_vae_text_filter_17w.csv,/ytech_m2v2_hdd/liujiwen/audio_v3/m2v-diffusers/id_data_480_720_1080_with_pose/avspeech_0925_merged_filter4_13d4w_ht_11d4w.csv,/ytech_m2v2_hdd/liujiwen/audio_v3/m2v-diffusers/id_data_480_720_1080_with_pose/avspeech_0925_merged_filter4_13d4w_i2v_2w.csv,/ytech_m2v2_hdd/liujiwen/audio_v3/m2v-diffusers/id_data_480_720_1080_with_pose/emo_20w_30fps_with_md5_vae_caption_te_193f_3d4w.csv,/ytech_m2v2_hdd/liujiwen/audio_v3/m2v-diffusers/id_data_480_720_1080_with_pose/filtered_AVspeech_data_human_check_8dot3w_0804_with_md5_vae_caption_te_193f_2d5w.csv,/ytech_m2v2_hdd/liujiwen/audio_v3/m2v-diffusers/id_data_480_720_1080_with_pose/movie_0812_split_17d2w_merged_filter_10w_ht_6d5w.csv,/ytech_m2v2_hdd/liujiwen/audio_v3/m2v-diffusers/id_data_480_720_1080_with_pose/movie_0812_split_17d2w_merged_filter_10w_i2v_3d6w.csv" \
  --data_file_keys "video_path" \
  --dataset_num_workers 0 \
  --save_steps 100 \
  --max_pixels 268800 \
  --num_frames 41 \
  --dataset_repeat 1 \
  --model_paths '[["/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors", "/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors", "/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors", "/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors","/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors","/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors","/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"], "/m2v_intern/mengzijie/DiffSynth-Studio/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors", "/m2v_intern/mengzijie/DiffSynth-Studio/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors", "/m2v_intern/mengzijie/DiffSynth-Studio/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"]' \
  --learning_rate 1e-5 \
  --num_epochs 10 \
  --trainable_models "dit" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/ytech_m2v4_hdd/mengzijie/DiffSynth-Studio/models/train/i2v_debug" \
  --extra_inputs "input_image" \
  --gradient_accumulation_steps 1 \
  --debug \
  --enable_id_grid \
  --id_grid_max_pixels 268800 \
  --id_grid_num_frames 1 \
  --id_drop_rate 0.2 \
  --use_swanlab \
  --swanlab_mode "local" \
  --swanlab_project "wan_video_id_grid" \
  --swanlab_run_name "debug_run" \
2>&1 | tee "logs/[Debug]_wan_train_$(date +'%Y.%m.%d_%H:%M:%S').log"

  # --use_gradient_checkpointing_offload \
  # --model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors,Wan-AI/Wan2.2-S2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-S2V-14B:Wan2.1_VAE.pth" \



# sleep  100d;
  # --dataset_metadata_path "/m2v_intern/mengzijie/DiffSynth-Studio/emo_ge81f_verified_test320.csv" \
