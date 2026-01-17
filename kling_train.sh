# export MODELSCOPE_CACHE=/m2v_intern/mengzijie/Wan2.2/
export http_proxy=http://10.66.16.238:11080 https_proxy=http://10.66.16.238:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
export PATH="/m2v_intern/mengzijie/env/wan2.2/bin:$PATH"

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_14B.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path "" \
  --dataset_metadata_path "/m2v_intern/mengzijie/DiffSynth-Studio/emo_ge81f_verified.csv" \
  --data_file_keys "video_path,audio_path" \
  --dataset_num_workers 4 \
  --save_steps 100 \
  --height 640 \
  --width 480 \
  --tgt_fps 15 \
  --num_frames 57 \
  --dataset_repeat 1 \
  --model_paths '[["/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/diffusion_pytorch_model-00001-of-00004.safetensors", "/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/diffusion_pytorch_model-00002-of-00004.safetensors", "/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/diffusion_pytorch_model-00003-of-00004.safetensors", "/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/diffusion_pytorch_model-00004-of-00004.safetensors"], "/m2v_intern/mengzijie/DiffSynth-Studio/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors", "/m2v_intern/mengzijie/DiffSynth-Studio/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.1_VAE.safetensors", "/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/model.safetensors"]' \
  --audio_processor_path "/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english" \
  --tokenizer_path "/m2v_intern/mengzijie/DiffSynth-Studio/models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --trainable_models "dit" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/m2v_intern/mengzijie/DiffSynth-Studio/models/train/Wan2.2-S2V-14B_debug_singlegpu" \
  --extra_inputs "input_image,input_audio" \
  --offload_optimizer_device "cpu" \
  # --debug \
  # --use_gradient_checkpointing_offload \
  # --model_id_with_origin_paths "Wan-AI/Wan2.2-S2V-14B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-S2V-14B:wav2vec2-large-xlsr-53-english/model.safetensors,Wan-AI/Wan2.2-S2V-14B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-S2V-14B:Wan2.1_VAE.pth" \