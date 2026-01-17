# export http_proxy=http://oversea-squid2.ko.txyun:11080 https_proxy=http://oversea-squid2.ko.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
export http_proxy=http://10.68.24.160:11080 https_proxy=http://10.68.24.160:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
# export MODELSCOPE_CACHE=/m2v_intern/mengzijie/Wan2.2/
# export DIFFSYNTH_MODEL_DIR=/m2v_intern/mengzijie/Wan2.2
# export DIFFSYNTH_HOME=/m2v_intern/mengzijie/Wan2.2

python /m2v_intern/mengzijie/DiffSynth-Studio/examples/wanvideo/model_training/validate_full/Wan2.2-S2V-14B.py
