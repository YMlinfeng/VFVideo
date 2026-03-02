import argparse
from diffsynth.core.data.unified_dataset import UnifiedDataset
from diffsynth.core.data.operators import LoadAudio
import os

def parse_args():
    parser = argparse.ArgumentParser()
    # 基础数据集参数
    parser.add_argument("--dataset_base_path", type=str, default="")
    parser.add_argument("--dataset_metadata_path", type=str, default="/m2v_intern/mengzijie/DiffSynth-Studio/emo_ge81f_verified.csv")
    parser.add_argument("--data_file_keys", type=str, default="video_path")
    parser.add_argument("--dataset_repeat", type=int, default=1)
    
    # 视频处理参数
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=560)
    parser.add_argument("--num_frames", type=int, default=41)
    parser.add_argument("--max_pixels", type=int, default=1920*1080)
    parser.add_argument("--tgt_fps", type=int, default=15)

    # ID 九宫格注入参数
    parser.add_argument("--enable_id_grid", action="store_true", default=True, help="测试默认开启")
    parser.add_argument("--id_grid_height", type=int, default=None)
    parser.add_argument("--id_grid_width", type=int, default=None)
    parser.add_argument("--id_grid_max_pixels", type=int, default=268800)
    parser.add_argument("--id_grid_aug_intensity", type=float, default=1.9)
    
    # Debug 参数
    parser.add_argument("--debug", action="store_true", default=True, help="测试默认开启")
    parser.add_argument("--debug_save_dir", type=str, default="./debug_vis")
    
    # 测试数量
    parser.add_argument("--num_test_samples", type=int, default=1, help="测试处理多少条数据")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Initializing UnifiedDataset...")
    print(f"Metadata path: {args.dataset_metadata_path}")
    print(f"Enable ID Grid: {args.enable_id_grid}")
    print(f"Debug Mode: {args.debug}")
    
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        num_frames=args.num_frames,
        tgt_fps=args.tgt_fps,
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "audio_path": LoadAudio(
                num_frames=args.num_frames, 
                tgt_fps=args.tgt_fps, 
                sr=16000
            ),
        },
        enable_id_grid=args.enable_id_grid,
        id_grid_height=args.id_grid_height,
        id_grid_width=args.id_grid_width,
        id_grid_max_pixels=args.id_grid_max_pixels,
        id_grid_aug_intensity=args.id_grid_aug_intensity,
        debug=args.debug,
        debug_save_dir=args.debug_save_dir,
    )
    
    print(f"Dataset initialized. Total samples: {len(dataset)}")
    
    # 获取几条数据测试
    for i in range(min(args.num_test_samples, len(dataset))):
        print(f"\nFetching sample {i}...")
        data = dataset[i]
        
        print("Sample keys:")
        for k, v in data.items():
            if hasattr(v, 'shape'):
                print(f"  - {k}: Tensor/Array of shape {v.shape}, dtype {v.dtype}")
            elif isinstance(v, list) and len(v) > 0 and hasattr(v[0], 'shape'):
                print(f"  - {k}: List of {len(v)} items, first item shape {v[0].shape}")
            else:
                print(f"  - {k}: {type(v)} (value: {str(v)[:50]}...)")
                
    print("\nDataset test finished. Check the debug_vis directory for outputs.")

if __name__ == "__main__":
    main()
