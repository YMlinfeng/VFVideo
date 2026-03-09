# -*- coding: utf-8 -*-

import os
import cv2
import torch
import pytorch_lightning as pl
import numpy as np
import argparse
from hmr4d.utils.pylogger import Log
import hydra
from hydra import initialize_config_module, compose
from pathlib import Path
from pytorch3d.transforms import quaternion_to_matrix

# 在解析路径时，显式处理 Unicode
from pathlib import Path

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    read_video_np,
    save_video,
    merge_videos_horizontal,
    get_writer,
    get_video_reader,
)
from hmr4d.utils.vis.cv2_utils import (
    draw_bbx_xyxy_on_image_batch,
    draw_coco17_skeleton_batch,
)

from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SLAMModel

from hmr4d.utils.geo.hmr_cam import (
    get_bbx_xys_from_xyxy,
    estimate_K,
    convert_K_to_K4,
    create_camera_sensor,
)
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.vis.renderer import (
    Renderer,
    get_global_cameras_static,
    get_ground_params_from_points,
)
from tqdm import tqdm
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from einops import einsum, rearrange
import json
import time
from tqdm import tqdm



CRF = 23  # 17 is lossless, every +6 halves the mp4 size


class SmplInfer:
    def __init__(self):
        self.tracker = Tracker()
        self.vitpose_extractor = VitPoseExtractor()
        self.extractor = Extractor()
        self.slam_height = None
        self.slam_width = None
        self.slam = None
        self.cfg = self.parse_args_to_cfg()
        self.model: DemoPL = hydra.utils.instantiate(self.cfg.model, _recursive_=False)
        print("model: DemoPL = hydra.utils.instantiate sucess")
        self.model.load_pretrained_model(self.cfg.ckpt_path)
        print("load_pretrained_model")
        self.model = self.model.eval().cuda()
        print("model.eval().cuda()")

    def parse_args_to_cfg(self):
        current_file_path = os.path.abspath(__file__)
        # 获取父目录路径
        parent_directory = os.path.dirname(current_file_path)

        # Input
        verbose = False
        static_cam = False

        with initialize_config_module(
            version_base="1.3", config_module=f"hmr4d.configs"
        ):
            overrides = [
                f"static_cam={static_cam}",
                f"verbose={verbose}",
            ]
            register_store_gvhmr()
            cfg = compose(config_name="demo", overrides=overrides)

        # cfg.video_path = video_path
        ckpt_path = cfg.get("ckpt_path", None)
        cfg.ckpt_path = parent_directory + "/" + ckpt_path
        return cfg

    @torch.no_grad()
    def run_preprocess(self, cfg):

        Log.info(f"[Preprocess] Start!")
        tic = Log.time()
        video_path = cfg.video_path
        paths = cfg.paths
        static_cam = cfg.static_cam
        verbose = cfg.verbose
        verbose = False
        # Get bbx tracking result
        if not Path(paths.bbx).exists():

            bbx_xyxy = self.tracker.get_one_track(str(video_path)).float()  # (L, 4)
            print("bbx_xyxy***************", bbx_xyxy)
            bbx_xys = get_bbx_xys_from_xyxy(
                bbx_xyxy, base_enlarge=1.2
            ).float()  # (L, 3) apply aspect ratio and enlarge
            torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        else:
            bbx_xys = torch.load(paths.bbx)["bbx_xys"]
            Log.info(f"[Preprocess] bbx (xyxy, xys) from {paths.bbx}")
        if verbose:
            video = read_video_np(str(video_path))
            bbx_xyxy = torch.load(paths.bbx)["bbx_xyxy"]
            video_overlay = draw_bbx_xyxy_on_image_batch(bbx_xyxy, video)
            save_video(video_overlay, cfg.paths.bbx_xyxy_video_overlay)

        # Get VitPose
        if not Path(paths.vitpose).exists():

            vitpose = self.vitpose_extractor.extract(str(video_path), bbx_xys)
            torch.save(vitpose, paths.vitpose)
        else:
            vitpose = torch.load(paths.vitpose)
            Log.info(f"[Preprocess] vitpose from {paths.vitpose}")
        if verbose:
            video = read_video_np(str(video_path))
            video_overlay = draw_coco17_skeleton_batch(video, vitpose, 0.5)
            save_video(video_overlay, paths.vitpose_video_overlay)

        # Get vit features
        if not Path(paths.vit_features).exists():

            vit_features = self.extractor.extract_video_features(
                str(video_path), bbx_xys
            )
            torch.save(vit_features, paths.vit_features)
        else:
            Log.info(f"[Preprocess] vit_features from {paths.vit_features}")

        # Get DPVO results
        if not static_cam:  # use slam to get cam rotation
            if not Path(paths.slam).exists():
                length, width, height = get_video_lwh(cfg.video_path)
                # 建议处理的数据按视频相同的HW来分批处理已减少SLAMModel的反复初始化加载
                # if (
                #     self.slam is None
                #     or height != self.slam_height
                #     or width != self.slam_width
                # ):
                K_fullimg = estimate_K(width, height)
                intrinsics = convert_K_to_K4(K_fullimg)
                # self.slam_height = height
                # self.slam_width = width
                self.slam = SLAMModel(
                    str(video_path),
                    width,
                    height,
                    intrinsics,
                    buffer=4000,
                    resize=0.5,
                )
                bar = tqdm(total=length, desc="DPVO")
                while True:
                    ret = self.slam.track()
                    if ret:
                        bar.update()
                    else:
                        break
                slam_results = self.slam.process()  # (L, 7), numpy
                torch.save(slam_results, paths.slam)
            else:
                Log.info(f"[Preprocess] slam results from {paths.slam}")

        Log.info(f"[Preprocess] End. Time elapsed: {Log.time()-tic:.2f}s")

    def load_data_dict(self, cfg):
        paths = cfg.paths
        length, width, height = get_video_lwh(cfg.video_path)
        if cfg.static_cam:
            R_w2c = torch.eye(3).repeat(length, 1, 1)
        else:
            traj = torch.load(cfg.paths.slam)
            traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
            R_w2c = quaternion_to_matrix(traj_quat).mT
        K_fullimg = estimate_K(width, height).repeat(length, 1, 1)
        # K_fullimg = create_camera_sensor(width, height, 26)[2].repeat(length, 1, 1)

        data = {
            "length": torch.tensor(length),
            "bbx_xys": torch.load(paths.bbx)["bbx_xys"],
            "kp2d": torch.load(paths.vitpose),
            "K_fullimg": K_fullimg,
            "cam_angvel": compute_cam_angvel(R_w2c),
            "f_imgseq": torch.load(paths.vit_features),
        }
        return data

    def render_incam(self, cfg):
        incam_video_path = Path(cfg.paths.incam_video)
        if incam_video_path.exists():
            Log.info(f"[Render Incam] Video already exists at {incam_video_path}")
            return

        pred = torch.load(cfg.paths.hmr4d_results)
        smplx = make_smplx("supermotion").cuda()
        smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
        faces_smpl = make_smplx("smpl").faces

        # smpl
        smplx_out = smplx(**to_cuda(pred["smpl_params_incam"]))
        pred_c_verts = torch.stack(
            [torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices]
        )

        # -- rendering code -- #
        video_path = cfg.video_path
        length, width, height = get_video_lwh(video_path)
        K = pred["K_fullimg"][0]

        # renderer
        renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
        reader = get_video_reader(video_path)  # (F, H, W, 3), uint8, numpy
        bbx_xys_render = torch.load(cfg.paths.bbx)["bbx_xys"]

        # -- render mesh -- #
        verts_incam = pred_c_verts
        writer = get_writer(incam_video_path, fps=30, crf=CRF)
        for i, img_raw in tqdm(
            enumerate(reader),
            total=get_video_lwh(video_path)[0],
            desc=f"Rendering Incam",
        ):
            img = renderer.render_mesh(verts_incam[i].cuda(), img_raw, [0.8, 0.8, 0.8])

            # # bbx
            # bbx_xys_ = bbx_xys_render[i].cpu().numpy()
            # lu_point = (bbx_xys_[:2] - bbx_xys_[2:] / 2).astype(int)
            # rd_point = (bbx_xys_[:2] + bbx_xys_[2:] / 2).astype(int)
            # img = cv2.rectangle(img, lu_point, rd_point, (255, 178, 102), 2)

            writer.write_frame(img)
        writer.close()
        reader.close()

    def render_global(self, cfg):
        global_video_path = Path(cfg.paths.global_video)
        if global_video_path.exists():
            Log.info(f"[Render Global] Video already exists at {global_video_path}")
            return

        debug_cam = False
        pred = torch.load(cfg.paths.hmr4d_results)
        smplx = make_smplx("supermotion").cuda()
        smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
        faces_smpl = make_smplx("smpl").faces
        J_regressor = torch.load(
            "hmr4d/utils/body_model/smpl_neutral_J_regressor.pt"
        ).cuda()

        # smpl
        smplx_out = smplx(**to_cuda(pred["smpl_params_global"]))
        pred_ay_verts = torch.stack(
            [torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices]
        )

        def move_to_start_point_face_z(verts):
            "XZ to origin, Start from the ground, Face-Z"
            # position
            verts = verts.clone()  # (L, V, 3)
            offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]  # (3)
            offset[1] = verts[:, :, [1]].min()
            verts = verts - offset
            # face direction
            T_ay2ayfz = compute_T_ayfz2ay(
                einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True
            )
            verts = apply_T_on_points(verts, T_ay2ayfz)
            return verts

        verts_glob = move_to_start_point_face_z(pred_ay_verts)
        joints_glob = einsum(
            J_regressor, verts_glob, "j v, l v i -> l j i"
        )  # (L, J, 3)
        global_R, global_T, global_lights = get_global_cameras_static(
            verts_glob.cpu(),
            beta=2.0,
            cam_height_degree=20,
            target_center_height=1.0,
        )

        # -- rendering code -- #
        video_path = cfg.video_path
        length, width, height = get_video_lwh(video_path)
        _, _, K = create_camera_sensor(width, height, 24)  # render as 24mm lens

        # renderer
        renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
        # renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K, bin_size=0)

        # -- render mesh -- #
        scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
        renderer.set_ground(scale * 1.5, cx, cz)
        color = torch.ones(3).float().cuda() * 0.8

        render_length = length if not debug_cam else 8
        writer = get_writer(global_video_path, fps=30, crf=CRF)
        for i in tqdm(range(render_length), desc=f"Rendering Global"):
            cameras = renderer.create_camera(global_R[i], global_T[i])
            img = renderer.render_with_ground(
                verts_glob[[i]], color[None], cameras, global_lights
            )
            writer.write_frame(img)
        writer.close()

    def infer(self, input_file, output_result_path):
        # 设置输入
        if len(output_result_path) > 0:

            # 确保路径是 Unicode 格式
            # output_result_path = output_result_path.encode('utf-8').decode('utf-8')

            # 获取视频的文件名（带扩展名）
            video_name_with_ext = os.path.basename(
                input_file
            )  # 获取文件名，例如 'video.mp4'

            # 去掉文件扩展名（.mp4）并加上 .json
            video_name_without_ext = os.path.splitext(video_name_with_ext)[0]                       # 去掉扩展名，得到 'video'
            file_type              = os.path.splitext(video_name_with_ext)[1]

            if file_type in ['.mp4', '.MP4', '.avi', '.AVI', ".mov", ".MOV"]:
                print("input is [video]")
            elif file_type in ['.png', ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]:
                print("input is [image]")
                
                # 将image转换成video
                input_file_video = input_file[:-len(file_type)] + ".mp4"
                cmd = "ffmpeg -loop 1 -i {} -t 1 {}".format(input_file, input_file_video)
                os.system(cmd)

                print("convert image to mp4, then infer SMPL based on mp4")
                input_file = input_file_video

            else:
                print("wrong input type, not [image] or [video], please check input file")
                exit()


            output_json_file_path = f"{output_result_path}/{video_name_without_ext}.json"  # 拼接成最终的输出路径

        else:
            # 设置输出
            output_json_file_folde_prefix = "/ytech_milm/Keling_HumanMotion/Smpl/"
            output_json_file_path = (
                input_file.replace(
                    "/ytech_milm/Keling_HumanMotion/", output_json_file_folde_prefix
                )[:-4]
                + ".json"
            )

            if os.path.exists(output_json_file_path):
                print(f"File {output_json_file_path} already exists. Skipping.")
                result = {"smpl3d": output_json_file_path}  # 跳过当前处理，继续下一个
                return json.dumps(result).encode("utf-8").decode("unicode_escape")

            os.makedirs(os.path.dirname(output_json_file_path), exist_ok=True)
            ########################################################################################################################
            # 这个固定死
            output_result_folder_prefix = "/ytech_milm/Keling_HumanMotion/Smpl/"  # 实际上应该是/ytech_milm/Keling_HumanMotion/smpl/, 去掉test后缀
            output_result_path = input_file.replace(
                "/ytech_milm/Keling_HumanMotion/", output_result_folder_prefix
            )

        output_folder = os.path.dirname(output_result_path)

        video_path = Path(input_file)
        assert video_path.exists(), f"Video not found at {video_path}"
        self.cfg.video_name = video_path.stem
        self.cfg.video_path = video_path
        output_path = Path(output_folder).absolute().as_posix()
        self.cfg.output_root = output_path
        Log.info(f"[Output Dir]: {self.cfg.output_dir}")
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

        paths = self.cfg.paths
        print("paths", paths)
        # ===== Preprocess and save to disk ===== #
        self.run_preprocess(self.cfg)
        data = self.load_data_dict(self.cfg)

        # ===== HMR4D ===== #
        if not Path(paths.hmr4d_results).exists():
            Log.info("[HMR4D] Predicting")
            tic = Log.sync_time()
            pred = self.model.predict(data, static_cam=self.cfg.static_cam)
            pred = detach_to_cpu(pred)
            data_time = data["length"] / 30
            Log.info(
                f"[HMR4D] Elapsed: {Log.sync_time() - tic:.2f}s for data-length={data_time:.1f}s"
            )
            torch.save(pred, paths.hmr4d_results)
        print("Done!")
        # save result
        result_json = {}
        result_json["hmr4d_results"] = paths.hmr4d_results
        result_json["bbx"] = paths.bbx
        result_json["vitpose"] = paths.vitpose
        result_json["vit_features"] = paths.vit_features
        result_json["slam"] = paths.slam

        with open(output_json_file_path, "w") as json_file:
            json.dump(
                result_json, json_file, indent=4, ensure_ascii=False
            )  # indent=4 格式化输出 ensure_ascii=False 来禁用 ASCII 转义，这样 JSON 文件中的中文字符就会正常显示

        print("final_output_json_path===========", output_json_file_path)

        # new_path = path.replace("'", '"')
        result = {"smpl3d": output_json_file_path}
        print(
            "json.dumps(result)=============",
            json.dumps(result).encode("utf-8").decode("unicode_escape"),
        )
        # self.render_incam(self.cfg)
        # self.render_global(self.cfg)
        return json.dumps(result).encode("utf-8").decode("unicode_escape")




# 确保只有当直接运行此脚本时才执行 main 函数
if __name__ == "__main__":
    # output_json_file_path 暂不支持给地址，需要按照需求改造中间产物存放地址，路径写死的（仅限于用于看keling）

    parser = argparse.ArgumentParser(description="gRPC server")
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument('--is_image', type=int, default=0)              # required=True, help="0:video, 1:image")

    args = parser.parse_args()


    smpl_infer = SmplInfer()
    # result = smpl_infer.infer('/ytech_milm/Keling_HumanMotion/data/one-man/livephoto-body/livephoto-body-Kwai_50k/all/ac/000476/000/video.mp4','')
    # result = smpl_infer.infer('/ytech_milm/xujialin/personal_project/smpl3d/grpc-python-raven/data/2s.mp4','/ytech_milm/xujialin/personal_project/smpl3d/grpc-python-raven/data')

    input_file = args.input_file
    output_dir = args.output_dir

        
    result = smpl_infer.infer(input_file, output_dir)

    # # print("time",time.time() -time0)
    # unicode_string = r'/ytech_milm/Keling_HumanMotion/smpljson_test/one-man/livephoto-body/livephoto-body-Kwai_50k/bodydance\u4eba\u4f53\u4e0b\u8e72/ac/000199/005/video.json'  # 这代表 "你好世界"

    # # 使用unicode_escape解码
    # decoded_string = unicode_string.encode('utf-8').decode('unicode_escape')

    # print(decoded_string)  # 输出: 你好世界

