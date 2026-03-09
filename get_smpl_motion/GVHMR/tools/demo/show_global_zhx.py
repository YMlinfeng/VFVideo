import cv2
import torch
import sys

import pytorch_lightning as pl
import numpy as np
import argparse
from hmr4d.utils.pylogger import Log
import hydra
from hydra import initialize_config_module, compose
from pathlib import Path
from pytorch3d.transforms import quaternion_to_matrix

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    read_video_np,
    save_video,
    merge_videos_horizontal,
    get_writer,
    get_video_reader,
)
from hmr4d.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch, draw_coco17_skeleton_batch

from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SLAMModel

from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, convert_K_to_K4, create_camera_sensor
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from tqdm import tqdm
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from einops import einsum, rearrange
import pandas as pd
import os

CRF = 23
def render_global():
    csv_file = '/vh_data/zhanghaoxian/m2v-diffusers/data_path/debug0224_depth_smpl.csv'
    df = pd.read_csv(csv_file)

    for idx, row in df.iterrows():
        file_path = row['smpl_result'][:-11]
        smpl_path = file_path + '/video/hmr4d_results.pt'
        video_path = file_path + '/video.mp4'

        smpl_path = smpl_path.replace('smpljson_test', 'smpl_test')
        video_path = video_path.replace('smpljson_test', 'data')

        video_vae_path = row['video_vae_pt']
        new_name = os.path.splitext(os.path.basename(video_vae_path))[0]+'.mp4'
        global_video_path = os.path.join('/vh_data/zhanghaoxian/m2v-diffusers/outputs/smpl_GT_0303/', new_name)

        '''debug'''
        # global_video_path = Path('/vh_data/zhanghaoxian/m2v-diffusers/outputs/smpl_GT_0303/debug.mp4')
        # smpl_path = '/ytech_milm/Keling_HumanMotion/smpl_test/one-man/livephoto-body/livephoto-body-Kwai_50k/bodydance_download_new/ac/002717/000/video/hmr4d_results.pt'
        # video_path = '/ytech_milm/Keling_HumanMotion/data/one-man/livephoto-body/livephoto-body-Kwai_50k/bodydance_download_new/ac/002717/000/video.mp4'

        #if global_video_path.exists():
        #    Log.info(f"[Render Global] Video already exists at {global_video_path}")
        #    return

        debug_cam = False
        pred = torch.load(smpl_path)
        smplx = make_smplx("supermotion").cuda()
        smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
        faces_smpl = make_smplx("smpl").faces
        J_regressor = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()

        # smpl
        print(pred["smpl_params_global"].keys())
        #pred["smpl_params_global"]['transl']*=0
        #pred["smpl_params_global"]['global_orient']*=0
        random_beta=torch.randn(pred["smpl_params_global"]['betas'][0].shape)
        for i in range(pred["smpl_params_global"]['betas'].shape[0]):
            pred["smpl_params_global"]['betas'][i] = random_beta
        print(pred["smpl_params_global"]['betas'],pred["smpl_params_global"]['betas'].shape,pred["smpl_params_global"]['betas'].min(),pred["smpl_params_global"]['betas'].max())
        print(pred["smpl_params_global"]['betas'][0]==pred["smpl_params_global"]['betas'][1])
        
        smplx_out = smplx(**to_cuda(pred["smpl_params_global"]))
        pred_ay_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])

        def move_to_start_point_face_z(verts):
            "XZ to origin, Start from the ground, Face-Z"
            # position
            verts = verts.clone()  # (L, V, 3)
            offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]  # (3)
            offset[1] = verts[:, :, [1]].min()
            verts = verts - offset
            # face direction
            T_ay2ayfz = compute_T_ayfz2ay(einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True)
            verts = apply_T_on_points(verts, T_ay2ayfz)
            return verts

        verts_glob = move_to_start_point_face_z(pred_ay_verts)
        joints_glob = einsum(J_regressor, verts_glob, "j v, l v i -> l j i")  # (L, J, 3)
        global_R, global_T, global_lights = get_global_cameras_static(
            verts_glob.cpu(),
            beta=2.0,
            cam_height_degree=20,
            target_center_height=1.0,
        )

        # -- rendering code -- #
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
            img = renderer.render_with_ground(verts_glob[[i]], color[None], cameras, global_lights)
            writer.write_frame(img)
        writer.close()

render_global()