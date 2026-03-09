import cv2
import torch
import sys
import json
import math
import pytorch_lightning as pl
import numpy as np
import argparse
import matplotlib
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
from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points, perspective_projection
from tqdm import tqdm
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from einops import einsum, rearrange


import numpy as np

def rodrigues_to_matrix(rot_vec):
    """
    将旋转向量转换为旋转矩阵（罗德里格斯公式）
    :param rot_vec: 旋转向量,格式为三维数组或np.ndarray
    :return: 3x3旋转矩阵
    """
    rot_vec = np.asarray(rot_vec, dtype=np.float64).flatten()
    theta = np.linalg.norm(rot_vec)
    
    if theta < 1e-10:
        return np.eye(3)
    
    axis = rot_vec / theta
    K = np.array([
        [0,        -axis[2], axis[1]],
        [axis[2],  0,        -axis[0]],
        [-axis[1], axis[0],  0]
    ])
    
    I = np.eye(3)
    R = I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R

def rotate_points(points, R):
    """
    使用旋转矩阵旋转三维点集
    :param points: Nx3的坐标点数组
    :param R: 3x3旋转矩阵
    :return: 旋转后的Nx3坐标点数组
    """
    return np.dot(points, R.T)  # 等价于 (R @ points.T).T



CRF = 23  # 17 is lossless, every +6 halves the mp4 size

# cfg.paths.incam_video = 'outputs/test.mp4'
# cfg.paths.hmr4d_results = 'inputs/hmr4d_results.pt'
# cfg.video_path = '/ytech_milm/Keling_HumanMotion/data/one-man/livephoto-body/livephoto-body-Kwai_50k/bodydance_caiji02_new/ae/000727/001/dwpose_style2.mp4'
# cfg.paths.bbx = ''/group/stonefan/smpl/4000line.json

def draw_dwpose(img,start,end,colors,stickwidth = 6):
    X = np.array([start[1],end[1]])
    Y = np.array([start[0],end[0]])
    mX = np.mean(X)
    mY = np.mean(Y)
    length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
    angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
    polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
    cv2.fillConvexPoly(img, polygon, colors)
    
    return img


def draw_lines(img,start,end,colorx,colory):
    img = cv2.circle(img, (int(start[0]), int(start[1])), 4, colorx, thickness=-1)
    
    max_span = np.max(np.abs(start-end))
    span = max_span//4+1
    for i in range(span):
        x,y = ((span-i)*start+i*end)//span
        cx = ((span-i)*colorx+i*colory)//span
        img = cv2.circle(img, (int(x), int(y)), 4, colory, thickness=-1)
    img = cv2.circle(img, (int(end[0]), int(end[1])), 4, colory, thickness=-1)
    return img
def render_incam(incam_video_path,pt_path,video_path,dw_path,smplx):
    
    # if incam_video_path.exists():
    #     Log.info(f"[Render Incam] Video already exists at {incam_video_path}")
    #     return
    ##dw_pose = np.load(dw_path)
    pred = torch.load(pt_path)
    print(pt_path)
    #test = np.zeros((1536,1024,3))
    n=92
    source_motion ='/ytech_milm/Keling_HumanMotion/processed/one-man/livephoto-body/livephoto-body-Kwai_50k/bodydance_caiji_new/ad/000604/001/video/hmr4d_results.pt'
    pred_src =torch.load(source_motion)
    print(pred["smpl_params_incam"].keys())
    print(pred_src["smpl_params_incam"]["global_orient"][0,...])
    '''
    for it in tt:
        print(it)
        test = cv2.circle(test,(int(it[0]*1024),int(it[1]*1536)),1,(255, 255, 255),2)
    cv2.imwrite('test.png',test)
    '''
    smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
    faces_smpl = make_smplx("smpl").faces
    for i in range(1,299):
        pred["smpl_params_incam"]["body_pose"][i,...] = pred_src["smpl_params_incam"]["body_pose"][i,...]
        pred["smpl_params_incam"]["transl"][i,...] = pred_src["smpl_params_incam"]["transl"][i,...]-pred_src["smpl_params_incam"]["transl"][0,...]+pred["smpl_params_incam"]["transl"][0,...]
        pred["smpl_params_incam"]["global_orient"][i,...] = pred_src["smpl_params_incam"]["global_orient"][i,...]
    
    # smpl
    smplx_out = smplx(**to_cuda(pred["smpl_params_incam"]))
    pred_c_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])
    rot = rodrigues_to_matrix(np.array([0,-np.pi/3,0]))
    trans =np.array([[2.2,0,1.3]]*6890)
    rot = torch.tensor(rot,dtype=torch.float32).cuda()
    trans = torch.tensor(trans,dtype=torch.float32).cuda()
    pred_c_verts[1:] = torch.stack([(torch.mm(rot, (v_).transpose(0,1))).transpose(0,1)+trans for v_ in pred_c_verts[1:]])
    pred_c_joints = smplx_out.joints[:,:,:]
    trans =np.array([[2.2,0,1.3]]*127)
    trans = torch.tensor(trans,dtype=torch.float32).cuda()
    pred_c_joints[1:] = torch.stack([(torch.mm(rot, (v_).transpose(0,1))).transpose(0,1)+trans for v_ in pred_c_joints[1:]])
    
    print(smplx_out.joints.shape,pred_c_verts.shape)
    
    
    


    # -- rendering code -- #
    length, width, height = get_video_lwh(video_path)
    K = pred["K_fullimg"][0]


    '''
    left 92-20
    thumb 93/96-37/39+66
    index 97/100-25/27+67
    middle 101/104-28/30+68
    ring 105/108-34/36+69
    pinky 109/112-31/33+70

    right 113-21
    thumb 114/117-52/54+71
    index 118/121-40/42+72
    middle 122/125-43/45+73
    ring 126/129-49/51+74
    pinky 130/133-46/48+75
    '''
    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    verts_incam = pred_c_verts
    reader = get_video_reader(video_path)  # (F, H, W, 3), uint8, numpy
    #bbx_xys_render = torch.load(cfg.paths.bbx)["bbx_xys"]
    joints = perspective_projection(pred_c_joints,K=K.cuda())
    joints = joints.detach().cpu().numpy()

    dwp0se_smplx=[20,37,38,39,66,25,26,27,67,28,29,30,68,34,35,36,69,31,32,33,70,
                21,52,53,54,71,40,41,42,72,43,44,45,73,49,50,51,74,46,47,48,75]#92-133
    # for iframe in range(length):
    #     if (dw_pose[iframe,0,92:113,:]>0.01).all() and (dw_pose[iframe,0,92:113,2]>0.4).all():
    #         for ijoint in range(93,113):
    #             joints[iframe,dwp0se_smplx[ijoint-92],0]=(dw_pose[iframe,0,ijoint,0]-dw_pose[iframe,0,92,0])*width+joints[iframe,dwp0se_smplx[0],0]
    #             joints[iframe,dwp0se_smplx[ijoint-92],1]=(dw_pose[iframe,0,ijoint,1]-dw_pose[iframe,0,92,1])*height+joints[iframe,dwp0se_smplx[0],1]
    #     if (dw_pose[iframe,0,113:134,:]>0.01).all() and (dw_pose[iframe,0,113:134,2]>0.4).all():
    #         for ijoint in range(114,134):
    #             joints[iframe,dwp0se_smplx[ijoint-92],0]=(dw_pose[iframe,0,ijoint,0]-dw_pose[iframe,0,113,0])*width+joints[iframe,dwp0se_smplx[21],0]
    #             joints[iframe,dwp0se_smplx[ijoint-92],1]=(dw_pose[iframe,0,ijoint,1]-dw_pose[iframe,0,113,1])*height+joints[iframe,dwp0se_smplx[21],1]
    joints =joints.astype(np.int32)
    pred_c_joints = pred_c_joints.detach().cpu().numpy()
    depth = pred_c_joints[...,2]
    depth = (depth-depth.min())/(depth.max()-depth.min())
    print(joints.shape, depth.shape)
    writer = get_writer(incam_video_path, fps=30, crf=CRF)
    joints[:,12] = (joints[:,16]+joints[:,17])//2
    newA = (3*joints[:,1]-joints[:,2])/2
    newB = (3*joints[:,2]-joints[:,1])/2
    joints[:,1] =newA
    joints[:,2] =newB
    '''
    [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [85, 85, 255],
            [85, 255, 0], [85, 255, 85], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255]]

    '''

    dw_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [85, 85, 255], \
            [85, 255, 0], [85, 255, 85], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255]]
    hand_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    
    dw_points =[15,12,17,19,21,16,18,20,2,5,8,1,4,7,56,57,58,59,11,10]
    lls = [ [12,17],[12,16],[17,19],[19,21],[16,18],[18,20],[12,2],[2,5],[5,8],[12,1],[1,4],[4,7],[12,15],[15,56],[56,58],[15,57],[57,59],[7,10],[8,11],
            [20, 37],   # left hand from my view
            [37, 38],   # left_thumb
            [38, 39],
            [39, 66],   # finger tips
            [20, 25],   
            [25, 26],   # left_index
            [26, 27],   
            [27, 67],   # finger tips
            [20, 28],
            [28, 29],   # left_middle
            [29, 30],   
            [30, 68],
            [20, 34],
            [34, 35],   # left_ring
            [35, 36],
            [36, 69],
            [20, 31],
            [31, 32],   # left_pinky
            [32, 33],
            [33, 70],
            [21, 52],   # right hand from my view
            [52, 53],   # right_thumb
            [53, 54],
            [54, 71],   # finger tips
            [21, 40],   
            [40, 41],   # right_index
            [41, 42],   
            [42, 72],   # finger tips
            [21, 43],
            [43, 44],   # right_middle
            [44, 45],   
            [45, 73],   # finger tips
            [21, 49],
            [49, 50],   # right_ring
            [50, 51],
            [51, 74],   # finger tips
            [21, 46],   
            [46, 47],   # right_pinky
            [47, 48],
            [48, 75],   # finger tips

        ]
    hand_point=set([20,37,38,39,66,25,26,27,67,28,29,30,68,34,35,36,69,31,32,33,70,
                21,52,53,54,71,40,41,42,72,43,44,45,73,49,50,51,74,46,47,48,75])
    raw0=[]
    for i, img_raw in tqdm(enumerate(reader), total=get_video_lwh(video_path)[0], desc=f"Rendering Incam"):
        if i>=299:
            break
        if i ==0:
            raw0= img_raw
        
        img = renderer.render_mesh(verts_incam[i].cuda(), raw0, [0.8, 0.8, 0.8])
        ## img = img_raw * 0
        # for j,de in list(zip(joints[i],depth[i]))[:55]:
            #img = cv2.circle(img,(j[0],j[1]),1,(np.ceil(255*(1-de)), 0, np.ceil(255*de)),2)
        # draw smpl
        # for ll in lls:
        #     colorx =(np.ceil(255*(1-depth[i,ll[0]])),0,np.ceil(255*depth[i,ll[0]]))
        #     colory =(np.ceil(255*(1-depth[i,ll[1]])),0,np.ceil(255*depth[i,ll[1]]))
        #     img = draw_lines(img,colorx,colory)
        # draw orient
        # left_joint = joints[i,[1,4,7,10,13,16,18,20]]
        # right_joint = joints[i,[2,5,8,11,14,17,19,21]]
        # for js in left_joint:
        #     img = cv2.circle(img,(js[0],js[1]),4,(127,127,127),4)
        # for js in right_joint:
        #     img = cv2.circle(img,(js[0],js[1]),4,(0,255,0),4)
        # writer.write_frame(img)
        for num,ll in enumerate(lls):
            if num>18:
                img = draw_dwpose(img,joints[i,ll[0]],joints[i, ll[1]], matplotlib.colors.hsv_to_rgb([(num-19)%20 / float(20), 1.0, 1.0]) * 255,3)
            else:
                img = draw_dwpose(img,joints[i,ll[0]],joints[i, ll[1]], dw_colors[num])
                if num ==18:
                    img = (img*0.6).astype(np.uint8)
            
            
        for ipp in range(20):
            x,y = joints[i,dw_points[ipp]]
            img =cv2.circle(img, (int(x), int(y)), 6, dw_colors[ipp], thickness=-1)
        for ipp in hand_point:
            x,y = joints[i,ipp]
            img = cv2.circle(img, (x, y), 3, (0, 0, 255), thickness=-1)
        writer.write_frame(img)
    writer.close()
    reader.close()

info = json.load(open('/group/stonefan/smpl/4000line.json','r'))
smplx = make_smplx("supermotion").cuda()
for indd in range(2,4000):
    dw_path = info[indd]['kps_style2_path'].replace('/ytech_milm/Keling_HumanMotion/data/one-man/','/ytech_data_ssd/m2v/')
    dw_path = dw_path.replace('dwpose_style2.mp4','dwpose.npy')
    print(indd)
    #try:
    print(info[indd]['video_path'])
    print(info[0].keys())
    a = torch.load(info[indd]['smplx_path'])
    print(a.keys())
    render_incam('/group/stonefan/smpl/GVHMR/outputs/test{}.mp4'.format(indd),
            '/ytech_milm/liujiwen/kling_motion_service/GVHMR/甄子丹/hmr4d_results.pt',
            #info[indd]['smplx_path'],
            #info[indd]['kps_style2_path'],
            '/ytech_milm/liujiwen/kling_motion_service/GVHMR/甄子丹/preprocess/vitpose_video_overlay.mp4',
            ##info[indd]['video_path'],
            dw_path,
            smplx=smplx
                )
    # except:
    #     print('err')
    #     pass
    break

#/ytech_data_ssd/m2v/livephoto-body/livephoto-body-Kwai_50k/bodydance_caiji02_new/ae/000727/001/dw_pose.npy
#"/ytech_milm/Keling_HumanMotion/data/one-man/livephoto-body/livephoto-body-Kwai_50k/bodydance_caiji02_new/ae/000727/001/dwpose_style2.mp4