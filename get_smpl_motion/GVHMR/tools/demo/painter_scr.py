import cv2
import torch
import sys
import json
import csv
import sys
import numpy as np
import argparse
#from pytorch3d.transforms import quaternion_to_matrix

from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    get_writer,
    get_video_reader,
)

from hmr4d.utils.net_utils import  to_cuda
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.vis.renderer import perspective_projection
from tqdm import tqdm


CRF = 23  # 17 is lossless, every +6 halves the mp4 size

# cfg.paths.incam_video = 'outputs/test.mp4'
# cfg.paths.hmr4d_results = 'inputs/hmr4d_results.pt'
# cfg.video_path = '/ytech_milm/Keling_HumanMotion/data/one-man/livephoto-body/livephoto-body-Kwai_50k/bodydance_caiji02_new/ae/000727/001/dwpose_style2.mp4'
# cfg.paths.bbx = ''/group/stonefan/smpl/4000line.json



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
    try:
        dw_pose = np.load(dw_path)
    except:
        dw_pose = np.load(dw_path.replace('/ytech_data_ssd/m2v/','/m2v/'))
    pred = torch.load(pt_path)
    #test = np.zeros((1536,1024,3))
    n=92
    '''
    for it in tt:
        print(it)
        test = cv2.circle(test,(int(it[0]*1024),int(it[1]*1536)),1,(255, 255, 255),2)
    cv2.imwrite('test.png',test)
    '''


    # smpl
    
    random_beta=torch.randn(pred["smpl_params_incam"]['betas'][0].shape)*2
    for i in range(pred["smpl_params_incam"]['betas'].shape[0]):
        pred["smpl_params_incam"]['betas'][i] += random_beta
    smplx_out = smplx(**to_cuda(pred["smpl_params_incam"]))
    print(smplx_out.joints.shape,)
    pred_c_joints = smplx_out.joints[:,:,:]
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
    reader = get_video_reader(video_path)  # (F, H, W, 3), uint8, numpy
    #bbx_xys_render = torch.load(cfg.paths.bbx)["bbx_xys"]
    joints = perspective_projection(pred_c_joints,K=K.cuda())
    joints = joints.detach().cpu().numpy()

    dwp0se_smplx=[20,37,38,39,66,25,26,27,67,28,29,30,68,34,35,36,69,31,32,33,70,
                21,52,53,54,71,40,41,42,72,43,44,45,73,49,50,51,74,46,47,48,75]#92-133
    for iframe in range(length):
        if (dw_pose[iframe,0,92:113,:]>0.01).all() and (dw_pose[iframe,0,92:113,2]>0.4).all():
            for ijoint in range(93,113):
                joints[iframe,dwp0se_smplx[ijoint-92],0]=(dw_pose[iframe,0,ijoint,0]-dw_pose[iframe,0,92,0])*width+joints[iframe,dwp0se_smplx[20],0]
                joints[iframe,dwp0se_smplx[ijoint-92],1]=(dw_pose[iframe,0,ijoint,1]-dw_pose[iframe,0,92,1])*height+joints[iframe,dwp0se_smplx[20],1]
        if (dw_pose[iframe,0,113:134,:]>0.01).all() and (dw_pose[iframe,0,113:134,2]>0.4).all():
            for ijoint in range(114,134):
                joints[iframe,dwp0se_smplx[ijoint-92],0]=(dw_pose[iframe,0,ijoint,0]-dw_pose[iframe,0,113,0])*width+joints[iframe,dwp0se_smplx[21],0]
                joints[iframe,dwp0se_smplx[ijoint-92],1]=(dw_pose[iframe,0,ijoint,1]-dw_pose[iframe,0,113,1])*height+joints[iframe,dwp0se_smplx[21],1]
    joints =joints.astype(np.int32)
    pred_c_joints = pred_c_joints.detach().cpu().numpy()
    depth = pred_c_joints[...,2]
    depth = (depth-depth.min())/(depth.max()-depth.min())
    print(joints.shape, depth.shape)
    writer = get_writer(incam_video_path, fps=30, crf=CRF)

    lls = [ [ 0, 1 ],
            [ 0, 2 ],
            [ 0, 3 ],
            [ 1, 4 ],
            [ 2, 5 ],
            [ 3, 6 ],
            [ 4, 7 ],
            [ 5, 8 ],
            [ 6, 9 ],
            [ 7, 10],
            [ 8, 11],
            [ 9, 12],
            [ 9, 13],
            [ 9, 14],
            [12, 15],
            [13, 16],
            [14, 17],
            [16, 18],
            [17, 19],
            [18, 20],
            [19, 21],
            [20, 25],   # left hand from my view
            [25, 26],   # left_index
            [26, 27],   
            [27, 67],   # finger tips
            [20, 28],
            [28, 29],   # left_middle
            [29, 30],   
            [30, 68],
            [20, 31],
            [31, 32],   # left_pinky
            [32, 33],
            [33, 70],
            [20, 34],
            [34, 35],   # left_ring
            [35, 36],
            [36, 69],
            [20, 37],
            [37, 38],   # left_thumb
            [38, 39],
            [39, 66],   # finger tips
            [21, 40],   # right hand from my view
            [40, 41],   # right_index
            [41, 42],   
            [42, 72],   # finger tips
            [21, 43],
            [43, 44],   # right_middle
            [44, 45],   
            [45, 73],
            [21, 46],   # right_pinky
            [46, 47],
            [47, 48],
            [48, 75],
            [21, 49],
            [49, 50],   # right_ring
            [50, 51],
            [51, 74],
            [21, 52],
            [52, 53],   # right_thumb
            [53, 54],
            [54, 71],   # finger tips
            [ 7, 60],
            [60, 61],
            [ 7, 62],
            [ 8, 63],
            [63, 64],
            [ 8, 65],
            [15,56],
            [15,57],
            [56,58],
            [57,59]
        ]
    for i, img_raw in tqdm(enumerate(reader), total=get_video_lwh(video_path)[0], desc=f"Rendering Incam"):
        img = img_raw * 0
        for j,de in list(zip(joints[i],depth[i]))[:55]:
            img = cv2.circle(img,(j[0],j[1]),1,(np.ceil(255*(1-de)), 0, np.ceil(255*de)),2)
        
        for ll in lls:
            colorx =(np.ceil(255*(1-depth[i,ll[0]])),0,np.ceil(255*depth[i,ll[0]]))
            colory =(np.ceil(255*(1-depth[i,ll[1]])),0,np.ceil(255*depth[i,ll[1]]))
            img = draw_lines(img,joints[i,ll[0]],joints[i,ll[1]],colorx,colory)

        left_joint=joints[i,[1,4,7,10,13,16,18,20]]
        right_joint=joints[i,[2,5,8,11,14,17,19,21]]
        for js in left_joint:
            img = cv2.circle(img,(js[0],js[1]),4,(127,127,127),4)
        for js in right_joint:
            img = cv2.circle(img,(js[0],js[1]),4,(0,255,0),4)
        writer.write_frame(img)
    writer.close()
    reader.close()

label_name = 'label/label-{}.txt'.format(sys.argv[1])
smplx = make_smplx("supermotion").cuda()
for i in open(label_name).readlines():
    smplx_path =i.strip()
    out_path = smplx_path.replace('hmr4d_results.pt','3d_pose_enh.mp4')
    video_path = smplx_path.replace('/hmr4d_results.pt','.mp4').replace('smpl_test','data')
    dw_path = smplx_path.replace('video/hmr4d_results.pt','dwpose.npy')
    dw_path = dw_path.replace('/ytech_milm/Keling_HumanMotion/smpl_test/one-man/','/ytech_data_ssd/m2v/')
    
    #try:
    render_incam(out_path,
            smplx_path,
            video_path,
            dw_path,
            smplx=smplx
            )
        print(out_path)
    #except:
    #    print('err',smplx_path)
'''
         /ytech_milm/Keling_HumanMotion/smpl_test/one-man/livephoto-body/livephoto-body-Kwai_50k/bodydance_caiji02_new/ac/000641/004/3d_pose.mp4
        "/ytech_milm/Keling_HumanMotion/data     /one-man/livephoto-body/livephoto-body-Kwai_50k/bodydance_caiji02_new/ad/001137/006/video.mp4",
        "/ytech_milm/Keling_HumanMotion/data     /one-man/livephoto-body/livephoto-body-Kwai_50k/bodydance_caiji02_new/ad/001137/006/dwpose_style2.mp4",
        "/ytech_milm/Keling_HumanMotion/processed/one-man/livephoto-body/livephoto-body-Kwai_50k/bodydance_caiji02_new/ad/001137/006/video/hmr4d_results.pt"
         /ytech_milm/Keling_HumanMotion/smpl_test/one-man/livephoto-body/livephoto-body-Kwai_50k/B站舞蹈/ad/000455/011/video/hmr4d_results.pt

info = json.load(open('/group/stonefan/smpl/4000line.json','r'))
print(len(info))

for indd in range(1,4000):
    dw_path = info[indd]['kps_style2_path'].replace('/ytech_milm/Keling_HumanMotion/data/one-man/','/ytech_data_ssd/m2v/')
    dw_path = dw_path.replace('dwpose_style2.mp4','dwpose.npy')
    print(indd)
    # try:
    render_incam('/group/stonefan/smpl/GVHMR/outputs/test{}.mp4'.format(indd),
            info[indd]['smplx_path'],
            info[indd]['kps_style2_path'],
            dw_path,
            smplx=smplx
            )
    # except:
    #     print('err')
    #     pass
dwpose_style2.mp4
cp /ytech_milm/Keling_HumanMotion/smpl_test/one-man/livephoto-body/livephoto-body-Kwai_50k/all/ah/000301/003/video/hmr4d_results.pt test1.pt
cp /ytech_milm/Keling_HumanMotion/smpl_test/one-man/livephoto-body/livephoto-body-Kwai_50k/all/ac/000386/003/video/hmr4d_results.pt test2.pt
cp /ytech_milm/Keling_HumanMotion/smpl_test/one-man/livephoto-body/livephoto-body-Kwai_50k/bodydance_caiji02_new/ac/000278/001/video/hmr4d_results.pt test3.pt
cp /ytech_milm/Keling_HumanMotion/smpl_test/one-man/livephoto-body/livephoto-body-Kwai_50k/B站舞蹈/ad/000447/010/video/hmr4d_results.pt test4.pt
cp /ytech_milm/Keling_HumanMotion/smpl_test/one-man/livephoto-body/livephoto-body-Kwai_50k/bodydance_download_new/ah/001701/001/video/hmr4d_results.pt test5.pt

cp /ytech_milm/Keling_HumanMotion/data/one-man/livephoto-body/livephoto-body-Kwai_50k/all/ah/000301/003/dwpose_style2.mp4 test1.mp4
cp /ytech_milm/Keling_HumanMotion/data/one-man/livephoto-body/livephoto-body-Kwai_50k/all/ac/000386/003/dwpose_style2.mp4 test2.mp4
cp /ytech_milm/Keling_HumanMotion/data/one-man/livephoto-body/livephoto-body-Kwai_50k/bodydance_caiji02_new/ac/000278/001/dwpose_style2.mp4 test3.mp4
cp /ytech_milm/Keling_HumanMotion/data/one-man/livephoto-body/livephoto-body-Kwai_50k/B站舞蹈/ad/000447/010/dwpose_style2.mp4 test4.mp4
cp /ytech_milm/Keling_HumanMotion/data/one-man/livephoto-body/livephoto-body-Kwai_50k/bodydance_download_new/ah/001701/001/dwpose_style2.mp4 test5.mp4


cp /ytech_data_ssd/m2v/livephoto-body/livephoto-body-Kwai_50k/all/ah/000301/003/dwpose.npy test1_dw.npy
cp /ytech_data_ssd/m2v/livephoto-body/livephoto-body-Kwai_50k/all/ac/000386/003/dwpose.npy test2_dw.npy
cp /ytech_data_ssd/m2v/livephoto-body/livephoto-body-Kwai_50k/bodydance_caiji02_new/ac/000278/001/dwpose.npy test3_dw.npy
cp /ytech_data_ssd/m2v/livephoto-body/livephoto-body-Kwai_50k/B站舞蹈/ad/000447/010/dwpose.npy test4_dw.npy
cp /ytech_data_ssd/m2v/livephoto-body/livephoto-body-Kwai_50k/bodydance_download_new/ah/001701/001/dwpose.npy test5_dw.npy


'''
#/ytech_data_ssd/m2v/livephoto-body/livephoto-body-Kwai_50k/bodydance_caiji02_new/ae/000727/001/dw_pose.npy
#"/ytech_milm/Keling_HumanMotion/data/one-man/livephoto-body/livephoto-body-Kwai_50k/bodydance_caiji02_new/ae/000727/001/dwpose_style2.mp4
