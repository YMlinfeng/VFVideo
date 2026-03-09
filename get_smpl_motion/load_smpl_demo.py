import torch

#单图：/ytech_milm/liujiwen/kling_motion_service/smpl_video_image/1c40492f-dd5d-48c8-8a4a-d6323caa84a3/all_result.pt
#视频：/ytech_milm/liujiwen/kling_motion_service/smpl_video_image/e1acaf2e-32b5-49ad-8b6e-320a876b4a47/all_result.pt
all_data = torch.load('/ytech_milm/liujiwen/kling_motion_service/get_smpl_motion/output/40c12197-7955-45a2-a02f-a84e853264d1//all_result.pt')

smpl_list = all_data['smpl'] #smpl
feature_list = all_data['data_list'] #2Dpose 等中间结果
'''
feature_list中的数据结构：
data = {
    "length": torch.tensor(length), #视频长度，帧数，单图就是1
    "bbx_xys": bbx_xys,
    "kp2d": vitpose,
    "K_fullimg": K_fullimg,
    "cam_angvel": compute_cam_angvel(R_w2c),
    "f_imgseq": vit_features,
    'width': width, #提取smpl和2Dpose过程中用的图片或视频size
    'height': height, #提取smpl和2Dpose过程中用的图片或视频size
}
'''
num_smpl = len(smpl_list)
image_ori = all_data['image_ori'] ##只有单图数据里面才有，是原始图片，在渲染motion时候应该能用到


for i in range(num_smpl):
    print (i)
    # for key in feature_list[i]:
    #     print (key)
    for key in smpl_list[i]:
        print (key)
        print (smpl_list[i]['smpl_params_global'])
        break

