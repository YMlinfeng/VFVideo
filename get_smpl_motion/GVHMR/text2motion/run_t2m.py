import os
import torch.nn as nn
from os.path import join as pjoin

import torch
import torch.nn.functional as F

from .models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from .models.vq.model import RVQVAE, LengthEstimator

from .options.eval_option import EvalT2MOptions
from .utils.get_opt import get_opt

from .utils.fixseed import fixseed
from .visualization.joints2bvh import Joint2BVHConvertor
from torch.distributions.categorical import Categorical

from .utils.motion_process import recover_from_ric
from .utils.plot_script import plot_3d_motion

from .utils.paramUtil import t2m_kinematic_chain
from .visualization.smplify import joints2smpl
import time
# import logging
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from .utils.filter_tools import OneEuroFilterQuaternion
from scipy.spatial.transform import Rotation as R
from ..hmr4d.model.gvhmr.utils.postprocess import process_ik, pp_static_joint
from ..hmr4d.utils.smplx_utils import make_smplx
from pytorch3d.transforms import (
    rotation_6d_to_matrix,
    matrix_to_axis_angle,
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from ..hmr4d.utils import matrix as matrix
# from hydra.utils import instantiate

import numpy as np
# clip_version = 'ViT-B/32'

def load_vq_model(vq_opt):
    # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_model = RVQVAE(vq_opt,
                vq_opt.dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_opt.name} Completed!')
    return vq_model, vq_opt

def load_trans_model(model_opt, opt, which_model, clip_version):
    t2m_transformer = MaskTransformer(code_dim=model_opt.code_dim,
                                      cond_mode='text',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      opt=model_opt)
    ckpt = torch.load(pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model),
                      map_location='cpu')
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    # print(ckpt.keys())
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer

def load_res_model(res_opt, vq_opt, opt, clip_version):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
                                            cond_mode='text',
                                            latent_dim=res_opt.latent_dim,
                                            ff_size=res_opt.ff_size,
                                            num_layers=res_opt.n_layers,
                                            num_heads=res_opt.n_heads,
                                            dropout=res_opt.dropout,
                                            clip_dim=512,
                                            shared_codebook=vq_opt.shared_codebook,
                                            cond_drop_prob=res_opt.cond_drop_prob,
                                            # codebook=vq_model.quantizer.codebooks[0] if opt.fix_token_emb else None,
                                            share_weight=res_opt.share_weight,
                                            clip_version=clip_version,
                                            opt=res_opt)

    ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', 'net_best_fid.tar'),
                      map_location=opt.device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer

def load_len_estimator(opt):
    model = LengthEstimator(512, 50)
    ckpt = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'length_estimator', 'model', 'finest.tar'),
                      map_location=opt.device)
    model.load_state_dict(ckpt['estimator'])
    print(f'Loading Length Estimator from epoch {ckpt["epoch"]}!')
    return model

class EnDecoder(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.smplx_model = make_smplx("supermotion_v437coco17", smpl_model_path=model_path)
        parents = self.smplx_model.parents[:22]
        self.register_buffer("parents_tensor", parents, False)
        self.parents = parents.tolist()

    def fk_v2(self, body_pose, betas, global_orient=None, transl=None, get_intermediate=False):
        """
        Args:
            body_pose: (B, L, 63)
            betas: (B, L, 10)
            global_orient: (B, L, 3)
        Returns:
            joints: (B, L, 22, 3)
        """
        print("body_pose:", body_pose.shape)
        print("betas:", betas.shape)
        print("global_orient:", global_orient.shape)
        B, L = body_pose.shape[:2]
        if global_orient is None:
            global_orient = torch.zeros((B, L, 3), device=body_pose.device)
        aa = torch.cat([global_orient, body_pose], dim=-1).reshape(B, L, -1, 3)
        rotmat = axis_angle_to_matrix(aa)  # (B, L, 22, 3, 3)

        skeleton = self.smplx_model.get_skeleton(betas)[..., :22, :]  # (B, L, 22, 3)
        local_skeleton = skeleton - skeleton[:, :, self.parents_tensor]
        local_skeleton = torch.cat([skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2)

        if transl is not None:
            local_skeleton[..., 0, :] += transl  # B, L, 22, 3

        mat = matrix.get_TRS(rotmat, local_skeleton)  # B, L, 22, 4, 4
        fk_mat = matrix.forward_kinematics(mat, self.parents)  # B, L, 22, 4, 4
        joints = matrix.get_position(fk_mat)  # B, L, 22, 3
        if not get_intermediate:
            return joints
        else:
            return joints, mat, fk_mat
    
class text2motion:
    def __init__(self, checkpoints='', device='cuda:0'):
        # parse args
        parser = EvalT2MOptions()
        opt = parser.parse()
        print (opt)
        if len(checkpoints) > 0:
            opt.checkpoints_dir = checkpoints
        fixseed(opt.seed)
        clip_version = pjoin(opt.checkpoints_dir, 'ViT-B-32.pt')

        # device
        opt.device = torch.device(device)
        torch.autograd.set_detect_anomaly(True)
        dim_pose = 251 if opt.dataset_name == 'kit' else 263
        opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22

        root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
        model_dir = pjoin(root_dir, 'model')
        result_dir = pjoin('./generation', opt.ext)
        joints_dir = pjoin(result_dir, 'joints')
        animation_dir = pjoin(result_dir, 'animations')
        os.makedirs(joints_dir, exist_ok=True)
        os.makedirs(animation_dir,exist_ok=True)
        model_opt_path = pjoin(root_dir, 'opt.txt')
        model_opt = get_opt(model_opt_path, device=opt.device)

        #######################
        ######Loading RVQ######
        #######################
        vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
        vq_opt = get_opt(vq_opt_path, device=opt.device)
        vq_opt.dim_pose = dim_pose
        vq_opt.checkpoints_dir = opt.checkpoints_dir
        self.vq_model, self.vq_opt = load_vq_model(vq_opt)

        model_opt.num_tokens = vq_opt.nb_code
        model_opt.num_quantizers = vq_opt.num_quantizers
        model_opt.code_dim = vq_opt.code_dim

        #################################
        ######Loading R-Transformer######
        #################################
        res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
        res_opt = get_opt(res_opt_path, device=opt.device)
        res_opt.checkpoints_dir = opt.checkpoints_dir
        self.res_model = load_res_model(res_opt, vq_opt, opt, clip_version=clip_version)

        assert res_opt.vq_name == model_opt.vq_name

        #################################
        ######Loading M-Transformer######
        #################################
        model_opt.checkpoints_dir = opt.checkpoints_dir
        self.t2m_transformer = load_trans_model(model_opt, opt, 'latest.tar', clip_version=clip_version)

        ##################################
        #####Loading Length Predictor#####
        ##################################
        self.length_estimator = load_len_estimator(model_opt)

        self.t2m_transformer.eval()
        self.vq_model.eval()
        self.res_model.eval()
        self.length_estimator.eval()

        self.res_model.to(opt.device)
        self.t2m_transformer.to(opt.device)
        self.vq_model.to(opt.device)
        self.length_estimator.to(opt.device)

        self.mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
        self.std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))
        self.opt = opt
        # self.endecoder = EnDecoder(model_path=os.path.join(checkpoints, '../checkpoints'))

    def inv_transform(self, data):
        return data * self.std + self.mean
        
    def is_valid_number(self, value):
        if isinstance(value, (int, float)) and 2 <= value <= 10:
            return True
        else:
            return False
    
    def slerp(self, result, old_fps=20, target_fps=25):
        if target_fps == old_fps:
            return result
        
        N = result["smpl_params_global"]["transl"].shape[0]
        t_old = np.linspace(0, (N-1)/old_fps, N)
        t_new = np.linspace(0, (N-1)/old_fps, int(N * target_fps/old_fps))

        transl_old = result["smpl_params_global"]["transl"].cpu().numpy()  # [N, 3]
        interp_transl = interp1d(t_old, transl_old, axis=0, kind='linear')
        transl_new = interp_transl(t_new)  # [M, 3]
        
        # 全局旋转插值
        global_orient_old = result["smpl_params_global"]["global_orient"].cpu().numpy()  # [N, 3]
        rots_global = Rotation.from_rotvec(global_orient_old)
        slerp_global = Slerp(t_old, rots_global)
        global_orient_new = slerp_global(t_new).as_rotvec()  # [M, 3]

        # 身体姿势插值（逐关节处理）
        body_pose_old = result["smpl_params_global"]["body_pose"].cpu().numpy()  # [N, 63]
        body_pose_new = np.zeros((len(t_new), 63))

        for j in range(21):  # 21个关节
            joint_rots = Rotation.from_rotvec(body_pose_old[:, j*3 : (j+1)*3])
            slerp_joint = Slerp(t_old, joint_rots)
            body_pose_new[:, j*3 : (j+1)*3] = slerp_joint(t_new).as_rotvec()
        
        betas_old = result["smpl_params_global"]["betas"].cpu().numpy()  # [N, 10]
        interp_betas = interp1d(t_old, betas_old, axis=0, kind='linear')
        betas_new = interp_betas(t_new)  # [M, 10]

        result_25fps = {
            "smpl_params_global": {
                "transl": torch.tensor(transl_new, dtype=torch.float32),
                "global_orient": torch.tensor(global_orient_new, dtype=torch.float32),
                "betas": torch.tensor(betas_new, dtype=torch.float32),
                "body_pose": torch.tensor(body_pose_new, dtype=torch.float32)
            }
        }
        return result_25fps

    def rot_filter(self, body_orient, one_euro_filter):
        def quaternion_to_rotation_vector_scipy(quat_vec):
            rot_vec_result = np.zeros((quat_vec.shape[0], quat_vec.shape[1], 3))
            for i in range(quat_vec.shape[0]):
                rot_vec_result[i] = R.from_quat(quat_vec[i]).as_rotvec()
            return rot_vec_result  # Convert Rotation object to rotation vector
        
        def rotation_vector_to_quaternion_scipy(rot_vec):
            rot_quat_result = np.zeros((rot_vec.shape[0], rot_vec.shape[1], 4))
            for i in range(rot_vec.shape[0]):
                # r = R.from_rotvec(rot)  # Convert rotation vector to Rotation object
                rot_quat_result[i] = R.from_rotvec(rot_vec[i]).as_quat()
            return rot_quat_result  # Convert Rotation object to quaternion (q_x, q_y, q_z, q_w)
        
        # body orient: tensor -> numpy
        body_orient_np = body_orient.cpu().numpy().reshape(body_orient.shape[0], -1, 3)

        # 将输入的四元数转换为旋转向量
        rot_quat_vec = rotation_vector_to_quaternion_scipy(body_orient_np)
        # print("rot shape : ", body_orient.shape, body_orient_np.shape, rot_quat_vec.shape)

        # 四元数滤波
        # print("one_euro_filter : ", len(one_euro_filter))
        rot_vec = np.zeros_like(body_orient_np)
        for i in range(rot_vec.shape[0]):
            for j in range(rot_vec.shape[1]):
                if j < 10 :
                    continue
                rot_quat_vec[i, j],_ = one_euro_filter[j].filter(rot_quat_vec[i, j], t=float(i)/25., reset=False, prev=None)
        
        rot_vec = quaternion_to_rotation_vector_scipy(rot_quat_vec) # Convert quaternion to rotation vector

        # numpy -> tensor
        rot_vec = torch.tensor(rot_vec, dtype=torch.float32).flatten(1)
        # print("rot_vec : ", rot_vec.shape)
        return rot_vec

    def infer(self, text, duration=None, target_fps=25, output_dir=''): # duration输入是毫秒单位
        # if text is None or len(text) == 0:
        #     logging.error("ERROR INPUT: text is null. \n")
        # else:
        #     logging.info("VALID INPUT: input text is ", text)
        
        # predict duration 
        time_0 = time.time()
        prompt_list=['a person is jump.', text, 'a person is walking in a circle'] # 这里 预测时长总是有问题。
        if duration is None or not is_valid_number(duration): 
            # logging.info("Since no motion length are specified, we will use estimated motion lengthes!!")
            text_embedding = self.t2m_transformer.encode_text(prompt_list)
            pred_dis = self.length_estimator(text_embedding)
            probs = F.softmax(pred_dis, dim=-1)  # (b, ntoken)
            token_lens = Categorical(probs).sample()  # (b, seqlen)
            # print("shapoe : ", token_lens.shape)
            token_lens = token_lens[1:2]
            prompt_list = [text]
        else:
            length_list=[duration]
            token_lens = torch.LongTensor(length_list) // 4
            token_lens = token_lens.to(self.opt.device).long()
        time_1 = time.time()
        # print("length predictor time: ", token_lens)
        
        # predict motion seq
        captions = prompt_list
        with torch.no_grad():
            # generate tranformer
            mids = self.t2m_transformer.generate(captions, token_lens,
                                            timesteps=self.opt.time_steps,
                                            cond_scale=self.opt.cond_scale,
                                            temperature=self.opt.temperature,
                                            topk_filter_thres=self.opt.topkr,
                                            gsample=self.opt.gumbel_sample)
            # print(mids)
            # print(mids.shape)
            # residual transformer
            mids = self.res_model.generate(mids, captions, token_lens, temperature=1, cond_scale=5)
            pred_motions = self.vq_model.forward_decoder(mids)

            pred_motions = pred_motions.detach().cpu().numpy()

            data = self.inv_transform(pred_motions)
        time_2 = time.time()
        
        # save results
        m_length = token_lens * 4
        for k, (caption, joint_data)  in enumerate(zip(captions, data)):
            joint_data = joint_data[:m_length[k]]
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()

            # joints to smpl
            # joints sample : 20->10
            sampled_joint = joint # [::2]

            # print("joint shape : ", self.opt.device)
            # split_joints = np.array_split(sampled_joint, len(self.j2s))
            # opt_dict = {}
            # for index, part in enumerate(split_joints):
            #     joint_last = None  # 初始化 joint_last 为 None
            #     last_init_param = None
            #     for frame_index in range(len(part)):
            #         time_s = time.time()
            #         smpl_param, last_init_param = self.j2s[index].joint2smpl(part[frame_index:frame_index+1], last_init_param)
            #         print("json2smpl time: ", time.time() - time_s)
            #         # save result
            #         if not opt_dict:
            #             opt_dict = last_init_param
            #         else:
            #             opt_dict = {k: torch.cat((opt_dict[k], v), dim=0) for k, v in last_init_param.items()}
                    
                    # print(result)  # 打印或做其他处理
            # j2s = joints2smpl(1, device=self.opt.device, SMPL_MODEL_DIR=self.opt.checkpoints_dir, fix_foot=True)
            time0 = time.time()
            j2s = joints2smpl(sampled_joint.shape[0], device=self.opt.device, SMPL_MODEL_DIR=self.opt.checkpoints_dir, fix_foot=True)
            smpl_param, opt_dict = j2s.joint2smpl(sampled_joint)
            # print("-------------------- time : ", time.time()-time0)
            opt_dict['betas'][:] = opt_dict['betas'][0] # fix shape
            result={
                "smpl_params_global": {
                    "transl":opt_dict['cam'][:,0,:],
                    "global_orient":opt_dict['pose'][:, :3],
                    "betas":opt_dict['betas'],
                    "body_pose":opt_dict['pose'][:, 3:66]
                }
            }
            
            new_result = self.slerp(result, old_fps=20, target_fps=target_fps)
            # print("shapoe : ", new_result["smpl_params_global"]["body_pose"].shape)
            filter = []
            for i in range(21):
                min_cutoff = 0.2
                beta = 0.1
                if i < 10:
                    min_cutoff = 0.5
                    beta = 0.3
                if i == 12 or i == 15:
                    min_cutoff = 0.05
                    beta = 0.02
                filter.append(OneEuroFilterQuaternion(min_cutoff=min_cutoff, beta=beta))
            new_result["smpl_params_global"]["body_pose"] = self.rot_filter(new_result["smpl_params_global"]["body_pose"], filter)
            filter2 = [OneEuroFilterQuaternion(min_cutoff=0.5, beta=0.2) for i in range(1)]
            new_result["smpl_params_global"]["global_orient"] = self.rot_filter(new_result["smpl_params_global"]["global_orient"], filter2)

            # # add ik module
            # new_result["pred_smpl_params_global"] = new_result["smpl_params_global"].copy() # ["transl"]
            # for key in new_result["pred_smpl_params_global"].keys():
            #     new_result["pred_smpl_params_global"][key] = new_result["pred_smpl_params_global"][key].unsqueeze(1)
                
            # new_result["pred_smpl_params_global"]["transl"] = pp_static_joint(new_result, self.endecoder)
            # body_pose = process_ik(new_result, self.endecoder)
            # new_result["smpl_params_global"]["body_pose"] = body_pose

            # add hand pose
            norm_hand_pose = np.load(os.path.join(self.opt.checkpoints_dir,'..', 'wilor_checkpoints/norm_hand.npy'))
            left_vector = torch.tensor(R.from_matrix(norm_hand_pose[0]).as_rotvec(), dtype=torch.float32).flatten()
            right_vector = torch.tensor(R.from_matrix(norm_hand_pose[1]).as_rotvec(), dtype=torch.float32).flatten()
            NUM = new_result["smpl_params_global"]["body_pose"].shape[0]
            new_result["smpl_params_global"]["left_hand_pose"] = left_vector.repeat(NUM, 1)
            new_result["smpl_params_global"]["right_hand_pose"] = right_vector.repeat(NUM, 1)

            # render
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                pt_path = os.path.join(output_dir, "result.pt")
                torch.save(new_result, pt_path)

                # from .utils.paramUtil import t2m_kinematic_chain
                # joints_path = os.path.join(output_dir, "joints.mp4")
                # kinematic_chain = t2m_kinematic_chain
                # converter = Joint2BVHConvertor()
                # _, joint = converter.convert(joint, filename='output.bvh', iterations=100, foot_ik=False)
                # plot_3d_motion(joints_path, kinematic_chain, joint, title=text, fps=20)

                # render 
                # import sys
                # os.chdir("/ytech_milm/lixiaohan/code/smpl/GVHMR")
                # sys.path.insert(0, "/ytech_milm/lixiaohan/code/smpl/GVHMR")
                # from tools.demo.show_global import render_global
                # video_path = os.path.join(output_dir, "output.mp4")
                # render_global('/ytech_milm/Keling_HumanMotion/open_source/Motion-X-cut/Motion-X++-20250226T133749Z-001/Motion-X++/video/animation/animation/Ways_to_Jump_+_Sit_+_Fall_Standing_Ovation_clip1.mp4', 
                #             pt_path=pt_path, output_video_path=video_path)
                time_4 = time.time()
            # # 'pose': new_opt_joints[0, :24].flatten().clone().detach(), 'betas': new_opt_betas.clone().detach(), 'cam'
            # for key, value in opt_dict.items():
            #     print("key : ", key, value.shape)
        time_3 = time.time()
        
        # print : time log
        print("predict motion length : ", time_1-time_0)
        print("predict motion : ", time_2-time_1)
        print("convert result to smpl param : ", time_3-time_2)
        # print("render time  : ", time_4-time_3)
        # return smpl_param
        return new_result

if __name__ == '__main__':
    model = text2motion("/ytech_milm/liujiwen/kling_motion_service/smpl_all_checkpoints/t2m")
    
    # # txt读取
    # file_path = "/ytech_milm/lixiaohan/code/kelingHumanMotion/get_smpl_motion/GVHMR/text2motion/test2.txt"
    # with open(file_path, "r") as file:
    #     lines = file.readlines()

    # output = '/ytech_milm/lixiaohan/code/kelingHumanMotion/get_smpl_motion/GVHMR/text2motion/output/test_chanpin_0407_filter2'
    # processed_names = []
    # for line in lines:
    #     line = line.strip()
    #     processed_name = line.replace(" ", "_")
    #     output_path = os.path.join(output, processed_name)
    #     os.makedirs(output_path, exist_ok=True)
    #     model.infer(text=line, duration=None, target_fps=20, output_dir=output_path)
    
    # result = model.infer(text="A person opens his arms and then spins around.", duration=None, target_fps=20, output_dir='/ytech_milm/lixiaohan/code/kelingHumanMotion/get_smpl_motion/GVHMR/text2motion/output/test')
    # result = model.infer(text="A person does Tai Chi and then stands at attention.", duration=None, target_fps=20, output_dir='/ytech_milm/lixiaohan/code/kelingHumanMotion/get_smpl_motion/GVHMR/text2motion/output/test')
    # result = model.infer(text="The man walked forward, spun right on one foot and walked back to his original position.", duration=None, target_fps=20, output_dir='/ytech_milm/lixiaohan/code/kelingHumanMotion/get_smpl_motion/GVHMR/text2motion/output/test')
    result = model.infer(text="A person jump.", duration=None, target_fps=20, output_dir='/ytech_milm/lixiaohan/code/kelingHumanMotion/get_smpl_motion/GVHMR/text2motion/output/test')

    