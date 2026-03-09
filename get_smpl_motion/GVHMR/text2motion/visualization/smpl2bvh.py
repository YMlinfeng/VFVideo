import torch
import numpy as np
import argparse
import pickle
import smplx
import os 
from .utils import bvh, quat
from scipy.spatial.transform import Rotation as R

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/ytech_milm/lixiaohan/code/smpl/GVHMR/inputs/checkpoints/body_models/")
    parser.add_argument("--model_type", type=str, default="smpl", choices=["smpl", "smplx"])
    parser.add_argument("--gender", type=str, default="MALE", choices=["MALE", "FEMALE", "NEUTRAL"])
    parser.add_argument("--num_betas", type=int, default=10, choices=[10, 300])
    parser.add_argument("--poses", type=str, default="/ytech_milm/lixiaohan/code/t2m/momask-codes/result_t2m.pt")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--output", type=str, default="./gWA_sFM_cAll_d27_mWA5_ch20.bvh")
    parser.add_argument("--mirror", action="store_true")
    return parser.parse_args()

def mirror_rot_trans(lrot, trans, names, parents):
    joints_mirror = np.array([(
        names.index("Left"+n[5:]) if n.startswith("Right") else (
        names.index("Right"+n[4:]) if n.startswith("Left") else 
        names.index(n))) for n in names])

    mirror_pos = np.array([-1, 1, 1])
    mirror_rot = np.array([1, 1, -1, -1])
    grot = quat.fk_rot(lrot, parents)
    trans_mirror = mirror_pos * trans
    grot_mirror = mirror_rot * grot[:,joints_mirror]
    
    return quat.ik_rot(grot_mirror, parents), trans_mirror

def smpl2bvh(model_path:str, poses:str, output:str, mirror:bool,
             model_type="smpl", gender="MALE",
             num_betas=10, fps=60, scaling=None) -> None:
    """Save bvh file created by smpl parameters.

    Args:
        model_path (str): Path to smpl models.
        poses (str): Path to npz or pkl file.
        output (str): Where to save bvh.
        mirror (bool): Whether save mirror motion or not.
        model_type (str, optional): I prepared "smpl" only. Defaults to "smpl".
        gender (str, optional): Gender Information. Defaults to "MALE".
        num_betas (int, optional): How many pca parameters to use in SMPL. Defaults to 10.
        fps (int, optional): Frame per second. Defaults to 30.
    """
    
    # names = [
    #     "Pelvis",
    #     "Left_hip",
    #     "Right_hip",
    #     "Spine1",
    #     "Left_knee",
    #     "Right_knee",
    #     "Spine2",
    #     "Left_ankle",
    #     "Right_ankle",
    #     "Spine3",
    #     "Left_foot",
    #     "Right_foot",
    #     "Neck",
    #     "Left_collar",
    #     "Right_collar",
    #     "Head",
    #     "Left_shoulder",
    #     "Right_shoulder",
    #     "Left_elbow",
    #     "Right_elbow",
    #     "Left_wrist",
    #     "Right_wrist",
    #     "Left_palm",
    #     "Right_palm",
    # ]

    names = [
        "Hips",
        "LeftUpLeg",
        "RightUpLeg",
        "Spine",
        "LeftLeg",
        "RightLeg",
        "Spine1",
        "LeftFoot",
        "RightFoot",
        "Spine2",
        "LeftToe",
        "RightToe",
        "Neck",
        "LeftShoulder",
        "RightShoulder",
        "Head",
        "LeftArm",
        "RightArm",
        "LeftForeArm",
        "RightForeArm",
        "LeftHand",
        "RightHand",
        "LeftThumb",
        "RightThumb",
    ]

    # I prepared smpl models only, 
    # but I will release for smplx models recently.
    model = smplx.create(model_path=model_path, 
                        model_type=model_type,
                        gender=gender, 
                        batch_size=1)
    
    parents = model.parents.detach().cpu().numpy()
    
    # You can define betas like this.(default betas are 0 at all.)
    rest = model(
        # betas = torch.randn([1, num_betas], dtype=torch.float32)
    )
    rest_pose = rest.joints.detach().cpu().numpy().squeeze()[:24,:]
    
    root_offset = rest_pose[0]
    offsets = rest_pose - rest_pose[parents]
    offsets[0] = root_offset
    offsets *= 100
    print("offsets init : ", offsets[0])
    
    scaling = scaling
    
    # Pose setting.
    if poses.endswith(".npz"):
        poses = np.load(poses)
        rots = np.squeeze(poses["poses"], axis=0) # (N, 24, 3)
        trans = np.squeeze(poses["trans"], axis=0) # (N, 3)

    elif poses.endswith(".pkl"):
        with open(poses, "rb") as f:
            poses = pickle.load(f)
            rots = poses["smpl_poses"] # (N, 72)
            rots = rots.reshape(rots.shape[0], -1, 3) # (N, 24, 3)
            scaling = poses["smpl_scaling"]  # (1,)
            trans = poses["smpl_trans"]  # (N, 3)
    elif poses.endswith(".pt"):
        with open(poses, 'rb') as f:
            poses = torch.load(f)
            global_rot = poses["smpl_params_global"]["global_orient"].cpu().numpy()
            trans = poses["smpl_params_global"]["transl"].cpu().numpy()
            rots = poses["smpl_params_global"]["body_pose"].cpu().numpy() # (N, )
            N = rots.shape[0]
            rots = rots.reshape(N, -1, 3)
            zeros = np.zeros((N, 2, 3))
            global_rot = np.expand_dims(global_rot, axis=1) 
            print("rots : ", rots.shape, zeros.shape, global_rot.shape)
            rots = np.concatenate((global_rot, rots, zeros), axis=1)
            print("rots : ", rots.shape)
    else:
        raise Exception("This file type is not supported!")
    
    if scaling is not None:
        trans /= scaling
    
    # to quaternion
    rots = quat.from_axis_angle(rots)
    
    order = "zyx"
    pos = offsets[None].repeat(len(rots), axis=0)
    positions = pos.copy()
    # positions[:,0] += trans * 10
    positions[:, 0] += trans
    print("positions : ", positions[0, 0])
    rotations = np.degrees(quat.to_euler(rots, order=order))
    
    bvh_data ={
        "rotations": rotations[:, :22],
        "positions": positions[:, :22],
        "offsets": offsets[:22],
        "parents": parents[:22],
        "names": names[:22],
        "order": order,
        "frametime": 1 / fps,
    }
    
    if not output.endswith(".bvh"):
        output = output + "result.bvh"
    
    bvh.save(output, bvh_data)
    
    if mirror:
        rots_mirror, trans_mirror = mirror_rot_trans(
                rots, trans, names, parents)
        positions_mirror = pos.copy()
        positions_mirror[:,0] += trans_mirror
        rotations_mirror = np.degrees(
            quat.to_euler(rots_mirror, order=order))
        
        bvh_data ={
            "rotations": rotations_mirror,
            "positions": positions_mirror,
            "offsets": offsets,
            "parents": parents,
            "names": names,
            "order": order,
            "frametime": 1 / fps,
        }
        
        output_mirror = output.split(".")[0] + "_mirror.bvh"
        bvh.save(output_mirror, bvh_data)

def smplx2bvh(model_path, poses, output, mirror=False,
             model_type="smplx", gender="NEUTRAL",
             num_betas=300, fps=24, scaling=0.01, use_trans=False, max_f=None) -> None:
    """Save bvh file created by smpl parameters.

    Args:
        model_path (str): Path to smpl models.
        poses (str): Path to npz or pkl file.
        output (str): Where to save bvh.
        mirror (bool): Whether save mirror motion or not.
        model_type (str, optional): I prepared "smpl" only. Defaults to "smpl".
        gender (str, optional): Gender Information. Defaults to "MALE".
        num_betas (int, optional): How many pca parameters to use in SMPL. Defaults to 10.
        fps (int, optional): Frame per second. Defaults to 30.
    """
    pre_model_path = model_path
    model_path = os.path.join(model_path, 'body_models/')
    names = ["pelvis",
            "left_hip",
            "right_hip",
            "spine1",
            "left_knee",
            "right_knee",
            "spine2",
            "left_ankle",
            "right_ankle",
            "spine3",
            "left_foot",
            "right_foot",
            "neck",
            "left_collar",
            "right_collar",
            "head",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "jaw",
            "left_eye_smplhf",
            "right_eye_smplhf",
            "left_index1",
            "left_index2",
            "left_index3",
            "left_middle1",
            "left_middle2",
            "left_middle3",
            "left_pinky1",
            "left_pinky2",
            "left_pinky3",
            "left_ring1",
            "left_ring2",
            "left_ring3",
            "left_thumb1",
            "left_thumb2",
            "left_thumb3",
            "right_index1",
            "right_index2",
            "right_index3",
            "right_middle1",
            "right_middle2",
            "right_middle3",
            "right_pinky1",
            "right_pinky2",
            "right_pinky3",
            "right_ring1",
            "right_ring2",
            "right_ring3",
            "right_thumb1",
            "right_thumb2",
            "right_thumb3",]

    # I prepared smpl models only, 
    # but I will release for smplx models recently.
    # print("model type : ", model_type , model_path)
    model = smplx.create(model_path=model_path, 
                        model_type=model_type,
                        gender=gender, 
                        batch_size=1)
    
    parents = model.parents.detach().cpu().numpy()
    
    # You can define betas like this.(default betas are 0 at all.)
    rest = model(
        # betas = torch.randn([1, num_betas], dtype=torch.float32)
    )
    rest_pose = rest.joints.detach().cpu().numpy().squeeze()[:55,:]
    
    root_offset = rest_pose[0]
    offsets = rest_pose - rest_pose[parents]
    offsets[0] = np.zeros(3) # root_offset # np.zeros(3) ## root_offset
    offsets *= 100
    # offsets = offsets[:]-offsets[0]
    # print("offsets init : ", offsets[0])
    
    scaling = scaling

    # 解析pose
    global_rot = poses["smpl_params_global"]["global_orient"].cpu().numpy()
    trans = poses["smpl_params_global"]["transl"].cpu().numpy()
    rots = poses["smpl_params_global"]["body_pose"].cpu().numpy() # (N, )
    rots = rots.reshape(rots.shape[0], -1, 3)
    global_rot = np.expand_dims(global_rot, axis=1)
    rots = np.concatenate((global_rot, rots), axis=1)
    n_zero = 55-rots.shape[1]
    if n_zero > 0:
        np_zero = np.zeros([rots.shape[0], n_zero, 3])
        rots = np.concatenate((rots, np_zero), axis=1)
    
    if "left_hand_pose" in poses["smpl_params_global"]:
        rots[:, 25:40, :] = poses["smpl_params_global"]["left_hand_pose"].cpu().numpy().reshape(-1, 15, 3)
    if "right_hand_pose" in poses["smpl_params_global"]:
        rots[:, 40:55, :] = poses["smpl_params_global"]["right_hand_pose"].cpu().numpy().reshape(-1, 15, 3)
    if "left_hand_pose" not in poses["smpl_params_global"] and "right_hand_pose" not in poses["smpl_params_global"]:
        norm_hand_pose = np.load(os.path.join(pre_model_path,'..', 'wilor_checkpoints/norm_hand.npy'))
        left_vector = R.from_matrix(norm_hand_pose[0]).as_rotvec()
        right_vector = R.from_matrix(norm_hand_pose[1]).as_rotvec()
        print(("left_vector : ", left_vector.shape, right_vector.shape))
        rots[:, 25:40, :] = left_vector
        rots[:, 40:55, :] = right_vector
    
    '''
    # Pose setting.
    if poses.endswith(".npz"):
        poses = np.load(poses)
        rots = np.squeeze(poses["poses"], axis=0) # (N, 24, 3)
        trans = np.squeeze(poses["trans"], axis=0) # (N, 3)

    elif poses.endswith(".pkl"):
        with open(poses, "rb") as f:
            poses = pickle.load(f)
            rots = poses["smpl_poses"] # (N, 72)
            rots = rots.reshape(rots.shape[0], -1, 3) # (N, 24, 3)
            scaling = poses["smpl_scaling"]  # (1,)
            trans = poses["smpl_trans"]  # (N, 3)
    elif poses.endswith(".pt"):
        with open(poses, 'rb') as f:
            poses = torch.load(f)
            global_rot = poses["smpl_params_global"]["global_orient"].cpu().numpy()
            trans = poses["smpl_params_global"]["transl"].cpu().numpy()
            rots = poses["smpl_params_global"]["body_pose"].cpu().numpy() # (N, )
            rots = rots.reshape(rots.shape[0], -1, 3)
            global_rot = np.expand_dims(global_rot, axis=1)
            rots = np.concatenate((global_rot, rots), axis=1)
            n_zero = 55-rots.shape[1]
            if n_zero > 0:
                np_zero = np.zeros([rots.shape[0], n_zero, 3])
                rots = np.concatenate((rots, np_zero), axis=1)
            # print("rots : ", rots.shape, trans.shape, global_rot.shape)
            # print("rots : ", rots.shape)
    else:
        raise Exception("This file type is not supported!")
    '''

    if scaling is not None:
        trans /= scaling
    
    # to quaternion
    rots = quat.from_axis_angle(rots)
    
    order = "zyx"
    pos = offsets[None].repeat(len(rots), axis=0)
    positions = pos.copy()
    # positions[:,0] += trans * 10
    # print("trans size : ", trans.shape, positions.shape, positions[:, 0].shape)
    if not use_trans:
        vector_diff = np.array([0.27621302, 0, -500]) - trans[0]
    if use_trans:
        vector_diff = np.zeros([3])
    trans = trans+vector_diff
    positions[:, 0] += trans
    # print("positions init : ", positions[0, 0])
    rotations = np.degrees(quat.to_euler(rots, order=order))

    if max_f is not None:
        rotations = rotations[:max_f]
        positions = positions[:max_f]
    
    bvh_data ={
        "rotations": rotations[:, :55],
        "positions": positions[:, :55],
        "offsets": offsets[:55],
        "parents": parents[:55],
        "names": names[:55],
        "order": order,
        "frametime": 1 / fps,
    }
    
    if not output.endswith(".bvh"):
        output = output + ".bvh"
    
    bvh.save(output, bvh_data)
    with open(output, "rb") as f:
        bvh_binary = f.read()
    return bvh_binary
    
    if mirror:
        rots_mirror, trans_mirror = mirror_rot_trans(
                rots, trans, names, parents)
        positions_mirror = pos.copy()
        positions_mirror[:,0] += trans_mirror
        rotations_mirror = np.degrees(
            quat.to_euler(rots_mirror, order=order))
        
        bvh_data ={
            "rotations": rotations_mirror,
            "positions": positions_mirror,
            "offsets": offsets,
            "parents": parents,
            "names": names,
            "order": order,
            "frametime": 1 / fps,
        }
        
        output_mirror = output.split(".")[0] + "_mirror.bvh"
        bvh.save(output_mirror, bvh_data)

# class smplx2bvh:
#     def __init__(self, model_path):

if __name__ == "__main__":
    args = parse_args()
    # smpl2bvh(model_path=args.model_path, model_type=args.model_type, 
    #          mirror = args.mirror, gender=args.gender,
    #          poses=args.poses, num_betas=args.num_betas, 
    #          fps=24, scaling=0.01, output=args.output)
    poses = torch.load("/ytech_milm/lixiaohan/code/t2m/momask-codes/result_t2m.pt")
    smplx2bvh(model_path="/ytech_milm/liujiwen/kling_motion_service/smpl_all_checkpoints/checkpoints", model_type='smplx', 
            mirror = args.mirror, gender="NEUTRAL",
            poses=poses, num_betas=300, 
            fps=24, scaling=0.01, output=args.output)
    print("finished!")
