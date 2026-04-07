# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
from . import util
from .wholebody import Wholebody
import cv2
import torch
from .tracker import Tracker
# def draw_pose(pose, H, W, bboxs=None):
#     bodies = pose['bodies']
#     faces = pose['faces']
#     hands = pose['hands']
#     candidate = bodies['candidate']
#     subset = bodies['subset']
#     canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

#     canvas = util.draw_bodypose_V2(canvas, candidate, subset)
#     canvas = util.draw_handpose(canvas, hands)
#     canvas = util.draw_facepose(canvas, faces)

#     if bboxs is not None:
#         for bbox in bboxs:
#             # 画框
#             cv2.rectangle(canvas, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

#     return canvas

def draw_pose_foot(pose, H, W, bboxs=None):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose_V2(canvas, candidate, subset)
    canvas = util.draw_handpose(canvas, hands)
    canvas = util.draw_facepose(canvas, faces)

    if bboxs is not None:
        for bbox in bboxs:
            # 画框
            cv2.rectangle(canvas, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

    return canvas

def putinfo(output, dic_info):
    height, width, _ = output.shape

    # 计算文本的起始位置（这里将文本绘制到图像中央）
    center_x = width // 2
    center_y = height // 2

    # 设置文本绘制的偏移量，以确保文本不会重叠
    text_offset = 40  # 每行文本之间的偏移
    
    l1 = dic_info['l1']
    l2 = dic_info['l2']
    l1_location1 =dic_info['l1_location1']
    l1_location2 = dic_info['l1_location2']

    r1 = dic_info['r1']
    r2 = dic_info['r2']

    r1_location1 = dic_info['r1_location1']
    r1_location2 = dic_info['r1_location2']

    #print(dic_info['visible'].shape)

    v_ll = dic_info['visible'][0,18]
    v_l2 = dic_info['visible'][0,19]

    v_rl = dic_info['visible'][0,21]
    v_r2 = dic_info['visible'][0,22]

    #print("v_ll, ", v_ll)
    # 在图像中央显示这些数值及坐标
    cv2.putText(output, f'l1: {l1:.2f}, v1: {v_ll:.2f}, l1_location1: ({l1_location1[0]:.2f}, {l1_location1[1]:.2f})', 
                (center_x - 200, center_y - 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(output, f'l2: {l2:.2f},  v2: {v_l2:.2f}, l1_location2: ({l1_location2[0]:.2f}, {l1_location2[1]:.2f})', 
                (center_x - 200, center_y - 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(output, f'r1: {r1:.2f}, v_rl: {v_rl:.2f}, r1_location1: ({r1_location1[0]:.2f}, {r1_location1[1]:.2f})', 
                (center_x - 200, center_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.putText(output, f'r2: {r2:.2f},  v_r2: {v_r2:.2f}, r1_location2: ({r1_location2[0]:.2f}, {r1_location2[1]:.2f})', 
                (center_x - 200, center_y + text_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return output

def draw_pose_133poitns(pose, H, W):

    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    N, num,loc = pose.shape
    for i in range(N):
        for j in range(num):
            x = int(pose[i,j,0])
            y = int(pose[i,j,1])
            cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (255, 255, 255)  # 白色
            thickness = 2
            cv2.putText(canvas, "%d"%(j), (x, y), font, font_scale, color, thickness)
    return canvas


class DWposeDetector:
    def __init__(self, pose_config, pose_ckpt, \
            det_config=None, det_ckpt=None, \
            det_onnx=None, pose_onnx=None, \
            det_trt=None, pose_trt=None, \
            device='cpu', type='pt', cuda_stream=torch.cuda.current_stream()): #todo
    # def __init__(self, pose_config, pose_ckpt, \
    #         det_config=None, det_ckpt=None, \
    #         det_onnx=None, pose_onnx=None, \
    #         det_trt=None, pose_trt=None, \
    #         device='cpu', type='pt', cuda_stream=None):


        self.pose_estimation = Wholebody(det_config, det_ckpt, pose_config, pose_ckpt, \
             det_onnx = det_onnx, pose_onnx=pose_onnx, \
             det_trt = det_trt, pose_trt=pose_trt, \
             device=device, type=type, cuda_stream=cuda_stream)
        
    #def __call__(self, image_np_hwc, show_face, show_hands, show_body, show_toes=True, threshold=5.5, plot=False):
    def __call__(self, image_np_hwc, show_face, show_hands, show_body, show_toes=True, threshold=5, plot=False, box_ext=None, xdwpose_result=None):
        image_np_hwc= image_np_hwc.copy()

        BODY_NUM = 18
        if show_toes:
            BODY_NUM = 20
        
        H, W, C = image_np_hwc.shape
        
        if xdwpose_result is None:
            candidate_133, subset_133, candidate, subset, visible, bbox = self.pose_estimation(image_np_hwc, box_ext=box_ext)
            return [candidate, subset, visible, bbox]
        else:
            candidate, subset, visible, bbox = xdwpose_result
            bbox = box_ext

        nums, keys, locs = candidate.shape
        candidate[..., 0] /= float(W)
        candidate[..., 1] /= float(H)
        body = candidate[:,:18].copy()
        body = body.reshape(nums*18, locs)
        score = subset[:,:BODY_NUM]

        #candidate (1, 134, 2)
        #subset (1, 134)

        if show_toes:
                    
            left_foottoe = np.mean(candidate[:, [18, 19]], axis=1)
            right_foottoe = np.mean(candidate[:, [21, 22]], axis=1)

            l1 = candidate[:, 18]
            l2 = candidate[:, 19]

            r1 = candidate[:, 21]
            r2 = candidate[:, 22]

            
            # 判断每个坐标的横纵坐标绝对值是否为 1
            is_inside_l = np.logical_and(np.all(np.abs(l1) < 1, axis=1),
                                         np.all(np.abs(l2) < 1, axis=1)).astype(int)
            is_inside_r = np.logical_and(np.all(np.abs(r1) < 1, axis=1),
                                         np.all(np.abs(r2) < 1, axis=1)).astype(int)


            left_foottoe_score = np.logical_and(
                subset[:, 18] > threshold,
                subset[:, 19] > threshold).astype(int)

            left_foottoe_score = np.logical_and(left_foottoe_score, is_inside_l).astype(int)
            right_foottoe_score = np.logical_and(
                subset[:, 21] > threshold,
                subset[:, 22] > threshold).astype(int)

            right_foottoe_score = np.logical_and(right_foottoe_score, is_inside_r).astype(int)


        l1 = subset[0, 18] 
        l2 = subset[0, 19] 
        l1_location1 = candidate[0, 18] #2维坐标
        l1_location2 = candidate[0, 19] #2维坐标

        r1 = subset[0, 21] 
        r2 = subset[0, 22] 

        r1_location1 = candidate[0, 21] #2维坐标
        r1_location2 = candidate[0, 22] #2维坐标
        dic_info = {}
        dic_info['l1'] = l1
        dic_info['l2'] = l2 
        dic_info['l1_location1'] = l1_location1
        dic_info['l1_location2'] = l1_location2

        dic_info['r1'] = r1
        dic_info['r2'] = r2

        dic_info['r1_location1'] = r1_location1
        dic_info['r1_location2'] = r1_location2

        dic_info['visible'] = visible

        for i in range(len(score)):
            for j in range(len(score[i])):

                if j < 18:
                    if score[i][j] > threshold:
                        score[i][j] = int((BODY_NUM)*i+j)
                    else:
                        score[i][j] = -1
                elif j == 18:
                    if left_foottoe_score[i] > 0: #这里eft_foottoe_score[i]已经是0，1值了:
                        score[i][j] = int((BODY_NUM)*i+j)
                    else:
                        score[i][j] = -1
                elif j == 19:
                    if right_foottoe_score[i] > 0:#这里eft_foottoe_score[i]已经是0，1值了:
                        score[i][j] = int((BODY_NUM)*i+j)
                    else:
                        score[i][j] = -1

        un_visible = subset < threshold
        candidate[un_visible] = -1

        foot = candidate[:,18:24]
        faces = candidate[:,24:92]
        hands = candidate[:,92:113]
        hands_2 =  candidate[:,113:]

        hands = np.vstack([hands, candidate[:,113:]])

        if show_toes:
            body =  np.concatenate((body, left_foottoe, right_foottoe), axis=0)

        #print("body.shape is ", body.shape)
        bodies = dict(candidate=body, subset=score)
        pose = dict(bodies=bodies, hands=hands, faces=faces)
        
        pose = dict(bodies=bodies if show_body else {'candidate':[], 'subset':[]}, faces=faces if show_face else [], hands=hands if show_hands else [])

        if plot:
            output = draw_pose_foot(pose, H, W, bboxs=bbox)
            return output[:,:,::-1]
            #output = putinfo(output, dic_info)
            #
            #output = cv2.hconcat([image_np_hwc, output])
            return candidate_133, subset_133, bbox, output[:,:,::-1]
        else:
            return candidate_133, subset_133, bbox, None

