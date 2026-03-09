import numpy as np

from .onnxdet import inference_detector as inference_detector_trt
from .onnxpose import inference_pose

from .onnxdet import inference_detector_onnx as inference_detector_onnx
from .onnxpose import inference_pose_onnx as inference_pose_onnx

import torch
import torch.nn.functional as F
import os

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.structures import merge_data_samples
from mmdet.apis import inference_detector, init_detector

import onnxruntime as ort

class Wholebody:
    def __init__(self, det_config=None, det_ckpt=None, 
                 pose_config=None, pose_ckpt=None,
                det_onnx=None, pose_onnx=None, \
                det_trt=None, pose_trt=None, \
                device="cpu", type='pth', cuda_stream=torch.cuda.current_stream()):
        
        #对应的score * 10
        self.is_rtmw = True        
        # if det_ckpt is None:
        #     det_ckpt = os.path.join("models", "yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth")
        #     #'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth'
        
        # if pose_ckpt is None:
        #     pose_ckpt = os.path.join("models", "dw-ll_ucoco_384.pth")
        #     #"https://huggingface.co/wanghaofan/dw-ll_ucoco_384/resolve/main/dw-ll_ucoco_384.pth"
        
        self.type = type
        self.cuda_stream = cuda_stream
        if self.type == 'pt' or self.type == 'pth':
            # build detector
            # self.detector = init_detector(det_config, det_ckpt, device=device)
            # self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)
        
            # build pose estimator
            self.pose_estimator = init_pose_estimator(
                pose_config,
                pose_ckpt,
                device=device)

        elif self.type == 'onnx':
            #TODO
            #device = 'cuda:0'
            providers = ['CPUExecutionProvider'
                    ] if device == 'cpu' else ['CUDAExecutionProvider']

            self.session_det = ort.InferenceSession(path_or_bytes=det_onnx, providers=providers)
            self.session_pose = ort.InferenceSession(path_or_bytes=pose_onnx, providers=providers)
       
        elif self.type == 'trt':
            from .trt_utilities import Engine
            self.engine = Engine(det_trt)
            self.engine.load()
            self.engine.activate()
            self.engine.allocate_buffers()

            self.engine2 = Engine(pose_trt)            
            self.engine2.load()
            self.engine2.activate()
            self.engine2.allocate_buffers()
    
    def __call__(self, oriImg,  box_ext, raw_133_ind_type=True, return_bbox=True):
        
        if self.type == 'pt' or self.type == 'pth':

            bboxes = box_ext
            # predict keypoints
            if len(bboxes) == 0:
                pose_results = inference_topdown(self.pose_estimator, oriImg)
            else:
                pose_results = inference_topdown(self.pose_estimator, oriImg, bboxes)
            preds = merge_data_samples(pose_results)
            preds = preds.pred_instances

            # preds = pose_results[0].pred_instances
            keypoints = preds.get('transformed_keypoints',
                                            preds.keypoints)
            if 'keypoint_scores' in preds:
                scores = preds.keypoint_scores
            else:
                scores = np.ones(keypoints.shape[:-1])

            if 'keypoints_visible' in preds:
                visible = preds.keypoints_visible
            else:
                visible = np.ones(keypoints.shape[:-1])
            keypoints_info = np.concatenate(
                (keypoints, scores[..., None], visible[..., None]),
                axis=-1)
            det_result =  bboxes  
            
        elif self.type == 'onnx':
            det_result = inference_detector_onnx(self.session_det, oriImg)
            keypoints, scores = inference_pose_onnx(self.session_pose, det_result, oriImg)
            keypoints_info = np.concatenate(
            (keypoints, scores[..., None]), axis=-1)

        elif self.type == 'trt':
            det_result = inference_detector_trt(
            engine=self.engine, cudaStream=self.cuda_stream, image_np_hwc=oriImg)
            keypoints, scores = inference_pose(
            engine=self.engine2, cudaStream=self.cuda_stream, out_bbox=det_result, image_np_hwc=oriImg)
        
            keypoints_info = np.concatenate(
                (keypoints, scores[..., None]), axis=-1)

        keypoints_info_133 = keypoints_info.copy()
        scores_133 = scores.copy()

        # compute neck joint
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        # neck score when visualizing pred
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        
        if self.is_rtmw:
            neck[:, 2:3] = neck[:, 2:3] * 10
        new_keypoints_info = np.insert(
            keypoints_info, 17, neck, axis=1)
        
        mmpose_idx = [
            17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
        ]
        openpose_idx = [
            1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
        ]
        new_keypoints_info[:, openpose_idx] = \
            new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info

        if self.type == 'trt' or self.type == 'onnx':
            keypoints, scores = keypoints_info[
                ..., :2], keypoints_info[..., 2]
        else:
            keypoints, scores, visible = keypoints_info[
            ..., :2], keypoints_info[..., 2], keypoints_info[..., 3]
   
        return keypoints_info_133, scores_133, keypoints, scores, visible,  det_result
