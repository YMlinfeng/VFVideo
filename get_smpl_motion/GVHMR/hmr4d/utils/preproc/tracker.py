from ultralytics import YOLO

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from ..seq_utils import (
    get_frame_id_list_from_mask,
    linear_interpolate_frame_ids,
    frame_id_to_mask,
    rearrange_by_mask,
)
from ..video_io_utils import get_video_lwh
from ..net_utils import moving_average_smooth

import cv2

class Tracker:
    def __init__(self, model_path) -> None:
        # https://docs.ultralytics.com/modes/predict/
        ##self.yolo = YOLO(PROJ_ROOT / "inputs/checkpoints/yolo/yolov8x.pt")
        self.yolo = YOLO(model_path)
        self.yolo_detact = YOLO(model_path)
        self.model_path = model_path

    def detect(self, image):
        ##input_img0 = cv2.imread('/ytech_milm/liujiwen/kling_motion_service/GVHMR/human.png')
        results = self.yolo_detact.predict(source=image, save=False, save_txt=False, device='cuda:0')[0].boxes.data.cpu().numpy()

        det_result = []
        for i in range(results.shape[0]):
            print(results[i],image.shape)
            x1, y1, x2, y2, score, ID = results[i]
            a = (x2-x1)*(y2-y1)
            print ('x1, y1, x2, y2, score, ID', x1, y1, x2, y2, score, ID)
            if ID == 0 and score>0.25:
                det_result.append([[x1, y1, x2, y2],a])
        try:
            det_result = sorted(det_result, key=lambda x: -x[1]) #按照面积排序
            ##det_result = [det[0] for det in det_result]
        except:
            pass

        return det_result

    
    def track(self, video_path):
        track_history = []
        cfg = {
            "device": "cuda",
            "conf": 0.25,  # default 0.25, wham 0.5
            "classes": 0,  # human
            "verbose": False,
            "stream": True,
        }
        results = self.yolo.track(video_path, **cfg, tracker=self.model_path.replace('/yolo11x.pt', '/botsort.yaml'))
        ##results = self.yolo.track(video_path, **cfg, tracker=self.model_path.replace('/yolo11x.engine', '/botsort.yaml'))

        # frame-by-frame tracking
        track_history = []
        for result in tqdm(results, total=get_video_lwh(video_path)[0], desc="YoloV11 Tracking"):
            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()  # (N)
                bbx_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4)
                result_frame = [{"id": track_ids[i], "bbx_xyxy": bbx_xyxy[i]} for i in range(len(track_ids))]
            else:
                result_frame = []
            track_history.append(result_frame)

        return track_history


    @staticmethod
    def sort_track_length(track_history, video_path):
        """This handles the track history from YOLO tracker."""
        id_to_frame_ids = defaultdict(list)
        id_to_bbx_xyxys = defaultdict(list)
        # parse to {det_id : [frame_id]}
        for frame_id, frame in enumerate(track_history):
            for det in frame:
                id_to_frame_ids[det["id"]].append(frame_id)
                id_to_bbx_xyxys[det["id"]].append(det["bbx_xyxy"])
        for k, v in id_to_bbx_xyxys.items():
            id_to_bbx_xyxys[k] = np.array(v)

        # Sort by length of each track (max to min)
        id_length = {k: len(v) for k, v in id_to_frame_ids.items()}
        id2length = dict(sorted(id_length.items(), key=lambda item: item[1], reverse=True))

        # Sort by area sum (max to min)
        id_area_sum = {}
        l, w, h = get_video_lwh(video_path)
        for k, v in id_to_bbx_xyxys.items():
            bbx_wh = v[:, 2:] - v[:, :2]
            id_area_sum[k] = (bbx_wh[:, 0] * bbx_wh[:, 1] / w / h).sum()
        id2area_sum = dict(sorted(id_area_sum.items(), key=lambda item: item[1], reverse=True))
        id_sorted = list(id2area_sum.keys())

        return id_to_frame_ids, id_to_bbx_xyxys, id_sorted

    def bbox_crop(self, human_bbox_list,h,w,center_crop_scale=1.2):
        for i in range(human_bbox_list.shape[0]):
            bbox = human_bbox_list[i]
            cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
            cw, ch = bbox[2]-bbox[0], bbox[3]-bbox[1]
            x1, y1 = int(max(0,cx-cw/2*center_crop_scale)), int(max(0,cy-ch/2*center_crop_scale))
            x2, y2 = int(min(w,cx+cw/2*center_crop_scale)), int(min(h,cy+ch/2*center_crop_scale))
            human_bbox_list[i][0] = x1
            human_bbox_list[i][1] = y1
            human_bbox_list[i][2] = x2
            human_bbox_list[i][3] = y2

        return human_bbox_list

    def get_one_track(self, video_path):
        # track
        track_history = self.track(video_path)

        # parse track_history & use top1 track
        id_to_frame_ids, id_to_bbx_xyxys, id_sorted = self.sort_track_length(track_history, video_path)
        if len (id_sorted) == 0:
            return None
        track_id = id_sorted[0]
        frame_ids = torch.tensor(id_to_frame_ids[track_id])  # (N,)
        bbx_xyxys = torch.tensor(id_to_bbx_xyxys[track_id])  # (N, 4)

        # interpolate missing frames
        L, W, H = get_video_lwh(video_path)
        mask = frame_id_to_mask(frame_ids, L)
        bbx_xyxy_one_track = rearrange_by_mask(bbx_xyxys, mask)  # (F, 4), missing filled with 0
        missing_frame_id_list = get_frame_id_list_from_mask(~mask)  # list of list

        ##bbx_xyxy_one_track = linear_interpolate_frame_ids(bbx_xyxy_one_track, missing_frame_id_list)
        ##assert (bbx_xyxy_one_track.sum(1) != 0).all()

        ##bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)
        ##bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)

        bbx_xyxy_one_track = self.bbox_crop(bbx_xyxy_one_track,H,W,center_crop_scale=1.1)

        return bbx_xyxy_one_track
