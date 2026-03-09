import cv2
import time
import torch
from multiprocessing import Process, Queue

try:
    from dpvo.utils import Timer
    from dpvo.dpvo import DPVO
    from dpvo.config import cfg
except:
    pass


from ..geo.hmr_cam import estimate_focal_length
from tqdm import tqdm

class SLAMModel(object):
    def __init__(self, video_np, width, height, intrinsics=None, stride=1, skip=0, buffer=2048, resize=0.5, model_path='./input/'):
        """
        Args:
            intrinsics: [fx, fy, cx, cy]
        """
        if intrinsics is None:
            print("Estimating focal length")
            focal_length = estimate_focal_length(width, height)
            intrinsics = torch.tensor([focal_length, focal_length, width / 2.0, height / 2.0])
        else:
            intrinsics = intrinsics.clone()

        self.dpvo_cfg = model_path.replace('/dpvo.pth', '/DPVO_default.yaml')
        self.dpvo_ckpt = model_path

        self.buffer = buffer
        self.times = []
        self.slam = None

        n, h, w = video_np.shape[:3]
        self.image_list = [cv2.resize(video_np[i],None,fx=resize,fy=resize,interpolation=cv2.INTER_AREA)[: h - h % 16, : w - w % 16] for i in range(n)]
        self.intrinsics = intrinsics * resize

    def track(self):
        n = len(self.image_list)
        intrinsics = self.intrinsics
        bar = tqdm(total=n, desc="DPVO")

        for i, image in enumerate(self.image_list):
            image = torch.from_numpy(image).permute(2, 0, 1).cuda()
            intrinsics = intrinsics.cuda()  # [fx, fy, cx, cy]

            if self.slam is None:
                cfg.merge_from_file(self.dpvo_cfg)
                cfg.BUFFER_SIZE = self.buffer
                self.slam = DPVO(cfg, self.dpvo_ckpt, ht=image.shape[1], wd=image.shape[2], viz=False)

            with Timer("SLAM", enabled=False):
                t = time.time()
                self.slam(t, image, intrinsics)
                self.times.append(time.time() - t)
            
            bar.update()

    def process(self):
        for _ in range(12):
            self.slam.update()

        return self.slam.terminate()[0]