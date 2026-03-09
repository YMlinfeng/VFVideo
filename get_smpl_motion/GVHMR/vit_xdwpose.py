import torch
import numpy as np
import cv2
import os, sys



def xdwpose2vitpose(xdwpose):
    n = xdwpose.shape[0]
    id_pair_dict = {0:0,1:15,2:14,3:17,4:16,5:5,6:2,7:6,8:3,9:7,10:4,11:11,12:8,13:12,14:9,15:13,16:10}
    vit_pose = np.ones([n,17,3])
    for i in range(n):
        for j in range(17):
            vit_pose[i,j,0] = xdwpose[i,0,id_pair_dict[j],0]
            vit_pose[i,j,1] = xdwpose[i,0,id_pair_dict[j],1]
            vit_pose[i,j,2] = xdwpose[i,0,id_pair_dict[j],2]
    print ('vit_pose2', vit_pose.shape)

    return vit_pose


vit_pose = torch.load('/ytech_milm/liujiwen/kling_motion_service/GVHMR/test/preprocess/vitpose.pt').numpy()
xdwpose = np.load('/ytech_milm/liujiwen/kling_motion_service/GVHMR/xdwpose.npy')

vit_pose = xdwpose2vitpose(xdwpose)

print ('vit_pose', vit_pose.shape)
print ('xdwpose', xdwpose.shape)



n = xdwpose.shape[0]
img0 = np.ones([1280, 720, 3]).astype('uint8')*255

for i in range(n):
    img = img0.copy()
    for j in range(17):
        x, y = int(vit_pose[i,j,0]), int(vit_pose[i,j,1])
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)  # 画点
        cv2.putText(img, str(j), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # 标 ID

        # x, y = int(xdwpose[i,0,j,0]), int(xdwpose[i,0,j,1])
        # cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)  # 画点
        # cv2.putText(img, str(j), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # 标 ID


    cv2.imwrite('./t1.png', img)
    break



    