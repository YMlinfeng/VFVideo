import cv2
import os
from dwpose import DWposeDetector
import time

time0 = time.time()
#frame = cv2.imread("./assets/usain-bolt.jpg")


#folder = '/mmu_audio_hdd/MILM_data/SpeakingVideosImage/oneSpeaker/vhuman_project_formal/koubo-xiaoxuan1/'
folder = '/mmu_audio_hdd/MILM_data/SpeakingVideosImage/oneSpeaker/vhuman_project_formal/koubo-zile3'

out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)
dwpose = DWposeDetector()

time_s = time.time()
t2 = 0
for i in range(1, 1+250):
    filename = f"{i:08d}.jpg"
    name = os.path.join(folder, filename)
    frame = cv2.imread(name)

    time1 = time.time()
    output = dwpose(image_np_hwc=frame, show_body=True,
                    show_face=True, show_hands=True)
    time2 = time.time()
    
    t2 += (time2 - time1) 
    cv2.imwrite(os.path.join(out_dir, filename), output)

time_e = time.time()


print("time infer,", t2/250)
print("time all,", time_e - time_s)