import json
import csv
import os
csvff= '/ytech_milm/Keling_HumanMotion/json_files/modified_0219_mvb3_video_motion_vae51_720_cap4_t5_valid_vae_depth_smpl.csv'
list_txt=[]
curr_list=[]
csvoo= '/ytech_milm/Keling_HumanMotion/json_files/modified_0219_mvb3_video_motion_vae51_720_cap4_t5_valid_vae_depth_smpl_3dpose.csv'
blank =0

with open(csvff, 'r', newline='', encoding='utf-8') as css:
  with open(csvoo, 'w', newline='', encoding='utf-8') as coo:
    reader=csv.reader(css)
    writer = csv.writer(coo)
    flag=0
    data=[]
    for row in reader:
        noflag=False
        if not flag:
            row.append('3dpose_path')
            writer.writerow(row)
            flag+=1
            continue
        if not flag%2000:
            print(flag)
            print(blank,'null')
        #     print(list_txt[0],len(list_txt[0]),len(list_txt))
        flag+=1
        try:
            a=json.load(open(row[-1],'r'))
        except:
            noflag=True
        if not noflag:
            dir_path = a['hmr4d_results'].replace('hmr4d_results.pt','3d_pose.mp4')
            if os.path.exists(dir_path):
                row.append(dir_path)
            else:
                row.append('')
                blank+=1
        else:
            row.append('')
            blank+=1
        writer.writerow(row)
print(blank)