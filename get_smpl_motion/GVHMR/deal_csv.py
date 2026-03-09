import json
import csv
csvff= '/ytech_milm/Keling_HumanMotion/json_files/modified_0219_mvb3_video_motion_vae51_720_cap4_t5_valid_vae_depth_smpl.csv'
list_txt=[]
curr_list=[]
with open(csvff, 'r', newline='', encoding='utf-8') as css:
    reader=csv.reader(css)
    flag=0
    
    for row in reader:
        
        if not flag:
            flag+=1
            continue
        if not flag%2000:
            print(flag)
        #     print(list_txt[0],len(list_txt[0]),len(list_txt))
        noflag = False
        try:
            a=json.load(open(row[-1],'r'))
        except:
            noflag=True
        if not noflag:
            curr_list.append(a['hmr4d_results'].strip())
        if len(curr_list)>=1000:
            list_txt.append(curr_list)
            curr_list=[]
        flag+=1
    list_txt.append(curr_list)
print(list_txt[0],len(list_txt[0]),len(list_txt))
n=0
for i in list_txt:
    aa= open('label-{}.txt'.format(n),'w')
    for j in i:
        aa.write(j+'\n')
    n+=1