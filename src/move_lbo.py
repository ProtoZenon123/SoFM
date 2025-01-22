import os
import shutil
import random

src_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/EYE'
dest_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_DOWNSIZE/train/LABEL_0/EYE'

num_img = 400
sub_folder = os.listdir(src_folder)
for sub_f in sub_folder:
    src_sub_folder = os.path.join(src_folder, sub_f)
    idx = sub_f[-1]
    tmp = None
    if sub_f[-1] == '0':
        tmp = sub_f[:-2] + 'e'
    else:
        tmp = sub_f[:-1] + 'e'
    tmp += idx
    dest_sub_folder = os.path.join(dest_folder, tmp)
    os.makedirs(dest_sub_folder, exist_ok=True)
    print(tmp)
    if len(os.listdir(dest_sub_folder)) == 0:
        cnt_1 = num_img
        cnt_2 = num_img
        list_img = os.listdir(src_sub_folder)
        random.shuffle(list_img)
        for img in list_img:
            img_name = img.split('.')[0]
            tail = img.split('.')[1]
            cpnt = img_name.split('_')[1]
            
            dest_img = None
            dest_idx = img_name[-1]
            if img_name[-1] == '0':
                dest_img = img_name[:-2] + 'e'
            else:
                dest_img = img_name[:-1] + 'e'
            dest_img += dest_idx
            
            # print((dest_img + f'.{tail}'))
            if cpnt[0] == '1' and cnt_1 > 0:
                shutil.copy(os.path.join(src_sub_folder, img), os.path.join(dest_sub_folder,(dest_img + f'.{tail}')))
                cnt_1 -= 1
            if cpnt[0] == '2' and cnt_2 > 0:
                shutil.copy(os.path.join(src_sub_folder, img), os.path.join(dest_sub_folder,(dest_img + f'.{tail}')))
                cnt_2 -= 1
            if cnt_2 ==  0 and cnt_1 == 0:
                break
        
        
    