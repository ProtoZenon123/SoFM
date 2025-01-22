import os
import shutil
import random
import copy

src_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA4TEST/LABEL1/LIP'
dest_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_DOWNSIZE/test/LABEL_1/LIP'

sub_folder = os.listdir(src_folder)
num_img = 110

for sub_f in sub_folder:
    dest_sub_folder = os.path.join(dest_folder, sub_f)
    os.makedirs(dest_sub_folder, exist_ok=True)
    if len(os.listdir(dest_sub_folder)) == 0:
        cnt_0 = copy.copy(num_img)
        cnt_1 = copy.copy(num_img)
        src_sub_folder = os.path.join(src_folder, sub_f)
        list_img = os.listdir(src_sub_folder)
        random.shuffle(list_img)
        for img in list_img:
            img_name = img.split('.')[0]
            if cnt_1 > 0 and img_name[-1] == '1':
                shutil.copy(os.path.join(src_sub_folder, img), os.path.join(dest_sub_folder,img))
                cnt_1 -= 1
            if cnt_0 > 0 and img_name[-1] == '0':
                shutil.copy(os.path.join(src_sub_folder, img), os.path.join(dest_sub_folder,img))    
                cnt_0 -= 1