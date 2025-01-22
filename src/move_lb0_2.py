import os
import shutil
import random

# src_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA4TEST/LABEL0/SMUGEDLIP'
# dest_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_DOWNSIZE/test/LABEL_0/SMUDGED_LIP'
# os.makedirs(dest_folder,exist_ok=True)

# num_img = 6
# sub_folder = os.listdir(src_folder)
# for sub_f in sub_folder:
#     src_sub_folder = os.path.join(src_folder, sub_f)
#     dest_sub_folder = os.path.join(dest_folder, sub_f)
#     os.makedirs(dest_sub_folder, exist_ok=True)
#     if len(os.listdir(dest_sub_folder)) == 0:
#         cnt_51_0 = num_img
#         cnt_51_1 = num_img
#         cnt_52_0 = num_img
#         cnt_52_1 = num_img
#         cnt_61_0 = num_img
#         cnt_61_1 = num_img
#         cnt_62_0 = num_img
#         cnt_62_1 = num_img
#         cnt_71_0 = num_img
#         cnt_71_1 = num_img
#         cnt_72_0 = num_img
#         cnt_72_1 = num_img
#         cnt_81_0 = num_img
#         cnt_81_1 = num_img
#         cnt_82_0 = num_img
#         cnt_82_1 = num_img
        
#         list_img = os.listdir(src_sub_folder)
#         random.shuffle(list_img)
#         for img in list_img:
#             img_name = img.split('.')[0]
#             tail = img.split('.')[1]
#             cpnt = img_name.split('_')
            
#             dest_img = img 
            
#             # print(dest_img)
#             if cpnt[1] == '0051' and cpnt[2][-1] == '0' and cnt_51_0 > 0:
#                 shutil.copy(os.path.join(src_sub_folder, img), os.path.join(dest_sub_folder,dest_img))
#                 cnt_51_0 -= 1
#             if cpnt[1] == '0051' and cpnt[2][-1] == '1' and cnt_51_1 > 0:
#                 shutil.copy(os.path.join(src_sub_folder, img), os.path.join(dest_sub_folder,dest_img))
#                 cnt_51_1 -= 1
#             if cpnt[1] == '0052' and cpnt[2][-1] == '0' and cnt_52_0 > 0:
#                 shutil.copy(os.path.join(src_sub_folder, img), os.path.join(dest_sub_folder,dest_img))
#                 cnt_52_0 -= 1
#             if cpnt[1] == '0052' and cpnt[2][-1] == '1' and cnt_52_1 > 0:
#                 shutil.copy(os.path.join(src_sub_folder, img), os.path.join(dest_sub_folder,dest_img))
#                 cnt_52_1 -= 1
#             if cpnt[1] == '0061' and cpnt[2][-1] == '0' and cnt_61_0 > 0:
#                 shutil.copy(os.path.join(src_sub_folder, img), os.path.join(dest_sub_folder,dest_img))
#                 cnt_61_0 -= 1
#             if cpnt[1] == '0061' and cpnt[2][-1] == '1' and cnt_61_1 > 0:
#                 shutil.copy(os.path.join(src_sub_folder, img), os.path.join(dest_sub_folder,dest_img))
#                 cnt_61_1 -= 1
#             if cpnt[1] == '0062' and cpnt[2][-1] == '0' and cnt_62_0 > 0:
#                 shutil.copy(os.path.join(src_sub_folder, img), os.path.join(dest_sub_folder,dest_img))
#                 cnt_62_0 -= 1
#             if cpnt[1] == '0062' and cpnt[2][-1] == '1' and cnt_62_1 > 0:
#                 shutil.copy(os.path.join(src_sub_folder, img), os.path.join(dest_sub_folder,dest_img))
#                 cnt_62_1 -= 1
#             if cpnt[1] == '0071' and cpnt[2][-1] == '0' and cnt_71_0 > 0:
#                 shutil.copy(os.path.join(src_sub_folder, img), os.path.join(dest_sub_folder,dest_img))
#                 cnt_71_0 -= 1
#             if cpnt[1] == '0071' and cpnt[2][-1] == '1' and cnt_71_1 > 0:
#                 shutil.copy(os.path.join(src_sub_folder, img), os.path.join(dest_sub_folder,dest_img))
#                 cnt_71_1 -= 1
#             if cpnt[1] == '0072' and cpnt[2][-1] == '0' and cnt_72_0 > 0:
#                 shutil.copy(os.path.join(src_sub_folder, img), os.path.join(dest_sub_folder,dest_img))
#                 cnt_72_0 -= 1
#             if cpnt[1] == '0072' and cpnt[2][-1] == '1' and cnt_72_1 > 0:
#                 shutil.copy(os.path.join(src_sub_folder, img), os.path.join(dest_sub_folder,dest_img))
#                 cnt_72_1 -= 1
#             if cpnt[1] == '0081' and cpnt[2][-1] == '0' and cnt_81_0 > 0:
#                 shutil.copy(os.path.join(src_sub_folder, img), os.path.join(dest_sub_folder,dest_img))
#                 cnt_81_0 -= 1
#             if cpnt[1] == '0081' and cpnt[2][-1] == '1' and cnt_81_1 > 0:
#                 shutil.copy(os.path.join(src_sub_folder, img), os.path.join(dest_sub_folder,dest_img))
#                 cnt_81_1 -= 1
#             if cpnt[1] == '0082' and cpnt[2][-1] == '0' and cnt_82_0 > 0:
#                 shutil.copy(os.path.join(src_sub_folder, img), os.path.join(dest_sub_folder,dest_img))
#                 cnt_82_0 -= 1
#             if cpnt[1] == '0082' and cpnt[2][-1] == '1' and cnt_82_1 > 0:
#                 shutil.copy(os.path.join(src_sub_folder, img), os.path.join(dest_sub_folder,dest_img))
#                 cnt_82_1 -= 1
                
folders = ['/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_DOWNSIZE/val/LABEL_0/LIP_SMUDGED',
           '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_DOWNSIZE/train/LABEL_0/LIP_SMUDGED',
           '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_DOWNSIZE/test/LABEL_0/LIP_SMUDGED']        
        
for folder in folders:
    for subfolder in os.listdir(folder):
        subpath = os.path.join(folder, subfolder)
        for img in os.listdir(subpath):
            pre = os.path.join(subpath, img)
            post = os.path.join(subpath, img + '.png')
            os.rename(pre, post)