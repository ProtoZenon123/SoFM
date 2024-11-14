import os
import cv2
from copy import deepcopy
import gc
from tqdm import tqdm

list_source_folder = ['/mnt/2E287FDD287FA28F/teamAI/MinhQuan/siamesenn/Symetry-of-Facial-Makeup/ProtoZenon123-Symmetry-of-Facial-Make-up/DATA_ALL/data_black_cheek/test/lb0/cheek',
                      '/mnt/2E287FDD287FA28F/teamAI/MinhQuan/siamesenn/Symetry-of-Facial-Makeup/ProtoZenon123-Symmetry-of-Facial-Make-up/DATA_ALL/data_black_cheek/test/lb1/cheek']
list_dest_folder = ['/mnt/2E287FDD287FA28F/teamAI/MinhQuan/siamesenn/Symetry-of-Facial-Makeup/ProtoZenon123-Symmetry-of-Facial-Make-up/DATA_ALL/data/test_data/label_0/cheek',
                    '/mnt/2E287FDD287FA28F/teamAI/MinhQuan/siamesenn/Symetry-of-Facial-Makeup/ProtoZenon123-Symmetry-of-Facial-Make-up/DATA_ALL/data/test_data/label_1/cheek']
    
for idx, source_folder in enumerate(list_source_folder):
    dest_folder = list_dest_folder[idx]
    os.makedirs(dest_folder, exist_ok=True)
    for source_image_folder in tqdm(os.listdir(source_folder), desc=f"{idx}_{os.path.basename(source_folder)}"):
        source_image_folder_path = os.path.join(source_folder, source_image_folder)
        dest_image_folder_path = os.path.join(dest_folder, source_image_folder)
        os.makedirs(dest_image_folder_path, exist_ok=True)
        for image_name in os.listdir(source_image_folder_path):
            image_path = os.path.join(source_image_folder_path, image_name)
            dest_image_path = os.path.join(dest_image_folder_path, image_name)
            image = cv2.imread(image_path)
            white_image = deepcopy(image)
            mask = (white_image == [0, 0, 0]).all(axis=2)
            white_image[mask] = [255, 255, 255]
            cv2.imwrite(dest_image_path, white_image)
            del image
            del white_image
            gc.collect()

            
        
    





