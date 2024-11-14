import os 
import shutil

input_folder = '/mnt/2E287FDD287FA28F/teamAI/MinhQuan/siamesenn/Symetry-of-Facial-Makeup/ProtoZenon123-Symmetry-of-Facial-Make-up/DATA_ALL/data4train/data/test_data/label_0/eye'
output_folder = '/mnt/2E287FDD287FA28F/teamAI/MinhQuan/siamesenn/Symetry-of-Facial-Makeup/redundant_2510'
os.makedirs(input_folder, exist_ok=True)

# cnt_make_eyeline = 13200
# cnt_nonmake_eyeline = 6600
cnt_make_eyeline = 4160
cnt_nonmake_eyeline = 2080
for image_name in sorted(os.listdir(input_folder)):
    image_path = os.path.join(input_folder, image_name)
    cpnt = image_name.split('_')
    if cnt_nonmake_eyeline > 0:
        if cpnt[1] == '300' or cpnt[1] == '400':
            shutil.move(image_path, output_folder)
            cnt_nonmake_eyeline -= 1
    if cnt_make_eyeline > 0:
        if cpnt[1] == '000':
            shutil.move(image_path, output_folder)
            cnt_make_eyeline -= 1
print(cnt_make_eyeline)
print(cnt_nonmake_eyeline)