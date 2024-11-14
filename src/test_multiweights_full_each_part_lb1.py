from utils import load_config_eye, load_config_lip, load_config_cheek, make_grid_of_image
from model import siameseNN
from final_crop_and_transform import crop_and_transform_six_parts
from PIL import Image
import torch
import cv2
from torchvision import transforms as transforms
import os
from tqdm import tqdm

def predict_labels_1(outputs, threshold=0.5):
    if outputs >= threshold:
        return 1
    return 0
    
def convert_to_pilimage(opencv_image):
    pilimage = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(pilimage)

config = load_config_eye()

num_features = config['num_features']
channels_img = config['channels_img']
threshold = config['threshold']
cuda_id = config['cuda_id']
folder_weights_dir_eye = config['folder_weights_dir']

config = load_config_cheek()
folder_weights_dir_cheek = config['folder_weights_dir']

config = load_config_lip()
folder_weights_dir_lip = config['folder_weights_dir']

print(folder_weights_dir_eye)
print(folder_weights_dir_cheek)
print(folder_weights_dir_lip)

transform = transforms.Compose([transforms.Resize((121, 121)),
                                transforms.ToTensor()
                                ])

cuda = 'cuda:' + f'{cuda_id}'
device = torch.device(cuda if torch.cuda.is_available() else "cpu")

model = siameseNN(channels_img, num_features)
model = model.to(device)

# image_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/FACE_DATA/DATA_700/DATA_700_LB0/700_elegant'
# res_txt_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/FACE_DATA/RESULT_700/elegant_lb0_0911_each_part/res.txt'
# res_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/FACE_DATA/RESULT_700/elegant_lb0_0911_each_part/img_res'
# image_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/FACE_DATA/DATA_700/DATA_700_LB1/700_elegant'
image_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/FACE_DATA/DATA_2K_FILTERED/FRONT'
res_txt_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/FACE_DATA/RESULT_700/elegant_lb1_0911_each_part/res.txt'
res_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/FACE_DATA/RESULT_700/elegant_lb1_0911_each_part/img_res'

os.makedirs(res_folder, exist_ok=True)
os.makedirs(res_txt_folder, exist_ok=True)
 
dir_name = os.path.dirname(image_folder)
base_name = os.path.basename(image_folder)
crop_folder = os.path.join(dir_name, f"{base_name}_crop6parts")

for weights in os.listdir(folder_weights_dir_eye):
    cnt_eye_lb1 = 0
    cnt_cheek_lb1 = 0
    cnt_lip_lb1 = 0
    cnt_eye_lb0 = 0
    cnt_cheek_lb0 = 0
    cnt_lip_lb0 = 0
    total_eye_lb1 = 0
    total_eye_lb0 = 0
    total_cheek_lb1 = 0
    total_cheek_lb0 = 0
    total_lip_lb1 = 0
    total_lip_lb0 = 0
    total = 0
    
    id_weigths = weights.split('.')[1]
    id_weigths = id_weigths.split('_')
    if len(id_weigths) == 1:
        id_weigths = weights.split('.')[0]
    else:
        id_weigths = id_weigths[1]
    
    result_with_each_w_folder = os.path.join(res_folder, id_weigths)  
    os.makedirs(result_with_each_w_folder, exist_ok=True)  
        
    weights_path_eye = os.path.join(folder_weights_dir_eye, weights)
    weights_path_cheek = os.path.join(folder_weights_dir_cheek, weights)
    weights_path_lip = os.path.join(folder_weights_dir_lip, weights)
    
    model.eval()
    
    res_txt_path = os.path.join(res_txt_folder, f"{id_weigths}.txt")
    if os.path.exists(res_txt_path):
        with open(res_txt_path, 'w') as file:
            file.write('')
    else:
        with open(res_txt_path, 'w') as file:
            pass
    with torch.no_grad():
        for image_name in tqdm(os.listdir(image_folder), desc='test image'):
            image = cv2.imread(os.path.join(image_folder, image_name))
            if image is not None:
                total += 1
                left_eye, right_eye, left_cheek, right_cheek, left_lip, right_lip = crop_and_transform_six_parts(image)
                
                #############################################################################################
                ##############                             WHITE CHEEK                   ####################
                #############################################################################################
                # left_cheek = white_cheek(left_cheek)
                # right_cheek = white_cheek(right_cheek)
                
                try:
                    left_eye = convert_to_pilimage(left_eye)
                    right_eye = convert_to_pilimage(right_eye)
                    left_cheek = convert_to_pilimage(left_cheek)
                    right_cheek = convert_to_pilimage(right_cheek)
                    left_lip = convert_to_pilimage(left_lip)
                    right_lip = convert_to_pilimage(right_lip)
                except Exception as e:
                    print(image_name)
                
                #save_crop_img    
                folder_crop_image = os.path.join(crop_folder, image_name.split('.')[0])
                os.makedirs(folder_crop_image, exist_ok=True)
                left_eye.save(os.path.join(folder_crop_image, "left_eye.png"))
                right_eye.save(os.path.join(folder_crop_image, "right_eye.png"))
                left_cheek.save(os.path.join(folder_crop_image, "left_cheek.png"))
                right_cheek.save(os.path.join(folder_crop_image, "right_cheek.png"))
                left_lip.save(os.path.join(folder_crop_image, "left_lip.png"))
                right_lip.save(os.path.join(folder_crop_image, "right_lip.png"))
                
                left_eye = transform(left_eye)
                right_eye = transform(right_eye)
                left_cheek = transform(left_cheek)
                right_cheek = transform(right_cheek)
                left_lip = transform(left_lip)
                right_lip = transform(right_lip)
                
                left_eye = left_eye.unsqueeze(0).to(device)
                right_eye = right_eye.unsqueeze(0).to(device)
                left_cheek = left_cheek.unsqueeze(0).to(device)
                right_cheek = right_cheek.unsqueeze(0).to(device)
                left_lip = left_lip.unsqueeze(0).to(device)
                right_lip = right_lip.unsqueeze(0).to(device)
                
                model.load_state_dict(torch.load(weights_path_eye))
                res_eye = model(left_eye, right_eye)
                model.load_state_dict(torch.load(weights_path_cheek))
                res_cheek = model(left_cheek, right_cheek)
                model.load_state_dict(torch.load(weights_path_lip))
                res_lip = model(left_lip, right_lip)
                
                res_eye = res_eye.item()
                res_cheek = res_cheek.item()
                res_lip = res_lip.item()
                
                cpnt_name = image_name.split('.')[0]
                cpnt_name = cpnt_name.split('_')[-1]
                
                # if cpnt_name[0] == '0':
                #     if predict_labels_1(res_eye) == 1: cnt_eye_lb1 += 1
                #     total_eye_lb1 += 1
                # else:
                #     if predict_labels_1(res_eye) == 0: cnt_eye_lb0 += 1
                #     total_eye_lb0 += 1
                # if cpnt_name[1] == '0':
                #     if predict_labels_1(res_cheek) == 1: cnt_cheek_lb1 += 1
                #     total_cheek_lb1 += 1
                # else:
                #     if predict_labels_1(res_cheek) == 0: cnt_cheek_lb0 += 1
                #     total_cheek_lb0 += 1
                # if cpnt_name[2] == '0':
                #     if predict_labels_1(res_lip) == 1: cnt_lip_lb1 += 1
                #     total_lip_lb1 += 1
                # else:
                #     if predict_labels_1(res_lip) == 0: cnt_lip_lb0 += 1
                #     total_lip_lb0 += 1
                if predict_labels_1(res_eye) == 1: cnt_eye_lb1 += 1
                total_eye_lb1 += 1
                if predict_labels_1(res_eye) == 0: cnt_eye_lb0 += 1
                total_eye_lb0 += 1
            
                if predict_labels_1(res_cheek) == 1: cnt_cheek_lb1 += 1
                total_cheek_lb1 += 1
            
                if predict_labels_1(res_cheek) == 0: cnt_cheek_lb0 += 1
                total_cheek_lb0 += 1
            
                if predict_labels_1(res_lip) == 1: cnt_lip_lb1 += 1
                total_lip_lb1 += 1
            
                if predict_labels_1(res_lip) == 0: cnt_lip_lb0 += 1
                total_lip_lb0 += 1
                
                with open(res_txt_path, 'a') as file:
                    file.write(f'with {weights}:')
                    file.write(f'\t{image_name}\n')
                    file.write(f'\teye: {res_eye}\n')
                    file.write(f'\tcheek: {res_cheek}\n')
                    file.write(f'\tlip: {res_lip}\n\n')
                
                grid_image = make_grid_of_image(folder_crop_image, [res_eye, res_cheek, res_lip], weights)
                grid_image.save(f'{os.path.join(result_with_each_w_folder, image_name)}')
        print(id_weigths)   
        print("\nlabel 1:")
        if total_eye_lb1 != 0:     
            print(f'\teye: {(cnt_eye_lb1 / total_eye_lb1 * 100):.2f}')
        else:
            print("No eye label 1 !")
        if total_cheek_lb1 != 0:
            print(f'\tcheek: {(cnt_cheek_lb1 / total_cheek_lb1 * 100):.2f}')
        else:
            print("No cheek label 1 !")
        if total_lip_lb1 != 0:
            print(f'\tlip: {(cnt_lip_lb1 / total_lip_lb1 * 100):.2f}')
        else:
            print("No lip label 1 !")
        
        print("\nlabel 0:")  
        if total_eye_lb0 != 0:   
            print(f'\teye: {(cnt_eye_lb0 / total_eye_lb0 * 100):.2f}')
        else:
            print("No eye label 0 !")
        if total_cheek_lb0 != 0:
            print(f'\tcheek: {(cnt_cheek_lb0 / total_cheek_lb0 * 100):.2f}')
        else:
            print("No cheek label 0 !")
        if total_lip_lb0 != 0:
            print(f'\tlip: {(cnt_lip_lb0 / total_lip_lb0 * 100):.2f}')
        else:
            print("No lip label 0 !")
        
       
        
    