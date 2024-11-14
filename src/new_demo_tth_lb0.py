import os
import cv2
from PIL import Image
import torch
from torchvision import transforms as transforms

from model import siameseNN
from new_crop import crop_six_parts
from utils import load_config
import argparse

def predict_labels_0(outputs, threshold=0.5):
    return (outputs < threshold)
def predict_labels_1(outputs, threshold=0.5):
    return (outputs >= threshold)
def convert_to_pilimage(opencv_image):
    pilimage = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(pilimage)

# image_folder = '/mnt/2E287FDD287FA28F/teamAI/MinhQuan/siamesenn/Symetry-of-Facial-Makeup/ProtoZenon123-Symmetry-of-Facial-Make-up/outstd'
# folder_weights_dir = "/mnt/2E287FDD287FA28F/teamAI/MinhQuan/siamesenn/Symetry-of-Facial-Makeup/weights_1010"
folder_weights_dir = "/mnt/2E287FDD287FA28F/teamAI/MinhQuan/siamesenn/Symetry-of-Facial-Makeup/ProtoZenon123-Symmetry-of-Facial-Make-up/weights_0810"
image_folder = "/mnt/2E287FDD287FA28F/teamAI/hong/EleGANt/label_so_0_REF/LB0"
# cuda_id = 0

#------------------------ prepare for checking the summetry of facial make up -------------------------
parser = argparse.ArgumentParser("nahhhhhhhhhh")
parser.add_argument("--label", type=int, default=1)
args = parser.parse_args()

config = load_config()

num_features = config['num_features']
channels_img = config['channels_img']
threshold = config['threshold']
cuda_id = config['cuda_id']
# folder_weights_dir = config['folder_weights_dir']

dir_name = os.path.dirname(image_folder)
base_name = os.path.basename(image_folder)
crop_folder = os.path.join(dir_name, f"{base_name}_crop6parts")


transform = transforms.Compose([transforms.Resize((121, 121)),
                                transforms.ToTensor()
                                ])

cuda = 'cuda:' + f'{cuda_id}'
device = torch.device(cuda if torch.cuda.is_available() else "cpu")

model_last_weight = siameseNN(channels_img, num_features)
model_min_loss_weight = siameseNN(channels_img, num_features)

# last_weights_path = os.path.join(folder_weights_dir, 'last_weights.pth')
# min_loss_weight_path = os.path.join(folder_weights_dir, 'min_loss_weights.pth')


last_weights_path = os.path.join(folder_weights_dir, 'last_weights.pth_15')
min_loss_weight_path = os.path.join(folder_weights_dir, 'min_loss_weights.pth')

model_last_weight.load_state_dict(torch.load(last_weights_path))
model_min_loss_weight.load_state_dict(torch.load(min_loss_weight_path))

#---------------------------------checking the symmetry of facial make up -----------------------------

cnt_mlw = 0
cnt_lw = 0
total = 0
cnt_eye_lw = 0
cnt_lip_lw = 0
cnt_cheek_lw = 0
cnt_eye_mlw = 0
cnt_lip_mlw = 0
cnt_cheek_mlw = 0

with torch.no_grad():
    model_last_weight.eval()
    model_min_loss_weight.eval()
    
    for image_name in os.listdir(image_folder):
        image = cv2.imread(os.path.join(image_folder, image_name))
        if image is not None:
            left_eye, right_eye, left_cheek, right_cheek, left_lip, right_lip = crop_six_parts(image)
            
            left_eye = convert_to_pilimage(left_eye)
            right_eye = convert_to_pilimage(right_eye)
            left_cheek = convert_to_pilimage(left_cheek)
            right_cheek = convert_to_pilimage(right_cheek)
            left_lip = convert_to_pilimage(left_lip)
            right_lip = convert_to_pilimage(right_lip)
            
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
            
            model_last_weight = model_last_weight.to(device)
            model_min_loss_weight = model_min_loss_weight.to(device)
            
            res_eye_mlw = model_min_loss_weight(left_eye, right_eye)
            res_cheek_mlw = model_min_loss_weight(left_cheek, right_cheek)
            res_lip_mlw = model_min_loss_weight(left_lip, right_lip)
            
            res_eye_lw = model_last_weight(left_eye, right_eye)
            res_cheek_lw = model_last_weight(left_cheek, right_cheek)
            res_lip_lw = model_last_weight(left_lip, right_lip)
            
            res_eye_lw = res_eye_lw.item()
            res_cheek_lw = res_cheek_lw.item()
            res_lip_lw = res_lip_lw.item()
            res_eye_mlw = res_eye_mlw.item()
            res_cheek_mlw = res_cheek_mlw.item()
            res_lip_mlw = res_lip_mlw.item()
            
            if args.label == 1:
                if predict_labels_0(res_eye_lw):
                    cnt_eye_lw += 1
                if predict_labels_0(res_cheek_lw):
                    cnt_cheek_lw += 1
                if predict_labels_0(res_lip_lw):
                    cnt_lip_lw += 1
                
                if predict_labels_0(res_eye_mlw):
                    cnt_eye_mlw += 1
                if predict_labels_0(res_cheek_mlw):
                    cnt_cheek_mlw += 1
                if predict_labels_0(res_lip_mlw):
                    cnt_lip_mlw += 1
                    
                if predict_labels_0(res_eye_lw) or predict_labels_0(res_cheek_lw) or predict_labels_0(res_lip_lw):
                    cnt_lw += 1
                
                if predict_labels_0(res_eye_mlw) or predict_labels_0(res_cheek_mlw) or predict_labels_0(res_lip_mlw):
                    cnt_mlw += 1
            
                    
            total += 1
                
            print('With min_loss_weights_aka_15:')
            print(f'{image_name}:\n\teye: {res_eye_mlw:.6f}\n\tcheek: {res_cheek_mlw:.6f}\n\tlip: {res_lip_mlw:.6f}')
            print('With last_weights_aka_29:')
            print(f'{image_name}:\n\teye: {res_eye_lw:.6f}\n\tcheek: {res_cheek_lw:.6f}\n\tlip: {res_lip_lw:.6f}')
            print('\n')
            
            
if args.label == 1:
    print(f"MLW: {(cnt_mlw / total * 100):.2f} %")
    print(f"\teye_MLW: {(cnt_eye_mlw / total * 100):.2f} %")
    print(f"\tcheek_MLW: {(cnt_cheek_mlw / total * 100):.2f} %")
    print(f"\tlip_MLW: {(cnt_lip_mlw / total * 100):.2f} %")

    print(f"LW: {(cnt_lw / total * 100):.2f} %")
    print(f"\teye_LW: {(cnt_eye_lw / total * 100):.2f} %")
    print(f"\tcheek_LW: {(cnt_cheek_lw / total * 100):.2f} %")
    print(f"\tlip_LW: {(cnt_lip_lw / total * 100):.2f} %")
# else:
#     print(f"MLW: {(cnt_mlw / total * 100):.2f} %")
#     print(f"eye_MLW: {(cnt_eye_mlw / total * 100):.2f} %")
#     print(f"cheek_MLW: {(cnt_cheek_mlw / total * 100):.2f} %")
#     print(f"lip_MLW: {(cnt_lip_mlw / total * 100):.2f} %")

#     print(f"LW: {(cnt_lw / total * 100):.2f} %")
#     print(f"eye_LW: {(cnt_eye_lw / total * 100):.2f} %")
#     print(f"cheek_LW: {(cnt_cheek_lw / total * 100):.2f} %")
#     print(f"lip_LW: {(cnt_lip_lw / total * 100):.2f} %")
