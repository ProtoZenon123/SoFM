import os
import argparse

import torch.nn as nn
import torch
import torch.nn.functional as func
import torchvision.transforms as transforms
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader

from utils import load_config, make_grid_of_image, create_image_grid
from siamese_cnn import siameseCNN_BCE, siameseCNN_Contrast_Origin, siameseCNN_Contrast_Inter
import dataset
from loss import contrastive_loss
from final_crop_and_transform import crop_and_transform_six_parts

def convert_to_pilimage(opencv_image):
    pilimage = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(pilimage)

def test(args, input_folder, output_folder):
    in_folder = input_folder
    out_folder += f"/{args.component}/{args.model_type}"
    os.makedirs(out_folder, exist_ok=True)
    config = load_config()
    
    num_features = config['num_features']
    channels_img = config['channels_img']
    threshold = config['threshold']
    cuda_id = config['cuda_id']
    margin = config['margin']
    folder_weights_dir = config['folder_weights_dir']
    
    folder_weights_dir += f"/{args.component}/{args.model_type}"
    
    # transform
    transform = None
    if args.model_type == "BCE" or args.model_type == "contrast_inter":
        transform = transforms.Compose([transforms.Resize((121, 121)),
                                        transforms.ToTensor()
                                        ])
    if args.model_type == "contrast_origin":
        transform = transforms.Compose([transforms.Resize((100, 100)),
                                        transforms.ToTensor()
                                        ])
    
    cuda = 'cuda:' + f'{cuda_id}'
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    
    sia_model = None
    if args.model_type == "BCE":
        sia_model = siameseCNN_BCE(channels_img, num_features)
    if args.model_type == "contrast_origin":
        sia_model = siameseCNN_Contrast_Origin(channels_img, num_features)
    if args.model_type == "contrast_inter":
        sia_model = siameseCNN_Contrast_Inter(channels_img, num_features)
    sia_model.to(device)
    
    for weights in folder_weights_dir:
        weights_path = os.path.join(folder_weights_dir, weights)
        sia_model.load_state_dict(torch.load(weights_path))
        sia_model.eval()
        
        out_folder_eachw = os.path.join(out_folder, f'{weights}')
        res_txt_path = os.path.join(out_folder_eachw, 'res.txt')
        os.makedirs(res_txt_path, exist_ok=True)
        
        print(f"USING {weights_path}")
        
        total = 0
        cnt_eye = 0
        cnt_cheek = 0
        cnt_lip = 0
        total_eye = 0
        total_cheek = 0
        total_lip = 0
        
        with torch.no_grad():
            for image_name in tqdm(os.listdir(in_folder), desc='testing: '):
                image = cv2.imread(os.path.join(in_folder, image_name))
                if image is not None:
                    left_eye, right_eye, left_cheek, right_cheek, left_lip, right_lip = crop_and_transform_six_parts(image)
                    left_eye = convert_to_pilimage(left_eye)
                    right_eye = convert_to_pilimage(right_eye)
                    left_cheek = convert_to_pilimage(left_cheek)
                    right_cheek = convert_to_pilimage(right_cheek)
                    left_lip = convert_to_pilimage(left_lip)
                    right_lip = convert_to_pilimage(right_lip)

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
                    res_eye = None
                    res_cheek = None
                    res_lip = None
                    if args.model_type == 'BCE':
                        if args.component == 'all':
                            res_eye = sia_model(left_eye, right_eye).item()
                            res_cheek = sia_model(left_cheek, right_cheek).item()
                            res_lip = sia_model(left_lip, right_lip).item()
                        elif args.component == 'eye':
                            res_eye = sia_model(left_eye, right_eye).item()
                        elif args.component == 'cheek':
                            res_cheek = sia_model(left_cheek, right_cheek).item()
                        else:
                            res_lip = sia_model(left_lip, right_lip).item()
                    else:
                        if args.component == 'all':
                            emb_vtr_1, emb_vtr_2 = sia_model(left_eye, right_eye)
                            res_eye = func.pairwise_distance(emb_vtr_1, emb_vtr_2).item()
                            emb_vtr_1, emb_vtr_2 = sia_model(left_cheek, right_cheek)
                            res_cheek = func.pairwise_distance(emb_vtr_1, emb_vtr_2).item()
                            emb_vtr_1, emb_vtr_2 = sia_model(left_lip, right_lip)
                            res_lip = func.pairwise_distance(emb_vtr_1, emb_vtr_2).item()
                        elif args.component == 'eye':
                            emb_vtr_1, emb_vtr_2 = sia_model(left_eye, right_eye)
                            res_eye = func.pairwise_distance(emb_vtr_1, emb_vtr_2).item()
                        elif args.component == 'cheek':
                            emb_vtr_1, emb_vtr_2 = sia_model(left_cheek, right_cheek)
                            res_cheek = func.pairwise_distance(emb_vtr_1, emb_vtr_2).item()
                        else:
                            emb_vtr_1, emb_vtr_2 = sia_model(left_lip, right_lip)
                            res_lip = func.pairwise_distance(emb_vtr_1, emb_vtr_2).item()

                    if args.label == 0:
                        mistake_code = image_name.split('.')[0]
                        mistake_code = mistake_code.split('_')[-1]
                        
                        if mistake_code[0] != 0 and res_eye is not None:
                            total_eye += 1
                            if args.model_type == 'BCE':
                                cnt_eye += int(res_eye > threshold)
                            else:
                                cnt_eye += int(res_eye <= margin)
                            with open(res_txt_path, 'a') as file:
                                file.write(f'{args.model_tpye} with {weights}:')
                                file.write(f'\t{image_name}\n')
                                file.write(f'\teye: {res_eye}\n')
                        if mistake_code[1] != 0 and res_cheek is not None:
                            total_cheek += 1
                            if args.model_type == 'BCE':
                                cnt_cheek += int(res_cheek > threshold)
                            else:
                                cnt_cheek += int(res_cheek <= margin)
                            with open(res_txt_path, 'a') as file:
                                file.write(f'{args.model_tpye} with {weights}:')
                                file.write(f'\t{image_name}\n')
                                file.write(f'\tcheek: {res_cheek}\n')
                        if mistake_code[2] != 0 and res_lip is not None:
                            total_lip += 1
                            if args.model_type == 'BCE':
                                cnt_lip += int(res_lip > threshold)
                            else:
                                cnt_lip += int(res_lip <= margin)
                            with open(res_txt_path, 'a') as file:
                                file.write(f'{args.model_tpye} with {weights}:')
                                file.write(f'\t{image_name}\n')
                                file.write(f'\tlip: {res_lip}\n\n')
                    else:
                        total += 1
                        if args.model_type == 'BCE':
                            cnt_eye += int(res_eye >= threshold)
                            cnt_cheek += int(res_cheek >= threshold)
                            cnt_lip += int(res_lip >= threshold)
                            with open(res_txt_path, 'a') as file:
                                file.write(f'{args.model_tpye} with {weights}:')
                                file.write(f'\t{image_name}\n')
                                file.write(f'\teye: {res_eye}\n')
                                file.write(f'\tcheek: {res_cheek}\n')
                                file.write(f'\tlip: {res_lip}\n\n')
                        else:
                            cnt_eye += int(res_eye <= margin)
                            cnt_cheek += int(res_cheek <= margin)
                            cnt_lip += int(res_lip <= margin)
                            with open(res_txt_path, 'a') as file:
                                file.write(f'{args.model_tpye} with {weights}:')
                                file.write(f'\t{image_name}\n')
                                file.write(f'\teye: {res_eye}\n')
                                file.write(f'\tcheek: {res_cheek}\n')
                                file.write(f'\tlip: {res_lip}\n\n')
            print('RESULTS:')
            if args.label == 0:
                print(f'\teye:{(cnt_eye/total_eye*100):.2f}%')
                print(f'\teye:{(cnt_cheek/total_cheek*100):.2f}%')
                print(f'\teye:{(cnt_lip/total_lip*100):.2f}%')
            else:
                print(f'\teye:{(cnt_eye/total*100):.2f}%')
                print(f'\teye:{(cnt_cheek/total*100):.2f}%')
                print(f'\teye:{(cnt_lip/total*100):.2f}%')
                    
if __name__ == '__main__':
    parser = argparse.ArgumentParser("setting before testing")
    parser.add_argument("--component", type=str,
                        choices=["eye", "cheek", "lip", "all"],
                        help="Choose which part of faces dataset to use (eye, cheek, lip, or all)",
                        default="all")
    parser.add_argument("--model_type", type=str,
                        choices=["BCE", "contrast_origin", "contrast_inter"],
                        help="Specify the type of model",
                        default="BCE")
    parser.add_argument("--label", type=int,
                        choices=[0, 1],
                        default=1)
    args = parser.parse_args()
    
    input_folder = ''
    output_folder = ''
    test(args,input_folder, output_folder)