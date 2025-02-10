import os
import argparse
from copy import deepcopy as dcopy

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

from utils import load_config, create_image_grid
from siamese_cnn import siameseCNN_BCE, siameseCNN_Contrast_Origin, siameseCNN_Contrast_Inter
import dataset
from loss import contrastive_loss
from final_crop_and_transform import crop_and_transform_six_parts

def convert_to_pilimage(opencv_image):
    '''
    Ảnh được cắt đọc bởi CV2, trước khi đưa vào model cần đọc bằng pil
    '''
    pilimage = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(pilimage)

def test(args):
    in_folder = args.input_folder
    out_folder = args.output_folder
    out_folder += f"/{args.component}/{args.model_type}"
    os.makedirs(out_folder, exist_ok=True)
    config = load_config()
    
    num_features = config['num_features']
    channels_img = config['channels_img']
    threshold = config['threshold']
    cuda_id = config['cuda_id']
    margin = config['margin']
    folder_weights_dir = config['folder_weights_dir']
    ################################# WARNING ###############################
    # folder_weights_dir = "/mnt/2E287FDD287FA28F/teamAI/MinhQuan/siamesenn/Symetry-of-Facial-Makeup/SoFM/WEIGHTS_ALL/weights_1312_datasm"
    # print(f'WARNING !!!! \nTHE DIR OF WEIGHTS FOLDER\n\t{folder_weights_dir}')
    # margin = 2.0
    ########################################################################
    
    folder_weights_dir += f"/{args.component}/{args.model_type}/{margin}"
    print(f'WARNING !!!! \nTHE DIR OF WEIGHTS FOLDER\n\t{folder_weights_dir}')

    
    in_folder = args.input_folder
    out_folder = args.output_folder
    out_folder += f"/{args.component}/{args.model_type}/{margin}"
    os.makedirs(out_folder, exist_ok=True)
    
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
    ###########################################################################
    cuda = 'cuda:' + f'{args.cuda}'
    print(f'WARNING !!!! \nCUDA:\n\t{cuda}')
    ###########################################################################
    
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    
    sia_model = None
    if args.model_type == "BCE":
        sia_model = siameseCNN_BCE(channels_img, num_features)
    if args.model_type == "contrast_origin":
        sia_model = siameseCNN_Contrast_Origin(channels_img, num_features)
    if args.model_type == "contrast_inter":
        sia_model = siameseCNN_Contrast_Inter(channels_img, num_features)
    sia_model.to(device)
    
    for weights in os.listdir(folder_weights_dir):
        weights_path = os.path.join(folder_weights_dir, weights)
        sia_model.load_state_dict(torch.load(weights_path))
        sia_model.eval()
        
        tmp = weights.split('.')
        out_folder_eachw = os.path.join(out_folder, f'{tmp[0]}_{tmp[1]}')
        os.makedirs(out_folder_eachw, exist_ok=True)
        res_txt_path_true = os.path.join(out_folder_eachw, 'true.txt')
        res_txt_path_false = os.path.join(out_folder_eachw, 'false.txt')
        res_folder_image_true = os.path.join(out_folder_eachw, 'res_image_true')
        res_folder_image_false = os.path.join(out_folder_eachw, 'res_image_false')
        os.makedirs(res_folder_image_true, exist_ok=True)
        os.makedirs(res_folder_image_false, exist_ok=True)
        
        print(f"USING {weights_path}")
        
        total = 0
        cnt_eye = 0
        cnt_cheek = 0
        cnt_lip = 0
        total_eye = 0
        total_cheek = 0
        total_lip = 0
        
        with torch.no_grad():
            for image_name in tqdm(sorted(os.listdir(in_folder)), desc='testing: '):
                image = cv2.imread(os.path.join(in_folder, image_name))
                if image is not None:
                    left_eye, right_eye, left_cheek, right_cheek, left_lip, right_lip = crop_and_transform_six_parts(image)
                    left_eye = convert_to_pilimage(left_eye)
                    right_eye = convert_to_pilimage(right_eye)
                    left_cheek = convert_to_pilimage(left_cheek)
                    right_cheek = convert_to_pilimage(right_cheek)
                    left_lip = convert_to_pilimage(left_lip)
                    right_lip = convert_to_pilimage(right_lip)

                    left_eye_ = dcopy(left_eye)
                    right_eye_ = dcopy(right_eye)
                    left_cheek_ = dcopy(left_cheek)
                    right_cheek_ = dcopy(right_cheek)
                    left_lip_ = dcopy(left_lip)
                    right_lip_ = dcopy(right_lip)
                    
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
                        
                        if mistake_code[0] != '0' and res_eye is not None:
                            total_eye += 1
                            if args.model_type == 'BCE':
                                cnt_eye += int(res_eye < threshold)
                            else:
                                cnt_eye += int(res_eye > margin)
                            res_txt_path = res_txt_path_true
                            res_folder_image = res_folder_image_true
                            if res_eye <= margin:
                                res_txt_path = res_txt_path_false
                                res_folder_image = res_folder_image_false
                            with open(res_txt_path, 'a') as file:
                                file.write(f'{args.model_type} with {weights}:\n')
                                file.write(f'\t{image_name}\n')
                                file.write(f'\teye: {res_eye}\n')
                            grid_image = create_image_grid(left_eye_, right_eye_, left_cheek_, right_cheek_, left_lip_, right_lip_, f"image {image_name}", [res_eye, res_cheek, res_lip])
                            grid_image.save(os.path.join(res_folder_image, image_name))
                        if mistake_code[1] != '0' and res_cheek is not None:
                            total_cheek += 1
                            if args.model_type == 'BCE':
                                cnt_cheek += int(res_cheek < threshold)
                            else:
                                cnt_cheek += int(res_cheek > margin)
                            res_txt_path = res_txt_path_true
                            res_folder_image = res_folder_image_true
                            if res_cheek <= margin:
                                res_txt_path = res_txt_path_false
                                res_folder_image = res_folder_image_false
                            with open(res_txt_path, 'a') as file:
                                file.write(f'{args.model_type} with {weights}:\n')
                                file.write(f'\t{image_name}\n')
                                file.write(f'\tcheek: {res_cheek}\n')
                            grid_image = create_image_grid(left_eye_, right_eye_, left_cheek_, right_cheek_, left_lip_, right_lip_, f"image {image_name}", [res_eye, res_cheek, res_lip])
                            grid_image.save(os.path.join(res_folder_image, image_name))
                        if mistake_code[2] != '0' and res_lip is not None:
                            total_lip += 1
                            if args.model_type == 'BCE':
                                cnt_lip += int(res_lip < threshold)
                            else:
                                cnt_lip += int(res_lip > margin)
                            res_txt_path = res_txt_path_true
                            res_folder_image = res_folder_image_true
                            if res_lip <= margin:
                                res_txt_path = res_txt_path_false
                                res_folder_image = res_folder_image_false
                            with open(res_txt_path, 'a') as file:
                                file.write(f'{args.model_type} with {weights}:\n')
                                file.write(f'\t{image_name}\n')
                                file.write(f'\tlip: {res_lip}\n\n')
                            grid_image = create_image_grid(left_eye_, right_eye_, left_cheek_, right_cheek_, left_lip_, right_lip_, f"image {image_name}", [res_eye, res_cheek, res_lip])
                            grid_image.save(os.path.join(res_folder_image, image_name))
                    else:
                        total += 1
                        if args.model_type == 'BCE':
                            cnt_eye += int(res_eye >= threshold)
                            cnt_cheek += int(res_cheek >= threshold)
                            cnt_lip += int(res_lip >= threshold)
                            res_txt_path = res_txt_path_false
                            res_folder_image = res_folder_image_false
                            if res_eye >= threshold and res_cheek >= threshold and res_lip >= threshold:
                                res_txt_path = res_txt_path_true
                                res_folder_image = res_folder_image_true
                            with open(res_txt_path, 'a') as file:
                                file.write(f'{args.model_type} with {weights}:\n')
                                file.write(f'\t{image_name}\n')
                                file.write(f'\teye: {res_eye}\n')
                                file.write(f'\tcheek: {res_cheek}\n')
                                file.write(f'\tlip: {res_lip}\n\n')
                            grid_image = create_image_grid(left_eye_, right_eye_, left_cheek_, right_cheek_, left_lip_, right_lip_, f"image {image_name}", [res_eye, res_cheek, res_lip])
                            grid_image.save(os.path.join(res_folder_image, image_name))
                        else:
                            cnt_eye += int(res_eye <= margin)
                            cnt_cheek += int(res_cheek <= margin)
                            cnt_lip += int(res_lip <= margin)
                            res_txt_path = res_txt_path_false
                            res_folder_image = res_folder_image_false
                            if res_eye <= margin and res_cheek <= margin and res_lip <= margin:
                                res_txt_path = res_txt_path_true
                                res_folder_image = res_folder_image_true
                            with open(res_txt_path, 'a') as file:
                                file.write(f'{args.model_type} with {weights}:\n')
                                file.write(f'\t{image_name}\n')
                                file.write(f'\teye: {res_eye}\n')
                                file.write(f'\tcheek: {res_cheek}\n')
                                file.write(f'\tlip: {res_lip}\n\n')
                            grid_image = create_image_grid(left_eye_, right_eye_, left_cheek_, right_cheek_, left_lip_, right_lip_, f"image {image_name}", [res_eye, res_cheek, res_lip])
                            grid_image.save(os.path.join(res_folder_image, image_name))
            print('RESULTS:')
            if args.label == 0:
                if total_eye != 0:
                    print(f'\teye:{(cnt_eye/total_eye*100):.2f}%')
                if total_cheek != 0:
                    print(f'\tcheek:{(cnt_cheek/total_cheek*100):.2f}%')
                if total_lip != 0:
                    print(f'\tlip:{(cnt_lip/total_lip*100):.2f}%')
            else:
                if total != 0:
                    print(f'\teye:{(cnt_eye/total*100):.2f}%')
                    print(f'\tcheek:{(cnt_cheek/total*100):.2f}%')
                    print(f'\tlip:{(cnt_lip/total*100):.2f}%')
                    
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
    parser.add_argument("--cuda", type=int,
                        default=1)
    parser.add_argument("--input_folder", type=str)
    parser.add_argument("--output_folder", type=str)
    args = parser.parse_args()
    
    # input_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/FACE_DATA/DATA_700/DATA_700_LB1/700_elegant'
    # output_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/FACE_DATA/RESULT_700/test_0712/0712'
    test(args)