import os
import time
import argparse
import torch.nn as nn
import torch
import torch.nn.functional as func
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from utils import config_manager, plot
from models.siamese_cnn import siameseCNN_BCE, siameseCNN_Contrast_Origin, siameseCNN_Contrast_Inter
import dataset
from contrastive_loss import contrastive_loss

import logging
parser = argparse.ArgumentParser("setting before start training")
parser.add_argument("--mode_split_data", type=int,
                    help="mode_split_data indicates whether test_data and val_data have been set up",
                    default=1)
parser.add_argument("--is_pretrained", type=int,
                    help="is_pretrained indicates whether the model is being trained from scratch or continues training with the existing weights.",
                    default=0)
parser.add_argument("--component", type=str,
                    choices=["eye", "cheek", "lip", "all"],
                    help="Choose which part of faces dataset to use (eye, cheek, lip, or all)",
                    default="all")
parser.add_argument("--model_type", type=str,
                    choices=["BCE", "contrast_origin", "contrast_inter"],
                    help="Specify the type of model",
                    default="contrast_inter")

parser.add_argument("--image_test", type = str, default="/mnt/2E287FDD287FA28F/teamAI/hong/01_PINK/res_0603/FOR_test_pose/CROPPED/tilted/CSD")
parser.add_argument("--margin", type=float)
parser.add_argument("--folder_weight", type = str, default = "/mnt/2E287FDD287FA28F/teamAI/MinhQuan/siamesenn/Symetry-of-Facial-Makeup/SoFM/WEIGHTS_ALL/2203_data_111/MIX")
args = parser.parse_args()


current_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
# log_dir = "/mnt/2E287FDD287FA28F/teamAI/MinhQuan/siamesenn/Symetry-of-Facial-Makeup/SoFM/FIGURES_ALL/LOG_test/wA_dtB"
log_dir = "/media/ivsr/data2/DATASET_A/log/test"

os.makedirs(log_dir, exist_ok = True)
log_filename = os.path.join(log_dir, f'TEST_{args.folder_weight.split(os.sep)[-1]}_{args.model_type}_{args.margin}_{args.image_test.split(os.sep)[-2]}_{args.image_test.split(os.sep)[-1]}_44.txt')
# Set up logging with timestamped filename
# log_filename = os.path.join(log_dir, f'TESTING_log_{current_timestamp}.txt')
logging.basicConfig(filename=log_filename,
                    level=logging.INFO,
                    format='%(asctime)s - %(message)s')

# Log the start of the training session
logging.info("Starting new training session")

def predict_labels(outputs, threshold, margin, type):
    '''
        predict output depend on threshold or margin
        args:
            outputs (tensor): output of the model 
            type = 0: using threshold
            type = 1: using margin
    '''
    if type == 'BCE':
        return outputs >= threshold
    if type == 'contrast_origin' or type == 'contrast_inter':
        return outputs < margin
    
def testing(test_dataloader,
            model, criterion, device,
            threshold_, margin_, model_type_):
    
    model.eval()
    threshold = threshold_
    margin = margin_
    model_type = model_type_
    
    test_loss = 0.0
    num_correct_labels = 0
    total = 0

    with torch.no_grad():
        for pimage in tqdm(test_dataloader, desc="Testing: "):
            image1 = pimage[0]
            image2 = pimage[1]
            label = pimage[2].float()
            label = label.view(len(label), -1)

            image1 = image1.to(device)
            image2 = image2.to(device)
            label = label.to(device)
            total += len(label)

            if model_type == 'BCE':
                output = model(image1, image2)
                predict_label = predict_labels(output, threshold, margin, model_type)
                num_correct_labels += (predict_label == label).sum().item()
                
                BCE_loss = criterion(output, label)
                test_loss += BCE_loss.item()
                
            if model_type == 'contrast_origin' or model_type == 'contrast_inter':
                output_01, output_02 = model(image1, image2)
                euclidean_distance = func.pairwise_distance(output_01, output_02)
                euclidean_distance = euclidean_distance.view(len(euclidean_distance), -1)
                predict_label = predict_labels(euclidean_distance, threshold, margin, model_type)
                num_correct_labels += (predict_label == label).sum().item()
                
                cont_loss = criterion(output_01, output_02, label)
                test_loss += cont_loss.item()
            
    accuracy = num_correct_labels / total * 100
    test_loss /= len(test_dataloader.dataset)
    
    return accuracy, test_loss

def evaluate(args):
    config = config_manager()
    test_dir = args.image_test
    test_batch_size = config['train_batch_size']
    num_features = config['num_features']
    channels_img = config['channels_img']
    threshold = config['threshold']
    # cuda_id = config['cuda_id']
    cuda_id = 0
    # margin = config['margin']
    margin = args.margin
    folder_weights_dir = args.folder_weight
    # folder_weights_dir = config['folder_weights_dir']
    
    folder_weights_dir += f"/{args.component}/{args.model_type}"
    if args.model_type == "contrast_inter" or args.model_type == "contrast_origin":
        folder_weights_dir += f"/{margin}"
    os.makedirs(folder_weights_dir, exist_ok=True)


    # train_dir = config['train_dir']
    # base_dir = os.path.dirname(train_dir)
    # test_dir = os.path.join(base_dir, 'test_data')
    
    min_loss_weights_filename = "min_loss_weights.pth"
    last_weights_filename = "last_weights.pth"
    min_loss_weights_path = os.path.join(folder_weights_dir, min_loss_weights_filename)
    last_weights_path = os.path.join(folder_weights_dir, last_weights_filename)

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

    # dataset
    
    
    data = dataset.SiameseNetworkDataset(image_folder_dir=test_dir,
                                         transform=transform,
                                         component=args.component)

    # dataloader
    dataloader = DataLoader(dataset=data,
                            batch_size=test_batch_size,
                            shuffle=True)
    
    cuda = 'cuda:' + f'{cuda_id}'
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    
    # model
    sia_model = None
    if args.model_type == "BCE":
        sia_model = siameseCNN_BCE(channels_img, num_features)
    if args.model_type == "contrast_origin":
        sia_model = siameseCNN_Contrast_Origin(channels_img, num_features)
    if args.model_type == "contrast_inter":
        sia_model = siameseCNN_Contrast_Inter(channels_img, num_features)
        
    # sia_model_2 = None
    # if args.model_type == "BCE":
    #     sia_model_2 = siameseCNN_BCE(channels_img, num_features)
    # if args.model_type == "contrast_origin":
    #     sia_model_2 = siameseCNN_Contrast_Origin(channels_img, num_features)
    # if args.model_type == "contrast_inter":
    #     sia_model_2 = siameseCNN_Contrast_Inter(channels_img, num_features)
    
    # criterion
    criterion = None
    if args.model_type == "BCE":
        criterion = nn.BCELoss()
    if args.model_type == "contrast_inter" or args.model_type == "contrast_origin":
        criterion = contrastive_loss(margin)

    sia_model.load_state_dict(torch.load(last_weights_path))
    sia_model = sia_model.to(device)
    accuracy, loss = testing(dataloader, sia_model, criterion, device, threshold, margin, args.model_type)
    logging.info(f"Type model * type dts: {log_dir.split(os.sep)[-1]}")
    logging.info(f"Model: {args.folder_weight.split(os.sep)[-1]}_{args.model_type}_{args.margin}")
    logging.info(f"Data test: {args.image_test.split(os.sep)[-2]}")
    logging.info(f"Accuracy with last weight: {accuracy:.2f}%")
    print(f"Accuracy with last weight: {accuracy:.2f}%")
    logging.info(f"Loss with last weight: {loss:.8f}")
    
    sia_model.load_state_dict(torch.load(min_loss_weights_path))
    sia_model = sia_model.to(device)
    accuracy, loss = testing(dataloader, sia_model, criterion, device, threshold, margin, args.model_type)
    logging.info(f"Accuracy with min_loss weight: {accuracy:.2f}%")
    print(f"Accuracy with min_loss weight: {accuracy:.2f}%")
    logging.info(f"Loss with min_loss weight: {loss:.8f}")

# if __name__ == '__main__':
    # parser = argparse.ArgumentParser("setting before start training")
    # parser.add_argument("--mode_split_data", type=int,
    #                     help="mode_split_data indicates whether test_data and val_data have been set up",
    #                     default=1)
    # parser.add_argument("--is_pretrained", type=int,
    #                     help="is_pretrained indicates whether the model is being trained from scratch or continues training with the existing weights.",
    #                     default=0)
    # parser.add_argument("--component", type=str,
    #                     choices=["eye", "cheek", "lip", "all"],
    #                     help="Choose which part of faces dataset to use (eye, cheek, lip, or all)",
    #                     default="all")
    # parser.add_argument("--model_type", type=str,
    #                     choices=["BCE", "contrast_origin", "contrast_inter"],
    #                     help="Specify the type of model",
    #                     default="contrast_inter")
    # args = parser.parse_args()
 
logging.info("Evaluate with two model_weights:")
evaluate(args)
