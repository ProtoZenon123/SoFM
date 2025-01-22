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

from utils import load_config, plot
from siamese_cnn import siameseCNN_BCE, siameseCNN_Contrast_Origin, siameseCNN_Contrast_Inter
import dataset
from loss import contrastive_loss

def predict_labels(outputs, threshold, margin, type):
    '''
        type = 0: using threshold
        type = 1: using margin
    '''
    if type == 'BCE':
        return outputs >= threshold
    if type == 'contrast_origin' or type == 'contrast_inter':
        return outputs < margin

def train_each_epoch(train_dataloader, val_dataloader,
                     model, criterion, optimizer, device,
                     threshold_, margin_, model_type_):
    '''
        model_type_ = 0: using BCE loss
        model_type_ = 1: using contrastive loss
    '''
    model.train()
    threshold = threshold_
    margin = margin_
    model_type = model_type_
    
    epoch_train_loss = 0.0
    num_correct_labels = 0
    total = 0

    for pimage in tqdm(train_dataloader, desc="Training"):
        image1 = pimage[0]
        image2 = pimage[1]
        label = pimage[2].float()
        label = label.view(len(label), -1)
        
        image1 = image1.to(device)
        image2 = image2.to(device)
        label = label.to(device)
        total += len(label)

        optimizer.zero_grad()
        
        if model_type == 'BCE':
            output = model(image1, image2)
            predict_label = predict_labels(output, threshold, margin, model_type)
            num_correct_labels += (predict_label == label).sum().item()
            
            BCE_loss = criterion(output, label)
            epoch_train_loss += BCE_loss.item()
            BCE_loss.backward()
            
        if model_type == 'contrast_origin' or model_type == 'contrast_inter':
            output_01, output_02 = model(image1, image2)
            euclidean_distance = func.pairwise_distance(output_01, output_02)
            euclidean_distance = euclidean_distance.view(len(euclidean_distance), -1)
            predict_label = predict_labels(euclidean_distance, threshold, margin, model_type)
            num_correct_labels += (predict_label == label).sum().item()
            
            cont_loss = criterion(output_01, output_02, label)
            epoch_train_loss += cont_loss.item()
            cont_loss.backward()
            
        optimizer.step()

    accuracy_train = num_correct_labels / total * 100
    epoch_train_loss /= total

    model.eval()
    epoch_val_loss = 0
    num_correct_labels = 0
    total = 0
    with torch.no_grad():
        for pimage in tqdm(val_dataloader, desc="Validating"):
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
                epoch_val_loss += BCE_loss.item()
                
            if model_type == 'contrast_origin' or model_type == 'contrast_inter':
                output_01, output_02 = model(image1, image2)
                euclidean_distance = func.pairwise_distance(output_01, output_02)
                euclidean_distance = euclidean_distance.view(len(euclidean_distance), -1)
                predict_label = predict_labels(euclidean_distance, threshold, margin, model_type)
                num_correct_labels += (predict_label == label).sum().item()
                
                cont_loss = criterion(output_01, output_02, label)
                epoch_val_loss += cont_loss.item()

    accuracy_val = num_correct_labels / total * 100
    epoch_val_loss /= total

    return epoch_train_loss, epoch_val_loss, accuracy_train, accuracy_val

def training(args):
    
    # load config
    config = load_config()
    train_batch_size = config['train_batch_size']
    train_num_epochs = config['train_num_epochs']
    learning_rate = config['learning_rate']
    num_features = config['num_features']
    channels_img = config['channels_img']
    threshold = config['threshold']
    cuda_id = config['cuda_id']
    margin = config['margin']
    weight_decay = config['weight_decay']
    train_dir = config['train_dir']
    folder_weights_dir = config['folder_weights_dir']
    global figures_dir
    figures_dir = config['figures_dir']
    
    folder_weights_dir += f"/{args.component}/{args.model_type}/{margin}"
    figures_dir += f"/{args.component}/{args.model_type}/{margin}"
    os.makedirs(folder_weights_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    base_dir = os.path.dirname(train_dir)
    validation_dir =os.path.join(base_dir, 'val_data')

    min_loss_weights_filename = "min_loss_weights.pth"
    last_weight_filename = "last_weights.pth"
    min_loss_weights_path = os.path.join(folder_weights_dir, min_loss_weights_filename)
    last_weight_path = os.path.join(folder_weights_dir, last_weight_filename)
    
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
    print("< Train dataset >")
    train_data = dataset.SiameseNetworkDataset(image_folder_dir=train_dir,
                                               transform=transform,
                                               component=args.component)
    print("< Validation dataset >")
    val_data = dataset.SiameseNetworkDataset(image_folder_dir=validation_dir,
                                             transform=transform,
                                             component=args.component)

    # data loader
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=train_batch_size,
                                  shuffle=True)
    val_dataloader = DataLoader(dataset=val_data,
                                batch_size=train_batch_size,
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
    print(sia_model)
    
    # criterion
    criterion = None
    if args.model_type == "BCE":
        criterion = nn.BCELoss()
    if args.model_type == "contrast_inter" or args.model_type == "contrast_origin":
        criterion = contrastive_loss(margin)

    # optimizer
    optimizer = None
    if args.model_type == "BCE" or args.model_type == "contrast_inter":
        optimizer = optim.Adam(sia_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if args.model_type == "contrast_origin":
        optimizer = optim.Adam(sia_model.parameters(), lr=0.0005)
    # if pretraining available
    if args.is_pretrained == 1:
        sia_model.load_state_dict(torch.load(last_weight_path))
        train_num_epochs = 10
        
    ##################################################
    ###                  TRAINING                  ###
    ##################################################
    global list_train_loss
    global list_val_loss
    global list_acc_train
    global list_acc_val
    list_train_loss = []
    list_val_loss = []
    list_acc_train = []
    list_acc_val = []
    min_loss_train = 0.0

    print("HYPER PARAM:")
    print(f'\tBATCH_SIZE: {train_batch_size}')
    print(f'\tNUM_EPOCH: {train_num_epochs}')
    print(f'\tDEVICE: {cuda}')
    print(f'\tCHANNELS_IMAGE: {channels_img}')
    print(f'\tNUM_FEATURES: {num_features}')
    print(f'\tMARGIN: {margin}')
    print(f'\tTHRESHOLD: {threshold}')
    print(f'\tCRITERION: {criterion}')
    print(f'\tOPTIMIZER: {optimizer}')

    print("START TRAINING:")
    start_time = time.time()
    
    sia_model = sia_model.to(device)
    for epoch in range(train_num_epochs):
        print(f"Epoch number {epoch}:")
        epoch_train_loss, epoch_val_loss, accuracy_train, accuracy_val = train_each_epoch(train_dataloader,
                                                                                          val_dataloader,
                                                                                          sia_model,
                                                                                          criterion,
                                                                                          optimizer,
                                                                                          device,
                                                                                          threshold,
                                                                                          margin,
                                                                                          args.model_type)

        list_train_loss.append(epoch_train_loss)
        list_val_loss.append(epoch_val_loss)
        list_acc_train.append(accuracy_train)
        list_acc_val.append(accuracy_val)

        print(f"train_loss = {epoch_train_loss}")
        print(f"val_loss =   {epoch_val_loss}")
        print(f"accuracy training:   {accuracy_train:.2f}%")
        print(f"accuracy validation: {accuracy_val:.2f}%")

        if epoch == 0:
            min_loss_train = epoch_train_loss
        if epoch_train_loss <= min_loss_train:
            min_loss_train = epoch_train_loss
            print(f'\tNOTE: Get the last min_loss_weights at epoch {epoch}')
            torch.save(sia_model.state_dict(), min_loss_weights_path)
        torch.save(sia_model.state_dict(), last_weight_path)
        torch.save(sia_model.state_dict(), f"{last_weight_path}_{epoch}")

    end_time = time.time()
    total_time = end_time - start_time

    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("Training finished.")
    print(
        f"The smallest loss achieved is {min_loss_train} at epoch {list_train_loss.index(min_loss_train)}.")
    print(
        f"Total time of training: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds.")
    print("--------------------------------------------------------------------------------------")

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
    config = load_config()

    test_batch_size = config['train_batch_size']
    num_features = config['num_features']
    channels_img = config['channels_img']
    threshold = config['threshold']
    cuda_id = config['cuda_id']
    margin = config['margin']
    train_dir = config['train_dir']
    folder_weights_dir = config['folder_weights_dir']
    
    folder_weights_dir += f"/{args.component}/{args.model_type}/{margin}"
    os.makedirs(folder_weights_dir, exist_ok=True)

    base_dir = os.path.dirname(train_dir)
    test_dir = os.path.join(base_dir, 'test_data')
    
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
    print("< Test dataset >")
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
        
    sia_model_2 = None
    if args.model_type == "BCE":
        sia_model_2 = siameseCNN_BCE(channels_img, num_features)
    if args.model_type == "contrast_origin":
        sia_model_2 = siameseCNN_Contrast_Origin(channels_img, num_features)
    if args.model_type == "contrast_inter":
        sia_model_2 = siameseCNN_Contrast_Inter(channels_img, num_features)
    
    # criterion
    criterion = None
    if args.model_type == "BCE":
        criterion = nn.BCELoss()
    if args.model_type == "contrast_inter" or args.model_type == "contrast_origin":
        criterion = contrastive_loss(margin)

    sia_model.load_state_dict(torch.load(last_weights_path))
    sia_model = sia_model.to(device)
    accuracy, loss = testing(dataloader, sia_model, criterion, device, threshold, margin, args.model_type)
    print(f"Accuracy with last weight: {accuracy:.2f}%")
    print(f"Loss with last weight: {loss:.8f}")
    
    sia_model.load_state_dict(torch.load(min_loss_weights_path))
    sia_model = sia_model.to(device)
    accuracy, loss = testing(dataloader, sia_model, criterion, device, threshold, margin, args.model_type)
    print(f"Accuracy with min_loss weight: {accuracy:.2f}%")
    print(f"Loss with min_loss weight: {loss:.8f}")

if __name__ == '__main__':
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
                        default="BCE")
    args = parser.parse_args()
    #----------------------------------------- TRAINING ----------------------------------------------
    try:
        training(args)
    except KeyboardInterrupt:
        print('Training Interrupted')
    
    #------------------------------------------ FIGURE -----------------------------------------------
    train_loss_figure_path = os.path.join(figures_dir, "train_loss.jpg")
    val_loss_figure_path = os.path.join(figures_dir, "validation_loss.jpg")
    val_acc_figure_path = os.path.join(figures_dir, "validation_acc.jpg")
    train_acc_figure_path = os.path.join(figures_dir, "training_acc.jpg")
    plot(list_train_loss, "Training loss", "Epoch",
         "Loss", "Training Loss", train_loss_figure_path)
    plot(list_val_loss, "Validation loss", "Epoch",
         "Loss", "Validation Loss", val_loss_figure_path)
    plot(list_acc_train, "Training accuracy", "Epoch",
         "Loss", "Training accuracy", train_acc_figure_path)
    plot(list_acc_val, "Validation accuracy", "Epoch",
         "Loss", "Validation accuracy", val_acc_figure_path)
    
    #------------------------------------------ EVALUATE ----------------------------------------------
    print("Evaluate with two model_weights:")
    evaluate(args)
