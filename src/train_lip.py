from utils import load_config_lip, plot, devide_data
import model
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
import dataset_lip
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import os
import time
import argparse

def predict_labels(outputs, threshold=0.5):
    return (outputs >= threshold)

def train_each_epoch(train_dataloader, val_dataloader, model, criterion, optimizer, device, threshold):
    model = model.to(device)
    model.train()
    
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0
    num_correct_labels = 0
    total = 0
    threshold = threshold

    for pimage in tqdm(train_dataloader, desc="Training"):
        
        image1 = pimage[0]
        image2 = pimage[1]
        label = pimage[2].float()
        label = label.view(len(label), -1)
        
        image1 = image1.to(device)
        image2 = image2.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(image1, image2)
        predict_label = predict_labels(output, threshold)
        total += len(label)
        num_correct_labels += (predict_label == label).sum().item()

        loss = criterion(output, label)
        epoch_train_loss += loss.item()
        loss.backward()
        optimizer.step()

    accuracy_train = num_correct_labels / total * 100
    epoch_train_loss /= total

    model.eval()
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

            output = model(image1, image2)
            predict_label = predict_labels(output, threshold)
            total += len(label)
            num_correct_labels += (predict_label == label).sum().item()

            loss = criterion(output, label)
            epoch_val_loss += loss.item()

    accuracy_val = num_correct_labels / total * 100
    epoch_val_loss /= total

    return epoch_train_loss, epoch_val_loss, accuracy_train, accuracy_val


def training(args):
    config = load_config_lip()

    train_batch_size = config['train_batch_size']
    train_num_epochs = config['train_num_epochs']
    learning_rate = config['learning_rate']
    num_features = config['num_features']
    channels_img = config['channels_img']
    threshold = config['threshold']
    cuda_id = config['cuda_id']
    train_dir = config['train_dir']
    folder_weights_dir = config['folder_weights_dir']
    global figures_dir
    figures_dir = config['figures_dir']
    
    os.makedirs(folder_weights_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    base_dir = os.path.dirname(train_dir)
    validation_dir = os.path.join(base_dir, 'val_data')

    min_loss_weights_path = os.path.join(folder_weights_dir, "min_loss_weights.pth")
    last_weight_path = os.path.join(folder_weights_dir, "last_weights.pth")
    
    # Split the training data to create validation data and test data (depend on mode_split_data)
    if args.mode_split_data == 0:
        devide_data(train_dir)

    # transform = transforms.Compose([transforms.Resize((121, 121)),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize([0.485, 0.456, 0.406],
    #                                                      [0.229, 0.224, 0.225]) 
    #                                 ])
    transform = transforms.Compose([transforms.Resize((121, 121)),
                                    transforms.ToTensor()
                                    ])

    print("< Train dataset >")
    train_data = dataset_lip.SiameseNetworkDataset(image_folder_dir=train_dir,
                                               transform=transform)
    print("< Validation dataset >")
    val_data = dataset_lip.SiameseNetworkDataset(image_folder_dir=validation_dir,
                                             transform=transform)

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=train_batch_size,
                                  shuffle=True)
    val_dataloader = DataLoader(dataset=val_data,
                                batch_size=train_batch_size,
                                shuffle=True)

    cuda = 'cuda:' + f'{cuda_id}'
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    sia_model = model.siameseNN(channels_img, num_features)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(sia_model.parameters(), lr=learning_rate)

    # if pretraining available
    if args.mode_train == 1:
        sia_model.load_state_dict(torch.load(last_weight_path))
        train_num_epochs = 15
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
    
    print("PATH:")
    print(f'\t weights dir: {folder_weights_dir}')
    print(f'\t figures dir: {figures_dir}')

    print("START TRAINING:")
    start_time = time.time()
    for epoch in range(train_num_epochs):
        print(f"Epoch number {epoch}:")
        epoch_train_loss, epoch_val_loss, accuracy_train, accuracy_val = train_each_epoch(train_dataloader,
                                                                                          val_dataloader,
                                                                                          sia_model,
                                                                                          criterion,
                                                                                          optimizer,
                                                                                          device,
                                                                                          threshold)

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
        if epoch_train_loss < min_loss_train:
            min_loss_train = epoch_train_loss
            print(f'\tNOTE: Get the last min_loss_weights at epoch {epoch}')
            torch.save(sia_model.state_dict(), min_loss_weights_path)
        torch.save(sia_model.state_dict(), last_weight_path)
        if (epoch >= 0 and epoch % 5 == 0) or epoch == 4 or epoch == 5 or epoch == 6 or epoch == 7:
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


def test(test_dataloader, model, criterion, device, threshold):
    model = model.to(device)
    model.eval()

    test_loss = 0.0
    num_correct_labels = 0
    total = 0
    threshold = threshold

    with torch.no_grad():
        for pimage in tqdm(test_dataloader, desc="Testing: "):
            image1 = pimage[0]
            image2 = pimage[1]
            label = pimage[2].float()
            label = label.view(len(label), -1)

            image1 = image1.to(device)
            image2 = image2.to(device)
            label = label.to(device)

            outputs = model(image1, image2)
            predict_label = predict_labels(outputs, threshold)

            total += len(label)
            num_correct_labels += (predict_label == label).sum().item()
            test_loss += criterion(outputs, label).item()
            
    accuracy = num_correct_labels / total * 100
    test_loss /= len(test_dataloader.dataset)
    
    return accuracy, test_loss

def evaluate():
    config = load_config_lip()

    test_batch_size = config['train_batch_size']
    num_features = config['num_features']
    channels_img = config['channels_img']
    threshold = config['threshold']
    cuda_id = config['cuda_id']
    train_dir = config['train_dir']
    folder_weights_dir = config['folder_weights_dir']

    base_dir = os.path.dirname(train_dir)
    test_dir = os.path.join(base_dir, 'test_data')

    transform = transforms.Compose([transforms.Resize((121, 121)),
                                    transforms.ToTensor()
                                    ])

    print("< Test dataset >")
    data = dataset_lip.SiameseNetworkDataset(image_folder_dir=test_dir,
                                         transform=transform)

    dataloader = DataLoader(dataset=data,
                            batch_size=test_batch_size,
                            shuffle=True)
    cuda = 'cuda:' + f'{cuda_id}'
    criterion = nn.BCELoss()
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")

    model_last_weight = model.siameseNN(channels_img, num_features)
    model_min_loss_weight = model.siameseNN(channels_img, num_features)

    last_weights_path = os.path.join(folder_weights_dir, 'last_weights.pth')
    min_loss_weight_path = os.path.join(
        folder_weights_dir, 'min_loss_weights.pth')

    model_last_weight.load_state_dict(torch.load(last_weights_path))
    model_min_loss_weight.load_state_dict(torch.load(min_loss_weight_path))

    accuracy_fnw, loss_fnw = test(dataloader, model_last_weight, criterion, device, threshold)
    print(f"Accuracy with final weight: {accuracy_fnw:.2f}%")
    print(f"Loss with final weight: {loss_fnw:.8f}")
    accuracy_mlw, loss_mlw = test(dataloader, model_min_loss_weight, criterion, device, threshold)
    print(f"Accuracy with min_loss weight: {accuracy_mlw:.2f}%")
    print(f"Loss with min_loss weight: {loss_mlw:.8f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("set mode_split_data")
    parser.add_argument("--mode_split_data", type=int,
                        help="mode_split_data indicates whether test_data and val_data have been set up",
                        default=1)
    parser.add_argument("--mode_train", type=int,
                        help="mode_train indicates whether the model is being trained from scratch or continues training with the existing weights.",
                        default=0)
    args = parser.parse_args()
    #----------------------------------------- TRAINING ----------------------------------------------
    try:
        training(args)
    except KeyboardInterrupt:
        print('Training Interrupted')
    
    #---------------------------------------- ILLUSTRATION ---------------------------------------------
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
    evaluate()
