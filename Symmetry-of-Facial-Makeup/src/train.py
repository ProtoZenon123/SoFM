import os
from dotenv import load_dotenv
import logging

import torch
import torch.nn.functional as func
import torchvision.transforms as transforms
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader

from utils import measuretime, ConfigManager
import dataset
from typing import List, Tuple
from criterion.bce_loss import BCELoss
from criterion.contrastive_loss import ContrastiveLoss
from models import SiameseCnnBCE, SiameseCnnContrastInter, SiameseCnnContrastOrigin

# import log
load_dotenv()


def predict_labels(outputs, threshold, margin, type):
    """
    predict output depend on threshold or margin
    args:
        outputs (tensor): output of the model
        type = 0: using threshold
        type = 1: using margin
    """
    if type == "BCE":
        return outputs >= threshold
    if type == "contrast_origin" or type == "contrast_inter":
        return outputs < margin


def train_each_epoch(
    train_dataloader,
    val_dataloader,
    model,
    criterion,
    optimizer,
    device,
    threshold_,
    margin_,
    model_type_,
):
    """
    Trains and validates a Siamese network for one epoch using either Binary Cross-Entropy (BCE)
    or contrastive loss.

    Args:
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        model (torch.nn.Module): The Siamese network model.
        criterion (torch.nn.Module): Loss function (BCE or contrastive loss).
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to run computations on (CPU/GPU).
        threshold_ (float): Threshold for classification.
        margin_ (float): Margin for contrastive loss.
        model_type_ (str): Type of model training, one of {'BCE', 'contrast_origin', 'contrast_inter'}.

    Returns:
        tuple: (epoch_train_loss, epoch_val_loss, accuracy_train, accuracy_val)
            - epoch_train_loss (float): Average training loss for the epoch.
            - epoch_val_loss (float): Average validation loss for the epoch.
            - accuracy_train (float): Training accuracy in percentage.
            - accuracy_val (float): Validation accuracy in percentage.

    Notes:
        - If `model_type_` is 'BCE', the function uses Binary Cross-Entropy loss.
        - If `model_type_` is 'contrast_origin' or 'contrast_inter', contrastive loss is applied.
    """
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

        if model_type == "BCE":
            output = model(image1, image2)
            predict_label = predict_labels(output, threshold, margin, model_type)
            num_correct_labels += (predict_label == label).sum().item()

            BCE_loss = criterion(output, label)
            epoch_train_loss += BCE_loss.item()
            BCE_loss.backward()

        if model_type == "contrast_origin" or model_type == "contrast_inter":
            output_01, output_02 = model(image1, image2)
            euclidean_distance = func.pairwise_distance(output_01, output_02)
            euclidean_distance = euclidean_distance.view(len(euclidean_distance), -1)
            predict_label = predict_labels(
                euclidean_distance, threshold, margin, model_type
            )
            num_correct_labels += (predict_label == label).sum().item()

            cont_loss = criterion(output_01, output_02, label)
            epoch_train_loss += cont_loss.item()
            cont_loss.backward()

        optimizer.step()

    accuracy_train = num_correct_labels / total * 100
    epoch_train_loss /= total

    # validation
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

            if model_type == "BCE":
                output = model(image1, image2)
                predict_label = predict_labels(output, threshold, margin, model_type)
                num_correct_labels += (predict_label == label).sum().item()

                BCE_loss = criterion(output, label)
                epoch_val_loss += BCE_loss.item()

            if model_type == "contrast_origin" or model_type == "contrast_inter":
                output_01, output_02 = model(image1, image2)
                euclidean_distance = func.pairwise_distance(output_01, output_02)
                euclidean_distance = euclidean_distance.view(
                    len(euclidean_distance), -1
                )
                predict_label = predict_labels(
                    euclidean_distance, threshold, margin, model_type
                )
                num_correct_labels += (predict_label == label).sum().item()

                cont_loss = criterion(output_01, output_02, label)
                epoch_val_loss += cont_loss.item()

    accuracy_val = num_correct_labels / total * 100
    epoch_val_loss /= total

    return epoch_train_loss, epoch_val_loss, accuracy_train, accuracy_val


@measuretime
def training(config: ConfigManager, logger: logging.Logger):
    logger.info("Training initializing...")
    # load config
    train_batch_size = int(config["hyper_param"]["TRAIN_BATCH_SIZE"])
    train_num_epochs = int(config["hyper_param"]["TRAIN_NUM_EPOCHS"])
    learning_rate = float(config["hyper_param"]["LEARNING_RATE"])
    num_features = int(config["hyper_param"]["NUM_FEATURES"])
    channels_img = int(config["hyper_param"]["CHANNELS_IMG"])
    margin = config["hyper_param"]["MARGIN"]
    threshold = float(config["hyper_param"]["THRESHOLD"])
    cuda_id = config["hyper_param"]["CUDA_ID"]
    weight_decay = config["hyper_param"]["WEIGHT_DECAY"]

    # checkpoints and figures
    checkpoints_folder = config["CHECKPOINTS"]
    component = config["DATASET"]["COMPONENT"]
    model_type = config["MODEL_TYPE"]

    checkpoints_dir = os.path.join(os.getenv("CHECKPOINTS_DIR"), checkpoints_folder)
    checkpoints_dir += f"/{component}/{model_type}"
    if model_type == "contrast_inter" or model_type == "contrast_origin":
        checkpoints_dir += f"/{margin}"

    os.makedirs(checkpoints_dir, exist_ok=True)

    logger.info("Train config loaded successfully.")

    # checkpoints detail
    min_loss_checkpoint_filename = "min_loss_checkpoint.pth"
    last_checkpoint_filename = "last_checkpoint.pth"
    min_loss_checkpoint_path = os.path.join(
        checkpoints_dir, min_loss_checkpoint_filename
    )
    last_checkpoint_path = os.path.join(checkpoints_dir, last_checkpoint_filename)

    # transform
    transform = None
    if model_type == "BCE" or model_type == "contrast_inter":
        transform = transforms.Compose(
            [transforms.Resize((121, 121)), transforms.ToTensor()]
        )
    if model_type == "contrast_origin":
        transform = transforms.Compose(
            [transforms.Resize((100, 100)), transforms.ToTensor()]
        )

    # dataset
    dataset_name = config["DATASET"]["NAME"]
    train_data = dataset.SiameseNetworkDataset(
        dataset_name=dataset_name,
        transform=transform,
        type="train",
        component=component,
    )
    logger.info("Train dataset loaded successfully.")

    val_data = dataset.SiameseNetworkDataset(
        dataset_name=dataset_name, transform=transform, type="val", component=component
    )
    logger.info("Validation dataset loaded successfully.")

    # dataloader
    train_dataloader = DataLoader(
        dataset=train_data, batch_size=train_batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=val_data, batch_size=train_batch_size, shuffle=True
    )

    # resource
    cuda = "cuda:" + f"{cuda_id}"
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")

    # model
    sia_model = None
    if model_type == "BCE":
        sia_model = SiameseCnnBCE(channels_img, num_features)
    if model_type == "contrast_origin":
        sia_model = SiameseCnnContrastOrigin(channels_img, num_features)
    if model_type == "contrast_inter":
        sia_model = SiameseCnnContrastInter(channels_img, num_features)
    logger.debug(sia_model)
    logger.info("Model initialized successfully.")

    # criterion
    criterion = None
    if model_type == "BCE":
        criterion = BCELoss()
    if model_type == "contrast_inter" or model_type == "contrast_origin":
        criterion = ContrastiveLoss(margin)

    # optimizer
    optimizer = None
    if model_type == "BCE" or model_type == "contrast_inter":
        optimizer = optim.Adam(
            sia_model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    if model_type == "contrast_origin":
        optimizer = optim.Adam(sia_model.parameters(), lr=0.0005)
    # if pretraining available

    logger.info("Criterion and Optimizer initialized successfully.")

    num_trained_epoch = 0
    if config["is_pretrained"] is True:
        sia_model.load_state_dict(torch.load(last_checkpoint_path))
        for checkpoint in os.listdir(checkpoints_dir):
            name = checkpoint.split(".")[0]
            name = name.split("_")
            ids = []
            if len(name) == 3 and name[0] == "last":
                ids.append(int(name[2]))
        num_trained_epoch = max(ids)
        train_num_epochs = 10

    ##################################################
    ###                  TRAINING                  ###
    ##################################################
    logger.info("Training started.")
    list_train_loss = []
    list_val_loss = []
    list_acc_train = []
    list_acc_val = []
    min_loss_train = 0.0

    sia_model = sia_model.to(device)
    for epoch in range(train_num_epochs):
        logger.debug(f"Epoch number {epoch + num_trained_epoch}:")
        epoch_train_loss, epoch_val_loss, accuracy_train, accuracy_val = (
            train_each_epoch(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                model=sia_model,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                threshold_=threshold,
                margin_=margin,
                model_type_=model_type,
            )
        )

        list_train_loss.append(epoch_train_loss)
        list_val_loss.append(epoch_val_loss)
        list_acc_train.append(accuracy_train)
        list_acc_val.append(accuracy_val)

        logger.info(f"Epoch {epoch + num_trained_epoch} finished.")
        logger.debug(f"train_loss = {epoch_train_loss}")
        logger.debug(f"val_loss =   {epoch_val_loss}")
        logger.info(f"accuracy training:   {accuracy_train:.2f}%")
        logger.info(f"accuracy validation: {accuracy_val:.2f}%")

        if epoch == 0:
            min_loss_train = epoch_train_loss
        if epoch_train_loss <= min_loss_train:
            min_loss_train = epoch_train_loss
            logger.debug(
                f"NOTE: Get the last min_loss_weights at epoch {epoch + num_trained_epoch}"
            )
            torch.save(sia_model.state_dict(), min_loss_checkpoint_path)
        torch.save(sia_model.state_dict(), last_checkpoint_path)
        epoch_weight_name = last_checkpoint_filename.split(".")[0]
        torch.save(
            sia_model.state_dict(),
            os.path.join(
                checkpoints_dir, f"{epoch_weight_name}_{epoch + num_trained_epoch}.pth"
            ),
        )

    logger.info("Training finished.")
    logger.debug(
        f"The smallest loss achieved is {min_loss_train} at epoch {list_train_loss.index(min_loss_train) + num_trained_epoch}."
    )
    logger.debug("-" * 70)
    return (
        tuple(list_train_loss),
        tuple(list_val_loss),
        tuple(list_acc_train),
        tuple(list_acc_val),
    )
