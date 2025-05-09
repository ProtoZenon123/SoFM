import os
from dotenv import load_dotenv
import logging

import torch
import torch.nn.functional as func
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import measuretime, ConfigManager
import dataset
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


def testing(
    test_dataloader, model, criterion, device, threshold_, margin_, model_type_
):
    """_summary_

    Args:
        test_dataloader (_type_): _description_
        model (_type_): _description_
        criterion (_type_): _description_
        device (_type_): _description_
        threshold_ (_type_): _description_
        margin_ (_type_): _description_
        model_type_ (_type_): _description_

    Returns:
        _type_: _description_
    """
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

            if model_type == "BCE":
                output = model(image1, image2)
                predict_label = predict_labels(output, threshold, margin, model_type)
                num_correct_labels += (predict_label == label).sum().item()

                BCE_loss = criterion(output, label)
                test_loss += BCE_loss.item()

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
                test_loss += cont_loss.item()

    accuracy = num_correct_labels / total * 100
    test_loss /= total

    return accuracy, test_loss

@measuretime
def evaluate(config: ConfigManager, logger: logging.Logger):

    logger.info("Testing initializing...")
    
    # load config
    test_batch_size = int(config["hyper_param"]["TRAIN_BATCH_SIZE"])
    num_features = int(config["hyper_param"]["NUM_FEATURES"])
    channels_img = int(config["hyper_param"]["CHANNELS_IMG"])
    margin = config["hyper_param"]["MARGIN"]
    threshold = float(config["hyper_param"]["THRESHOLD"])
    cuda_id = config["hyper_param"]["CUDA_ID"]

    # checkpoints
    checkpoints_folder = config["CHECKPOINTS"]
    component = config["DATASET"]["COMPONENT"]
    model_type = config["MODEL_TYPE"]

    checkpoints_dir = os.path.join(os.getenv("CHECKPOINTS_DIR"), checkpoints_folder)
    checkpoints_dir += f"/{component}/{model_type}"
    if model_type == "contrast_inter" or model_type == "contrast_origin":
        checkpoints_dir += f"/{margin}"

    logger.info("Test config loaded successfully.")
    
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
    test_data = dataset.SiameseNetworkDataset(
        dataset_name=dataset_name,
        transform=transform,
        type="test",
        component=component
    )
    logger.info("Test dataset loaded successfully.")

    # dataloader
    test_dataloader = DataLoader(
        dataset=test_data, batch_size=test_batch_size, shuffle=True
    )
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
        
    # last checkpoint
    sia_model.load_state_dict(torch.load(last_checkpoint_path))
    sia_model = sia_model.to(device)
    accuracy, loss = testing(test_dataloader=test_dataloader,
                             model=sia_model,
                             criterion=criterion,
                             device=device,
                             threshold_=threshold,
                             margin_=margin,
                             model_type_=model_type)
    logger.debug(f"Accuracy with last checkpoint: {accuracy:.2f}%")
    logger.debug(f"Loss with last checkpoint: {loss:.8f}")

    sia_model.load_state_dict(torch.load(min_loss_checkpoint_path))
    sia_model = sia_model.to(device)
    accuracy, loss = testing(test_dataloader=test_dataloader,
                             model=sia_model,
                             criterion=criterion,
                             device=device,
                             threshold_=threshold,
                             margin_=margin,
                             model_type_=model_type)
    logger.debug(f"Accuracy with min_loss checkpoint: {accuracy:.2f}%")
    logger.debug(f"Loss with min_loss checkpoint: {loss:.8f}")
    logger.info("Evaluating finished.")

    