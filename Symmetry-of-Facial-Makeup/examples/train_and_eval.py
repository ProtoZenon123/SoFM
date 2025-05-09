import os
from train import training
from evaluate import evaluate
from utils import plot, creat_logger, load_config

# initialize logger
logger = creat_logger("train_and_evaluate")

# load config and training
config = load_config()
list_train_loss, list_val_loss, list_acc_train, list_acc_val = training(
    config=config, logger=logger
)

# report training and validation loss and accuracy
figures_name = config["FIGURES"]
margin = config["hyper_param"]["MARGIN"]
component = config["DATASET"]["COMPONENT"]
model_type = config["MODEL_TYPE"]

figures_dir = os.path.join(os.getenv("FIGURES_DIR"), figures_name)
figures_dir += f"/{component}/{model_type}"
if model_type == "contrast_inter" or model_type == "contrast_origin":
    figures_dir += f"/{margin}"
os.makedirs(figures_dir, exist_ok=True)


train_loss_figure_path = os.path.join(figures_dir, "train_loss.jpg")
val_loss_figure_path = os.path.join(figures_dir, "validation_loss.jpg")
val_acc_figure_path = os.path.join(figures_dir, "validation_acc.jpg")
train_acc_figure_path = os.path.join(figures_dir, "training_acc.jpg")
plot(
    list_train_loss,
    "Training loss",
    "Epoch",
    "Loss",
    "Training Loss",
    train_loss_figure_path,
)
plot(
    list_val_loss,
    "Validation loss",
    "Epoch",
    "Loss",
    "Validation Loss",
    val_loss_figure_path,
)
plot(
    list_acc_train,
    "Training accuracy",
    "Epoch",
    "Loss",
    "Training accuracy",
    train_acc_figure_path,
)
plot(
    list_acc_val,
    "Validation accuracy",
    "Epoch",
    "Loss",
    "Validation accuracy",
    val_acc_figure_path,
)

logger.debug("Evaluate with two model_weights:")
evaluate(config=config, logger=logger)
