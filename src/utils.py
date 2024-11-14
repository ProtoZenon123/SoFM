import matplotlib.pyplot as plt
import os
import config, config_eye, config_cheek, config_lip
import shutil
import random
import math
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy


def plot(data, label, xlabel, ylabel, title, save_path):
    plt.figure()

    # min_value = min(data)
    X = range(1, len(data) + 1)
    plt.plot(X, data, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def load_config():
    config_dict = {
        'train_batch_size': config.TRAIN_BATCH_SIZE,
        'train_num_epochs': config.TRAIN_NUM_EPOCHS,
        'learning_rate': config.LEARNING_RATE,
        'num_features': config.NUM_FEATURES,
        'channels_img': config.CHANNELS_IMG,
        'margin': config.MARGIN,
        'threshold': config.THRESHOLD,
        'cuda_id': config.CUDA_ID,
        'train_dir': config.TRAIN_DIR,
        'folder_weights_dir': config.FOLDER_WEIGHTS_DIR,
        'figures_dir': config.FIGURES_DIR,
    }
    return config_dict
def load_config_eye():
    config_dict = {
        'train_batch_size': config_eye.TRAIN_BATCH_SIZE,
        'train_num_epochs': config_eye.TRAIN_NUM_EPOCHS,
        'learning_rate': config_eye.LEARNING_RATE,
        'num_features': config_eye.NUM_FEATURES,
        'channels_img': config_eye.CHANNELS_IMG,
        'margin': config_eye.MARGIN,
        'threshold': config_eye.THRESHOLD,
        'cuda_id': config_eye.CUDA_ID,
        'train_dir': config_eye.TRAIN_DIR,
        'folder_weights_dir': config_eye.FOLDER_WEIGHTS_DIR,
        'figures_dir': config_eye.FIGURES_DIR,
    }
    return config_dict
def load_config_cheek():
    config_dict = {
        'train_batch_size': config_cheek.TRAIN_BATCH_SIZE,
        'train_num_epochs': config_cheek.TRAIN_NUM_EPOCHS,
        'learning_rate': config_cheek.LEARNING_RATE,
        'num_features': config_cheek.NUM_FEATURES,
        'channels_img': config_cheek.CHANNELS_IMG,
        'margin': config_cheek.MARGIN,
        'threshold': config_cheek.THRESHOLD,
        'cuda_id': config_cheek.CUDA_ID,
        'train_dir': config_cheek.TRAIN_DIR,
        'folder_weights_dir': config_cheek.FOLDER_WEIGHTS_DIR,
        'figures_dir': config_cheek.FIGURES_DIR,
    }
    return config_dict
def load_config_lip():
    config_dict = {
        'train_batch_size': config_lip.TRAIN_BATCH_SIZE,
        'train_num_epochs': config_lip.TRAIN_NUM_EPOCHS,
        'learning_rate': config_lip.LEARNING_RATE,
        'num_features': config_lip.NUM_FEATURES,
        'channels_img': config_lip.CHANNELS_IMG,
        'margin': config_lip.MARGIN,
        'threshold': config_lip.THRESHOLD,
        'cuda_id': config_lip.CUDA_ID,
        'train_dir': config_lip.TRAIN_DIR,
        'folder_weights_dir': config_lip.FOLDER_WEIGHTS_DIR,
        'figures_dir': config_lip.FIGURES_DIR,
    }
    return config_dict


def move_folder(src_folder, dest_folder, percentage):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    subfolders = [f.path for f in os.scandir(src_folder) if f.is_dir()]

    num_select_subfolders = int(len(subfolders) * (percentage / 100))

    selected_folders = random.sample(subfolders, num_select_subfolders)

    for folder in selected_folders:
        folder_name = os.path.basename(folder)
        shutil.move(folder, os.path.join(dest_folder, folder_name))


def devide_data(train_dir):

    base_dir = os.path.dirname(train_dir)
    test_dir = os.path.join(base_dir, 'test_data')
    validation_dir = os.path.join(base_dir, 'val_data')

    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    labels = ['label_0', 'label_1']
    comps = ['eye', 'cheek', 'lip']
    for label in labels:
        for comp in comps:
            os.makedirs(os.path.join(test_dir, label, comp))
            os.makedirs(os.path.join(validation_dir, label, comp))
            move_folder(os.path.join(train_dir, label, comp), os.path.join(test_dir, label, comp), 20)
            move_folder(os.path.join(train_dir, label, comp), os.path.join(validation_dir, label, comp), 20)
    

def calculate_grid_size(num_pairs):
    if num_pairs <= 4:
        return (1, num_pairs)
    elif num_pairs <= 6:
        return (2, math.ceil(num_pairs / 2))
    elif num_pairs <= 9:
        return (2, math.ceil(num_pairs / 2))
    elif num_pairs <= 12:
        return (2, math.ceil(num_pairs / 2))
    elif num_pairs <= 16:
        return (4, math.ceil(num_pairs / 4))
    else:
        return (4, 5)


def create_image_grid(list_pairs_img, save_path_lastW, save_path_minW, gap=2):
    num_pairs = len(list_pairs_img)
    image_pairs = []
    score_minW = []
    score_lastW = []
    for item in list_pairs_img:
        image_pairs.append(item[:2])
        score_lastW.append(item[2])
        score_minW.append(item[3])

    grid_size = calculate_grid_size(num_pairs)

    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(16, 8))

    for idx, (image_pair, score) in enumerate(zip(image_pairs, score_lastW)):
        row = idx // grid_size[1]
        col = idx % grid_size[1]

        ax = axes[row, col] if grid_size[0] > 1 else axes[col]
        ax.axis('off')

        img1 = Image.open(image_pair[0])
        img2 = Image.open(image_pair[1])

        max_height = max(img1.size[1], img2.size[1])
        total_width = img1.size[0] + img2.size[0] + gap

        combined_img = Image.new('RGB', (total_width, max_height))
        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (img1.size[0] + gap, 0))

        ax.imshow(combined_img)

        ax.set_title(f"Score: {score:.8f}", fontsize=12)

    for i in range(num_pairs, grid_size[0] * grid_size[1]):
        row = i // grid_size[1]
        col = i % grid_size[1]
        if grid_size[0] > 1:
            axes[row, col].axis('off')
        else:
            axes[col].axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.savefig(save_path_lastW, bbox_inches='tight', pad_inches=0.1)
    plt.close()
 
    plt.figure()
    grid_size = calculate_grid_size(num_pairs)

    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(16, 8))
    for idx, (image_pair, score) in enumerate(zip(image_pairs, score_minW)):
        row = idx // grid_size[1]
        col = idx % grid_size[1]

        ax = axes[row, col] if grid_size[0] > 1 else axes[col]
        ax.axis('off')

        img1 = Image.open(image_pair[0])
        img2 = Image.open(image_pair[1])

        max_height = max(img1.size[1], img2.size[1])
        total_width = img1.size[0] + img2.size[0] + gap

        combined_img = Image.new('RGB', (total_width, max_height))
        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (img1.size[0] + gap, 0))

        ax.imshow(combined_img)

        ax.set_title(f"Score: {score:.8f}", fontsize=12)

    for i in range(num_pairs, grid_size[0] * grid_size[1]):
        row = i // grid_size[1]
        col = i % grid_size[1]
        if grid_size[0] > 1:
            axes[row, col].axis('off')
        else:
            axes[col].axis('off')

    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.savefig(save_path_minW, bbox_inches='tight', pad_inches=0.1)
    plt.close()

from PIL import Image, ImageDraw, ImageFont

def make_grid_of_image(image_folder, res_cpnt, weights_name):
    horizontal_spacing = 20 
    vertical_spacing = 40 

    image_files = [("left_eye.png", "right_eye.png"),
                   ("left_cheek.png", "right_cheek.png"),
                   ("left_lip.png", "right_lip.png")]

    res_cpnt = [f"Eyes: {res_cpnt[0]:.8f}", f"Cheeks: {res_cpnt[1]:.8f}", f"Lips: {res_cpnt[2]:.8f}"]

    images = [(Image.open(f'{image_folder}/{left}'), Image.open(f'{image_folder}/{right}')) for left, right in image_files]

    max_pair_width = max(left.size[0] + horizontal_spacing + right.size[0] for left, right in images)
    total_height = sum(max(left.size[1], right.size[1]) + vertical_spacing for left, right in images)

    title_height = 50
    label_height = 20

    grid_img = Image.new('RGB', (max_pair_width, total_height + title_height + len(res_cpnt) * label_height), color=(255, 255, 255))

    draw = ImageDraw.Draw(grid_img)
    
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()
    
    title_bbox = draw.textbbox((0, 0), weights_name, font=title_font)
    title_width = title_bbox[2] - title_bbox[0] 
    title_x = (max_pair_width - title_width) // 2
    title_y = 10
    draw.text((title_x, title_y), weights_name, font=title_font, fill=(0, 0, 0))

    y_offset = title_height + label_height

    for i, (left, right) in enumerate(images):
        left_width, left_height = left.size
        right_width, right_height = right.size

        left_x = int((max_pair_width - horizontal_spacing)/2 - left_width)
        right_x = left_x + left_width + horizontal_spacing

        grid_img.paste(left, (left_x, y_offset))
        grid_img.paste(right, (right_x, y_offset))

        text_x = 0 

        text_y = y_offset - label_height 

        draw.text((text_x, text_y), res_cpnt[i], font=font, fill=(0, 0, 0))

        y_offset += max(left_height, right_height) + vertical_spacing

    return grid_img

def white_cheek(image):
    # print(image.shape)
    white_image = deepcopy(image)
    mask = (white_image == [0, 0, 0]).all(axis=2)
    white_image[mask] = [1, 1, 1]
    return white_image

