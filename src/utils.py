import matplotlib.pyplot as plt
import os
import config
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
        'weight_decay':config.WEIGHT_DECAY
    }
    return config_dict

def create_image_grid(left_eye, right_eye, left_cheek, right_cheek, left_lip, right_lip, title, scores):
    font = ImageFont.load_default()

    # Lấy kích thước của các ảnh (width, height)
    img_width = max(left_eye.size[0], right_eye.size[0], left_cheek.size[0], right_cheek.size[0], left_lip.size[0], right_lip.size[0])
    img_height = max(left_eye.size[1], right_eye.size[1], left_cheek.size[1], right_cheek.size[1], left_lip.size[1], right_lip.size[1])

    # Kích thước của ảnh lưới
    grid_width = img_width * 2 + 20  # 2 ảnh + khoảng cách giữa
    grid_height = img_height * 3 + 130   # 3 hàng ảnh + khoảng cách giữa các cặp ảnh

    # Tạo một ảnh trắng làm nền cho lưới ảnh
    grid_image = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid_image)

    # In title vào phần đầu của ảnh
    title_bbox = draw.textbbox((0, 0), title, font=font)
    title_width, title_height = title_bbox[2] - title_bbox[0], title_bbox[3] - title_bbox[1]
    title_x = (grid_width - title_width) // 2
    draw.text((title_x, 10), title, font=font, fill=(0, 0, 0))

    # Vị trí y cho các cặp ảnh
    y_offset = 40  # Khoảng cách giữa title và ảnh

    # Vẽ các cặp ảnh với scores vào trong lưới ảnh
    for i, (left_img, right_img, score) in enumerate([(left_eye, right_eye, scores[0]),
                                                      (left_cheek, right_cheek, scores[1]),
                                                      (left_lip, right_lip, scores[2])]):
        # In score
        if score is None:
            continue
        score_text = f"Score: {score:.6f}"
        score_bbox = draw.textbbox((0, 0), score_text, font=font)
        score_width, score_height = score_bbox[2] - score_bbox[0], score_bbox[3] - score_bbox[1]
        score_x = (grid_width - score_width) // 2
        draw.text((score_x, y_offset), score_text, font=font, fill=(0, 0, 0))

        # Vẽ ảnh cặp
        left_x = (grid_width // 2) - img_width - 10
        right_x = (grid_width // 2) + 10
        grid_image.paste(left_img, (left_x, y_offset + score_height + 10))
        grid_image.paste(right_img, (right_x, y_offset + score_height + 10))

        # Cập nhật y_offset cho hàng tiếp theo
        y_offset += img_height + score_height + 30  # Khoảng cách giữa các cặp ảnh

    # Trả về ảnh lưới đã tạo
    return grid_image