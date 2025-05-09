import os
import shutil
import random

def get_images_by_char(char, source_folder):
    """Lấy các file ảnh chứa ký tự `char` trong tên file"""
    # images = [f for f in os.listdir(source_folder) if f.endswith('.png') and char in f]
    images = [f for f in os.listdir(source_folder) if char in f]
    return images

def move_images_to_folder(image_list, sub_path, sub_dest_path, num_to_move):
    """Di chuyển một số ảnh đã chọn vào thư mục đích"""
    total_images = len(image_list)
    if total_images == 0:
        return
    # Randomly select 'num_to_move' images
    selected_images = random.sample(image_list, num_to_move)
    
    # Di chuyển các ảnh đã chọn
    for image in selected_images:
        src_image_path = os.path.join(sub_path, image)
        dest_image_path = os.path.join(sub_dest_path, image)
        shutil.move(src_image_path, dest_image_path)

def create_smaller_folder(src_folder, dest_folder, ratio=0.1):
    """
    Tạo một thư mục nhỏ hơn có cấu trúc tương tự thư mục nguồn, 
    mỗi subfolder chứa `ratio` phần trăm của các ảnh `a_images`, `b_images`, `c_images`.
    """
    # Tạo thư mục đích nếu chưa tồn tại
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Duyệt qua các subfolder trong thư mục nguồn
    for sub_folder in os.listdir(src_folder):
        sub_path = os.path.join(src_folder, sub_folder)
        
        # Tạo subfolder tương ứng trong thư mục đích
        dest_subfolder = os.path.join(dest_folder, sub_folder)
        os.makedirs(dest_subfolder, exist_ok=True)

        # Lấy các ảnh thuộc các nhóm a, b, c
        a_images = get_images_by_char('a', sub_path)
        b_images = get_images_by_char('b', sub_path)
        c_images = get_images_by_char('c', sub_path)

        # Tính số lượng ảnh cần di chuyển theo tỷ lệ
        def calculate_num_to_move(total_images, ratio):
            return max(1, int(total_images * ratio))  # Ít nhất 1 ảnh

        # Tính số lượng ảnh cần di chuyển cho mỗi nhóm
        a_to_move = calculate_num_to_move(len(a_images), ratio)
        b_to_move = calculate_num_to_move(len(b_images), ratio)
        c_to_move = calculate_num_to_move(len(c_images), ratio)

        # Di chuyển các ảnh vào thư mục đích
        move_images_to_folder(a_images, sub_path, dest_subfolder, a_to_move)
        move_images_to_folder(b_images, sub_path, dest_subfolder, b_to_move)
        move_images_to_folder(c_images, sub_path, dest_subfolder, c_to_move)
       

# # Example usage
# source_folder = ''  # Cập nhật đường dẫn thư mục nguồn
# destination_folder = '/path/to/destination/folder'  # Cập nhật đường dẫn thư mục đích
# ratio = 0.1  # Tỷ lệ phần trăm ảnh cần lấy từ mỗi nhóm a, b, c

# create_smaller_folder(source_folder, destination_folder, ratio)

source_folder_comp = "/mnt/2E287FDD287FA28F/teamAI/hong/01_PINK/res_0603/res_0603_total_FULL/LB1/SKIN/CSD"
res_folder_comp_TVT = "/mnt/2E287FDD287FA28F/teamAI/hong/01_PINK/res_0603/2203/train/LB1/CHEEK/CSD"
ratio = 1
# os.makedirs(res_folder_comp_TVT, exist_ok= True)

# for sub in os.listdir(source_folder_comp):
#     sub_path = os.path.join(source_folder_comp, sub)
#     res_sub_path = os.path.join(res_folder_comp_TVT, sub)
create_smaller_folder(source_folder_comp, res_folder_comp_TVT, ratio)