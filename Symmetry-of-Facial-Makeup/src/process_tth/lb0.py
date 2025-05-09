import os
import shutil
import random

def get_images_by_style(source_folder):
    """Lấy tất cả các file ảnh từ thư mục nguồn."""
    images = [f for f in os.listdir(source_folder) if f.endswith('.png')]
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

def split_images_by_non_zero_count(images):
    """
    Chia các ảnh thành hai nhóm: nhóm có số ký tự khác 0 lẻ và nhóm có số ký tự khác 0 chẵn
    """
    odd_images = []
    even_images = []
    
    for image in images:
        # Lấy phần tên ảnh mà không có đuôi .png
        image_name = image.split('.')[0]
        # Lấy ký tự sau dấu "_"
        number_part = image_name.split('_')[1]
        
        # Đếm số ký tự khác 0
        non_zero_count = sum(1 for char in number_part if char != '0')
        
        # Phân loại ảnh vào nhóm lẻ hoặc chẵn
        if non_zero_count % 2 == 0:
            even_images.append(image)
        else:
            odd_images.append(image)
    
    return odd_images, even_images

def create_smaller_folder(src_folder, dest_folder, ratio=0.1):
    """
    Tạo một thư mục nhỏ hơn có cấu trúc tương tự thư mục nguồn, 
    nhưng chỉ lấy x% ảnh của nhóm có số ký tự khác 0 lẻ và x% ảnh của nhóm có số ký tự khác 0 chẵn.
    """
    # Tạo thư mục đích nếu chưa tồn tại
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Duyệt qua các thư mục con trong thư mục nguồn
    for sub_folder in os.listdir(src_folder):
        sub_path = os.path.join(src_folder, sub_folder)
        
        # Tạo thư mục con tương ứng trong thư mục đích
        dest_subfolder = os.path.join(dest_folder, sub_folder)
        os.makedirs(dest_subfolder, exist_ok=True)

        # Lấy tất cả ảnh từ thư mục con
        all_images = get_images_by_style(sub_path)

        # Chia các ảnh thành nhóm lẻ và chẵn
        odd_images, even_images = split_images_by_non_zero_count(all_images)

        # Tính số lượng ảnh cần di chuyển theo tỷ lệ
        odd_to_move = max(1, int(len(odd_images) * ratio))
        even_to_move = max(1, int(len(even_images) * ratio))

        # Di chuyển ảnh từ nhóm lẻ vào thư mục đích
        move_images_to_folder(odd_images, sub_path, dest_subfolder, odd_to_move)

        # Di chuyển ảnh từ nhóm chẵn vào thư mục đích
        move_images_to_folder(even_images, sub_path, dest_subfolder, even_to_move)

# Example usage
source_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/01_PINK/res_0603/res_0603_total_FULL/LB0/EYE/eyeline_for_Ele'  # Cập nhật đường dẫn thư mục nguồn
destination_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/01_PINK/res_0603/2203/test/LB0/EYE/EleGANt/eyeline'  # Cập nhật đường dẫn thư mục đích
ratio = 0.2

create_smaller_folder(source_folder, destination_folder, ratio)
