import os
import shutil

# Specify the source and destination folders
source_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/01_PINK/res_0603/res_0603_total_FULL/LB0/LIP/EleGANt/lip_4_Ele'
destination_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/01_PINK/res_0603/2203/train/LB0/LIP/EleGANt/otherlip'
os.makedirs(destination_folder, exist_ok=True)

# Specify the percentage of images to move (e.g., 50% of images)
percentage = 100

# List all files in the source folder
images = [f for f in os.listdir(source_folder) if f.lower().endswith(('jpg', 'jpeg', 'png', 'gif'))]

# Calculate the number of images to move
num_images_to_move = int(len(images) * percentage / 100)

# Select the first 'num_images_to_move' images
images_to_move = images[:num_images_to_move]

# Move the selected images to the destination folder
for image in images_to_move:
    source_path = os.path.join(source_folder, image)
    destination_path = os.path.join(destination_folder, image)
    shutil.move(source_path, destination_path)

print(f"Moved {len(images_to_move)} images from {source_folder} to {destination_folder}")
