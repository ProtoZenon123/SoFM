import os
import shutil
import random

def split_and_copy_folder(src_folder, train_folder, val_folder, test_folder, train_ratio=0.6, val_ratio=0.2):
    """
    Split all files in the source folder into train (60%), val (20%), and test (20%) folders.
    """
    # Create destination folders if they don’t exist
    for folder in [train_folder, val_folder, test_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Get list of all files in the source folder
    all_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    
    # Shuffle the files randomly
    random.shuffle(all_files)
    
    # Calculate split sizes
    total_files = len(all_files)
    if total_files == 0:
        print("No files found in source folder!")
        return
    
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    test_count = total_files - train_count - val_count  # Ensure all files are assigned

    # Split files into three groups
    train_files = all_files[:train_count]
    val_files = all_files[train_count:train_count + val_count]
    test_files = all_files[train_count + val_count:]

    # Copy files to respective folders
    for file in train_files:
        shutil.copy(os.path.join(src_folder, file), os.path.join(train_folder, file))
    
    for file in val_files:
        shutil.copy(os.path.join(src_folder, file), os.path.join(val_folder, file))
    
    for file in test_files:
        shutil.copy(os.path.join(src_folder, file), os.path.join(test_folder, file))

    print(f"Split complete: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test files")

# Define folder paths
source_folder = "/mnt/2E287FDD287FA28F/teamAI/hong/CSD_part2/CSD-MT/CSD-MT/script_bach/lb1/lip/src_lb1_non"
train_folder = "/mnt/2E287FDD287FA28F/teamAI/hong/01_PINK/bach_split_data/split_train/lb1/lip/csd/src_lb1_non"
val_folder = "/mnt/2E287FDD287FA28F/teamAI/hong/01_PINK/bach_split_data/split_val/lb1/lip/csd/src_lb1_non"
test_folder = "/mnt/2E287FDD287FA28F/teamAI/hong/01_PINK/bach_split_data/split_test/lb1/lip/csd/src_lb1_non"

# Execute the split
split_and_copy_folder(source_folder, train_folder, val_folder, test_folder, train_ratio=0.6, val_ratio=0.2)