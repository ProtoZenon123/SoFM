import os
import shutil
import random

def get_image_groups(source_folder):
    """Group images by their index number (e.g., '00001' from '00001_l1a_l1')"""
    image_dict = {}
    for filename in os.listdir(source_folder):
        if any(char in filename for char in ['a', 'b', 'c']):
            # Extract the index number (assumes it's the first part before '_')
            index = filename.split('_')[0]
            if index not in image_dict:
                image_dict[index] = []
            image_dict[index].append(filename)
    return image_dict

def copy_images_to_folder(image_list, sub_path, sub_dest_path):
    """Copy a list of images to the destination folder"""
    for image in image_list:
        src_image_path = os.path.join(sub_path, image)
        dest_image_path = os.path.join(sub_dest_path, image)
        shutil.copy(src_image_path, dest_image_path)

def is_in_range(index, ranges):
    """Check if an index falls within any of the specified ranges"""
    index_num = int(index)  # Convert string index to integer
    for start, end in ranges:
        if start <= index_num <= end:
            return True
    return False

def split_and_copy_folder(src_folder, train_folder, val_folder, test_folder, train_ratio=0.6, val_ratio=0.2):
    """
    Split images into train (60%), val (20%), and test (20%) folders from specific index ranges,
    keeping images with the same index number together.
    """
    # Define the index ranges
    index_ranges = [
        (0, 1368), (10000, 10062), (10125, 10188), (10252, 10309),
        (10368, 10423), (10479, 10545), (10614, 10681), (10749, 10806),
        (10865, 10921), (10978, 11087) , (11198, 11307), (11418, 11527),
        (11638, 11747), (11858, 11967), (12078, 12187), (12298, 12407),
        (12518, 12627)
    ]

    # Create destination folders if they don’t exist
    for folder in [train_folder, val_folder, test_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Process each subfolder in the source folder
    for sub_folder in os.listdir(src_folder):
        sub_path = os.path.join(src_folder, sub_folder)
        if not os.path.isdir(sub_path):
            continue  # Skip if not a directory

        # Create corresponding subfolders in train, val, test
        train_subfolder = os.path.join(train_folder, sub_folder)
        val_subfolder = os.path.join(val_folder, sub_folder)
        test_subfolder = os.path.join(test_folder, sub_folder)
        os.makedirs(train_subfolder, exist_ok=True)
        os.makedirs(val_subfolder, exist_ok=True)
        os.makedirs(test_subfolder, exist_ok=True)

        # Group images by index number
        image_groups = get_image_groups(sub_path)
        group_list = list(image_groups.keys())  # List of index numbers

        # Filter groups to only include those in the specified ranges
        filtered_groups = [index for index in group_list if is_in_range(index, index_ranges)]

        # Shuffle the filtered groups randomly
        random.shuffle(filtered_groups)

        # Calculate split sizes based on filtered groups
        total_groups = len(filtered_groups)
        if total_groups == 0:
            continue  # Skip if no groups are in the range

        train_count = int(total_groups * train_ratio)
        val_count = int(total_groups * val_ratio)
        test_count = total_groups - train_count - val_count  # Ensure all groups are assigned

        # Assign groups to train, val, test
        train_groups = filtered_groups[:train_count]
        val_groups = filtered_groups[train_count:train_count + val_count]
        test_groups = filtered_groups[train_count + val_count:]

        # Copy train images
        for index in train_groups:
            copy_images_to_folder(image_groups[index], sub_path, train_subfolder)

        # Copy val images
        for index in val_groups:
            copy_images_to_folder(image_groups[index], sub_path, val_subfolder)

        # Copy test images
        for index in test_groups:
            copy_images_to_folder(image_groups[index], sub_path, test_subfolder)

# Example usage
source_folder = "/mnt/2E287FDD287FA28F/teamAI/hong/CSD_part2/CSD-MT/CSD-MT/script_bach/lb1/lip"
train_folder = "/mnt/2E287FDD287FA28F/teamAI/hong/01_PINK/bach_split_data/split_train/lb1/lip/csd"
val_folder = "/mnt/2E287FDD287FA28F/teamAI/hong/01_PINK/bach_split_data/split_val/lb1/lip/csd"
test_folder = "/mnt/2E287FDD287FA28F/teamAI/hong/01_PINK/bach_split_data/split_test/lb1/lip/csd"

split_and_copy_folder(source_folder, train_folder, val_folder, test_folder, train_ratio=0.6, val_ratio=0.2)