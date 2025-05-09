import os
import shutil
import random

def get_image_groups(source_folder, valid_middle_numbers):
    """Group images by their first number (e.g., '01370') and filter by valid middle numbers"""
    image_dict = {}
    for filename in os.listdir(source_folder):
        parts = filename.split('_')
        if len(parts) >= 2:  # Ensure there's at least a first and middle part
            first_number = parts[0]  # e.g., '01370'
            middle_number = parts[1]  # e.g., '0052'
            # Only include if middle number is in the valid list
            if middle_number in valid_middle_numbers:
                if first_number not in image_dict:
                    image_dict[first_number] = []
                image_dict[first_number].append(filename)
    return image_dict

def copy_images_to_folder(image_list, sub_path, sub_dest_path):
    """Copy a list of images to the destination folder"""
    for image in image_list:  # Fixed typo from 'image in image_dict'
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
    Split images into train (60%), val (20%), and test (20%) folders based on first numbers,
    only including images with specific middle numbers, and respecting index ranges.
    """
    # Define the valid middle numbers
    valid_middle_numbers = {
        '100', '200', '300', '400', '010', '020', '001', '002', '003', '004',
        '0051', '0052', '0061', '0062', '0071', '0072', '0081', '0082'
    }

    # Define the index ranges for the first part
    index_ranges = [
        (1369, 2737), (10063, 10124), (10189, 10251), (10310, 10367),
        (10424, 10478), (10546, 10613), (10682, 10748), (10807, 10864),
        (10922, 10977), (11088, 11197), (11308, 11417), (11528, 11637),
        (11748, 11857), (11968, 12077), (12188, 12297), (12408, 12517),
        (12628, 12737)
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

        # Group images by first number, filtering by valid middle numbers
        image_groups = get_image_groups(sub_path, valid_middle_numbers)
        group_list = list(image_groups.keys())  # List of first numbers

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
source_folder = "/mnt/2E287FDD287FA28F/teamAI/hong/EleGANt/script_bach/lb0/LIP/smudge_lip"
train_folder = "/mnt/2E287FDD287FA28F/teamAI/hong/01_PINK/bach_split_data/split_train/lb0/lip/elegant/smudge_lip"
val_folder = "/mnt/2E287FDD287FA28F/teamAI/hong/01_PINK/bach_split_data/split_val/lb0/lip/elegant/smudge_lip"
test_folder = "/mnt/2E287FDD287FA28F/teamAI/hong/01_PINK/bach_split_data/split_test/lb0/lip/elegant/smudge_lip"

split_and_copy_folder(source_folder, train_folder, val_folder, test_folder, train_ratio=0.6, val_ratio=0.2)