import os

def count_images_in_subfolders(src_folder):
    # Initialize counters for 'a', 'b', 'c' styles
    count_a = 0
    count_b = 0
    count_c = 0

    # Traverse through all subfolders and files in the src_folder
    for subdir, dirs, files in os.walk(src_folder):
        for image_name in files:
            if image_name.endswith(".png"):
                # Check if the image contains 'a', 'b', or 'c' in the filename
                if 'a' in image_name:
                    count_a += 1
                elif 'b' in image_name:
                    count_b += 1
                elif 'c' in image_name:
                    count_c += 1
    
    # Return the total counts
    return count_a, count_b, count_c

# Example usage
source_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/01_DATA_internet/01_PINK/res_0603/res_0603_train/LB1/SKIN/EleGANt'  # Replace with the actual path

count_a, count_b, count_c = count_images_in_subfolders(source_folder)

# Print the total counts of images with 'a', 'b', and 'c'
print(f"Total 'a' images: {count_a}")
print(f"Total 'b' images: {count_b}")
print(f"Total 'c' images: {count_c}")
