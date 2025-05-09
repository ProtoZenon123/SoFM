import os
import sys
import cv2

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from utils import crop_and_transform_six_parts 

input_image_folder = "/mnt/2E287FDD287FA28F/teamAI/MinhQuan/FaceEnhancementAndMakeup/bach_downgrade/after"
def is_number_only(text):
    """
    Checks if the given text contains only numeric characters.

    Args:
        text (str): The input text to check.

    Returns:
        bool: True if the text contains only numbers, False otherwise.
    """
    return text.isdigit()
for image_name in os.listdir(input_image_folder):
    output_image_folder = "/mnt/2E287FDD287FA28F/teamAI/MinhQuan/FaceEnhancementAndMakeup/bach_downgrade/after_cropped"
    try:
        input_image_path = os.path.join(input_image_folder, image_name)
        
        image = cv2.imread(input_image_path)
        
        if image is not None:
            # Call the function with the image
            left_eye, right_eye, left_cheek, right_cheek, left_lip, right_lip = crop_and_transform_six_parts(image)
            
            # Save the cropped image
            image_name_without_tail = image_name.split(".")[0]
            if len(image_name.split(".")) > 2:
                image_name_without_tail += "." + image_name.split(".")[1]
            info = image_name_without_tail.split("_")[3]
            label = is_number_only(info)
            
            if label:
                output_image_folder = os.path.join(output_image_folder, "label_0")
                if info[0] != "0":
                    output_image_folder = os.path.join(output_image_folder, "eye")
                    output_image_folder = os.path.join(output_image_folder, image_name)
                    os.makedirs(output_image_folder, exist_ok=True)
                    cv2.imwrite(os.path.join(output_image_folder, 'left_eye.png'), left_eye)
                    cv2.imwrite(os.path.join(output_image_folder, 'right_eye.png'), right_eye)
                elif info[1] != "0":
                    output_image_folder = os.path.join(output_image_folder, "cheek")
                    output_image_folder = os.path.join(output_image_folder, image_name)
                    os.makedirs(output_image_folder, exist_ok=True)
                    cv2.imwrite(os.path.join(output_image_folder, 'left_cheek.png'), left_cheek)
                    cv2.imwrite(os.path.join(output_image_folder, 'right_cheek.png'), right_cheek)
                elif info[2] != "0":
                    output_image_folder = os.path.join(output_image_folder, "lip")
                    output_image_folder = os.path.join(output_image_folder, image_name)
                    os.makedirs(output_image_folder, exist_ok=True)
                    cv2.imwrite(os.path.join(output_image_folder, 'left_lip.png'), left_lip)
                    cv2.imwrite(os.path.join(output_image_folder, 'right_lip.png'), right_lip)
            else:
                output_image_folder = os.path.join(output_image_folder, "label_1")
                if info[0] == "e":
                    output_image_folder = os.path.join(output_image_folder, "eye")
                    output_image_folder = os.path.join(output_image_folder, image_name)
                    os.makedirs(output_image_folder, exist_ok=True)
                    cv2.imwrite(os.path.join(output_image_folder, 'left_eye.png'), left_eye)
                    cv2.imwrite(os.path.join(output_image_folder, 'right_eye.png'), right_eye)
                elif info[0] == "s":   
                    output_image_folder = os.path.join(output_image_folder, "cheek")
                    output_image_folder = os.path.join(output_image_folder, image_name)
                    os.makedirs(output_image_folder, exist_ok=True)
                    cv2.imwrite(os.path.join(output_image_folder, 'left_cheek.png'), left_cheek)
                    cv2.imwrite(os.path.join(output_image_folder, 'right_cheek.png'), right_cheek)
                elif info[0] == "l":
                    output_image_folder = os.path.join(output_image_folder, "lip")
                    output_image_folder = os.path.join(output_image_folder, image_name)
                    os.makedirs(output_image_folder, exist_ok=True)
                    cv2.imwrite(os.path.join(output_image_folder, 'left_lip.png'), left_lip)
                    cv2.imwrite(os.path.join(output_image_folder, 'right_lip.png'), right_lip)
    except Exception as e:
        print(f"Error processing {image_name}: {e}")
                
        
            
        
    

    
        
        
        