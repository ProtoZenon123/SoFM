import os
import cv2
import dlib
import statistics
import numpy as np
from utils import load_config
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

def mean_(list_):
    return int(statistics.mean(list_))

def left_eye_bounding_box(hor_, ver_):
    '''
        detect bouding box of left eye
    '''
    # top_left_corner_x = mean_([hor_[0], hor_[17]])
    top_left_corner_x = hor_[17]
    top_left_corner_y = mean_([ver_[19], ver_[24]]) - 5
    bottom_right_corner_x = mean_([hor_[21], hor_[31]])
    bottom_right_corner_y = mean_([mean_([ver_[41], ver_[46]]), ver_[30]])
    return top_left_corner_x, top_left_corner_y, bottom_right_corner_x, bottom_right_corner_y

def right_eye_bounding_box(hor_, ver_):
    '''
        detect bounding box of right eye
    '''
    top_left_corner_x = mean_([hor_[22], hor_[35]])
    top_left_corner_y = mean_([ver_[19], ver_[24]]) - 5
    # bottom_right_corner_x = mean_([hor_[26], hor_[16]])
    bottom_right_corner_x = hor_[26]
    bottom_right_corner_y = mean_([mean_([ver_[41], ver_[46]]), ver_[30]])
    return top_left_corner_x, top_left_corner_y, bottom_right_corner_x, bottom_right_corner_y

def crop_left_cheek(image, hor_, ver_):
    '''
        detect left cheek
    '''
    image_ = image.copy()
    polygon_points = []
    polygon_points.append([mean_([hor_[21], hor_[31]]),mean_([mean_([ver_[41], ver_[46]]), ver_[30]])])
    polygon_points.append([hor_[36],mean_([mean_([ver_[41], ver_[46]]), ver_[30]])])
    polygon_points.append([hor_[0] + 5,ver_[0]])
    polygon_points.append([hor_[1] + 5,ver_[1]])
    polygon_points.append([hor_[2] + 5,ver_[2]])
    polygon_points.append([hor_[36], ver_[33]])
    # polygon_points.append([hor_[3] + 5,ver_[3]])
    # polygon_points.append([hor_[4] + 5,ver_[4]])
    polygon_points.append([mean_([hor_[39], hor_[40]]),ver_[30]])
    
    polygon_points = np.array(polygon_points)
    top_left_corner_x = np.min(polygon_points[:, 0])
    top_left_corner_y = np.min(polygon_points[:, 1])
    bottom_right_corner_x = np.max(polygon_points[:, 0])
    bottom_right_corner_y = np.max(polygon_points[:, 1])

    mask = np.zeros_like(image_)
    cv2.fillPoly(mask, [polygon_points], (255, 255, 255))
    masked_image = cv2.bitwise_and(image_, mask)
    output_image = masked_image[top_left_corner_y:bottom_right_corner_y, top_left_corner_x:bottom_right_corner_x]
    for point in polygon_points:
        cv2.circle(image_, tuple(point), radius=2, color=(0, 0, 255), thickness=-1)
    
    return output_image

def crop_right_cheek(image, hor_, ver_):
    '''
        detect right cheek
    '''
    image_ = image.copy()
    polygon_points = []
    polygon_points.append([mean_([hor_[22], hor_[35]]),mean_([mean_([ver_[41], ver_[46]]), ver_[30]])])
    polygon_points.append([hor_[45],mean_([mean_([ver_[41], ver_[46]]), ver_[30]])])
    polygon_points.append([hor_[16]-5,ver_[16]])
    polygon_points.append([hor_[15]-5,ver_[15]])
    polygon_points.append([hor_[14]-5,ver_[14]])
    polygon_points.append([hor_[45], ver_[33]])
    # polygon_points.append([hor_[13] - 5,ver_[13]])
    # polygon_points.append([hor_[12] - 5,ver_[12]])
    polygon_points.append([mean_([hor_[42], hor_[47]]),ver_[30]])
    
    polygon_points = np.array(polygon_points)
    top_left_corner_x = np.min(polygon_points[:, 0])
    top_left_corner_y = np.min(polygon_points[:, 1])
    bottom_right_corner_x = np.max(polygon_points[:, 0])
    bottom_right_corner_y = np.max(polygon_points[:, 1])

    mask = np.zeros_like(image_)
    cv2.fillPoly(mask, [polygon_points], (255, 255, 255))
    masked_image = cv2.bitwise_and(image_, mask)
    output_image = masked_image[top_left_corner_y:bottom_right_corner_y, top_left_corner_x:bottom_right_corner_x]
    for point in polygon_points:
        cv2.circle(image_, tuple(point), radius=2, color=(0, 0, 255), thickness=-1)
    
    return output_image

def left_lip_bounding_box(hor_, ver_):
    '''
        detect bounding box of left lip
    '''
    top_left_corner_x = mean_([hor_[48], hor_[36]])
    # top_left_corner_x = 
    top_left_corner_y = ver_[33]
    bottom_right_corner_x = mean_([hor_[51], hor_[62], hor_[66], hor_[57]])
    bottom_right_corner_y = mean_([mean_([ver_[48], ver_[54]]), ver_[8]])
    return top_left_corner_x, top_left_corner_y, bottom_right_corner_x, bottom_right_corner_y

def right_lip_bounding_box(hor_, ver_):
    '''
        detect bounding box of right lip
    '''
    top_left_corner_x = mean_([hor_[51], hor_[62], hor_[66], hor_[57]])
    top_left_corner_y = ver_[33]
    bottom_right_corner_x = mean_([hor_[54], hor_[45]])
    bottom_right_corner_y = mean_([mean_([ver_[48], ver_[54]]), ver_[8]])
    return top_left_corner_x, top_left_corner_y, bottom_right_corner_x, bottom_right_corner_y

# def left_nose_bounding_box(hor_, ver_):
#     '''
#         detect bounding box of right eye
#     '''
#     top_left_corner_x = ver_[22]
#     top_left_corner_y = mean_([mean_([hor_[41], hor_[46]]), 0])
#     bottom_right_corner_x = mean_([ver_[26], ver_[16]])
#     bottom_right_corner_y = mean_([mean_([hor_[41], hor_[46]]), hor_[30]])
#     return list(top_left_corner_x, top_left_corner_y, bottom_right_corner_x, bottom_right_corner_y)

# def right_nose_bounding_box(hor_, ver_):
#     '''
#         detect bounding box of right eye
#     '''
#     top_left_corner_x = ver_[22]
#     top_left_corner_y = mean_([mean_([hor_[41], hor_[46]]), 0])
#     bottom_right_corner_x = mean_([ver_[26], ver_[16]])
#     bottom_right_corner_y = mean_([mean_([hor_[41], hor_[46]]), hor_[30]])
#     return list(top_left_corner_x, top_left_corner_y, bottom_right_corner_x, bottom_right_corner_y)

def crop_six_parts(image):
    '''
    crop a face image to 6 parts: left eye, right eye, left cheek,
                                  right cheek, left lip, right lip
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = hog_face_detector(gray)
    face4cr = faces[0]
    face_landmarks = dlib_facelandmark(gray, face4cr)

    ver_ = []
    hor_ = []
    for n in range(68):
        hor_.append(face_landmarks.part(n).x)
        ver_.append(face_landmarks.part(n).y)
        
    # crop eye
    x_min, y_min, x_max, y_max = left_eye_bounding_box(hor_, ver_)
    left_eye = image[y_min:y_max, x_min:x_max]
    x_min, y_min, x_max, y_max = right_eye_bounding_box(hor_, ver_)
    right_eye = image[y_min:y_max, x_min:x_max]
    
    # crop cheek
    left_cheek = crop_left_cheek(image, hor_, ver_)
    right_cheek = crop_right_cheek(image, hor_, ver_)
    
    # crop lip
    x_min, y_min, x_max, y_max = left_lip_bounding_box(hor_, ver_)
    left_lip = image[y_min:y_max, x_min:x_max]
    x_min, y_min, x_max, y_max = right_lip_bounding_box(hor_, ver_)
    right_lip = image[y_min:y_max, x_min:x_max]
    
    # crop nose
    # image_07_01 = image[mean_([hor_[21], hor_[22]]):hor_[33], mean_([ver_[21], ver_[31]]):mean_([ver_[22], ver_[35]])]
    # image_07_02 = image[hor_[28]:hor_[33], mean_([ver_[39], ver_[40]]):mean_([ver_[42], ver_[47]])]
    
    return left_eye, right_eye, left_cheek, right_cheek, left_lip, right_lip

if __name__ == '__main__':
    
    config = load_config()
    output_folder = config['train_dir']
    
    #------------------------------------------------- Label 0 -------------------------------------------------------------------
    input_folders = ['/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/CHEEK/sub_1_skin1',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/CHEEK/sub_2_skin2',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/CHEEK/sub_4_skin4',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/CHEEK/sub_5_skin5',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/CHEEK/sub_6_skin6',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/EYE/sub_1_1',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/EYE/sub_2_2',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/EYE/sub_4_4',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/EYE/sub_5_5',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/EYE/sub_6_6',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/LIP/sub_1_1',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/LIP/sub_2_2',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/LIP/sub_4_4',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/LIP/sub_5_5',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/LIP/sub_6_6',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/EYELINE/makeup/right_eyeline/sub_1_e0',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/EYELINE/makeup/right_eyeline/sub_2_e2',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/EYELINE/makeup/right_eyeline/sub_4_e4',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/EYELINE/makeup/right_eyeline/sub_5_e5',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/EYELINE/makeup/right_eyeline/sub_6_e6',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/EYELINE/makeup/left_eyeline/sub_1_e0',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/EYELINE/makeup/left_eyeline/sub_2_e2',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/EYELINE/makeup/left_eyeline/sub_4_e4',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/EYELINE/makeup/left_eyeline/sub_5_e5',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/EYELINE/makeup/left_eyeline/sub_6_e6',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL0/EYELINE/nonmake'
                     ]
    
    output_folder_lb0 = os.path.join(output_folder, 'label_0')

    eye_folder = os.path.join(output_folder_lb0, 'eye')
    cheek_folder = os.path.join(output_folder_lb0, 'cheek')
    lip_folder = os.path.join(output_folder_lb0, 'lip')
    
    os.makedirs(eye_folder, exist_ok=True)
    os.makedirs(cheek_folder, exist_ok=True)
    os.makedirs(lip_folder, exist_ok=True)
    
    for input_folder in input_folders:
        for image_name in os.listdir(input_folder):
            image_path = os.path.join(input_folder, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, (512, 512))
                
                left_eye, right_eye, left_cheek, right_cheek, left_lip, right_lip = crop_six_parts(image)
                
                name = image_name.split('.')[0]
                mistake_code = name.split('_')[1]
                
                if(mistake_code[0] != '0'):
                    save_folder = os.path.join(eye_folder, name)
                    os.makedirs(save_folder, exist_ok=True)
                    cv2.imwrite(os.path.join(save_folder, 'left_eye.png'), left_eye)
                    cv2.imwrite(os.path.join(save_folder, 'right_eye.png'), right_eye)
                if(mistake_code[1] != '0'):
                    save_folder = os.path.join(cheek_folder, name)
                    os.makedirs(save_folder, exist_ok=True)
                    cv2.imwrite(os.path.join(save_folder, 'left_cheek.png'), left_cheek)
                    cv2.imwrite(os.path.join(save_folder, 'right_cheek.png'), right_cheek)
                if(mistake_code[2] != '0'):
                    save_folder = os.path.join(lip_folder, name)
                    os.makedirs(save_folder, exist_ok=True)
                    cv2.imwrite(os.path.join(save_folder, 'left_lip.png'), left_lip)
                    cv2.imwrite(os.path.join(save_folder, 'right_lip.png'), right_lip)

    #------------------------------------------------- Label 1 -------------------------------------------------------------------    
    input_folders = ['/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL1/CHEEK/sub_1_c0',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL1/CHEEK/sub_2_c2',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL1/CHEEK/sub_4_c4',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL1/CHEEK/sub_5_c5',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL1/CHEEK/sub_6_c6',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL1/EYE/sub_1_e0',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL1/EYE/sub_2_e2',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL1/EYE/sub_4_e4',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL1/EYE/sub_5_e5',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL1/EYE/sub_6_e6',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL1/LIP/sub_1_l0',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL1/LIP/sub_2_l2',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL1/LIP/sub_4_l4',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL1/LIP/sub_5_l5',
                     '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL1/LIP/sub_6_l6'
                     ]
    
    output_folder_lb1 = os.path.join(output_folder, 'label_1')

    eye_folder = os.path.join(output_folder_lb1, 'eye')
    cheek_folder = os.path.join(output_folder_lb1, 'cheek')
    lip_folder = os.path.join(output_folder_lb1, 'lip')
    
    os.makedirs(eye_folder, exist_ok=True)
    os.makedirs(cheek_folder, exist_ok=True)
    os.makedirs(lip_folder, exist_ok=True)
    
    for input_folder in input_folders:
        for image_name in os.listdir(input_folder):
            image_path = os.path.join(input_folder, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, (512, 512))
                
                left_eye, right_eye, left_cheek, right_cheek, left_lip, right_lip = crop_six_parts(image)
                
                name = image_name.split('.')[0]
                
                save_folder = os.path.join(eye_folder, name)
                os.makedirs(save_folder, exist_ok=True)
                cv2.imwrite(os.path.join(save_folder, 'left_eye.png'), left_eye)
                cv2.imwrite(os.path.join(save_folder, 'right_eye.png'), right_eye)
                save_folder = os.path.join(cheek_folder, name)
                os.makedirs(save_folder, exist_ok=True)
                cv2.imwrite(os.path.join(save_folder, 'left_cheek.png'), left_cheek)
                cv2.imwrite(os.path.join(save_folder, 'right_cheek.png'), right_cheek)
                save_folder = os.path.join(lip_folder, name)
                os.makedirs(save_folder, exist_ok=True)
                cv2.imwrite(os.path.join(save_folder, 'left_lip.png'), left_lip)
                cv2.imwrite(os.path.join(save_folder, 'right_lip.png'), right_lip)