import os
import cv2
import dlib
import statistics
import numpy as np
from utils import load_config
import math, copy
from tqdm import tqdm
import gc

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

output_folder_image_test = '/media/nguyenminhquan/fshit/SymmetricalMakeup/test'

def mean_(list_):
    return int(statistics.mean(list_))
def calc_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
def rotate_image_with_landmarks(image, hor_, ver_, idx_point_1, idx_point_2, degree = 0):
    image_cp = copy.deepcopy(image)
    hor_cp = copy.deepcopy(hor_)
    ver_cp = copy.deepcopy(ver_)
    
    # rotate image depend on left eye
    point_1 = (hor_cp[idx_point_1], ver_cp[idx_point_1])
    point_2 = (hor_cp[idx_point_2], ver_cp[idx_point_2])
    delta_x = point_2[0] - point_1[0]
    delta_y = point_2[1] - point_1[1]
    
    angle_radians = math.atan2(delta_y, delta_x)  
    angle_degrees = math.degrees(angle_radians)  
    rotation_angle = angle_degrees - degree
    (h, w) = image_cp.shape[:2]
    center = (w // 2, h // 2) 
    
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_image = cv2.warpAffine(image_cp, rotation_matrix, (w, h))
    
    # rotate landmarks
    for idx in range(68):
        tmp_point = np.array([hor_cp[idx], ver_cp[idx], 1])
        tmp_point = np.dot(rotation_matrix, tmp_point.T)
        hor_cp[idx] = int(tmp_point[0])
        ver_cp[idx] = int(tmp_point[1])
    # return rotated_image, hor_cp, ver_cp
    return image, hor_, ver_

def draw(image, key_points_hor, key_points_ver):
    image_cp = copy.deepcopy(image)
    for idx in range(len(key_points_hor)):
        cv2.circle(image_cp, (key_points_hor[idx], key_points_ver[idx]), 2, (250, 0, 0), -1)
    return image_cp
def draw2(image, key_points):
    image_cp = copy.deepcopy(image)
    for idx, points in enumerate(key_points):
        # cv2.circle(image_cp, points, 10, (250, idx*30, 250 - idx*30), -1)
        int_points = (int(points[0]), int(points[1]))
        # cv2.circle(image_cp, int_points, 2, (0, 250, 0), -1)
        cv2.circle(image_cp, int_points, 3, (250, idx*30, 250 - idx*30), -1)
        # cv2.circle(image_cp, tuple(points), 10, (250, 250, 250), -1)
    return image_cp

def draw3(image, key_points):
    image_cp = copy.deepcopy(image)
    for idx, points in enumerate(key_points):
        # cv2.circle(image_cp, points, 10, (250, idx*30, 250 - idx*30), -1)
        int_points = (int(points[0]), int(points[1]))
        cv2.circle(image_cp, int_points, 2, (250, 0, 0), -1)
        # cv2.circle(image_cp, tuple(points), 10, (250, 250, 250), -1)
    return image_cp
    
#####################################################
###    Determine the bounding box of each part    ###
#####################################################
def left_eye_bounding_box(hor_, ver_):
    '''
        detect bouding box of left eye
    '''
    top_left_corner_x = hor_[17]
    top_left_corner_y = mean_([ver_[19], ver_[24]])
    bottom_right_corner_x = mean_([hor_[21], hor_[31]])
    bottom_right_corner_y = mean_([mean_([ver_[41], ver_[46]]), ver_[30]])
    return top_left_corner_x, top_left_corner_y, bottom_right_corner_x, bottom_right_corner_y

def right_eye_bounding_box(hor_, ver_):
    '''
        detect bounding box of right eye
    '''
    top_left_corner_x = mean_([hor_[22], hor_[35]])
    top_left_corner_y = mean_([ver_[19], ver_[24]])
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
    polygon_points.append([hor_[0],ver_[0]])
    polygon_points.append([hor_[1],ver_[1]])
    polygon_points.append([hor_[2],ver_[2]])
    polygon_points.append([hor_[3],ver_[3]])
    polygon_points.append([hor_[36], ver_[33]])
    polygon_points.append([mean_([hor_[39], hor_[40]]),ver_[30]])
    
    polygon_points = np.array(polygon_points)
    top_left_corner_x = np.min(polygon_points[:, 0])
    top_left_corner_y = np.min(polygon_points[:, 1])
    bottom_right_corner_x = np.max(polygon_points[:, 0])
    bottom_right_corner_y = np.max(polygon_points[:, 1])
    left_cheek_bbox = [top_left_corner_x, top_left_corner_y, bottom_right_corner_x, bottom_right_corner_y]

    mask = np.zeros_like(image_)
    cv2.fillPoly(mask, [polygon_points], (255, 255, 255))
    masked_image = cv2.bitwise_and(image_, mask)
    output_image = masked_image[top_left_corner_y:bottom_right_corner_y, top_left_corner_x:bottom_right_corner_x]
    
    return np.array(output_image), polygon_points, left_cheek_bbox

def crop_right_cheek(image, hor_, ver_):
    '''
        detect right cheek
    '''
    image_ = image.copy()
    polygon_points = []
    polygon_points.append([mean_([hor_[22], hor_[35]]),mean_([mean_([ver_[41], ver_[46]]), ver_[30]])])
    polygon_points.append([hor_[45],mean_([mean_([ver_[41], ver_[46]]), ver_[30]])])
    polygon_points.append([hor_[16],ver_[16]])
    polygon_points.append([hor_[15],ver_[15]])
    polygon_points.append([hor_[14],ver_[14]])
    polygon_points.append([hor_[13],ver_[13]])
    polygon_points.append([hor_[45], ver_[33]])
    polygon_points.append([mean_([hor_[42], hor_[47]]),ver_[30]])
    
    polygon_points = np.array(polygon_points)
    top_left_corner_x = np.min(polygon_points[:, 0])
    top_left_corner_y = np.min(polygon_points[:, 1])
    bottom_right_corner_x = np.max(polygon_points[:, 0])
    bottom_right_corner_y = np.max(polygon_points[:, 1])
    right_cheek_bbox = [top_left_corner_x, top_left_corner_y, bottom_right_corner_x, bottom_right_corner_y]

    mask = np.zeros_like(image_)
    cv2.fillPoly(mask, [polygon_points], (255, 255, 255))
    masked_image = cv2.bitwise_and(image_, mask)
    output_image = masked_image[top_left_corner_y:bottom_right_corner_y, top_left_corner_x:bottom_right_corner_x]
    
    return np.array(output_image), polygon_points, right_cheek_bbox

def left_lip_bounding_box(hor_, ver_):
    '''
        detect bounding box of left lip
    '''
    top_left_corner_x = mean_([hor_[48], hor_[36]])
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

#####################################################
###         Crop and transform each part          ###
#####################################################

def transform_image(resource_image, dest_image, key_points_of_src, key_points_of_dest, source_bbox, is_cheek, source_image):
    key_points_of_src = np.array(key_points_of_src, dtype=np.float32)
    key_points_of_dest = np.array(key_points_of_dest, dtype=np.float32)
    
    key_points_of_src_cp = copy.deepcopy(key_points_of_src)
    key_points_of_dest_cp = copy.deepcopy(key_points_of_dest)
    
    h, w, _ = dest_image.shape
    lim_h, lim_w, _ = resource_image.shape
    
    dest_image = cv2.flip(dest_image, 1)
    for point in key_points_of_dest_cp:
        point[0] = w - point[0] - 1
        
    trans_matrix, _ = cv2.estimateAffinePartial2D(key_points_of_src_cp, key_points_of_dest_cp)
    transformed_source_image = np.zeros_like(dest_image)
    trans_matrix_inv = cv2.invertAffineTransform(trans_matrix)
    x_min, y_min, x_max, y_max = source_bbox
    for row in range(h):
        for col in range(w):
            if not is_cheek or (is_cheek and not np.array_equal(dest_image[row, col], [0, 0, 0])):
                transformed_pixel = np.dot(trans_matrix_inv, np.array([col, row, 1]))
                
                original_x = int(transformed_pixel[0]) + x_min
                original_y = int(transformed_pixel[1]) + y_min
                
                if 0 <= original_x < lim_w and 0 <= original_y < lim_h:
                    transformed_source_image[row, col] = resource_image[original_y, original_x]
    return transformed_source_image

def transform_image_2(resource_image, dest_image, key_points_of_src, key_points_of_dest, source_bbox, is_cheek, source_image):
    key_points_of_src = np.array(key_points_of_src, dtype=np.float32)
    key_points_of_dest = np.array(key_points_of_dest, dtype=np.float32)
    
    key_points_of_src_cp = copy.deepcopy(key_points_of_src)
    key_points_of_dest_cp = copy.deepcopy(key_points_of_dest)
    
    h, w, _ = dest_image.shape
    h_src, w_src, _ = source_image.shape
    lim_h, lim_w, _ = resource_image.shape
    
    dest_image = cv2.flip(dest_image, 1)
    for point in key_points_of_dest_cp:
        point[0] = w - point[0] - 1
    
    src_img = copy.deepcopy(source_image)
    trans_matrix, _ = cv2.estimateAffinePartial2D(key_points_of_src_cp, key_points_of_dest_cp)

    transformed_source_image = cv2.warpAffine(src_img, trans_matrix, (w, h))
    trans_matrix_inv = cv2.invertAffineTransform(trans_matrix)
    x_min, y_min, x_max, y_max = source_bbox
    # for row in range(h):
    #     for col in range(w):
    #         if not is_cheek or (is_cheek and not np.array_equal(dest_image[row, col], [0, 0, 0])):
    #             transformed_pixel = np.dot(trans_matrix_inv, np.array([col, row, 1]))

    #             original_x = int(transformed_pixel[0]) + x_min
    #             original_y = int(transformed_pixel[1]) + y_min
                
    #             if x_min <= original_x < x_max and y_min <= original_y < y_max:
    #                 transformed_source_image[row, col] = resource_image[original_y, original_x]
    # ones_column = np.ones((key_points_of_src_resize.shape[0], 1))
    # key_points_with_ones = np.hstack([key_points_of_src_resize, ones_column])

    # transformed_keypoints = np.dot(trans_matrix, key_points_with_ones.T).T
        
    # transformed_source_image_cp = draw2(transformed_source_image, transformed_keypoints)
    # print(transformed_keypoints)
    # cv2.imwrite(os.path.join(toi_khong_sai_folder, 'trans_image_kp.png'), transformed_source_image_cp)
    return transformed_source_image

def crop_and_transform_six_parts(image):
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
    
    #-------------------------#
    #----- crop left eye -----#
    #-------------------------#
    
    # rotate image and landmarks depend on left eye
    rotated_left_eye_image, hor_cp, ver_cp = rotate_image_with_landmarks(image, hor_, ver_, 36, 39, 0)
     
    x_min, y_min, x_max, y_max = left_eye_bounding_box(hor_cp, ver_cp)
    left_eye_bbox = [x_min, y_min, x_max, y_max]
    
    # crop left eye
    left_eye = rotated_left_eye_image[y_min:y_max, x_min:x_max]
    
    # determine the key points of left eye
    left_eye_key_points = [[hor_cp[36], ver_cp[36]], [hor_cp[37], ver_cp[37]], [hor_cp[39], ver_cp[39]], [hor_cp[40], ver_cp[40]],
                          [hor_cp[17], ver_cp[17]], [hor_cp[19], ver_cp[19]], [hor_cp[21], ver_cp[21]]]
    for point in left_eye_key_points:
        point[0] -= x_min
        point[1] -= y_min
    
    #--------------------------#
    #----- crop right eye -----#
    #--------------------------#
    
    # rotate image and landmarks depend on right eye
    rotated_right_eye_image, hor_cp, ver_cp = rotate_image_with_landmarks(image, hor_, ver_, 42, 45, 0)
    
    x_min, y_min, x_max, y_max = right_eye_bounding_box(hor_cp, ver_cp)
    right_eye_bbox = [x_min, y_min, x_max, y_max]
    
    # crop right eye
    right_eye = rotated_right_eye_image[y_min:y_max, x_min:x_max]
    
    # determine the key points of right eye
    right_eye_key_points = [[hor_cp[45], ver_cp[45]], [hor_cp[44], ver_cp[44]], [hor_cp[42], ver_cp[42]], [hor_cp[47], ver_cp[47]],
                          [hor_cp[26], ver_cp[26]], [hor_cp[24], ver_cp[24]], [hor_cp[22], ver_cp[22]]]
    for point in right_eye_key_points:
        point[0] -= x_min
        point[1] -= y_min   
    
    #------------------------------#
    #----- crop lip and cheek -----#
    #------------------------------#
    
    # rotate image and landmarks depend on the lips
    rotated_lip_image, hor_cp, ver_cp = rotate_image_with_landmarks(image, hor_, ver_, 51, 57, 90)
    
    x_min, y_min, x_max, y_max = left_lip_bounding_box(hor_cp, ver_cp)
    left_lip_bbox = [x_min, y_min, x_max, y_max]
    
    # crop left lip
    left_lip = rotated_lip_image[y_min:y_max, x_min:x_max]
    
    # determine the key points of left lip
    left_lip_key_points = [[hor_cp[48], ver_cp[48]], [hor_cp[49], ver_cp[49]], [hor_cp[50], ver_cp[50]], [hor_cp[51], ver_cp[51]], [hor_cp[62], ver_cp[62]],
                           [hor_cp[66], ver_cp[66]], [hor_cp[57], ver_cp[57]], [hor_cp[58], ver_cp[58]], [hor_cp[59], ver_cp[59]]]

    for point in left_lip_key_points:
        point[0] -= x_min
        point[1] -= y_min   
    
    x_min, y_min, x_max, y_max = right_lip_bounding_box(hor_cp, ver_cp)
    right_lip_bbox = [x_min, y_min, x_max, y_max]
    
    # crop right lip
    right_lip = rotated_lip_image[y_min:y_max, x_min:x_max]
    
    # determine the key points of right lip
    right_lip_key_points = [[hor_cp[54], ver_cp[54]], [hor_cp[53], ver_cp[53]], [hor_cp[52], ver_cp[52]], [hor_cp[51], ver_cp[51]], [hor_cp[62], ver_cp[62]],
                            [hor_cp[66], ver_cp[66]], [hor_cp[57], ver_cp[57]], [hor_cp[56], ver_cp[56]], [hor_cp[55], ver_cp[55]]]

    for point in right_lip_key_points:
        point[0] -= x_min
        point[1] -= y_min
    
    # crop cheek
    left_cheek, left_cheek_key_points, left_cheek_bbox = crop_left_cheek(rotated_lip_image, hor_cp, ver_cp)
    right_cheek, right_cheek_key_points, right_cheek_bbox = crop_right_cheek(rotated_lip_image, hor_cp, ver_cp)
    x_min, y_min, x_max, y_max = left_cheek_bbox
    for point in left_cheek_key_points:
        point[0] -= x_min
        point[1] -= y_min
    x_min, y_min, x_max, y_max = right_cheek_bbox
    for point in right_cheek_key_points:
        point[0] -= x_min
        point[1] -= y_min
        
    #--------------------------------#
    #----- Transforms each part -----#
    #--------------------------------#
    
    # transform eye
    if calc_area(left_eye_bbox) > calc_area(right_eye_bbox):
        left_eye = transform_image(rotated_left_eye_image, right_eye, left_eye_key_points, right_eye_key_points, left_eye_bbox, False, left_eye)
    else:
        right_eye = transform_image(rotated_right_eye_image, left_eye, right_eye_key_points, left_eye_key_points, right_eye_bbox, False, right_eye)
        
    #transform cheek
    if calc_area(left_cheek_bbox) > calc_area(right_cheek_bbox):
        left_cheek = transform_image(rotated_lip_image, right_cheek, left_cheek_key_points, right_cheek_key_points, left_cheek_bbox, True, left_cheek)
    else:
        right_cheek = transform_image(rotated_lip_image, left_cheek, right_cheek_key_points, left_cheek_key_points, right_cheek_bbox, True, right_cheek)
        
    # transform lip
    if calc_area(left_lip_bbox) > calc_area(right_lip_bbox):
        left_lip = transform_image(rotated_lip_image, right_lip, left_lip_key_points, right_lip_key_points, left_lip_bbox, False, left_lip)
    else:
        right_lip = transform_image(rotated_lip_image, left_lip, right_lip_key_points, left_lip_key_points, right_lip_bbox, False, right_lip)

    return left_eye, right_eye, left_cheek, right_cheek, left_lip, right_lip

if __name__ == '__main__':
    
    ######################################################################################################
    ###############            skip crop cheek
    ######################################################################################################
    
    
    # folder_image = '/mnt/2E287FDD287FA28F/teamAI/hong/FACE_DATA/1210_DATA_POSE/make/make_good_mini'
    # output_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/FACE_DATA/1210_DATA_POSE/make'
    # os.makedirs(output_folder, exist_ok=True)
    # for image_name in os.listdir(folder_image):
    #     image_path = os.path.join(folder_image, image_name)
    #     image = cv2.imread(image_path)
    #     output_folder_image = os.path.join(output_folder, image_name)
    #     os.makedirs(output_folder_image, exist_ok=True)
    #     left_eye, right_eye, left_cheek, right_cheek, left_lip, right_lip = crop_and_transform_six_parts(image)
        
    #     cv2.imwrite(os.path.join(output_folder_image, 'left_e.jpg'), left_eye)
    #     cv2.imwrite(os.path.join(output_folder_image, 'right_e.jpg'), right_eye)
    #     cv2.imwrite(os.path.join(output_folder_image, 'left_c.jpg'), left_cheek)
    #     cv2.imwrite(os.path.join(output_folder_image, 'right_c.jpg'), right_cheek)
    #     cv2.imwrite(os.path.join(output_folder_image, 'left_l.jpg'), left_lip)
    #     cv2.imwrite(os.path.join(output_folder_image, 'right_l.jpg'), right_lip)
    
    #-------------------------------------------------------------------------------------------------------
    #------------------------------------------- LABEL 1 ---------------------------------------------------
    #-------------------------------------------------------------------------------------------------------
    
    # input_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA_2109/LABEL1'
    # output_folder = '/mnt/2E287FDD287FA28F/teamAI/MinhQuan/siamesenn/Symetry-of-Facial-Makeup/ProtoZenon123-Symmetry-of-Facial-Make-up/DATA_ALL/data/train_data'
    # output_folder_lb1 = os.path.join(output_folder, 'label_1')

    # eye_folder = os.path.join(output_folder_lb1, 'eye')
    # cheek_folder = os.path.join(output_folder_lb1, 'cheek')
    # lip_folder = os.path.join(output_folder_lb1, 'lip')
    
    # os.makedirs(eye_folder, exist_ok=True)
    # os.makedirs(cheek_folder, exist_ok=True)
    # os.makedirs(lip_folder, exist_ok=True)
    
    # for cpnt_folder in os.listdir(input_folder):
    #     cpnt_folder_dir = os.path.join(input_folder, cpnt_folder)
    #     for sub_folder in os.listdir(cpnt_folder_dir):
    #         sub_folder_dir = os.path.join(cpnt_folder_dir, sub_folder)
    #         for image_name in tqdm(os.listdir(sub_folder_dir), desc="crop for train_data lb1", leave=False):
    #             image_path = os.path.join(sub_folder_dir, image_name)
    #             image = cv2.imread(image_path)
    #             if image is not None:
    #                 if cpnt_folder[0] == 'E':
    #                     image = cv2.resize(image, (512, 512))
                        
    #                     left_eye, right_eye, left_cheek, right_cheek, left_lip, right_lip = crop_and_transform_six_parts(image)
                        
    #                     name = image_name.split('.')[0]
                        
    #                     save_folder = os.path.join(eye_folder, name)
    #                     os.makedirs(save_folder, exist_ok=True)
    #                     cv2.imwrite(os.path.join(save_folder, 'left_eye.png'), left_eye)
    #                     cv2.imwrite(os.path.join(save_folder, 'right_eye.png'), right_eye)
                    
    #             if image is not None:
    #                 if cpnt_folder[0] == 'C':
    #                     image = cv2.resize(image, (512, 512))
                        
    #                     left_eye, right_eye, left_cheek, right_cheek, left_lip, right_lip = crop_and_transform_six_parts(image)
                        
    #                     name = image_name.split('.')[0]
                        
    #                     save_folder = os.path.join(cheek_folder, name)
    #                     os.makedirs(save_folder, exist_ok=True)
    #                     cv2.imwrite(os.path.join(save_folder, 'left_cheek.png'), left_cheek)
    #                     cv2.imwrite(os.path.join(save_folder, 'right_cheek.png'), right_cheek)
            
    #             if image is not None:
    #                 if cpnt_folder[0] == 'L':
    #                     image = cv2.resize(image, (512, 512))
                        
    #                     left_eye, right_eye, left_cheek, right_cheek, left_lip, right_lip = crop_and_transform_six_parts(image)
                        
    #                     name = image_name.split('.')[0]
                        
    #                     save_folder = os.path.join(lip_folder, name)
    #                     os.makedirs(save_folder, exist_ok=True)
    #                     cv2.imwrite(os.path.join(save_folder, 'left_lip.png'), left_lip)
    #                     cv2.imwrite(os.path.join(save_folder, 'right_lip.png'), right_lip)
    #             del image
    #             gc.collect()
                        
    # input_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA4VAL/LABEL1'
    # output_folder = '/mnt/2E287FDD287FA28F/teamAI/MinhQuan/siamesenn/Symetry-of-Facial-Makeup/ProtoZenon123-Symmetry-of-Facial-Make-up/DATA_ALL/data/val_data'
    # output_folder_lb1 = os.path.join(output_folder, 'label_1')

    # eye_folder = os.path.join(output_folder_lb1, 'eye')
    # cheek_folder = os.path.join(output_folder_lb1, 'cheek')
    # lip_folder = os.path.join(output_folder_lb1, 'lip')
    
    # os.makedirs(eye_folder, exist_ok=True)
    # os.makedirs(cheek_folder, exist_ok=True)
    # os.makedirs(lip_folder, exist_ok=True)
    
    # for cpnt_folder in os.listdir(input_folder):
    #     cpnt_folder_dir = os.path.join(input_folder, cpnt_folder)
    #     for sub_folder in os.listdir(cpnt_folder_dir):
    #         sub_folder_dir = os.path.join(cpnt_folder_dir, sub_folder)
    #         for image_name in tqdm(os.listdir(sub_folder_dir), desc="crop for val_data lb1", leave=False):
    #             image_path = os.path.join(sub_folder_dir, image_name)
    #             image = cv2.imread(image_path)
    #             if image is not None:
    #                 if cpnt_folder[0] == 'E':
    #                     image = cv2.resize(image, (512, 512))
                        
    #                     left_eye, right_eye, left_cheek, right_cheek, left_lip, right_lip = crop_and_transform_six_parts(image)
                        
    #                     name = image_name.split('.')[0]
                        
    #                     save_folder = os.path.join(eye_folder, name)
    #                     os.makedirs(save_folder, exist_ok=True)
    #                     cv2.imwrite(os.path.join(save_folder, 'left_eye.png'), left_eye)
    #                     cv2.imwrite(os.path.join(save_folder, 'right_eye.png'), right_eye)
                    
    #             if image is not None:
    #                 if cpnt_folder[0] == 'C':
    #                     image = cv2.resize(image, (512, 512))
                        
    #                     left_eye, right_eye, left_cheek, right_cheek, left_lip, right_lip = crop_and_transform_six_parts(image)
                        
    #                     name = image_name.split('.')[0]
                        
    #                     save_folder = os.path.join(cheek_folder, name)
    #                     os.makedirs(save_folder, exist_ok=True)
    #                     cv2.imwrite(os.path.join(save_folder, 'left_cheek.png'), left_cheek)
    #                     cv2.imwrite(os.path.join(save_folder, 'right_cheek.png'), right_cheek)
            
    #             if image is not None:
    #                 if cpnt_folder[0] == 'L':
    #                     image = cv2.resize(image, (512, 512))
                        
    #                     left_eye, right_eye, left_cheek, right_cheek, left_lip, right_lip = crop_and_transform_six_parts(image)
                        
    #                     name = image_name.split('.')[0]
                        
    #                     save_folder = os.path.join(lip_folder, name)
    #                     os.makedirs(save_folder, exist_ok=True)
    #                     cv2.imwrite(os.path.join(save_folder, 'left_lip.png'), left_lip)
    #                     cv2.imwrite(os.path.join(save_folder, 'right_lip.png'), right_lip)
    #             del image
    #             gc.collect()
    
    input_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/DATA4TEST/LABEL1'
    output_folder = '/mnt/2E287FDD287FA28F/teamAI/MinhQuan/siamesenn/Symetry-of-Facial-Makeup/ProtoZenon123-Symmetry-of-Facial-Make-up/DATA_ALL/data/test_data'
    output_folder_lb1 = os.path.join(output_folder, 'label_1')

    eye_folder = os.path.join(output_folder_lb1, 'eye')
    cheek_folder = os.path.join(output_folder_lb1, 'cheek')
    lip_folder = os.path.join(output_folder_lb1, 'lip')
    
    os.makedirs(eye_folder, exist_ok=True)
    os.makedirs(cheek_folder, exist_ok=True)
    os.makedirs(lip_folder, exist_ok=True)
    
    for cpnt_folder in os.listdir(input_folder):
        cpnt_folder_dir = os.path.join(input_folder, cpnt_folder)
        for sub_folder in os.listdir(cpnt_folder_dir):
            sub_folder_dir = os.path.join(cpnt_folder_dir, sub_folder)
            for image_name in tqdm(os.listdir(sub_folder_dir), desc="crop for test_data_lb1", leave=False):
                image_path = os.path.join(sub_folder_dir, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    if cpnt_folder[0] == 'E':
                        image = cv2.resize(image, (512, 512))
                        
                        left_eye, right_eye, left_cheek, right_cheek, left_lip, right_lip = crop_and_transform_six_parts(image)
                        
                        name = image_name.split('.')[0]
                        
                        save_folder = os.path.join(eye_folder, name)
                        os.makedirs(save_folder, exist_ok=True)
                        cv2.imwrite(os.path.join(save_folder, 'left_eye.png'), left_eye)
                        cv2.imwrite(os.path.join(save_folder, 'right_eye.png'), right_eye)
                    
                if image is not None:
                    if cpnt_folder[0] == 'C':
                        image = cv2.resize(image, (512, 512))
                        
                        left_eye, right_eye, left_cheek, right_cheek, left_lip, right_lip = crop_and_transform_six_parts(image)
                        
                        name = image_name.split('.')[0]
                        
                        save_folder = os.path.join(cheek_folder, name)
                        os.makedirs(save_folder, exist_ok=True)
                        cv2.imwrite(os.path.join(save_folder, 'left_cheek.png'), left_cheek)
                        cv2.imwrite(os.path.join(save_folder, 'right_cheek.png'), right_cheek)
            
                if image is not None:
                    if cpnt_folder[0] == 'L':
                        image = cv2.resize(image, (512, 512))
                        
                        left_eye, right_eye, left_cheek, right_cheek, left_lip, right_lip = crop_and_transform_six_parts(image)
                        
                        name = image_name.split('.')[0]
                        
                        save_folder = os.path.join(lip_folder, name)
                        os.makedirs(save_folder, exist_ok=True)
                        cv2.imwrite(os.path.join(save_folder, 'left_lip.png'), left_lip)
                        cv2.imwrite(os.path.join(save_folder, 'right_lip.png'), right_lip)
                del image
                gc.collect()
                
    output_folder = '/mnt/2E287FDD287FA28F/teamAI/MinhQuan/siamesenn/Symetry-of-Facial-Makeup/ProtoZenon123-Symmetry-of-Facial-Make-up/DATA_ALL/data/train_data'
    input_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/GANFORMER'
    output_folder_lb1 = os.path.join(output_folder, 'label_1')

    eye_folder = os.path.join(output_folder_lb1, 'eye')
    cheek_folder = os.path.join(output_folder_lb1, 'cheek')
    lip_folder = os.path.join(output_folder_lb1, 'lip')
    
    os.makedirs(eye_folder, exist_ok=True)
    os.makedirs(cheek_folder, exist_ok=True)
    os.makedirs(lip_folder, exist_ok=True)
    for image_name in tqdm(os.listdir(input_folder), desc="crop for train_data lb1 non-make", leave=False):
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, (512, 512))
            
            left_eye, right_eye, left_cheek, right_cheek, left_lip, right_lip = crop_and_transform_six_parts(image)
            
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
        del image
        gc.collect()    
                    
    output_folder = '/mnt/2E287FDD287FA28F/teamAI/MinhQuan/siamesenn/Symetry-of-Facial-Makeup/ProtoZenon123-Symmetry-of-Facial-Make-up/DATA_ALL/data/val_data'
    input_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/GANFORMER4VAL'
    output_folder_lb1 = os.path.join(output_folder, 'label_1')

    eye_folder = os.path.join(output_folder_lb1, 'eye')
    cheek_folder = os.path.join(output_folder_lb1, 'cheek')
    lip_folder = os.path.join(output_folder_lb1, 'lip')
    
    os.makedirs(eye_folder, exist_ok=True)
    os.makedirs(cheek_folder, exist_ok=True)
    os.makedirs(lip_folder, exist_ok=True)
    for image_name in tqdm(os.listdir(input_folder), desc="crop for val_data lb1 non-make", leave=False):
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, (512, 512))
            
            left_eye, right_eye, left_cheek, right_cheek, left_lip, right_lip = crop_and_transform_six_parts(image)
            
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
        del image
        gc.collect()
        
    output_folder = '/mnt/2E287FDD287FA28F/teamAI/MinhQuan/siamesenn/Symetry-of-Facial-Makeup/ProtoZenon123-Symmetry-of-Facial-Make-up/DATA_ALL/data/test_data'
    input_folder = '/mnt/2E287FDD287FA28F/teamAI/hong/DATA4SOFM/GANFORMER4TEST'
    output_folder_lb1 = os.path.join(output_folder, 'label_1')

    eye_folder = os.path.join(output_folder_lb1, 'eye')
    cheek_folder = os.path.join(output_folder_lb1, 'cheek')
    lip_folder = os.path.join(output_folder_lb1, 'lip')
    
    os.makedirs(eye_folder, exist_ok=True)
    os.makedirs(cheek_folder, exist_ok=True)
    os.makedirs(lip_folder, exist_ok=True)
    for image_name in tqdm(os.listdir(input_folder), desc="crop for test_data lb1 non-make", leave=False):
        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, (512, 512))
            
            left_eye, right_eye, left_cheek, right_cheek, left_lip, right_lip = crop_and_transform_six_parts(image)
            
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
        del image
        gc.collect()
