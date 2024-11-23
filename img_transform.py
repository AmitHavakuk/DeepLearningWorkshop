import torch
from torchvision import transforms
from PIL import Image
import random
import os
import cv2
import dlib
import numpy as np
import random
import time
import shutil

from utils_sunglasses_fitting import (
    detect_face,
    draw_landmarks,
    get_orientation,
    load_assets,
    rotate_along_axis,
)

SUNGLASSES_PARAMS = [(1524,578), (514,277), (296,235), (507,231), (600,170)] # save (x,y) for each glasses
#SUNGLASSES_PARAMS = [(1524,578), (600,277), (296,235), (507,231)] # save (x,y) for each glasses
SUNGLASSES_PATHS = ["./additions/padded_black_sunglasses.png",
                    "./additions/cyan_sunglasses.png",
                    "./additions/purple_sunglasses.png",
                    "./additions/red_sunglasses.png",
                    "./additions/brown_sunglasses.png"]

SUNGLASSES_IMG = []  # leave empty here
#last_sunglasses = 0

def add_save_sunglases(img_path, sunglasses_index, detector_faces, detector_landmarks):
    #global last_sunglasses
    sunglasses = SUNGLASSES_IMG[sunglasses_index]
    #print(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #detector_faces = dlib.get_frontal_face_detector()

    face_rectangles = detector_faces(img_rgb, 0)
    #detector_landmarks = dlib.shape_predictor(
    #    "./auxiliary_models/shape_predictor_68_face_landmarks.dat")
    #print(len(face_rectangles))
    for i in range(len(face_rectangles)):
        rect = dlib.rectangle(
                        int(face_rectangles[i].left()),
                        int(face_rectangles[i].top()),
                        int(face_rectangles[i].right()),
                        int(face_rectangles[i].bottom()),
                    )
        landmarks = detector_landmarks(img_rgb, rect)

        # nose top, left and right face end points
        x = int(landmarks.parts()[27].x)
        y = int(landmarks.parts()[27].y)
        x_18 = int(landmarks.parts()[17].x)-10
        x_27 = int(landmarks.parts()[26].x)+10

        #print("A", x,y,x_18,x_27)

        if (x_18 < 0) or (x_27 >= img.shape[1]):
            print("Outside of frame, skibidipped glasses.")
            continue

        # if last_sunglasses != sunglasses.shape:
        #     print("Nigger")
        # last_sunglasses = sunglasses.shape

        sun_h, sun_w, _ = sunglasses.shape
        # calculate new width and height, moving distance for adjusting sunglasses
        width = int(abs(x_18 - x_27))
        scale = width / sun_w
        height = int(sun_h * scale)

        #CRITICAL INFORMATION
        nose_point = 27   # landmark of nose where eyeglasses should be fit
        glasses_mid_x = SUNGLASSES_PARAMS[sunglasses_index][0]
        glasses_mid_y = SUNGLASSES_PARAMS[sunglasses_index][1]

        move_x = int(glasses_mid_x * scale)
        move_y = int(glasses_mid_y * scale)

        _h, _w, _ = img.shape
        _, roll, yaw = get_orientation(_w, _h, landmarks.parts())
        edited_sunglasses = rotate_along_axis(sunglasses, width, height, phi=yaw, gamma=roll)
        #edited_sunglasses = cv2.resize(edited_sunglasses, (width, height))

        # get region of interest on the face to change
        roi_color = img[(y - move_y):(y + height - move_y), (x - move_x):(x + width - move_x)]

        # find all non-transparent points
        index = np.argwhere(edited_sunglasses[:, :, 3] > 0)
        #print("B", edited_sunglasses.shape, roi_color.shape)
        for j in range(3):
            roi_color[index[:, 0], index[:, 1], j] = edited_sunglasses[index[:, 0], index[:, 1], j]

        # set the area of the image of the changed region with sunglasses
        img[(y - move_y):(y + height - move_y), (x - move_x):(x + width - move_x)] = roi_color

    # print("nigga in end")
    # cv2.imshow("Image Window", img)  # Display the image
    # cv2.waitKey(0)  # Wait for a key press to close the window
    # cv2.destroyAllWindows()  # Close the image window
    last_name = os.path.basename(img_path)
    img_folder = os.path.dirname(img_path)

    result_filename = f"sunglasses{sunglasses_index}_{last_name}"
    #result_filepath = os.path.join(img_folder, "sunglasses", result_filename)
    result_filepath = os.path.join(img_folder, "sunglasses",result_filename)
    #print(result_filepath)
    cv2.imwrite(result_filepath, img)


def add_salt_and_pepper_noise(img, prob=0.05):
    """Applies salt and pepper noise to an image."""
    np_img = np.array(img)
    # Generate random mask for salt and pepper noise
    salt_pepper_mask = np.random.rand(*np_img.shape[:2])

    # Apply salt (white)
    np_img[salt_pepper_mask < (prob / 2)] = 255
    # Apply pepper (black)
    np_img[salt_pepper_mask > (1 - prob / 2)] = 0

    return Image.fromarray(np_img)


def add_gaussian_noise(img, mean=0, std=10.0):
    """Applies Gaussian noise to an image."""
    np_img = np.array(img).astype(np.float32)  # Convert to float to avoid overflow
    gaussian_noise = np.random.normal(mean, std, np_img.shape)
    np_img += gaussian_noise
    # Clip the values to be valid pixel values and convert back to uint8
    np_img = np.clip(np_img, 0, 255).astype(np.uint8)

    return Image.fromarray(np_img)



# Define augmentation pipeline
def augment_image(input_path, result_folder):
    # Transformation pipeline
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(degrees=10),
        transforms.Lambda(lambda img: transforms.GaussianBlur((3, 3), sigma=(0.1, 2.0))(img) if random.random() < 0.5 else img),
        transforms.Lambda(lambda img: add_gaussian_noise(img, mean=0, std=0.02) if random.random() < 0.5 else img),
        transforms.Lambda(lambda img: add_salt_and_pepper_noise(img) if random.random() < 0.5 else img),
        #transforms.ToTensor()
    ])

    # Load the image
    # print("B", input_path)
    image = Image.open(input_path)

    # Apply transformations
    augmented_image = transform(image)

    # Save the augmented image
    last_name = os.path.basename(input_path)
    img_folder = os.path.dirname(input_path)

    result_filename = f"augmented_{last_name}"
    result_filepath = os.path.join(result_folder, result_filename)
    # print(result_filepath)
    augmented_image.save(result_filepath)  # may not work do cv2.imsave or something instead


# # Example usage
# input_image_path = './stuff/my_face2.jpg'  # Path to your input image
# output_image_path = './results/augmented_my_face1.jpg'  # Path to save the augmented image
#augment_image(input_image_path, output_image_path)

source_folder = "./img_align_celeba_full/eval_partition/2_identity"  # Folder containing test identity subfolders

detector_faces = dlib.get_frontal_face_detector()
detector_landmarks = dlib.shape_predictor("./auxiliary_models/shape_predictor_68_face_landmarks.dat")

for idx in range(len(SUNGLASSES_PATHS)):
    SUNGLASSES_IMG.append(cv2.imread(SUNGLASSES_PATHS[idx], cv2.IMREAD_UNCHANGED))

# sorted_identities = sorted(os.listdir(source_folder), key=lambda x: int(os.path.splitext(x)[0]))

start = time.perf_counter()

# # img_cnt = 0
# # ppl_cnt = 0
# # max_ppl_cnt = 100
# #start_from = 182638  # comment / un-comment IF CONTINUE
# for identity in sorted_identities:
#     if int(identity) % 10 == 0:
#         print("current identity: ", identity)
#     # if int(identity) < start_from:
#     #     continue
#     identity_path = os.path.join(source_folder, identity)
#     images_list = os.listdir(identity_path)
#     sung_path = os.path.join(identity_path, "sunglasses")
#     os.makedirs(sung_path, exist_ok=True)
#     #prev_idx = -1
#     for img in images_list:
#         img_path = os.path.join(identity_path, img)
#         #print("img_path: ", img_path)
#         random_idx = random.randint(0,len(SUNGLASSES_PATHS)-1)
#         # while random_idx == prev_idx:
#         #     random_idx = random.randint(0, len(SUNGLASSES_PATHS) - 1)
#         #prev_idx = random_idx
#         #print(img_path)
#         add_save_sunglases(img_path, random_idx, detector_faces, detector_landmarks)
#     #     img_cnt+=1
#     # ppl_cnt+=1
#
#     # if ppl_cnt > max_ppl_cnt:
#     #     break


# for identity in sorted_identities:
#     if int(identity) % 10 == 0:
#         print("current identity: ", identity)
#     identity_path = os.path.join(source_folder, identity)
#     nat_path = os.path.join(identity_path, "natural")
#     os.makedirs(nat_path, exist_ok=True)
#     for img in os.listdir(identity_path):
#         img_path = os.path.join(identity_path, img)
#         if not os.path.isfile(img_path):
#             continue
#         new_path = os.path.join(nat_path, img)
#         shutil.move(img_path, new_path)




# DONT USE THIS CHUNK OF CODE
# for identity in sorted_identities:
#     #if int(identity) % 100 == 0:
#         #print("current identity: ", identity)
#     identity_path = os.path.join(source_folder, identity)
#     for img in os.listdir(identity_path):
#         img_path = os.path.join(identity_path, img)
#         #print("img_path: ", img_path)
#         augment_image(img_path)
#         img_cnt += 1
#     ppl_cnt += 1
#     #print(identity, ppl_cnt)
#     if ppl_cnt > max_ppl_cnt:
#         break




# for identity in sorted_identities:
#     if int(identity) % 10 == 0:
#         print("current identity: ", identity)
#     identity_path = os.path.join(source_folder, identity)
#
#     for fold_name in os.listdir(identity_path):
#         if fold_name != "mask":
#             continue
#         fold_path = os.path.join(identity_path, fold_name)
#         new_folder = os.path.join(identity_path, f"augmented_{fold_name}")
#         os.makedirs(new_folder, exist_ok=True)
#         for img in os.listdir(fold_path):
#             img_path = os.path.join(fold_path, img)
#             # if not os.path.isfile(img_path):
#             #     continue
#             #print("A", img_path)
#             augment_image(img_path, new_folder)


end = time.perf_counter()

# print("ppl cnt: ", ppl_cnt)
# print("img cnt: ", img_cnt)
print("Total time: ", end-start)