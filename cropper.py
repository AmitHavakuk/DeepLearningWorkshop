import numpy as np
import cv2
import dlib


def align_crop_opencv(img,
                      src_landmarks,
                      standard_landmarks,
                      crop_size=512,
                      face_factor=0.7,
                      align_type='similarity',
                      order=3,
                      mode='edge'):

    inter = {0: cv2.INTER_NEAREST, 1: cv2.INTER_LINEAR, 2: cv2.INTER_AREA,
             3: cv2.INTER_CUBIC, 4: cv2.INTER_LANCZOS4, 5: cv2.INTER_LANCZOS4}
    border = {'constant': cv2.BORDER_CONSTANT, 'edge': cv2.BORDER_REPLICATE,
              'symmetric': cv2.BORDER_REFLECT, 'reflect': cv2.BORDER_REFLECT101,
              'wrap': cv2.BORDER_WRAP}

    # check
    assert align_type in ['affine', 'similarity'], 'Invalid `align_type`! Allowed: %s!' % ['affine', 'similarity']
    assert order in [0, 1, 2, 3, 4, 5], 'Invalid `order`! Allowed: %s!' % [0, 1, 2, 3, 4, 5]
    assert mode in ['constant', 'edge', 'symmetric', 'reflect', 'wrap'], 'Invalid `mode`! Allowed: %s!' % ['constant', 'edge', 'symmetric', 'reflect', 'wrap']

    # crop size
    if isinstance(crop_size, (list, tuple)) and len(crop_size) == 2:
        crop_size_h = crop_size[0]
        crop_size_w = crop_size[1]
    elif isinstance(crop_size, int):
        crop_size_h = crop_size_w = crop_size
    else:
        raise Exception('Invalid `crop_size`! `crop_size` should be 1. int for (crop_size, crop_size) or 2. (int, int) for (crop_size_h, crop_size_w)!')

    # estimate transform matrix
    trg_landmarks = standard_landmarks * max(crop_size_h, crop_size_w) * face_factor + np.array([crop_size_w // 2, crop_size_h // 2])
    if align_type == 'affine':
        tform = cv2.estimateAffine2D(trg_landmarks, src_landmarks, ransacReprojThreshold=np.Inf)[0]
    else:
        tform = cv2.estimateAffinePartial2D(trg_landmarks, src_landmarks, ransacReprojThreshold=np.Inf)[0]

    # warp image by given transform
    output_shape = (crop_size_h, crop_size_w)
    img_crop = cv2.warpAffine(img, tform, output_shape[::-1], flags=cv2.WARP_INVERSE_MAP + inter[order], borderMode=border[mode])

    # get transformed landmarks
    tformed_landmarks = cv2.transform(np.expand_dims(src_landmarks, axis=0), cv2.invertAffineTransform(tform))[0]

    return img_crop, tformed_landmarks


def crop_by_path(img1_path):
    move_w = 0.
    move_h = 0.25
    detector_landmarks = dlib.shape_predictor("./auxiliary_models/shape_predictor_68_face_landmarks.dat")
    detector_faces = dlib.get_frontal_face_detector()
    standard_landmarks = np.genfromtxt("./auxiliary_models/standard_landmark_68pts.txt",dtype=float).reshape(68, 2)
    standard_landmarks[:, 0] += move_w
    standard_landmarks[:, 1] += move_h
    img1 = cv2.imread(img1_path)

    face_rectangles = detector_faces(img1, 0)
    rect = dlib.rectangle(
        int(face_rectangles[0].left()),
        int(face_rectangles[0].top()),
        int(face_rectangles[0].right()),
        int(face_rectangles[0].bottom()),
    )
    src_landmarks = detector_landmarks(img1, rect).parts()
    src_landmarks = [(point.x, point.y) for point in src_landmarks]
    # Convert the list to a NumPy array
    src_landmarks = np.array(src_landmarks)

    cropped_img1, _ = align_crop_opencv(img1,
                      src_landmarks,
                      standard_landmarks,
                      crop_size=(218,178),
                      face_factor=0.45,
                      align_type='similarity',
                      order=4,
                      mode='edge')
    return cropped_img1
    # cv2.imshow("hi", cropped_img1)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

