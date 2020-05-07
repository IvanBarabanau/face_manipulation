from collections import OrderedDict
import numpy as np
from scipy import ndimage
from skimage import draw
from skimage.morphology import binary_dilation
from imutils import face_utils
from torchvision import transforms


FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

FACIAL_ATTRIBUTE_TO_LANDMARKS = OrderedDict([
    ("smile", ("mouth", )),
    ("eyeglasses", ("right_eyebrow", "left_eyebrow", "right_eye", "left_eye"))
])

NUM_LANDMARKS = 68

def get_relevant_landmarks(group, landmarks):
    landmarks_use_mask = np.zeros(NUM_LANDMARKS, dtype=bool)

    start_idx, end_idx = FACIAL_LANDMARKS_68_IDXS[group]
    landmarks_use_mask[start_idx : end_idx] = True

    relevant_landmarks = landmarks[landmarks_use_mask]
    return relevant_landmarks

def extend_bbox_by_landmarks(face_bbox, landmarks):
    bb_x_min, bb_y_min, w, h = face_utils.rect_to_bb(face_bbox)
    bb_x_max, bb_y_max = bb_x_min + w, bb_y_min + h

    landmark_x_min, landmark_y_min = landmarks.min(0)
    landmark_x_max, landmark_y_max = landmarks.max(0)

    x_min = min(bb_x_min, landmark_x_min)
    x_max = max(bb_x_max, landmark_x_max)
    y_min = min(bb_y_min, landmark_y_min)
    y_max = max(bb_y_max, landmark_y_max)

    return np.array([x_min, y_min, x_max, y_max])

def get_mask(attribute, landmarks, img_size):
    landmarks_groups = FACIAL_ATTRIBUTE_TO_LANDMARKS[attribute]
    bin_mask = np.zeros(img_size[:2])

    for group in landmarks_groups:
        relevant_landmarks = get_relevant_landmarks(group, landmarks)

        contour = draw.polygon(relevant_landmarks[:, 0], relevant_landmarks[:, 1])
        ys, xs = contour[1], contour[0]
        for y, x in zip(ys, xs):
            bin_mask[y, x] = 1

    bin_mask = binary_dilation(bin_mask, selem=np.ones((20, 20)))
    mask = ndimage.gaussian_filter(bin_mask * 255, 3) / 255

    return mask

def restore_normalized_img(img):
    min_ = float(img.min())
    max_ = float(img.max())
    img = img.clamp_(min_, max_)
    img = (img - min_) / (max_ - min_ + 1e-5)
    img = img.numpy().transpose((1, 2, 0))
    return (img * 255).astype('uint8')

def extend_bbox(bbox, img_shape, h_ratio, w_ratio):
    img_h, img_w = img_shape[:2]
    x_min, y_min, x_max, y_max = bbox
    h_step, w_step = int(img_h * h_ratio), int(img_w * w_ratio)
    return [x_min - w_step, y_min - h_step, x_max + w_step, y_max]

def get_preinference_transforms(cfg):
    img_size = cfg['img_inference_shape']
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        SetRange])

    return transform

def combine_images(orig_img, gen_img, bbox, mask):
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, -1)

    x_min, y_min, x_max, y_max = bbox
    full_size_gen_img = np.zeros(orig_img.shape)
    full_size_gen_img[y_min : y_max, x_min : x_max] = gen_img
    inverse_mask = 1 - mask
    combined = orig_img * inverse_mask + full_size_gen_img * mask

    return combined.astype('uint8')
