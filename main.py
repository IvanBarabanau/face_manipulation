import argparse
import sys

import cv2
import numpy as np
import PIL
import torch
import torchvision.transforms.functional as F
import yaml

import dlib
from imutils import face_utils

from utils import (combine_images, get_mask, get_preinference_transforms,
                   extend_bbox, extend_bbox_by_landmarks, restore_normalized_img)

sys.path.append('PyTorch-VAE/')  # dirty hack to make repo accessible
from models.vanilla_vae import VanillaVAE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c',
                        dest='cfg', default='configs/face_manipulation_cfg.yaml')

    args = parser.parse_args()
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    img_path = cfg['img_path']
    facial_landmarks_model_path = cfg['facial_landmarks_model_path']
    face_attribute = cfg['face_attribute']

    vae_latent_vector = torch.load(cfg['latent_vector'][face_attribute])

    face_detector = dlib.get_frontal_face_detector()
    landmarks_predictor = dlib.shape_predictor(facial_landmarks_model_path)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detected_face = face_detector(grayscale_img, 1)
    num_faces_detected = len(detected_face)
    if num_faces_detected >= 1:
        detected_face = detected_face[0]

    facial_landmarks = landmarks_predictor(grayscale_img, detected_face)
    facial_landmarks = face_utils.shape_to_np(facial_landmarks)
    face_bbox = extend_bbox_by_landmarks(detected_face, facial_landmarks)

    face_attribute_mask = get_mask(face_attribute, facial_landmarks, img.shape)

    face_bbox = extend_bbox(face_bbox, img.shape, 0.25, 0.2)
    cropped_by_face_img = F.crop(PIL.Image.fromarray(
        img), face_bbox[1], face_bbox[0], face_bbox[3] - face_bbox[1], face_bbox[2] - face_bbox[0])

    with open(cfg['vae_cfg_path'], 'r') as f:
        vae_cfg = yaml.safe_load(f)

    model = VanillaVAE(**vae_cfg['model_params'])
    state_dict = torch.load(cfg['vae_model_path'], map_location=torch.device('cpu'))['state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    inference_transform = get_preinference_transforms(cfg)
    restored, _, mu, _ = model.forward(inference_transform(cropped_by_face_img).unsqueeze(0))

    scale = cfg['face_attribute_change_scale']
    generated_img = restore_normalized_img(model.decode(mu - scale * vae_latent_vector).detach()[0])
    generated_img = F.resize(PIL.Image.fromarray(generated_img), cropped_by_face_img.size[::-1])
    generated_img = np.array(generated_img)

    modified_image = combine_images(img, generated_img, face_bbox, face_attribute_mask)

    idx = img_path.index('.')
    new_img_path = img_path[:idx] + '_' + face_attribute + '_scale_' + str(scale) + img_path[idx:]

    cv2.imwrite(new_img_path, cv2.cvtColor(modified_image, cv2.COLOR_RGB2BGR))
