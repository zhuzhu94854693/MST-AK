
import torch
import json
import numpy as np
import tifffile as tiff
from natsort import natsorted
from torchvision.transforms import ToTensor, Normalize
from torchvision.transforms import Compose as tvCompose
from matplotlib import pyplot as plt
from matplotlib import axes

from albumentations import (
    PadIfNeeded,
    Compose,
    OneOf,

    HorizontalFlip,
    VerticalFlip,
    RandomRotate90,
    Transpose,
    Rotate,
    ShiftScaleRotate,

    ColorJitter,
    RandomBrightnessContrast,

    RandomResizedCrop,
    RandomCrop,
    RandomSizedCrop,
)

import cv2
import numpy as np
import random
import torch
from albumentations import (
    Compose, VerticalFlip, HorizontalFlip, Transpose, RandomRotate90, ShiftScaleRotate,
    ColorJitter, GaussNoise, CoarseDropout
)
from torchvision.transforms import ToTensor, Normalize, Compose as tvCompose


def augmentation_compose(sample):

    augmented = Compose([

        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        Transpose(p=0.5),
        RandomRotate90(p=0.5),
        ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0, value=0),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1, p=0.5),
    ], p=0.8, additional_targets={'image1': 'image', 'mask1': 'mask', 'mask2': 'mask'})

    tf_sample = augmented(image=sample['img1'], image1=sample['img2'], mask=sample['mask1'], mask1=sample['mask2'], mask2=sample['gt_mask'])
    img1 = tf_sample['image']
    img2 = tf_sample['image1']
    mask1 = tf_sample['mask']
    mask2 = tf_sample['mask1']
    gt_mask = tf_sample['mask2']
    return {'img1': img1, 'img2': img2, 'mask1': mask1, 'mask2': mask2, 'gt_mask': gt_mask}


def augmentation(image, mask):
    oh, ow = image.shape[:2]
    augmented = Compose([
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        RandomRotate90(p=0.5),
        ShiftScaleRotate(scale_limit=0.5, rotate_limit=45, shift_limit=0.1, p=0.5, border_mode=0, value=0),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    ], p=0.5, additional_targets={'image1': 'image'})

    tf_sample = augmented(image=image[:, :, :3], image1=image[:, :, 3:], mask=mask[0, :, :].astype(np.uint8))
    tf_mask = np.expand_dims(tf_sample['mask'], axis=0)
    tf_img = np.concatenate((tf_sample['image'], tf_sample['image1']), axis=-1)

    return tf_img, tf_mask


def transformation(image, mask=None):
   
    mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]

    transformed = tvCompose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])

    tf_image = transformed(image)
    tf_label = torch.from_numpy(mask.astype(np.float32))

    return tf_image, tf_label
