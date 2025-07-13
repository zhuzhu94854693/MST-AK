from dat.augmentation import augmentation_compose
import numpy as np
import os
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json

num_classes = 7
ST_COLORMAP = [[255,255,255], [0,0,255], [128,128,128], [0,128,0], [0,255,0], [128,0,0], [255,0,0]]
ST_CLASSES = ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']

MEAN_A = np.array([113.40, 114.08, 116.45])
STD_A  = np.array([48.30,  46.27,  48.14])
MEAN_B = np.array([111.07, 114.04, 118.18])
STD_B  = np.array([49.41,  47.01,  47.94])

colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(ST_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

def Colorls2Index(ColorLabels):
    IndexLabels = []
    for i, data in enumerate(ColorLabels):
        IndexMap = Color2Index(data)
        IndexLabels.append(IndexMap)
    return IndexLabels

def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    IndexMap = IndexMap * (IndexMap < num_classes)
    return IndexMap

def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]



class ChangeDetection_SECOND_label(Dataset):
    CLASSES = ['未变化区域', '水体', '地面', '低矮植被', '树木', '建筑物', '运动场']

    def __init__(self, root, mode, use_pseudo_label=False):
        super(ChangeDetection_SECOND_label, self).__init__()
        self.root = root
        self.mode = mode

        if mode == 'train':
            self.root = os.path.join('/home/user/zly/data3/SECOND/sample/t0.10/', 'train')
            self.ids = os.listdir(os.path.join(self.root, "im1"))
            self.ids.sort()


        self.transform = augmentation_compose
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def load_json(self, img_name, mode):

        json_file_path_1 = os.path.join('/home/user/zly/data3/SECOND/sample/t0.10/', mode, 'im1_clipcls_56_vit16.json')
        json_file_path_2 = os.path.join('/home/user/zly/data3/SECOND/sample/t0.10/', mode, 'im2_clipcls_56_vit16.json')

        def parse_json(json_file, img_name):

            if not os.path.exists(json_file):
                print(f"Error: JSON file {json_file} does not exist.")
                return [], []

            with open(json_file, 'r') as f:
                data = json.load(f)

            for item in data:
                if item.get("image_path", "").endswith(f"/{img_name}"):
                    class_names = []
                    confidences = []
                    for key, value in item.items():
                        if key != "image_path":
                            class_names.append(key)
                            confidences.append(float(value))
                    return class_names, confidences

            print(f"Error: No data found for image {img_name} in {json_file}")
            return [], []

        class_names_1, confidences_1 = parse_json(json_file_path_1, img_name)
        class_names_2, confidences_2 = parse_json(json_file_path_2, img_name)

        return (class_names_1, confidences_1), (class_names_2, confidences_2)

    def __getitem__(self, index):
        id = self.ids[index]

        img1 = np.array(Image.open(os.path.join(self.root, 'im1', id)))
        img2 = np.array(Image.open(os.path.join(self.root, 'im2', id)))

        (class_names_1, confidences_1), (class_names_2, confidences_2) = self.load_json(id, self.mode)

        mask1 = np.array(Image.open(os.path.join(self.root, 'label1', id)))
        mask2 = np.array(Image.open(os.path.join(self.root, 'label2', id)))

        mask_bin = np.zeros_like(mask1)
        mask_bin[mask1 != 0] = 1

        if self.mode == 'train':
            sample = self.transform({'img1': img1, 'img2': img2, 'mask1': mask1, 'mask2': mask2,
                                     'gt_mask': mask_bin})
            img1, img2, mask1, mask2, mask_bin = sample['img1'], sample['img2'], sample['mask1'], \
                sample['mask2'], sample['gt_mask']
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)

        mask1 = torch.from_numpy(np.array(mask1)).long()
        mask2 = torch.from_numpy(np.array(mask2)).long()
        mask_bin = torch.from_numpy(np.array(mask_bin)).float()

        return img1, img2, mask1, mask2, mask_bin, class_names_1, confidences_1, class_names_2, confidences_2, id

    def __len__(self):
        return len(self.ids)



class ChangeDetection_SECOND_nolabel(Dataset):
    CLASSES = ['未变化区域', '水体', '地面', '低矮植被', '树木', '建筑物', '运动场']

    def __init__(self, root, mode, use_pseudo_label=False):
        super(ChangeDetection_SECOND_nolabel, self).__init__()
        self.root = root
        self.mode = mode

        if mode == 'train':
            self.root = os.path.join('/home/user/zly/data3/SECOND/sample/SECOND_orgion/', 'train')
            self.ids = os.listdir(os.path.join(self.root, "im1"))
            self.ids.sort()
        elif mode == 'val':
            self.root = os.path.join('/home/user/zly/data3/SECOND/sample/SECOND_orgion/', 'val')
            self.ids = os.listdir(os.path.join(self.root, 'im1'))
            self.ids.sort()
        elif mode == 'test':
            self.root = os.path.join('/home/user/zly/data3/SECOND/sample/SECOND_orgion/', 'test')
            self.ids = os.listdir(os.path.join(self.root, 'im1'))
            self.ids.sort()

        self.transform = augmentation_compose
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def load_json(self, img_name, mode):

        json_file_path_1 = os.path.join('/home/user/zly/data3/SECOND/sample/SECOND_orgion/', mode, 'im1_clipcls_56_vit16.json')
        json_file_path_2 = os.path.join('/home/user/zly/data3/SECOND/sample/SECOND_orgion/', mode, 'im2_clipcls_56_vit16.json')

        def parse_json(json_file, img_name):

            if not os.path.exists(json_file):
                print(f"Error: JSON file {json_file} does not exist.")
                return [], []

            with open(json_file, 'r') as f:
                data = json.load(f)

            for item in data:
                if item.get("image_path", "").endswith(f"/{img_name}"):  # 确保匹配文件名
                    class_names = []
                    confidences = []
                    for key, value in item.items():
                        if key != "image_path":
                            class_names.append(key)
                            confidences.append(float(value))  
                    return class_names, confidences

            print(f"Error: No data found for image {img_name} in {json_file}")
            return [], []

        class_names_1, confidences_1 = parse_json(json_file_path_1, img_name)
        class_names_2, confidences_2 = parse_json(json_file_path_2, img_name)

        return (class_names_1, confidences_1), (class_names_2, confidences_2)

    def __getitem__(self, index):
        id = self.ids[index]

        img1 = np.array(Image.open(os.path.join(self.root, 'im1', id)))
        img2 = np.array(Image.open(os.path.join(self.root, 'im2', id)))

        (class_names_1, confidences_1), (class_names_2, confidences_2) = self.load_json(id, self.mode)

        mask1 = np.array(Image.open(os.path.join(self.root, 'label1', id)))
        mask2 = np.array(Image.open(os.path.join(self.root, 'label2', id)))

        mask_bin = np.zeros_like(mask1)
        mask_bin[mask1 != 0] = 1

        if self.mode == 'train':
            sample = self.transform({'img1': img1, 'img2': img2, 'mask1': mask1, 'mask2': mask2,
                                     'gt_mask': mask_bin})
            img1, img2, mask1, mask2, mask_bin = sample['img1'], sample['img2'], sample['mask1'], \
                sample['mask2'], sample['gt_mask']
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)

        mask1 = torch.from_numpy(np.array(mask1)).long()
        mask2 = torch.from_numpy(np.array(mask2)).long()
        mask_bin = torch.from_numpy(np.array(mask_bin)).float()

        return img1, img2, mask1, mask2, mask_bin, class_names_1, confidences_1, class_names_2, confidences_2, id

    def __len__(self):
        return len(self.ids)

