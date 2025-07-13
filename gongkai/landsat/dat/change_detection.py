from dat.augmentation import augmentation_compose
import numpy as np
import os
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json

num_classes = 5


class ChangeDetection_Landsat_label(Dataset):
    CLASSES = ['未变化区域', '农田', '沙漠', '建筑物', '水体']
    def __init__(self, root, mode, use_pseudo_label=False):
        super(ChangeDetection_Landsat_label, self).__init__()
        self.root = root

        self.mode = mode

        if mode == 'train':
            self.root = os.path.join('/home/user/zly/data3/Landsat/sample/t0.10/', 'train')
            self.ids = os.listdir(os.path.join(self.root, "im1"))
            self.ids.sort()

        self.transform = augmentation_compose
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def load_json(self, img_name, mode):

        json_file_path_1 = os.path.join('/home/user/zly/data3/Landsat/sample/t0.10/', mode, 'im1_clipcls_56_vit16.json')
        json_file_path_2 = os.path.join('/home/user/zly/data3/Landsat/sample/t0.10/', mode, 'im2_clipcls_56_vit16.json')

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

        img1 = self.normalize(img1)
        img2 = self.normalize(img2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_edge = cv2.GaussianBlur(mask_bin * 255, (3, 3), 0)
        mask_edge = cv2.Canny(mask_edge, 50, 150)
        mask_edge = cv2.dilate(mask_edge, kernel, iterations=2)
        mask_edge = torch.from_numpy(np.array(mask_edge) // 255).long()

        mask1 = torch.from_numpy(np.array(mask1)).long()
        mask2 = torch.from_numpy(np.array(mask2)).long()
        mask_bin = torch.from_numpy(np.array(mask_bin)).float()

        return img1, img2, mask1, mask2, mask_bin, class_names_1, confidences_1, class_names_2, confidences_2, id

    def __len__(self):
        return len(self.ids)



class ChangeDetection_Landsat_nolabel(Dataset):
    CLASSES = ['未变化区域', '农田', '沙漠', '建筑物', '水体']
    def __init__(self, root, mode, use_pseudo_label=False):
        super(ChangeDetection_Landsat_nolabel, self).__init__()
        self.root = root

        self.mode = mode

        if mode == 'train':
            self.root = os.path.join('/home/user/zly/data3/Landsat/sample/Landsat_orgion/', 'train')
            self.ids = os.listdir(os.path.join(self.root, "im1"))
            self.ids.sort()
        elif mode == 'val':
            self.root = os.path.join('/home/user/zly/data3/Landsat/sample/Landsat_orgion/', 'val')
            self.ids = os.listdir(os.path.join(self.root, 'im1'))
            self.ids.sort()
        elif mode == 'test':
            self.root = os.path.join('/home/user/zly/data3/Landsat/sample/Landsat_orgion/', 'test')
            self.ids = os.listdir(os.path.join(self.root, 'im1'))
            self.ids.sort()

        self.transform = augmentation_compose
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def load_json(self, img_name, mode):

        json_file_path_1 = os.path.join('/home/user/zly/data3/Landsat/sample/Landsat_orgion/', mode, 'im1_clipcls_56_vit16.json')
        json_file_path_2 = os.path.join('/home/user/zly/data3/Landsat/sample/Landsat_orgion/', mode, 'im2_clipcls_56_vit16.json')

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

        img1 = self.normalize(img1)
        img2 = self.normalize(img2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask_edge = cv2.GaussianBlur(mask_bin * 255, (3, 3), 0)
        mask_edge = cv2.Canny(mask_edge, 50, 150)
        mask_edge = cv2.dilate(mask_edge, kernel, iterations=2)
        mask_edge = torch.from_numpy(np.array(mask_edge) // 255).long()

        mask1 = torch.from_numpy(np.array(mask1)).long()
        mask2 = torch.from_numpy(np.array(mask2)).long()
        mask_bin = torch.from_numpy(np.array(mask_bin)).float()

        return img1, img2, mask1, mask2, mask_bin, class_names_1, confidences_1, class_names_2, confidences_2, id

    def __len__(self):
        return len(self.ids)

