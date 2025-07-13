
from pathlib import Path
import torch
import matplotlib.image as mping
import glob
from PIL import Image
import torch.utils.data
from torchvision.transforms import Compose, Normalize
from skimage import io
from torchvision.transforms import transforms
import numpy as np
import os


import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import ImageEnhance
from torch.utils.data import Dataset
from datasets.augmentation import augmentation_compose




num_classes = 7
ST_COLORMAP = [[255,255,255], [0,0,255], [128,128,128], [0,128,0], [0,255,0], [128,0,0], [255,0,0]]
ST_CLASSES = ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']


class Test_ChangeDataset(Dataset):
    CLASSES = ['未变化区域', '水体', '地面', '低矮植被', '树木', '建筑物', '运动场']

    def __init__(self, root, mode):
        super(Test_ChangeDataset, self).__init__()
        self.root = root
        self.mode = mode

        if mode == 'train':
            self.root_t0_10 = os.path.join('/home/user/zly/data3/SECOND/sample/t0.10/train/')  

            # 读取有标签的图像文件
            self.ids_t0_10 = os.listdir(os.path.join(self.root_t0_10, "im1"))
            self.ids_t0_10.sort()



            self.hebing = self.ids_t0_10
            self.hebing = sorted(self.hebing)

            self.gt_list = []
            for image_path in self.ids_t0_10:
                if image_path in self.ids_t0_10:
                    self.gt_list.append(self.hebing.index(image_path))



        self.transform = augmentation_compose
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __getitem__(self, index):
        if self.mode == 'train':
            if index < len(self.hebing):  
                id = self.hebing[index]
                img1 = np.array(Image.open(os.path.join(self.root_t0_10, 'im1', id)))
                img2 = np.array(Image.open(os.path.join(self.root_t0_10, 'im2', id)))

                mask1 = np.array(Image.open(os.path.join(self.root_t0_10, 'label1', id)))
                mask2 = np.array(Image.open(os.path.join(self.root_t0_10, 'label2', id)))

                mask_bin = np.zeros_like(mask1)
                mask_bin[mask1 != 0] = 1

                sample = self.transform({'img1': img1, 'img2': img2, 'mask1': mask1, 'mask2': mask2,
                                         'gt_mask': mask_bin})
                img1, img2, mask1, mask2, mask_bin = sample['img1'], sample['img2'], sample['mask1'], \
                                                     sample['mask2'], sample['gt_mask']

                img1 = self.normalize(img1)
                img2 = self.normalize(img2)

                mask1 = torch.from_numpy(np.array(mask1)).long()
                mask2 = torch.from_numpy(np.array(mask2)).long()
                mask_bin = torch.from_numpy(np.array(mask_bin)).float()


            return img1, img2, mask1, mask2


    def __len__(self):
        if self.mode == 'train':
            return len(self.ids_t0_10)
        else:
            return len(self.ids)

