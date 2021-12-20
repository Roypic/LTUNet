import cv2
import os
import os.path as osp
import logging
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Callable
from torchvision import transforms as T
import skimage


def correct_dims(*images):
    corr_images = []
    # print(images)
    for img in images:
        if len(img.shape) == 2:
            corr_images.append(np.expand_dims(img, axis=2))
        else:
            corr_images.append(img)

    if len(corr_images) == 1:
        return corr_images[0]
    else:
        return corr_images


class ImageToImage2D(Dataset):
    def __init__(self, dataset_path: str, joint_transform: Callable = None, one_hot_mask: int = False) -> None:
        self.dataset_path = dataset_path
        self.input_path = os.path.join(dataset_path, 'img')
        self.output_path = os.path.join(dataset_path, 'labelcol')
        self.images_list = os.listdir(self.input_path)
        self.one_hot_mask = one_hot_mask

        if joint_transform:
            self.joint_transform = joint_transform
        else:
            to_tensor = T.ToTensor()
            self.joint_transform = lambda x, y: (to_tensor(x), to_tensor(y))

    def __len__(self):
        return len(os.listdir(self.input_path))

    def __getitem__(self, idx):
        image_filename = self.images_list[idx]

        image = cv2.imread(os.path.join(self.input_path, image_filename))

        mask = cv2.imread(os.path.join(self.output_path, image_filename[: -3] + "bmp"), 0)
        image = cv2.resize(image, (224,224))
        mask = cv2.resize(mask, (224,224))


        mask[mask < 1] = 0
        mask[mask >= 1] = 1
        # correct dimensions if needed
        image, mask= correct_dims(image, mask)


        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)


        if self.one_hot_mask:
            assert self.one_hot_mask > 0, 'one_hot_mask must be nonnegative'
            mask = torch.zeros((self.one_hot_mask, mask.shape[1], mask.shape[2])).scatter_(0, mask.long(), 1)

        return {'image': image, 'mask': mask}
