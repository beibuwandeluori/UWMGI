import numpy as np
import pandas as pd

pd.options.plotting.backend = "plotly"
import random
from glob import glob
import os
import shutil
from tqdm import tqdm

tqdm.pandas()
import time
import copy
import joblib
from collections import defaultdict
import gc
from IPython import display as ipd

# visualization
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Sklearn
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

# TensorFlow
import tensorflow as tf

import timm

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# import rasterio
from joblib import Parallel, delayed

# For colored terminal text
from colorama import Fore, Back, Style


# _______________________________Image____________________________________


# def load_img(path, mask=False):
#     img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#     img = img.astype('float32')  # original is uint16
#     if not mask:
#         mx = np.max(img)
#         if mx:
#             img = (img / mx)  # scale image to [0, 1]
#         img = np.expand_dims(img, axis=-1)
#     else:
#         img = img / 255.0  # mask ->  [0, 1]
#     return img


def load_img(path, mask=False):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if not mask:
        img = np.tile(img[..., None], [1, 1, 3])  # gray to rgb
        img = img.astype('float32')  # original is uint16
        mx = np.max(img)
        if mx:
            img /= mx  # scale image to [0, 1]
    else:
        img = img.astype('float32')
        img /= 255.0

    return img


def load_msk(path):
    msk = np.load(path)
    msk = msk.astype('float32')
    msk /= 255.0
    return msk


# _______________________________transforms____________________________________
def get_transforms(img_size=256, phase='train'):
    data_transforms = {
        "train": A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=5, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                # #             A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),
            A.CoarseDropout(max_holes=8, max_height=img_size // 20, max_width=img_size // 20,
                            min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0)
        ], p=1.0),

        "valid": A.Compose([
            A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0)
        ], p=1.0)
    }
    return data_transforms[phase]


# _______________________________Dataset____________________________________
def prepare_loaders(df, fold, batch_size=32, img_size=256, return_dataset=False):
    train_df = df[df.fold != fold].reset_index(drop=True)
    valid_df = df[df.fold == fold].reset_index(drop=True)
    print(f'train_df:{train_df.shape}, valid_df:{valid_df.shape}')
    train_dataset = BuildDataset(train_df, transforms=get_transforms(img_size=img_size, phase='train'))
    valid_dataset = BuildDataset(valid_df, transforms=get_transforms(img_size=img_size, phase='valid'))
    if return_dataset:
        return train_dataset, valid_dataset

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=4, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              num_workers=4, shuffle=False, pin_memory=True)

    return train_loader, valid_loader


def read_csv(csv_path, n_fold=5):
    df = pd.read_csv(csv_path)
    df['empty'] = df.segmentation.map(lambda x: int(pd.isna(x)))

    df2 = df.groupby(['id'])['class'].agg(list).to_frame().reset_index()
    df2 = df2.merge(df.groupby(['id'])['segmentation'].agg(list), on=['id'])
    # df = df[['id','case','day','image_path','mask_path','height','width', 'empty']]

    df = df.drop(columns=['segmentation', 'class'])
    df = df.groupby(['id']).head(1).reset_index(drop=True)
    df = df.merge(df2, on=['id'])
    df.head()

    skf = StratifiedGroupKFold(n_splits=n_fold, shuffle=True, random_state=2022)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['empty'], groups=df["case"])):
        df.loc[val_idx, 'fold'] = fold

    return df


class BuildDataset(Dataset):
    def __init__(self, df, label=True, transforms=None):
        self.df = df
        self.label = label
        self.img_paths = df['image_path'].tolist()
        self.msk_paths = df['mask_path'].tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = load_img(img_path)

        if self.label:
            msk_path = self.msk_paths[index]
            msk = load_img(msk_path, mask=True)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img = data['image']
                msk = data['mask']
            img = torch.from_numpy(img.transpose(2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))
            return img, msk
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img = data['image']
            img = torch.from_numpy(img.transpose(2, 0, 1))
            return img


if __name__ == '__main__':
    df = read_csv(csv_path='/data1/chenby/dataset/UWMGI/train_mask.csv')
    # print(df.shape, df.head())
    start = time.time()
    train_loader, valid_loader = prepare_loaders(df, fold=0, batch_size=32)
    for i, (x, y) in enumerate(valid_loader):
        print(x.size(), y.shape, torch.unique(y))
        if i == 10:
            break

    end = time.time()
    print('End iterate, DataLoader total time: %fs' % (end - start))
