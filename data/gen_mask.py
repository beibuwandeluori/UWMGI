# code from https://www.kaggle.com/code/awsaf49/uwmgi-mask-data/notebook
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
import gc
from IPython import display as ipd
from joblib import Parallel, delayed

# visualization
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import tensorflow as tf


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def get_metadata(row):
    data = row['id'].split('_')
    case = int(data[0].replace('case', ''))
    day = int(data[1].replace('day', ''))
    slice_ = int(data[-1])
    row['case'] = case
    row['day'] = day
    row['slice'] = slice_
    return row


def path2info(row):
    path = row['image_path']
    data = path.split('/')
    slice_ = int(data[-1].split('_')[1])
    case = int(data[-3].split('_')[0].replace('case', ''))
    day = int(data[-3].split('_')[1].replace('day', ''))
    width = int(data[-1].split('_')[2])
    height = int(data[-1].split('_')[3])
    row['height'] = height
    row['width'] = width
    row['case'] = case
    row['day'] = day
    row['slice'] = slice_
    return row


def id2mask(id_):
    idf = df[df['id'] == id_]
    wh = idf[['height', 'width']].iloc[0]
    shape = (wh.height, wh.width, 3)
    mask = np.zeros(shape, dtype=np.uint8)
    for i, class_ in enumerate(['stomach', 'large_bowel', 'small_bowel']):
        cdf = idf[idf['class'] == class_]
        rle = cdf.segmentation.squeeze()
        if len(cdf) and not pd.isna(rle):
            mask[..., i] = rle_decode(rle, shape[:2])
    return mask


def rgb2gray(mask):
    pad_mask = np.pad(mask, pad_width=[(0, 0), (0, 0), (1, 0)])
    gray_mask = pad_mask.argmax(-1)
    return gray_mask


def gray2rgb(mask):
    rgb_mask = tf.keras.utils.to_categorical(mask, num_classes=4)
    return rgb_mask[..., 1:].astype(mask.dtype)


def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img.astype('float32')  # original is uint16
    img = (img - img.min()) / (img.max() - img.min()) * 255.0  # scale image to [0, 255]
    img = img.astype('uint8')
    return img


def show_img(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    #     plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='bone')

    if mask is not None:
        # plt.imshow(np.ma.masked_where(mask!=1, mask), alpha=0.5, cmap='autumn')
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0, 0), 1, 1, color=_c) for _c in
                   [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]]
        labels = ["Stomach", "Large Bowel", "Small Bowel"]
        plt.legend(handles, labels)
    plt.axis('off')


def save_mask(id_):
    idf = df[df['id'] == id_]
    mask = id2mask(id_) * 255
    image_path = idf.image_path.iloc[0]
    mask_path = image_path.replace('/data1/chenby/dataset/UWMGI/train/',
                                   '/data1/chenby/dataset/UWMGI/train_mask/')
    mask_folder = mask_path.rsplit('/', 1)[0]
    os.makedirs(mask_folder, exist_ok=True)
    cv2.imwrite(mask_path, mask, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return mask_path


if __name__ == '__main__':
    # Train
    df = pd.read_csv('/data1/chenby/dataset/UWMGI/train.csv')
    df = df.progress_apply(get_metadata, axis=1)
    print(df.head())

    # Path
    paths = glob('/data1/chenby/dataset/UWMGI/train/*/*/*/*')
    path_df = pd.DataFrame(paths, columns=['image_path'])
    path_df = path_df.progress_apply(path2info, axis=1)
    df = df.merge(path_df, on=['case', 'day', 'slice'])
    print(df.head())

    # Save Metadata
    df['mask_path'] = df.image_path.str.replace('/data1/chenby/dataset/UWMGI/train',
                                                '/data1/chenby/dataset/UWMGI/train_mask')
    df.to_csv('/data1/chenby/dataset/UWMGI/train_mask.csv', index=False)

    # Write Mask
    ids = df['id'].unique()
    _ = Parallel(n_jobs=-1, backend='threading')(delayed(save_mask)(id_) for id_ in tqdm(ids, total=len(ids)))

    # Check Saved Mask
    # i = 250
    # img = load_img(df.image_path.iloc[i])
    # mask_path = df['image_path'].iloc[i].replace('/data1/chenby/dataset/UWMGI/train/',
    #                                              '/data1/chenby/dataset/UWMGI/train_mask/')
    # mask = plt.imread(mask_path)
    # plt.figure(figsize=(5, 5))
    # # show_img(img, mask=mask)
    # plt.savefig('visual.png')


