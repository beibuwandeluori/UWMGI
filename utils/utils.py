import tensorboardX
import torch
import numpy as np
import glob
import cv2
from sklearn.metrics import accuracy_score, precision_score, average_precision_score, roc_auc_score, recall_score, \
    f1_score
import pandas as pd
from sklearn.metrics import confusion_matrix


class Test_time_agumentation(object):

    def __init__(self, is_rotation=True):
        self.is_rotation = is_rotation

    def __rotation(self, img):
        """
        clockwise rotation 90 180 270
        """
        img90 = img.rot90(-1, [2, 3])  # 1 逆时针； -1 顺时针
        img180 = img.rot90(-1, [2, 3]).rot90(-1, [2, 3])
        img270 = img.rot90(1, [2, 3])
        return [img90, img180, img270]

    def __inverse_rotation(self, img90, img180, img270):
        """
        anticlockwise rotation 90 180 270
        """
        img90 = img90.rot90(1, [2, 3])  # 1 逆时针； -1 顺时针
        img180 = img180.rot90(1, [2, 3]).rot90(1, [2, 3])
        img270 = img270.rot90(-1, [2, 3])
        return img90, img180, img270

    def __flip(self, img):
        """
        Flip vertically and horizontally
        """
        return [img.flip(2), img.flip(3)]

    def __inverse_flip(self, img_v, img_h):
        """
        Flip vertically and horizontally
        """
        return img_v.flip(2), img_h.flip(3)

    def tensor_rotation(self, img):
        """
        img size: [H, W]
        rotation degree: [90 180 270]
        :return a rotated list
        """
        # assert img.shape == (1024, 1024)
        return self.__rotation(img)

    def tensor_inverse_rotation(self, img_list):
        """
        img size: [H, W]
        rotation degree: [90 180 270]
        :return a rotated list
        """
        # assert img.shape == (1024, 1024)
        return self.__inverse_rotation(img_list[0], img_list[1], img_list[2])

    def tensor_flip(self, img):
        """
        img size: [H, W]
        :return a flipped list
        """
        # assert img.shape == (1024, 1024)
        return self.__flip(img)

    def tensor_inverse_flip(self, img_list):
        """
        img size: [H, W]
        :return a flipped list
        """
        # assert img.shape == (1024, 1024)
        return self.__inverse_flip(img_list[0], img_list[1])


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, model_name, header):
        self.header = header
        self.writer = tensorboardX.SummaryWriter(model_name)

    def __del(self):
        self.writer.close()

    def log(self, phase, values):
        epoch = values['epoch']

        for col in self.header[1:]:
            self.writer.add_scalar(phase + "/" + col, float(values[col]), int(epoch))


def calculate_dice(outputs, targets):
    return dice_coefficient(outputs, targets)


def dice_coefficient(y_pred, y_truth, eps=1e-8):
    batch_size = y_truth.size(0)
    y_pred = y_pred.type(torch.FloatTensor)
    dice = 0.
    for i in range(batch_size):
        intersection = torch.sum(torch.mul(y_pred[i], y_truth[i])) + eps / 2
        union = torch.sum(y_pred[i]) + torch.sum(y_truth[i]) + eps
        dice += 2 * intersection / union

    return dice / batch_size


def calculate_metric_score(outputs, targets, metric_name=None):
    # iou_score = compute_iou(outputs, targets, metric_name=metric_name)
    # batch_size > 1, 每个图片分别计算得分求平均
    iou_score = compute_iou(outputs[0], targets[0], metric_name=metric_name)
    iou_score = list(iou_score) if isinstance(iou_score, tuple) else iou_score
    for i in range(1, outputs.shape[0]):
        iou_score_t = compute_iou(outputs[i], targets[i], metric_name=metric_name)
        if isinstance(iou_score_t, tuple):
            iou_score_t = list(iou_score_t)
            for j in range(len(iou_score)):
                iou_score[j] += iou_score_t[j]
        else:
            iou_score += iou_score_t
    if isinstance(iou_score, list):
        for k in range(len(iou_score)):
            iou_score[k] = iou_score[k] / outputs.shape[0]
    else:
        iou_score = iou_score / outputs.shape[0]

    return iou_score


def compute_iou(premask, groundtruth, metric_name=None):
    # 通过矩阵的逻辑运算分别计算出tp,tn,fp,fn
    #  pre   p   n
    #  con p tp  fn
    #      n fp  tn

    # 判原始为0，判篡改为1
    # print(premask.shape, groundtruth.shape)
    premask = np.array(premask > 0.5, dtype=premask.dtype)
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = float(np.logical_and(premask, groundtruth).sum())  # float for division
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(premask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, groundtruth).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    acc = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

    cross = np.logical_and(premask, groundtruth)
    union = np.logical_or(premask, groundtruth)
    iou = np.sum(cross) / (np.sum(union) + 1e-6)

    if metric_name == 'acc':
        return acc
    elif metric_name == 'f1':
        return f1
    elif metric_name == 'iou':
        return iou
    elif metric_name is None:
        return acc, f1, iou

    return f1 + iou


def compute_f1(premask, groundtruth):
    premask = np.array(premask > 0.5).flatten().astype(int)
    groundtruth = groundtruth.flatten().astype(int)
    # print(premask.shape, groundtruth.shape, np.unique(premask), np.unique(groundtruth))
    f1 = f1_score(groundtruth, premask)

    return f1


def dice_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


def iou_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou


if __name__ == '__main__':
    pass
