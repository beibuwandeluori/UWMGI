import torch
from torch.nn.modules.loss import _Loss
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class SoftDiceLoss(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''

    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()

    def forward(self, y_pred, y_true, eps=1e-8):
        # print(y_pred.shape, y_true.shape)
        y_pred = torch.squeeze(y_pred)
        y_true = torch.squeeze(y_true)
        assert y_pred.size() == y_true.size(), "the size of predict and target must be equal."
        intersection = torch.sum(torch.mul(y_pred, y_true))
        union = torch.sum(torch.mul(y_pred, y_pred)) + torch.sum(torch.mul(y_true, y_true)) + eps

        dice = 2 * intersection / union
        dice_loss = 1.0 - dice
        # print(intersection.item(), union.item(),'dice:', dice.item(), dice_loss.item())
        return dice_loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    @staticmethod
    def forward(output, target, smooth=1e-5):
        batch = target.size(0)
        input_flat = output.view(batch, -1)
        target_flat = target.view(batch, -1)

        intersection = input_flat * target_flat
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / batch
        return loss


class CombinedLoss(_Loss):
    def __init__(self, vae_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.vae_weight = vae_weight
        self.dice_loss = SoftDiceLoss()
        self.l2_loss = nn.MSELoss()

    def forward(self, y_pred, y_true, vae_pred, vae_true):
        dice_loss = self.dice_loss(y_pred, y_true)
        l2_loss = self.l2_loss(vae_pred, vae_true)
        combined_loss = dice_loss + self.vae_weight * l2_loss

        return combined_loss


class BCEDicedLoss(_Loss):
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super(BCEDicedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = SoftDiceLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, y_pred, y_true):
        dice_loss = self.dice_loss(y_pred, y_true)
        bce_loss = self.bce_loss(y_pred, y_true)
        combined_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss

        return combined_loss


class BCEDicedLossV2(_Loss):
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super(BCEDicedLossV2, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = smp.losses.DiceLoss(mode='multilabel')
        self.bce_loss = smp.losses.SoftBCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        dice_loss = self.dice_loss(y_pred, y_true)
        bce_loss = self.bce_loss(y_pred, y_true)
        combined_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss

        return combined_loss


class BCETverskyLoss(_Loss):
    def __init__(self, bce_weight=0.5, tversky_weight=0.5):
        super(BCETverskyLoss, self).__init__()
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
        self.tversky_loss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)
        self.bce_loss = smp.losses.SoftBCEWithLogitsLoss()

    def forward(self, y_pred, y_true):
        dice_loss = self.tversky_loss(y_pred, y_true)
        bce_loss = self.bce_loss(y_pred, y_true)
        combined_loss = self.tversky_weight * dice_loss + self.bce_weight * bce_loss

        return combined_loss


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.05):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):

        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)

            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)

    def forward(self, preds, labels):
        if self.ignore_index is not None:
            mask = labels != self.ignore
            labels = labels[mask]
            preds = preds[mask]

        logpt = -self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt

        return loss


def get_losses(loss_name='LovaszLoss'):
    losses = {'JaccardLoss': smp.losses.JaccardLoss(mode='multilabel'),
              'DiceLoss': smp.losses.DiceLoss(mode='multilabel'),
              'BCELoss': smp.losses.SoftBCEWithLogitsLoss(),
              'LovaszLoss': smp.losses.LovaszLoss(mode='multilabel', per_image=False),
              'TverskyLoss': smp.losses.TverskyLoss(mode='multilabel', log_loss=False),
              }
    return losses[loss_name]
