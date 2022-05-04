import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .Attention import ChannelAttention, SELayer, SpatialAttention


def srm_init(shape=(5, 5, 1, 3), dtype=np.float32):
    hpf = np.zeros(shape, dtype=dtype)

    hpf[:, :, 0, 0] = np.array(
        [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]]) / 4.0
    hpf[:, :, 0, 1] = np.array(
        [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]) / 12.
    hpf[:, :, 0, 2] = np.array(
        [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]) / 2.0

    return hpf


# Truncation operation
class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()
        self.threshold = threshold

    def forward(self, input):
        output = torch.clamp(input, min=-self.threshold, max=self.threshold)
        return output


# 自定义SRM Layer, 可以即插即用
class SRM_Layer(nn.Module):
    def __init__(self, TLU_threshold=3.0, out_channel=3, use_se=False, dilation=1, requires_grad=False):
        super(SRM_Layer, self).__init__()
        self.out_channel = out_channel
        self.requires_grad = requires_grad
        self.hpf = nn.Conv2d(1, self.out_channel, kernel_size=5, padding=2, bias=False, dilation=dilation)
        self.hpf.weight = self.init_SRM(self.out_channel, self.requires_grad)
        # Truncation, threshold = 3
        self.tlu = TLU(threshold=TLU_threshold)
        self.use_se = use_se
        if self.use_se:
            # self.se = SELayer(channel=self.out_channel, reduction=3)
            self.se = ChannelAttention(in_planes=self.out_channel, ratio=3)

    @staticmethod
    def init_SRM(out_channel, requires_grad=False):
        if out_channel == 30:

            current_work_dir = os.path.dirname(__file__)  # 当前文件所在的目录
            srm_kernel = np.load(os.path.join(current_work_dir, 'SRM_Kernels.npy'))  # shape=(5, 5, 1, 30)
            # 归一化
            # for i in range(srm_kernel.shape[3]):
            #     srm_kernel[:, :, :, i] = srm_kernel[:, :, :, i] / np.max(np.abs(srm_kernel[:, :, :, i]))

            srm_kernel = srm_kernel.transpose(3, 2, 0, 1)  # shape=(30, 1, 5, 5)
            hpf_weight = nn.Parameter(torch.Tensor(srm_kernel).view(30, 1, 5, 5), requires_grad=requires_grad)
        else:
            srm_kernel = srm_init(shape=(5, 5, 1, 3), dtype=np.float32)
            srm_kernel = srm_kernel.transpose(3, 2, 0, 1)  # shape=(3, 1, 5, 5)
            hpf_weight = nn.Parameter(torch.Tensor(srm_kernel).view(3, 1, 5, 5), requires_grad=requires_grad)

        return hpf_weight

    def forward(self, x):
        for i in range(x.size(1)):
            out = self.hpf(x[:, i:i+1, :, :])
            out = self.tlu(out)
            if self.use_se:
                out = self.se(out)
            if i == 0:
                outs = out
            else:
                outs = torch.cat([outs, out], dim=1)
        # print(outs.shape)
        # 恢复原来的尺寸
        diffY = x.size()[2] - outs.size()[2]
        diffX = x.size()[3] - outs.size()[3]

        outs = F.pad(outs, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        # print(outs.shape)
        return outs

