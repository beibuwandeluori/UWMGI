import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .Attention import ChannelAttention, SELayer, SpatialAttention


# 自定义DCT Layer, 可以即插即用
class DCT_Layer(nn.Module):
    def __init__(self,use_se=False, dilation=1, requires_grad=False):
        super(DCT_Layer, self).__init__()
        self.dct = nn.Conv2d(1, 16, kernel_size=4, padding=2, bias=False, dilation=dilation)
        self.dct.weight = self.init_DCT(requires_grad=requires_grad)
        self.use_se = use_se
        if self.use_se:
            # self.se = SELayer(channel=16, reduction=16)
            self.se = ChannelAttention(in_planes=16, ratio=4)

    # DCT
    def init_DCT(self, shape=(4, 4, 1, 16), requires_grad=False):
        PI = math.pi
        DCT_kernel = np.zeros(shape, dtype=np.float32)  # [height,width,input,output], shape=(4, 4, 1, 16)
        u = np.ones([4], dtype=np.float32) * math.sqrt(2.0 / 4.0)
        u[0] = math.sqrt(1.0 / 4.0)
        for i in range(0, 4):
            for j in range(0, 4):
                for k in range(0, 4):
                    for l in range(0, 4):
                        DCT_kernel[i, j, :, k * 4 + l] = u[k] * u[l] * math.cos(PI / 8.0 * k * (2 * i + 1)) * math.cos(
                            PI / 8.0 * l * (2 * j + 1))
        DCT_kernel = DCT_kernel.transpose(3, 2, 0, 1)
        dct_weight = nn.Parameter(torch.Tensor(DCT_kernel).view(16, 1, 4, 4), requires_grad=requires_grad)

        return dct_weight

    # Trancation operation for DCT
    @staticmethod
    def DCT_Trunc(x):
        trunc = -(F.relu(-x + 8) - 8)
        return trunc

    def forward(self, x):
        for i in range(x.size(1)):
            out = self.dct(x[:, i:i+1, :, :])
            out = self.DCT_Trunc(torch.abs(out))
            if self.use_se:
                out = self.se(out)
            if i == 0:
                outs = out
            else:
                outs = torch.cat([outs, out], dim=1)
        # if self.use_se:
        #     outs = self.se(outs)
        # 恢复原来的尺寸
        diffY = x.size()[2] - outs.size()[2]
        diffX = x.size()[3] - outs.size()[3]

        outs = F.pad(outs, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        # print(outs.shape)
        return outs

