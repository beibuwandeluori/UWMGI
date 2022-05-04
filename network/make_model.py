"""
@ github: https://github.com/milesial/Pytorch-UNet
@ model: EfficientUNet
@ author: Baoying Chen
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import init
from efficientnet_pytorch import EfficientNet
import timm

from .module import SRM_Layer, DCT_Layer


# fc layer weight init
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def get_efficientnet_ns(model_name='tf_efficientnet_b3_ns', pretrained=True):
    net = timm.create_model(model_name, pretrained=pretrained)

    return net


def get_resnet(model_name='resnet18', pretrained=True):
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=pretrained)
    elif model_name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=pretrained)
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pretrained)
    elif model_name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=pretrained)
    elif model_name == 'resnext101_32x8d':
        model = torchvision.models.resnext101_32x8d(pretrained=pretrained)
    elif model_name == 'resnext50_32x4d':
        model = torchvision.models.resnext50_32x4d(pretrained=pretrained)
    else:
        print(f'No model with name {model_name}')

    return model


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, is_efn=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            if is_efn:
                self.up = nn.ConvTranspose2d(in_channels-out_channels, in_channels-out_channels, kernel_size=2, stride=2)
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Last_Up(nn.Module):
    """Upscaling"""
    def __init__(self, in_channels, scale_factor=2, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=scale_factor+1, stride=scale_factor)

    def forward(self, x1, original_size):
        x1 = self.up(x1)
        # input is CHW
        diffY = original_size[2] - x1.size()[2]
        diffX = original_size[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        return x1


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.Softmax(dim=1)):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = activation

    def forward(self, x):
        if self.activation is not None:
            return self.activation(self.conv(x))
        return self.conv(x)


class RRU_double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RRU_double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2),
            nn.GroupNorm(4, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=2, dilation=2),
            nn.GroupNorm(4, out_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class RRU_up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(RRU_up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2),
                nn.GroupNorm(4, in_ch // 2))

        self.conv = RRU_double_conv(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(4, out_ch))
        self.res_conv_back = nn.Sequential(
            nn.Conv2d(out_ch, in_ch, kernel_size=1, bias=False))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffY, 0, diffX, 0))

        x = self.relu(torch.cat([x2, x1], dim=1))

        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(self.res_conv(x) + ft1)
        # the second ring conv
        ft2 = self.res_conv_back(r1)
        x = torch.mul(1 + F.sigmoid(ft2), x)
        # the third ring conv
        ft3 = self.conv(x)
        r3 = self.relu(ft3 + self.res_conv(x))

        return r3


# ———————————————UNet———————————————————
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=True, activation=nn.Softmax(dim=1),
                 start_down=False, dct=False, srm=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.start_down = start_down
        self.dct = dct
        self.srm = srm
        if self.srm:
            self.srm = SRM_Layer(TLU_threshold=3.0, out_channel=30)
            n_channels = 90
        if self.dct:
            self.dct_layer = DCT_Layer()
            n_channels = 48

        if self.start_down:
            # self.start_down = nn.MaxPool2d(2)
            # self.start_down = nn.AvgPool2d(2)
            self.start_down = nn.Sequential(nn.Conv2d(n_channels, 64, kernel_size=3, stride=2, bias=False),
                                            nn.BatchNorm2d(64))
            self.last_up = Last_Up(in_channels=64, scale_factor=2, bilinear=self.bilinear)
        else:
            self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes, activation)

    def forward(self, x):
        original_size = x.size()
        if self.srm:
            x = self.srm(x)
        if self.dct:
            x = self.dct_layer(x)
        if self.start_down:
            x1 = self.start_down(x)
        else:
            x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.start_down:
            x = self.last_up(x, original_size)
        logits = self.outc(x)

        return logits


class EfficientEncoder(nn.Module):
    def __init__(self, efn, start, end):
        super(EfficientEncoder, self).__init__()
        self.blocks = efn._blocks[start:end]
        self.blocks_len = len(efn._blocks)
        self.drop_connect_rate = efn._global_params.drop_connect_rate
        self.start = start
        self.end = end

    def forward(self, x):
        for idx, block in enumerate(self.blocks):
            drop_connect_rate = self.drop_connect_rate
            if self.drop_connect_rate:
                drop_connect_rate *= float(self.start + idx) / self.blocks_len
            x = block(x, drop_connect_rate=drop_connect_rate)
        return x


class EfficientUNet(nn.Module):
    def __init__(self, model_name='efficientnet-b0', n_channels=3, n_classes=2, bilinear=True,  pre_trained=True,
                 activation=nn.Softmax(dim=1), start_down=False, scale_factor=2, dct=False, srm=False, use_se=False,
                 dilation=1, learnable=False, init_weights=False, use_edge=False, use_rru_up=False):
        super(EfficientUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.init_weights = init_weights
        self.start_down = start_down
        self.dct = dct
        self.srm = srm
        self.use_edge = use_edge
        self.use_rru_up = use_rru_up

        if self.srm:
            self.srm = SRM_Layer(TLU_threshold=3.0, out_channel=30, use_se=use_se, dilation=dilation, requires_grad=learnable)
            n_channels = 90
        if self.dct:
            self.dct_layer = DCT_Layer(use_se=use_se, dilation=dilation, requires_grad=learnable)
            n_channels = 48

        efn_params = {
            'efficientnet-b0': {'filters': [32, 24, 40, 80, 192], 'ends': [3, 5, 8, 15]},
            'efficientnet-b1': {'filters': [32, 24, 40, 80, 192], 'ends': [5, 8, 12, 21]},
            'efficientnet-b2': {'filters': [32, 24, 48, 88, 208], 'ends': [5, 8, 12, 21]},
            'efficientnet-b3': {'filters': [40, 32, 48, 96, 232], 'ends': [5, 8, 13, 24]},
            'efficientnet-b4': {'filters': [48, 32, 56, 112, 272], 'ends': [6, 10, 16, 30]},
            'efficientnet-b5': {'filters': [48, 40, 64, 128, 304], 'ends': [8, 13, 20, 36]},
            'efficientnet-b6': {'filters': [56, 40, 72, 144, 344], 'ends': [9, 15, 23, 42]},
            'efficientnet-b7': {'filters': [64, 48, 80, 160, 384], 'ends': [11, 18, 28, 51]},
            'tf_efficientnet_b0_ns': {'filters': [32, 24, 40, 80, 192], 'ends': [3, 5, 8, 15]},
            'tf_efficientnet_b1_ns': {'filters': [32, 24, 40, 80, 192], 'ends': [5, 8, 12, 21]},
            'tf_efficientnet_b2_ns': {'filters': [32, 24, 48, 88, 208], 'ends': [5, 8, 12, 21]},
            'tf_efficientnet_b3_ns': {'filters': [40, 32, 48, 96, 232], 'ends': [5, 8, 13, 24]},
            'tf_efficientnet_b4_ns': {'filters': [48, 32, 56, 112, 272], 'ends': [6, 10, 16, 30]},
            'tf_efficientnet_b5_ns': {'filters': [48, 40, 64, 128, 304], 'ends': [8, 13, 20, 36]},
            'tf_efficientnet_b6_ns': {'filters': [56, 40, 72, 144, 344], 'ends': [9, 15, 23, 42]},
            'tf_efficientnet_b7_ns': {'filters': [64, 48, 80, 160, 384], 'ends': [11, 18, 28, 51]},
        }
        filters = efn_params[model_name]['filters']  # 通道数
        ends = efn_params[model_name]['ends']  # 模块的位置
        if 'ns' in model_name:
            efn = timm.create_model(model_name, pretrained=pre_trained)
            print('Using noise student weights!')
        else:
            if pre_trained:
                efn = EfficientNet.from_pretrained(model_name=model_name)
            else:
                efn = EfficientNet.from_name(model_name=model_name)

        if self.start_down:
            # self.start_down = nn.MaxPool2d(2)
            # self.start_down = nn.AvgPool2d(2)
            self.start_down = nn.Sequential(nn.Conv2d(n_channels, filters[0], kernel_size=scale_factor+1, stride=scale_factor, bias=False),
                                            nn.BatchNorm2d(filters[0]))
            # self.start_down = nn.Sequential(
            #     nn.Conv2d(n_channels, filters[0], kernel_size=scale_factor + 1, stride=scale_factor, padding=1),
            #     nn.BatchNorm2d(filters[0]),
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
            #     nn.BatchNorm2d(filters[0]),
            #     nn.ReLU(inplace=True)
            # )
            self.last_up = Last_Up(in_channels=filters[0], scale_factor=scale_factor, bilinear=self.bilinear)
        else:
            self.inc = DoubleConv(n_channels, filters[0])
        # self.firstconv = efn._conv_stem
        # self.firstbn = efn._bn0
        if 'ns' in model_name:
            self.down1 = efn.blocks[:2]  # block1 + block2
            self.down2 = efn.blocks[2:3]  # block3
            self.down3 = efn.blocks[3:4]  # block4
            self.down4 = efn.blocks[4:6]  # block5 + block6
        else:
            self.down1 = EfficientEncoder(efn, start=0, end=ends[0])  # block1 + block2
            self.down2 = EfficientEncoder(efn, start=ends[0], end=ends[1])  # block3
            self.down3 = EfficientEncoder(efn, start=ends[1], end=ends[2])  # block4
            self.down4 = EfficientEncoder(efn, start=ends[2], end=ends[3])  # block5 + block6

        if not self.use_rru_up:
            self.up1 = Up(filters[4] + filters[3], filters[3], bilinear, is_efn=True)
            self.up2 = Up(filters[3] + filters[2], filters[2], bilinear, is_efn=True)
            self.up3 = Up(filters[2] + filters[1], filters[1], bilinear, is_efn=True)
            self.up4 = Up(filters[1] + filters[0], filters[0], bilinear, is_efn=True)
        else:
            self.up1 = RRU_up(filters[4] + filters[3], filters[3], bilinear)
            self.up2 = RRU_up(filters[3] + filters[2], filters[2], bilinear)
            self.up3 = RRU_up(filters[2] + filters[1], filters[1], bilinear)
            self.up4 = RRU_up(filters[1] + filters[0], filters[0], bilinear)

        self.outc = OutConv(filters[0], n_classes, activation)
        if self.use_edge:
            self.outc_edge = OutConv(filters[0], n_classes, activation)

        if self.init_weights:
            if self.start_down:
                self.start_down.apply(weights_init_kaiming)
            self.up1.conv.double_conv.apply(weights_init_kaiming)
            self.up2.conv.double_conv.apply(weights_init_kaiming)
            self.up3.conv.double_conv.apply(weights_init_kaiming)
            self.up4.conv.double_conv.apply(weights_init_kaiming)
            self.outc.conv.apply(weights_init_kaiming)
            print('weights init kaiming!!!')

    def forward(self, x):
        original_size = x.size()
        if self.srm:
            x = self.srm(x)
        if self.dct:
            x = self.dct_layer(x)

        if self.start_down:
            x1 = self.start_down(x)
        else:
            x1 = self.inc(x)
        # x1 = self.firstconv(x)
        # x1 = self.firstbn(x1)  # 32

        x2 = self.down1(x1)  # 24
        x3 = self.down2(x2)  # 40
        x4 = self.down3(x3)  # 80
        x5 = self.down4(x4)  # 112

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.start_down:
            x = self.last_up(x, original_size)
        logits = self.outc(x)

        if self.use_edge:
            edge_logits = self.outc_edge(x)
            return logits, edge_logits

        return logits


class ResNetUNet(nn.Module):
    def __init__(self, model_name='resnet18', n_channels=3, n_classes=2, bilinear=True,  pre_trained=True,
                 activation=nn.Softmax(dim=1), start_down=False, scale_factor=2, dct=False, srm=False, use_se=False,
                 dilation=1, learnable=False, init_weights=False):
        super(ResNetUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.init_weights = init_weights
        self.start_down = start_down
        self.dct = dct
        self.srm = srm

        if self.srm:
            self.srm = SRM_Layer(TLU_threshold=3.0, out_channel=30, use_se=use_se, dilation=dilation, requires_grad=learnable)
            n_channels = 90
        if self.dct:
            self.dct_layer = DCT_Layer(use_se=use_se, dilation=dilation, requires_grad=learnable)
            n_channels = 48

        filters = [64, 64, 128, 256, 512]  # 通道数

        resnet = get_resnet(model_name=model_name, pretrained=pre_trained)

        if self.start_down:
            # self.start_down = nn.MaxPool2d(2)
            # self.start_down = nn.AvgPool2d(2)
            self.start_down = nn.Sequential(nn.Conv2d(n_channels, filters[0], kernel_size=scale_factor+1, stride=scale_factor, bias=False),
                                            nn.BatchNorm2d(filters[0]))
            # self.start_down = nn.Sequential(
            #     nn.Conv2d(n_channels, filters[0], kernel_size=scale_factor + 1, stride=scale_factor, padding=1),
            #     nn.BatchNorm2d(filters[0]),
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
            #     nn.BatchNorm2d(filters[0]),
            #     nn.ReLU(inplace=True)
            # )
            self.last_up = Last_Up(in_channels=filters[0], scale_factor=scale_factor, bilinear=self.bilinear)
        else:
            self.inc = DoubleConv(n_channels, filters[0])

        self.down1 = resnet.layer1
        self.down2 = resnet.layer2
        self.down3 = resnet.layer3
        self.down4 = resnet.layer4

        self.up1 = Up(filters[4] + filters[3], filters[3], bilinear)
        self.up2 = Up(filters[3] + filters[2], filters[2], bilinear)
        self.up3 = Up(filters[2] + filters[1], filters[1], bilinear)
        self.up4 = Up(filters[1] + filters[0], filters[0], bilinear)

        self.outc = OutConv(filters[0], n_classes, activation)

        if self.init_weights:
            if self.start_down:
                self.start_down.apply(weights_init_kaiming)
            self.up1.conv.double_conv.apply(weights_init_kaiming)
            self.up2.conv.double_conv.apply(weights_init_kaiming)
            self.up3.conv.double_conv.apply(weights_init_kaiming)
            self.up4.conv.double_conv.apply(weights_init_kaiming)
            self.outc.conv.apply(weights_init_kaiming)
            print('weights init kaiming!!!')

    def forward(self, x):
        original_size = x.size()
        if self.srm:
            x = self.srm(x)
        if self.dct:
            x = self.dct_layer(x)

        if self.start_down:
            x1 = self.start_down(x)
        else:
            x1 = self.inc(x)
        # x1 = self.firstconv(x)
        # x1 = self.firstbn(x1)  # 32

        x2 = self.down1(x1)  # 24
        x3 = self.down2(x2)  # 40
        x4 = self.down3(x3)  # 80
        x5 = self.down4(x4)  # 112

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.start_down:
            x = self.last_up(x, original_size)
        logits = self.outc(x)

        return logits
