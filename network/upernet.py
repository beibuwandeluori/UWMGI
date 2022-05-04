"""
@ github: https://github.com/yassouali/pytorch-segmentation/blob/master/models/upernet.py
@ paper: https://arxiv.org/pdf/1807.10221.pdf
@ model: UperNet
@ author: Baoying Chen
@ time: 2022/2/24
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .backbone.coatnet import coatnet_0
from .backbone.efficientnet import EfficientNet, tf_efficientnet_ns_feature_extractor
from .backbone.hrnet import tf_hrnet_feature_extractor
from .backbone.convnext import get_convnext
from .backbone.swin_transformer import get_swin_transformer
from .module import SRM_Layer, DCT_Layer


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y


class PSPModule(nn.Module):
    # In the original inmplementation they use precise RoI pooling
    # Instead of using adaptative average pooling
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s)
                                     for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(bin_sizes)), in_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                       align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class ResNet(nn.Module):
    def __init__(self, in_channels=3, output_stride=16, backbone='resnet101', pretrained=True):
        super(ResNet, self).__init__()
        model = getattr(models, backbone)(pretrained)
        if not pretrained or in_channels != 3:
            self.initial = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            initialize_weights(self.initial)
        else:
            self.initial = nn.Sequential(*list(model.children())[:4])

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        if output_stride == 16:
            s3, s4, d3, d4 = (2, 1, 1, 2)
        elif output_stride == 8:
            s3, s4, d3, d4 = (1, 1, 2, 4)

        if output_stride == 8:
            for n, m in self.layer3.named_modules():
                if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'downsample.0' in n:
                    m.stride = (s3, s3)

        for n, m in self.layer4.named_modules():
            if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'downsample.0' in n:
                m.stride = (s4, s4)

    def forward(self, x):
        x = self.initial(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]


class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                                      for ft_size in feature_channels[1:]])
        self.smooth_conv = nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)]
                                         * (len(feature_channels) - 1))
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels) * fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]
        P = [up_and_add(features[i], features[i - 1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1])  # P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x


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


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.Softmax(dim=1)):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = activation

    def forward(self, x):
        if self.activation is not None:
            return self.activation(self.conv(x))
        return self.conv(x)


class UperNet(nn.Module):
    # Implementing only the object path
    def __init__(self, num_classes=1, in_channels=3, backbone='resnet101', pretrained=True,
                 activation=nn.Sigmoid(), use_edge=False, use_se=False, dct=False, srm=False, scale=4,
                 input_size=512, efn_start_down=True, use_roi=False, use_double_conv=False):
        super(UperNet, self).__init__()
        self.use_edge = use_edge
        self.use_roi = use_roi
        self.dct = dct
        self.srm = srm
        self.convnext_scale = scale  # UperNet第一个卷积缩放的比例
        self.use_double_conv = use_double_conv

        if self.srm:
            self.srm = SRM_Layer(TLU_threshold=3.0, out_channel=30, use_se=use_se)
            in_channels = 90
        if self.dct:
            self.dct_layer = DCT_Layer(use_se=use_se)
            in_channels = 48
        if 'resnet' in backbone:
            if backbone == 'resnet34' or backbone == 'resnet18':
                feature_channels = [64, 128, 256, 512]
            else:
                feature_channels = [256, 512, 1024, 2048]
            self.backbone = ResNet(in_channels, backbone=backbone, pretrained=pretrained)
        elif 'convnext' in backbone:
            self.backbone = get_convnext(in_chans=in_channels, model_name=backbone, pretrained=pretrained,
                                         scale=self.convnext_scale)
            feature_channels = self.backbone.dims
        elif 'hrnet' in backbone:
            self.backbone, feature_channels = tf_hrnet_feature_extractor(model_name=backbone,
                                                                         pretrained=pretrained,
                                                                         in_channel=in_channels)
        elif 'tf_efficientnet' in backbone:
            self.backbone, feature_channels = tf_efficientnet_ns_feature_extractor(model_name=backbone,
                                                                                   pretrained=pretrained,
                                                                                   in_channel=in_channels)
        elif 'efficientnet' in backbone:
            self.backbone = EfficientNet(model_name=backbone, pretrained=pretrained, start_down=efn_start_down,
                                         n_channels=in_channels)
            feature_channels = self.backbone.channels
        elif 'swin' in backbone:
            self.backbone = get_swin_transformer(model_name=backbone, pretrained=pretrained, in_chans=in_channels,
                                                 image_size=input_size)
            feature_channels = self.backbone.channels
        elif 'coatnet' in backbone:
            assert input_size % 16 == 0, '输入尺寸必须是16的倍数'
            self.backbone = coatnet_0(image_size=input_size, pretrained=pretrained, in_channels=in_channels)
            feature_channels = self.backbone.channels[:4]
        else:
            pass

        fpn_out = feature_channels[0]
        self.PPN = PSPModule(feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=fpn_out)
        self.head = nn.Conv2d(fpn_out, num_classes, kernel_size=3, padding=1)

        if self.use_double_conv:
            self.double_conv = DoubleConv(num_classes, num_classes)

        self.outc = OutConv(num_classes, num_classes, activation)
        if self.use_edge:
            self.outc_edge = OutConv(num_classes, num_classes, activation)
        if self.use_roi:
            self.outc_roi = OutConv(num_classes, num_classes, activation)

        # self.head = nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)
        # self.outc = OutConv(fpn_out, num_classes, activation)
        # if self.use_edge:
        #     self.outc_edge = OutConv(fpn_out, num_classes, activation)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        if self.srm:
            x = self.srm(x)
        if self.dct:
            x = self.dct_layer(x)

        features = self.backbone(x)
        # print([feature.shape for feature in features])
        features[-1] = self.PPN(features[-1])
        x = self.FPN(features)

        x = self.head(x)
        x = F.interpolate(x, size=input_size, mode='bilinear')
        if self.use_double_conv:
            x = self.double_conv(x)
        logits = self.outc(x)

        out = [logits]
        if self.use_edge:
            edge_logits = self.outc_edge(x)
            out.append(edge_logits)
        if self.use_roi:
            roi_logits = self.outc_roi(x)
            out.append(roi_logits)
            logits = logits * torch.sigmoid(roi_logits)
            out[0] = logits

        return out if len(out) > 1 else out[0]


class NoiseUperNet(nn.Module):
    # Implementing only the object path
    def __init__(self, num_classes=1, in_channels=3, backbone='resnet101', pretrained=True,
                 activation=nn.Sigmoid(), use_edge=False, use_se=True, dct=False, srm=False, scale=4,
                 input_size=512, efn_start_down=True):
        super(NoiseUperNet, self).__init__()
        self.use_edge = use_edge
        self.dct = dct
        self.srm = srm
        self.convnext_scale = scale  # UperNet第一个卷积缩放的比例

        if self.srm:
            self.srm = SRM_Layer(TLU_threshold=3.0, out_channel=30, use_se=use_se)
            self.outc_noise = OutConv(90, num_classes, activation)
        if self.dct:
            self.dct_layer = DCT_Layer(use_se=use_se)
            self.outc_noise = OutConv(48, num_classes, activation)

        if 'resnet' in backbone:
            if backbone == 'resnet34' or backbone == 'resnet18':
                feature_channels = [64, 128, 256, 512]
            else:
                feature_channels = [256, 512, 1024, 2048]
            self.backbone = ResNet(in_channels, backbone=backbone, pretrained=pretrained)
        elif 'convnext' in backbone:
            self.backbone = get_convnext(in_chans=in_channels, model_name=backbone, pretrained=pretrained,
                                         scale=self.convnext_scale)
            feature_channels = self.backbone.dims
        elif 'tf_efficientnet' in backbone:
            self.backbone, feature_channels = tf_efficientnet_ns_feature_extractor(model_name=backbone,
                                                                                   pretrained=pretrained,
                                                                                   in_channel=in_channels)
        elif 'efficientnet' in backbone:
            self.backbone = EfficientNet(model_name=backbone, pretrained=pretrained, start_down=efn_start_down,
                                         n_channels=in_channels)
            feature_channels = self.backbone.channels
        elif 'swin' in backbone:
            self.backbone = get_swin_transformer(model_name=backbone, pretrained=pretrained, in_chans=in_channels,
                                                 image_size=input_size)
            feature_channels = self.backbone.channels
        elif 'coatnet' in backbone:
            assert input_size % 16 == 0, '输入尺寸必须是16的倍数'
            self.backbone = coatnet_0(image_size=input_size, pretrained=pretrained, in_channels=in_channels)
            feature_channels = self.backbone.channels[:4]
        else:
            pass

        fpn_out = feature_channels[0]
        self.PPN = PSPModule(feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=fpn_out)
        self.head = nn.Conv2d(fpn_out, num_classes, kernel_size=3, padding=1)

        last_in = num_classes * 2 if self.srm or self.dct else num_classes
        self.outc = OutConv(last_in, num_classes, activation)
        if self.use_edge:
            self.outc_edge = OutConv(last_in, num_classes, activation)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        if self.srm:
            x_srm = self.srm(x)
            x_noise = self.outc_noise(x_srm)
        if self.dct:
            x_dct = self.dct_layer(x)
            x_noise = self.outc_noise(x_dct)

        features = self.backbone(x)
        # print([feature.shape for feature in features])
        features[-1] = self.PPN(features[-1])
        x = self.FPN(features)

        x = self.head(x)
        x = F.interpolate(x, size=input_size, mode='bilinear')
        if self.srm or self.dct:
            x = torch.cat([x, x_noise], dim=1)

        logits = self.outc(x)
        out = [logits]
        if self.use_edge:
            edge_logits = self.outc_edge(x)
            out.append(edge_logits)

        return out if len(out) > 1 else out[0]