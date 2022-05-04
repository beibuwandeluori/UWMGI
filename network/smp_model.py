import segmentation_models_pytorch as smp
import torch.nn as nn
import torch
import torch.nn.functional as F

try:
    from .module import SRM_Layer, DCT_Layer
except:
    from module import SRM_Layer, DCT_Layer


def get_smp_model(model_name='timm-efficientnet-b4',
                  num_classes=3,
                  in_channels=1,
                  encoder_weights='imagenet',
                  activation='sigmoid', model_type='unet'):
    if model_type == 'unet++':
        model = smp.UnetPlusPlus(
            encoder_name=model_name,
            classes=num_classes,
            in_channels=in_channels,
            encoder_weights=encoder_weights,
            activation=activation,
        )
    elif model_type == 'unet':
        model = smp.Unet(
            encoder_name=model_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=encoder_weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,  # model output channels (number of classes in your dataset)
            activation=activation,
        )

    return model


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.Softmax(dim=1)):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = activation

    def forward(self, x):
        if self.activation is not None:
            return self.activation(self.conv(x))
        return self.conv(x)


class SMPModel(nn.Module):
    # Implementing only the object path
    def __init__(self, num_classes=1, in_channels=3, backbone='timm-efficientnet-b5', pretrained=True,
                 activation=nn.Sigmoid(), use_edge=False, use_se=False, dct=False, srm=False, model_type='unetpp'):
        super(SMPModel, self).__init__()
        self.use_edge = use_edge
        self.dct = dct
        self.srm = srm
        self.model_type = model_type

        if self.srm:
            self.srm = SRM_Layer(TLU_threshold=3.0, out_channel=30, use_se=use_se)
            in_channels = 90
        if self.dct:
            self.dct_layer = DCT_Layer(use_se=use_se)
            in_channels = 48
        encoder_weights = 'imagenet' if pretrained else None
        if model_type == 'unetpp':
            self.model = smp.UnetPlusPlus(encoder_name=backbone,
                                          classes=num_classes,
                                          encoder_weights=encoder_weights,
                                          in_channels=in_channels,
                                          activation=None)
        elif model_type == 'psp':
            self.model = smp.PSPNet(encoder_name=backbone,
                                    classes=num_classes,
                                    encoder_weights=encoder_weights,
                                    in_channels=in_channels,
                                    activation=None)
        elif model_type == 'pan':
            self.model = smp.PAN(encoder_name=backbone,
                                 classes=num_classes,
                                 encoder_weights=encoder_weights,
                                 in_channels=in_channels,
                                 activation=None)
        else:
            pass

        self.outc = OutConv(num_classes, num_classes, activation)
        if self.use_edge:
            self.outc_edge = OutConv(num_classes, num_classes, activation)

    def forward(self, x):
        if self.model_type == 'unetpp':
            assert x.size()[2] % 32 == 0 and x.size()[3] % 32 == 0  # 要被32整除
        input_size = (x.size()[2], x.size()[3])
        if self.srm:
            x = self.srm(x)
        if self.dct:
            x = self.dct_layer(x)

        x = self.model(x)

        x = F.interpolate(x, size=input_size, mode='bilinear')
        logits = self.outc(x)
        if self.use_edge:
            edge_logits = self.outc_edge(x)
            return logits, edge_logits

        return logits


if __name__ == '__main__':
    image_size = 600
    use_edge = True
    # model = get_smp_model(model_name='timm-efficientnet-b2',
    #                       num_classes=1,
    #                       encoder_weights='imagenet',
    #                       activation='sigmoid')
    model = SMPModel(backbone='timm-efficientnet-b5', pretrained=True, use_edge=use_edge, use_se=False, dct=False,
                     srm=False, model_type='pan')
    print(model)
    model = model.to(torch.device('cpu'))
    model.eval()
    print(model)

    # input_s = (3, image_size, image_size)
    # print(summary(model, input_s, device='cpu'))

    img = torch.randn(4, 3, image_size, image_size)  # your high resolution picture
    outs = model(img)
    if not use_edge:
        print(outs.shape)
    else:
        print([out.shape for out in outs])
