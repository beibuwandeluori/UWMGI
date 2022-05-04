from efficientnet_pytorch import EfficientNet as EffNet
import torch
import torch.nn as nn
import timm


# 当in_channel != 3 时，初始化模型的第一个Conv的weight， 把之前的通道copy input_chaneel/3 次
def init_imagenet_weight(_conv_stem_weight, input_channel=3):
    if input_channel == 1:
        _conv_stem_weight_new = _conv_stem_weight[:, 0:1:, :, :]
    else:
        for i in range(input_channel // 3):
            if i == 0:
                _conv_stem_weight_new = _conv_stem_weight
            else:
                _conv_stem_weight_new = torch.cat([_conv_stem_weight_new, _conv_stem_weight], axis=1)

    return torch.nn.Parameter(_conv_stem_weight_new)


class EfficientNet(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, model_name='efficientnet-b0', pretrained=False, start_down=True, n_channels=3):
        super(EfficientNet, self).__init__()
        self.start_down = start_down  # 是否使用EfficientNet的第一层下采卷积
        if pretrained:
            model = EffNet.from_pretrained(model_name)
        else:
            model = EffNet.from_name(model_name)

        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

        efn_params = {
            'efficientnet-b0': [16, 24, 40, 112],
            'efficientnet-b1': [16, 24, 40, 112],
            'efficientnet-b2': [16, 24, 48, 120],
            'efficientnet-b3': [24, 32, 48, 136],
            'efficientnet-b4': [24, 32, 56, 160],
            'efficientnet-b5': [24, 40, 64, 176],
            'efficientnet-b6': [32, 40, 72, 200],
            'efficientnet-b7': [32, 48, 80, 224]
        }
        self.channels = efn_params[model_name]

        if n_channels != 3:
            self.model._conv_stem.in_channels = n_channels
            self.model._conv_stem.weight = init_imagenet_weight(self.model._conv_stem.weight, n_channels)
        if not self.start_down:
            self.model._conv_stem.stride = 1

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)

        feature_maps = []

        # TODO: temporarily storing extra tensor last_x and del it later might not be a good idea,
        #  try recording stride changing when creating efficientnet,
        #  and then apply it here.
        last_x = None
        block_len = len(self.model._blocks)
        for idx, block in enumerate(self.model._blocks[:block_len]):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks[:block_len]) - 1:
                feature_maps.append(x)
            last_x = x
            if len(feature_maps) == 4:
                break
        del last_x
        return feature_maps


def tf_efficientnet_ns_feature_extractor(model_name='tf_efficientnet_b0_ns',
                                         pretrained=True,
                                         features_only=True,
                                         in_channel=3):
    model = timm.create_model(model_name, pretrained=pretrained, features_only=features_only)
    model_params = {
        'tf_efficientnet_b0_ns': [16, 24, 40, 112],
        'tf_efficientnet_b1_ns': [16, 24, 40, 112],
        'tf_efficientnet_b2_ns': [16, 24, 48, 120],
        'tf_efficientnet_b3_ns': [24, 32, 48, 136],
        'tf_efficientnet_b4_ns': [24, 32, 56, 160],
        'tf_efficientnet_b5_ns': [24, 40, 64, 176],
        'tf_efficientnet_b6_ns': [32, 40, 72, 200],
        'tf_efficientnet_b7_ns': [32, 48, 80, 224],
        'tf_efficientnet_b8': [32, 56, 88, 248],
        'tf_efficientnet_l2_ns': [72, 104, 176, 480],
        'tf_efficientnetv2_s_in21ft1k': [24, 48, 64, 160],
        'tf_efficientnetv2_m_in21ft1k': [24, 48, 80, 176],
        'tf_efficientnetv2_l_in21ft1k': [32, 64, 96, 224],
    }
    channels = model_params[model_name]
    model.blocks = model.blocks[:5]  # 只需要提取前四个特征
    if in_channel != 3:
        model.conv_stem.in_channels = in_channel
        model.conv_stem.weight = init_imagenet_weight(model.conv_stem.weight, in_channel)

    return model, channels


if __name__ == '__main__':
    image_size = 224
    from timm.models.efficientnet import tf_efficientnet_b0_ns
    # model = EfficientNet(model_name='efficientnet-b8', pretrain=True, start_down=False, n_channels=3)
    # model = tf_efficientnet_b0_ns(pretrain=True, features_only=True)
    model = timm.create_model('tf_efficientnetv2_l_in21ft1k', pretrained=True, features_only=True)
    # model, channels = tf_efficientnet_ns_feature_extractor(model_name='tf_efficientnetv2_s_in21ft1k',
    #                                                        pretrained=True,
    #                                                        features_only=True)
    model = model.to(torch.device('cpu'))
    model.eval()
    print(model)
    # print(channels, len(model.blocks))

    # from torchsummary import summary
    # input_s = (3, image_size, image_size)
    # print(summary(model, input_s, device='cpu'))

    img = torch.randn(1, 3, image_size, image_size)  # your high resolution picture
    features = model(img)
    print([feature.shape for feature in features], len(features))
