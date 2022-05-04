import timm
import torch
import torch.nn as nn


# 当in_channel != 3 时，初始化模型的第一个Conv的weight， 把之前的通道copy input_chaneel/3 次
def init_imagenet_weight(_conv_stem_weight, input_channel=3):
    for i in range(input_channel // 3):
        if i == 0:
            _conv_stem_weight_new = _conv_stem_weight
        else:
            _conv_stem_weight_new = torch.cat([_conv_stem_weight_new, _conv_stem_weight], axis=1)

    return torch.nn.Parameter(_conv_stem_weight_new)


def tf_hrnet_feature_extractor(model_name='hrnet_w18',
                                         pretrained=True,
                                         features_only=True,
                                         in_channel=3):
    assert model_name in ['hrnet_w18', 'hrnet_w30', 'hrnet_w32', 'hrnet_w40', 'hrnet_w44', 'hrnet_w48', 'hrnet_w64']
    model = timm.create_model(model_name, pretrained=pretrained, features_only=features_only, out_indices=(0, 1, 2, 3))
    channels = [64, 128, 256, 512]
    if in_channel != 3:
        model.conv_stem.in_channels = in_channel
        model.conv_stem.weight = init_imagenet_weight(model.conv_stem.weight, in_channel)

    return model, channels


if __name__ == '__main__':
    image_size = 224
    model, channels = tf_hrnet_feature_extractor(model_name='hrnet_w18', pretrained=True, features_only=True)
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
