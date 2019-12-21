"""
reference: https://pytorch.org/docs/stable/_modules/torchvision/models/densenet.html
           https://zhuanlan.zhihu.com/p/37189203
author: alazycoder
data: 2019.12.20
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class DenseLayer(nn.Module):
    def __int__(self, num_input_features, growth_rate, bn_size, drop_rate):
        """
        :param num_input_features: number of channels of DenseLayer's input
        :param growth_rate: number of channels of DenseLayer's output
        :param bn_size: bn means bottle neck, the number of channels of bottle neck layers's output is
                        bn_size * growth_rate
        :param drop_rate: dropout rate of last conv layer' output
        """
        super().__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3,
                                           stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, inputs):
        """
        :param inputs: [N*C*W*H], list of previous's layers outputs
        :return:  N*C*W*H
        """
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]
        # channel wise concat
        concated_features = torch.cat(inputs, 1)
        # bottle neck
        bottle_neck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        # new feature
        new_features = self.conv2(self.relu2(self.norm2(bottle_neck_output)))
        # dropout
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class DenseBlock(nn.Module):
    def __int__(self, num_layers, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.layers = nn.ModuleDict()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.layers[f"denselayer{i+1}"] = layer

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.layers.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))
    # 因为Transition是nn.Sequential的子类，nn.Sequential有实现forward方法，input会依次通过各个layer，所以不需要再重写forward了


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
                 bn_size=4, drop_rate=0, num_classes=1000):
        super().__init__()
        # first convolution
        self.features = nn.Sequential(
            OrderDict([
                ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                ("norm0", nn.BatchNorm2d(num_init_features)),
                ("relu0", nn.ReLU(inplace=True)),
                ("pool0", nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
            ]))
        # Each DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_features, growth_rate, bn_size, drop_rate)
            self.features.add_module(f"denseblock{i+1}", block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition(num_features, num_features//2)
                self.features.add_module(f"transition{i+1}", trans)
                num_features = num_features // 2
        # Final batch norm
        # 因为当前最后是Transition的conv 和 pool, 所以要在加一层norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.features.add_module("relu5", nn.ReLU(inplace=True))
        self.classifier = nn.Linear(num_features, num_classes)

        # init params
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

