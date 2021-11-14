#encoding=utf8
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle import nn

from model.context_block import MultiAspectGCAttention
from model.initializers import kaiming_normal_, constant_


# CNN for Feature Extraction + Multi-Aspect GCAttention

def conv3x3(in_planes, out_planes, stride=1):
    # "3x3 convolution with padding"
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias_attr=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias_attr=False)


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_gcb=False, gcb_config=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2D(planes, momentum=0.9)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2D(planes, momentum=0.9)
        self.downsample = downsample
        self.stride = stride
        self.use_gcb = use_gcb

        if self.use_gcb:
            gcb_ratio = gcb_config['ratio']
            gcb_headers = gcb_config['headers']
            att_scale = gcb_config['att_scale']
            fusion_type = gcb_config['fusion_type']
            self.context_block = MultiAspectGCAttention(inplanes=planes,
                                                        ratio=gcb_ratio,
                                                        headers=gcb_headers,
                                                        att_scale=att_scale,
                                                        fusion_type=fusion_type)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_gcb:
            out = self.context_block(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2D(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Layer):

    def __init__(self, block, layers, zero_init_residual=False, gcb=None, in_channels=1):
        super(ResNet, self).__init__()
        gcb_config = gcb

        self.inplanes = 128
        self.conv1 = nn.Conv2D(in_channels, 64, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2D(64, 128, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(128)
        self.relu2 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2D(kernel_size=2, stride=2)

        self.layer1 = self._make_layer(block, 256, layers[0], stride=1, gcb_config=gcb_config,
                                       use_gcb=gcb_config['layers'][0])

        self.conv3 = nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(256)
        self.relu3 = nn.ReLU()

        self.maxpool2 = nn.MaxPool2D(kernel_size=2, stride=2)

        self.layer2 = self._make_layer(block, 256, layers[1], stride=1, gcb_config=gcb_config,
                                       use_gcb=gcb_config['layers'][1])

        self.conv4 = nn.Conv2D(256, 256, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn4 = nn.BatchNorm2D(256)
        self.relu4 = nn.ReLU()

        self.maxpool3 = nn.MaxPool2D(kernel_size=(2, 1), stride=(2, 1))

        self.layer3 = self._make_layer(block, 512, layers[2], stride=1, gcb_config=gcb_config,
                                       use_gcb=gcb_config['layers'][2])

        self.conv5 = nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn5 = nn.BatchNorm2D(512)
        self.relu5 = nn.ReLU()

        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, gcb_config=gcb_config,
                                       use_gcb=gcb_config['layers'][3])

        self.conv6 = nn.Conv2D(512, 512, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn6 = nn.BatchNorm2D(512)
        self.relu6 = nn.ReLU()

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2D):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.sublayers():
                if isinstance(m, Bottleneck):
                    constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, use_gcb=False, gcb_config=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2D(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_gcb=use_gcb, gcb_config=gcb_config))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)
        x = self.layer1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool2(x)
        x = self.layer2(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.maxpool3(x)
        x = self.layer3(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.layer4(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        return x


def resnet50(gcb_kwargs, in_channels=1):
    model = ResNet(BasicBlock, [1, 2, 5, 3], gcb=gcb_kwargs, in_channels=in_channels)
    return model


class ConvEmbeddingGC(nn.Layer):

    def __init__(self, gcb_kwargs, in_channels=1):
        super().__init__()
        self.backbone = resnet50(gcb_kwargs, in_channels=in_channels)

    def forward(self, x):
        feature = self.backbone(x)
        b, c, h, w = feature.shape  # （B， C， H/8, W/4）
        feature = feature.reshape((b, c, h * w))
        feature = feature.transpose((0, 2, 1))
        return feature
