# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 10/4/2020 14:17

import paddle
from paddle import nn
import paddle.nn.functional as F


class MultiAspectGCAttention(nn.Layer):

    def __init__(self,
                 inplanes,
                 ratio,
                 headers,
                 pooling_type='att',
                 att_scale=False,
                 fusion_type='channel_add'):
        super(MultiAspectGCAttention, self).__init__()
        assert pooling_type in ['avg', 'att']

        assert fusion_type in ['channel_add', 'channel_mul', 'channel_concat']
        assert inplanes % headers == 0 and inplanes >= 8  # inplanes must be divided by headers evenly

        self.headers = headers
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_type = fusion_type
        self.att_scale = False

        self.single_header_inplanes = int(inplanes / headers)

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2D(self.single_header_inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(axis=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2D(1)

        if fusion_type == 'channel_add':
            self.channel_add_conv = nn.Sequential(
                nn.Conv2D(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(),
                nn.Conv2D(self.planes, self.inplanes, kernel_size=1))
        elif fusion_type == 'channel_concat':
            self.channel_concat_conv = nn.Sequential(
                nn.Conv2D(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(),
                nn.Conv2D(self.planes, self.inplanes, kernel_size=1))
            # for concat
            self.cat_conv = nn.Conv2D(2 * self.inplanes, self.inplanes, kernel_size=1)
        elif fusion_type == 'channel_mul':
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2D(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(),
                nn.Conv2D(self.planes, self.inplanes, kernel_size=1))

    def spatial_pool(self, x):
        batch, channel, height, width = x.shape
        if self.pooling_type == 'att':
            # [N*headers, C', H , W] C = headers * C'
            x = x.reshape([batch * self.headers, self.single_header_inplanes, height, width])
            input_x = x

            # [N*headers, C', H * W] C = headers * C'
            # input_x = input_x.view(batch, channel, height * width)
            input_x = input_x.reshape([batch * self.headers, self.single_header_inplanes, height * width])

            # [N*headers, 1, C', H * W]
            input_x = input_x.unsqueeze(1)
            # [N*headers, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N*headers, 1, H * W]
            context_mask = context_mask.reshape([batch * self.headers, 1, height * width])

            # scale variance
            if self.att_scale and self.headers > 1:
                context_mask = context_mask / paddle.sqrt(self.single_header_inplanes)

            # [N*headers, 1, H * W]
            context_mask = self.softmax(context_mask)

            # [N*headers, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N*headers, 1, C', 1] = [N*headers, 1, C', H * W] * [N*headers, 1, H * W, 1]
            context = paddle.matmul(input_x, context_mask)

            # [N, headers * C', 1, 1]
            context = context.reshape([batch, self.headers * self.single_header_inplanes, 1, 1])
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x

        if self.fusion_type == 'channel_mul':
            # [N, C, 1, 1]
            channel_mul_term = F.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        elif self.fusion_type == 'channel_add':
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        else:
            # [N, C, 1, 1]
            channel_concat_term = self.channel_concat_conv(context)

            # use concat
            _, C1, _, _ = channel_concat_term.shape
            N, C2, H, W = out.shape

            out = paddle.concat([out, channel_concat_term.expand([-1, -1, H, W])], axis=1)
            out = self.cat_conv(out)
            out = F.layer_norm(out, [self.inplanes, H, W])
            out = F.relu(out)

        return out
