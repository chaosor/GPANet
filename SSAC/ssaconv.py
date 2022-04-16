#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time : 2021/9/2 9:56
# @Author : zyc17
# @File : saconv.py
# @Software: PyCharm
import torch

from .conv_ws import ConvAWS2d

class SSAC(ConvAWS2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # self.switch = torch.nn.Conv2d(
        #     self.in_channels,
        #     1,
        #     kernel_size=1,
        #     stride=stride,
        #     bias=True)
        # self.switch.weight.data.fill_(0)
        # self.switch.bias.data.fill_(1)
        # self.weight_diff = torch.nn.Parameter(torch.Tensor(self.weight.size()))
        # self.weight_diff.data.zero_()
        self.soft =torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channels,1,kernel_size=3,stride=1,padding=1,bias=True),
            torch.nn.Sigmoid()
        )
        # self.soft.weight.data.fill_(0)
        # self.soft.bias.data.fill_(1)
        # self.weight_diff = torch.nn.Parameter(torch.Tensor(self.weight.size()))
        # self.weight_diff.data.zero_()

        self.pre_context = torch.nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=1,
            bias=True)
        self.pre_context.weight.data.fill_(0)
        self.pre_context.bias.data.fill_(0)
        self.post_context = torch.nn.Conv2d(
            self.out_channels,
            self.out_channels,
            kernel_size=1,
            bias=True)
        self.post_context.weight.data.fill_(0)
        self.post_context.bias.data.fill_(0)

    def forward(self, x):
        # pre-context
        avg_x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)
        avg_x = self.pre_context(avg_x)
        avg_x = avg_x.expand_as(x)
        x = x + avg_x
        # switch
        # avg_x = torch.nn.functional.pad(x, pad=(2, 2, 2, 2), mode="constant")
        # avg_x = torch.nn.functional.avg_pool2d(avg_x, kernel_size=5, stride=1, padding=0)
        soft =  self.soft(x)
        # switch = self.switch(avg_x)

        # sac
        weight = self._get_weight(self.weight)

        out_s = super()._conv_forward(x, weight, bias=self.bias)
        ori_p = self.padding
        ori_d = self.dilation
        self.padding = tuple(3 * p for p in self.padding)
        self.dilation = tuple(3 * d for d in self.dilation)
        # weight = weight + self.weight_diff

        out_l = super()._conv_forward(x, weight, bias=self.bias)
        out = soft * out_s + (1 - soft) * out_l
        self.padding = ori_p
        self.dilation = ori_d
        # post-context
        avg_x = torch.nn.functional.adaptive_avg_pool2d(out, output_size=1)
        avg_x = self.post_context(avg_x)
        avg_x = avg_x.expand_as(out)
        out = out + avg_x
        return out
