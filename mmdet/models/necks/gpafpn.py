import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16
import torch
import math
from ..builder import NECKS
from .fpn import FPN
from mmcv.ops import SAConv2d
from SSAC.ssaconv import SSAC

class ECA(nn.Module):
    def __init__(self,gamma = 2,b=1):
        super(ECA, self).__init__()
        t=int(abs((math.log(256, 2) + b) / gamma))
        k=t if t % 2 else t + 1
        #k=3
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size = k, padding = int(k/2), bias = False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1,-2))
        y = y.transpose(-1,-2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class SEBlock(nn.Module):
    def __init__(self,channel=256,r = 16):
        super(SEBlock,self).__init__()
        self.avg_pool =nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,channel//r,bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel//r,channel,bias = False),
            nn.Sigmoid(),
        )
    def forward(self,x):
        b,c,_,_ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b,c)
        # Excitation
        y = self.fc(y).view(b,c,1,1)
        # Fscale
        y = torch.mul(x,y)
        return y

class ChannelAttention(nn.Module):
    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(256, 16, 1, bias = False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(16, 256, 1, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        y = self.sigmoid(out)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding = padding, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim = 1, keepdim = True)
        max_out, _ = torch.max(x, dim = 1, keepdim = True)
        y = torch.cat([avg_out, max_out], dim = 1)
        y = self.conv1(y)
        y= self.sigmoid(y)
        return x * y.expand_as(x)

class CBAM(nn.Module):
    def __init__(self,):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention()
        self.sa = SpatialAttention()

    def forward(self, x):
        ChannelAttention_down_aft = self.ca(x)
        ChannelAttention_after = torch.einsum('bnoo,bnwh->bnwh', ChannelAttention_down_aft, x)
        SpatialAttention_after = self.sa(ChannelAttention_after)
        down_after_gate = torch.einsum('bowh,bnwh->bnwh', SpatialAttention_after, ChannelAttention_after)

        return down_after_gate


@NECKS.register_module()
class GPAFPN(FPN):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PAFPN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): Whether to add conv layers on top of the
            original feature maps. Default: False.
        extra_convs_on_inputs (bool): Whether to apply extra conv on
            the original feature from the backbone. Default: False.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 # extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 attention_ways = "other",
                 using_ssac = False,
                 where_using_ssac = 0,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(GPAFPN, self).__init__(in_channels,
                             out_channels,
                             num_outs,
                             start_level=start_level,
                             end_level=end_level,
                             add_extra_convs=add_extra_convs,
                             relu_before_extra_convs=relu_before_extra_convs,
                             no_norm_on_lateral=no_norm_on_lateral,
                             conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg
                             )
        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()

        self.attention_ways = attention_ways
        self.using_ssac = using_ssac
        self.where_using_ssac = where_using_ssac

        for i in range(self.start_level + 1, self.backbone_end_level):
            d_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            pafpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)
        self.sac_conv=SAConv2d(
                in_channels = out_channels,
                out_channels = out_channels,
                kernel_size = 1,
            )
        self.ssac = SSAC(in_channels=out_channels,
                 out_channels=out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        #########################################################
        ECA1 = ECA()
        ECA1.cuda()
        SEBlock1 = SEBlock()
        SEBlock1.cuda()
        CBAM1 = CBAM()
        CBAM1.cuda()
        #########################################################

        #part 1: build top-down path
        used_backbone_levels = len(laterals)

        #original
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')  # 'bilinear'

        # build outputs
        # part 1: from original levels

        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)    ###gai   sac_convs,self.fpn_convs[i]
        ]

        if self.attention_ways == 'SE':

            for i in range(0,used_backbone_levels-1):
                down_aft = self.downsample_convs[i](inter_outs[i])
                SE_down_after = SEBlock1(down_aft)
                inter_outs[i + 1] += SE_down_after + down_aft
        elif self.attention_ways == 'ECA':

            for i in range(0, used_backbone_levels - 1):
                down_after = self.downsample_convs[i](inter_outs[i])
                ECA_down_after = ECA1(down_after)
                inter_outs[i + 1] += ECA_down_after + down_after
        elif self.attention_ways == 'CBAM':

            for i in range(0, used_backbone_levels - 1):
                down_after = self.downsample_convs[i](inter_outs[i])
                CBAM_down_after = CBAM1(down_after)
                inter_outs[i + 1] += CBAM_down_after + down_after

        else:

            for i in range(0, used_backbone_levels - 1):
                inter_outs[i + 1] += self.downsample_convs[i](inter_outs[i])


        # outs = []
        # outs.extend( sac_laterals[i] for i in range(4))

        outs = []
        outs.append(inter_outs[0])
        outs.extend([
            self.pafpn_convs[i - 1](inter_outs[i])
            for i in range(1, used_backbone_levels)
        ])

        # part 3: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)  where SAC and sac_conv can exchange
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    if self.using_ssac == True and self.where_using_ssac == 1:
                        outs1 = self.ssac(outs[-1])
                        outs.append(F.max_pool2d(outs1, 1, stride=2))
                    elif self.using_ssac == False and self.where_using_ssac == 1:
                        outs.append(F.max_pool2d(outs[-1], 1, stride=2))

                    elif self.using_ssac == True and self.where_using_ssac == 2:
                        outs2 = self.ssac(outs[-2])
                        outs.append(F.max_pool2d(outs2, 1, stride=4))
                    elif self.using_ssac == False and self.where_using_ssac == 2:
                        outs.append(F.max_pool2d(outs[-2], 1, stride=4))

                    elif self.using_ssac == True and self.where_using_ssac == 3:
                        outs3 = self.ssac(outs[-3])
                        outs.append(F.max_pool2d(outs3, 1, stride=8))
                    elif self.using_ssac == False and self.where_using_ssac == 3:
                        outs.append(F.max_pool2d(outs[-3], 1, stride=8))

                    else:
                        outs.append(F.max_pool2d(outs[-1], 1, stride=2))

            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
