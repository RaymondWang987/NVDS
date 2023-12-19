import os
import numpy as np
import torch.nn as nn
import torch
# from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict
# from mmseg.ops import resize
# from builder import HEADS
from decode_head import BaseDecodeHead, BaseDecodeHead_clips, BaseDecodeHead_clips_flow
# from mmseg.models.utils import *
import attr
from IPython import embed
from stabilization_attention import BasicLayer3d3
import cv2
from networks import *
import warnings
# from mmcv.utils import Registry, build_from_cfg
from torch import nn
from backbone import *
from stabilization_network import *
from collections import OrderedDict

class NVDS(nn.Module):
    def __init__(self,use_pretrain='False'):
        super().__init__()

        self.backbone = mit_b5()
        self.seq_len = 4
        self.use_pretrain = use_pretrain

        if self.use_pretrain is True:
            self.backbone.init_weights('/xxx/mit_b5.pth')

        old_conv = self.backbone.patch_embed1.proj
        new_conv = nn.Conv2d(old_conv.in_channels + 1, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding)

        new_conv.weight[:, :3, :, :].data.copy_(old_conv.weight.clone())
        self.backbone.patch_embed1.proj = new_conv

        self.Stabilizer = Stabilization_Network_Cross_Attention(in_channels=[64, 128, 320, 512],
                    in_index=[0, 1, 2, 3],
                    feature_strides=[4, 8, 16, 32],
                    channels=128,
                    dropout_ratio=0.1,
                    num_classes=1,
                    align_corners=False,
                    decoder_params=dict(embed_dim=256, depths=4),
                    num_clips=4,
                    norm_cfg = dict(type='SyncBN', requires_grad=True))
        self.edge_conv = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1, bias=True),\
                                  nn.ReLU(inplace=True))
        self.edge_conv1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2, bias=True),\
                                  nn.ReLU(inplace=True))



    def forward(self, inputs, num_clips=None, imgs=None):

        edge_feat = self.edge_conv(inputs[:,-1,0:-1,:,:])
        edge_feat1 = self.edge_conv1(edge_feat)

        x = self.backbone(inputs)
        outputs = self.Stabilizer(x,edge_feat,edge_feat1,num_clips=4)
        outputs = F.relu(outputs)

        return outputs