#!/usr/bin/env python3
# coding: utf-8


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from networks import *

class Decoder(nn.Module):
    def __init__(self, inchannels = [256, 512, 1024, 2048], midchannels = [256, 256, 256, 512], upfactors = [2,2,2,2], outchannels = 1):
        super(Decoder, self).__init__()
        self.inchannels = inchannels
        self.midchannels = midchannels
        self.upfactors = upfactors
        self.outchannels = outchannels

        self.conv = FTB(inchannels=self.inchannels[3], midchannels=self.midchannels[3])
        self.conv1 = nn.Conv2d(in_channels=self.midchannels[3], out_channels=self.midchannels[2], kernel_size=3, padding=1, stride=1, bias=True)
        self.upsample = nn.Upsample(scale_factor=self.upfactors[3], mode='bilinear', align_corners=True)

        self.ffm2 = FFM(inchannels=self.inchannels[2], midchannels=self.midchannels[2], outchannels = self.midchannels[2], upfactor=self.upfactors[2])
        self.ffm1 = FFM(inchannels=self.inchannels[1], midchannels=self.midchannels[1], outchannels = self.midchannels[1], upfactor=self.upfactors[1])
        self.ffm0 = FFM(inchannels=self.inchannels[0], midchannels=self.midchannels[0], outchannels = self.midchannels[0], upfactor=self.upfactors[0])

        self.outconv = AO(inchannels=self.inchannels[0], outchannels=self.outchannels, upfactor=2)

        self._init_params()

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                #init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                #init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                #init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): #nn.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, features):
        _,_,h,w = features[3].size()
        feat=[]
        x = self.conv(features[3])
        x = self.conv1(x)
        x = self.upsample(x)
        feat.append(x)
        x = self.ffm2(features[2], x)
        feat.append(x)
        x = self.ffm1(features[1], x)
        feat.append(x)
        x = self.ffm0(features[0], x)
        feat.append(x)
        

        #-----------------------------------------
        x = self.outconv(x)
        return x,feat
