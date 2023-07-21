#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Ke Xian
Email: kexian@hust.edu.cn
Date: 2020/07/20
'''

import torch
import torch.nn as nn
import torch.nn.init as init

# ==============================================================================================================

class FTB(nn.Module):
    def __init__(self, inchannels, midchannels=512):
        super(FTB, self).__init__()
        self.in1 = inchannels
        self.mid = midchannels

        self.conv1 = nn.Conv2d(in_channels=self.in1, out_channels=self.mid, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv_branch = nn.Sequential(nn.ReLU(inplace=True),\
                                         nn.Conv2d(in_channels=self.mid, out_channels=self.mid, kernel_size=3, padding=1, stride=1, bias=True),\
                                         #nn.BatchNorm2d(num_features=self.mid),\
                                         nn.ReLU(inplace=True),\
                                         nn.Conv2d(in_channels=self.mid, out_channels= self.mid, kernel_size=3, padding=1, stride=1, bias=True))
        self.relu = nn.ReLU(inplace=True)

        self.init_params()

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.conv_branch(x)
        x = self.relu(x)

        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                #init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                # init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):  #nn.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class ATA(nn.Module):
    def __init__(self, inchannels, reduction = 8):
        super(ATA, self).__init__()
        self.inchannels = inchannels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(self.inchannels*2, self.inchannels // reduction),
                                nn.ReLU(inplace=True),
                                nn.Linear(self.inchannels // reduction, self.inchannels),
                                nn.Sigmoid())
        self.init_params()

    def forward(self, low_x, high_x):
        n, c, _, _ = low_x.size()
        x = torch.cat([low_x, high_x], 1)
        x = self.avg_pool(x)
        x = x.view(n, -1)
        x = self.fc(x).view(n,c,1,1)
        x = low_x * x + high_x

        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #init.kaiming_normal_(m.weight, mode='fan_out')
                #init.normal(m.weight, std=0.01)
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                #init.kaiming_normal_(m.weight, mode='fan_out')
                #init.normal_(m.weight, std=0.01)
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): #nn.BatchNorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class FFM(nn.Module):
    def __init__(self, inchannels, midchannels, outchannels, upfactor=2):
        super(FFM, self).__init__()
        self.inchannels = inchannels
        self.midchannels = midchannels
        self.outchannels = outchannels
        self.upfactor = upfactor

        self.ftb1 = FTB(inchannels=self.inchannels, midchannels=self.midchannels)
        self.ftb2 = FTB(inchannels=self.midchannels, midchannels=self.outchannels)

        self.upsample = nn.Upsample(scale_factor=self.upfactor, mode='bilinear', align_corners=True)

        self.init_params()
        #self.p1 = nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False)
        #self.p2 = nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False)
        #self.p3 = nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False)

    def forward(self, low_x, high_x):

        x = self.ftb1(low_x)
        
        '''
        x = torch.cat((x,high_x),1)
        if x.shape[2] == 12:
            x = self.p1(x)
        elif x.shape[2] == 24:
            x = self.p2(x)
        elif x.shape[2] == 48:
            x = self.p3(x)
        '''
        x = x + high_x            ###high_x
        x = self.ftb2(x)
        x = self.upsample(x)

        return x

    def init_params(self):
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
            elif isinstance(m, nn.BatchNorm2d): #nn.Batchnorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)



class noFFM(nn.Module):
    def __init__(self, inchannels, midchannels, outchannels, upfactor=2):
        super(noFFM, self).__init__()
        self.inchannels = inchannels
        self.midchannels = midchannels
        self.outchannels = outchannels
        self.upfactor = upfactor

        self.ftb2 = FTB(inchannels=self.midchannels, midchannels=self.outchannels)

        self.upsample = nn.Upsample(scale_factor=self.upfactor, mode='bilinear', align_corners=True)

        self.init_params()
        #self.p1 = nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False)
        #self.p2 = nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False)
        #self.p3 = nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False)

    def forward(self, low_x, high_x):

        #x = self.ftb1(low_x)
        x = high_x            ###high_x
        x = self.ftb2(x)
        x = self.upsample(x)

        return x

    def init_params(self):
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
            elif isinstance(m, nn.BatchNorm2d): #nn.Batchnorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)




class AO(nn.Module):
    # Adaptive output module
    def __init__(self, inchannels, outchannels, upfactor=2):
        super(AO, self).__init__()
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.upfactor = upfactor

        """
        self.adapt_conv = nn.Sequential(nn.Conv2d(in_channels=self.inchannels, out_channels=self.inchannels//2, kernel_size=3, padding=1, stride=1, bias=True),\
                                  nn.BatchNorm2d(num_features=self.inchannels//2),\
                                  nn.ReLU(inplace=True),\
                                  nn.Conv2d(in_channels=self.inchannels//2, out_channels=self.outchannels, kernel_size=3, padding=1, stride=1, bias=True),\
                                  nn.Upsample(scale_factor=self.upfactor, mode='bilinear', align_corners=True) )#,\
                                  #nn.ReLU(inplace=True))  ## get positive values
        """
        self.adapt_conv = nn.Sequential(nn.Conv2d(in_channels=self.inchannels, out_channels=self.inchannels//2, kernel_size=3, padding=1, stride=1, bias=True),\
                                  #nn.BatchNorm2d(num_features=self.inchannels//2),\
                                  nn.ReLU(inplace=True),\
                                  nn.Upsample(scale_factor=self.upfactor, mode='bilinear', align_corners=True), \
                                  nn.Conv2d(in_channels=self.inchannels//2, out_channels=self.outchannels, kernel_size=1, padding=0, stride=1))

                                  #nn.ReLU(inplace=True))  ## get positive values

        self.init_params()

    def forward(self, x):
        x = self.adapt_conv(x)
        return x

    def init_params(self):
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
            elif isinstance(m, nn.BatchNorm2d): #nn.Batchnorm2d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class ASPP(nn.Module):
    def __init__(self, inchannels=256, planes=128, rates = [1, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.inchannels = inchannels
        self.planes = planes
        self.rates = rates
        self.kernel_sizes = []
        self.paddings = []
        for rate in self.rates:
            if rate == 1:
                self.kernel_sizes.append(1)
                self.paddings.append(0)
            else:
                self.kernel_sizes.append(3)
                self.paddings.append(rate)
        self.atrous_0 = nn.Sequential(nn.Conv2d(in_channels=self.inchannels, out_channels=self.planes, kernel_size=self.kernel_sizes[0],
                                                     stride=1, padding=self.paddings[0], dilation=self.rates[0], bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(num_features=self.planes)
                                      )
        self.atrous_1 = nn.Sequential(nn.Conv2d(in_channels=self.inchannels, out_channels=self.planes, kernel_size=self.kernel_sizes[1],
                                                     stride=1, padding=self.paddings[1], dilation=self.rates[1], bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(num_features=self.planes),
                                      )
        self.atrous_2 = nn.Sequential(nn.Conv2d(in_channels=self.inchannels, out_channels=self.planes, kernel_size=self.kernel_sizes[2],
                                                     stride=1, padding=self.paddings[2], dilation=self.rates[2], bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(num_features=self.planes),
                                      )
        self.atrous_3 = nn.Sequential(nn.Conv2d(in_channels=self.inchannels, out_channels=self.planes, kernel_size=self.kernel_sizes[3],
                                                     stride=1, padding=self.paddings[3], dilation=self.rates[3], bias=True),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(num_features=self.planes),
                                      )

        #self.conv = nn.Conv2d(in_channels=self.planes * 4, out_channels=self.inchannels, kernel_size=3, padding=1, stride=1, bias=True)
    def forward(self, x):
        x = torch.cat([self.atrous_0(x), self.atrous_1(x), self.atrous_2(x), self.atrous_3(x)],1)
        #x = self.conv(x)

        return x

# ==============================================================================================================


class ResidualConv(nn.Module):
    def __init__(self, inchannels):
        super(ResidualConv, self).__init__()
        #nn.BatchNorm2d
        self.conv = nn.Sequential(
                                  #nn.BatchNorm2d(num_features=inchannels),
                                  nn.ReLU(inplace=False),
                                  #nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=3, padding=1, stride=1, groups=inchannels,bias=True),
                                  #nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=1, padding=0, stride=1, groups=1,bias=True)
                                  nn.Conv2d(in_channels=inchannels, out_channels=inchannels//2, kernel_size=3, padding=1, stride=1, bias=False),
                                  nn.BatchNorm2d(num_features=inchannels//2),
                                  nn.ReLU(inplace=False),
                                  nn.Conv2d(in_channels=inchannels//2, out_channels=inchannels, kernel_size=3, padding=1, stride=1, bias=False)
                                  )
        self.init_params()

    def forward(self, x):
        x = self.conv(x)+x
        return x

    def init_params(self):
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


class FeatureFusion(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(FeatureFusion, self).__init__()
        self.conv = ResidualConv(inchannels=inchannels)
        #nn.BatchNorm2d
        self.up = nn.Sequential(ResidualConv(inchannels=inchannels),
                                nn.ConvTranspose2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3,stride=2, padding=1, output_padding=1),
                                nn.BatchNorm2d(num_features=outchannels),
                                nn.ReLU(inplace=True))

    def forward(self, lowfeat, highfeat):
        return self.up(highfeat + self.conv(lowfeat))

    def init_params(self):
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


class SenceUnderstand(nn.Module):
    def __init__(self, channels):
        super(SenceUnderstand, self).__init__()
        self.channels = channels
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
                                   nn.ReLU(inplace = True))
        self.pool = nn.AdaptiveAvgPool2d(8)
        self.fc = nn.Sequential(nn.Linear(512*8*8, self.channels),
                                nn.ReLU(inplace = True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, padding=0),
                                   nn.ReLU(inplace=True))
        self.initial_params()

    def forward(self, x):
        n,c,h,w = x.size()
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(n,-1)
        x = self.fc(x)
        x = x.view(n, self.channels, 1, 1)
        x = self.conv2(x)
        x = x.repeat(1,1,h,w)
        return x

    def initial_params(self, dev=0.01):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #print torch.sum(m.weight)
                m.weight.data.normal_(0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.ConvTranspose2d):
                #print torch.sum(m.weight)
                m.weight.data.normal_(0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, dev)
