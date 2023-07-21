import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
import cv2



def normalize_prediction_robust(target, mask):
    ssum = torch.sum(mask, (1, 2))
    valid = ssum > 0

    m = torch.zeros_like(ssum)
    s = torch.ones_like(ssum)

    m[valid] = torch.median(
        (mask[valid] * target[valid]).view(valid.sum(), -1), dim=1
    ).values
    target = target - m.view(-1, 1, 1)

    sq = torch.sum(mask * target.abs(), (1, 2))
    s[valid] = torch.clamp((sq[valid] / ssum[valid]), min=1e-6)

    return target / (s.view(-1, 1, 1))


def normalize_prediction_test(target,mask):
    
    scale = torch.median(target)
    
    target = target - scale
    
    shift = torch.mean(torch.abs(target))

    target = target / shift

    return target


class flow_warping_loss_align(nn.Module):
    def __init__(self, alpha=50):
        super(flow_warping_loss_align, self).__init__()
        self.alpha = alpha

    def forward(self,warp_rgb,rgb,warp_depth,depth,device):#,mask_flow):
        
        #print(warp_depth.shape)
        #exit(1)
        warp_depth = normalize_prediction_robust(warp_depth.squeeze(1),torch.ones((warp_depth.shape[0],384,384)).to(device)).unsqueeze(1)
        depth = normalize_prediction_robust(depth.squeeze(1),torch.ones((warp_depth.shape[0],384,384)).to(device)).unsqueeze(1)
        
        #print(torch.mean(warp_depth),torch.mean(depth))
        diff_depth = torch.abs(warp_depth - depth)
        
        diff_rgb = (warp_rgb - rgb)**2
        
        #print(warp_depth.shape,depth.shape,diff_depth.shape)
        mask_rgb = torch.exp(-(self.alpha*diff_rgb))
        #print(mask_rgb.shape,diff_rgb.shape,warp_rgb.shape,rgb.shape)
        mask_rgb = torch.sum(mask_rgb,dim=1,keepdim=True)
        #print(mask_rgb.shape)
        weight_diff = torch.mul(mask_rgb,diff_depth)
        #print(weight_diff.shape)
        loss_one_pair = 10*torch.mean(weight_diff)

        return loss_one_pair




class flow_warping_loss_align_test(nn.Module):
    def __init__(self, infer_h,infer_w,alpha=50):
        super(flow_warping_loss_align_test, self).__init__()
        self.alpha = alpha
        self.infer_h = infer_h
        self.infer_w = infer_w
        
    def forward(self,warp_rgb,rgb,warp_depth,depth,device):#,mask_flow):
        
        #print(warp_depth.shape)
        #exit(1)
        warp_depth = normalize_prediction_robust(warp_depth.squeeze(1),torch.ones((warp_depth.shape[0],self.infer_h,self.infer_w)).to(device)).unsqueeze(1)
        depth = normalize_prediction_robust(depth.squeeze(1),torch.ones((warp_depth.shape[0],self.infer_h,self.infer_w)).to(device)).unsqueeze(1)
        
        #print(torch.mean(warp_depth),torch.mean(depth))
        diff_depth = torch.abs(warp_depth - depth)
        
        diff_rgb = (warp_rgb - rgb)**2
        
        #print(warp_depth.shape,depth.shape,diff_depth.shape)
        mask_rgb = torch.exp(-(self.alpha*diff_rgb))
        #print(mask_rgb.shape,diff_rgb.shape,warp_rgb.shape,rgb.shape)
        mask_rgb = torch.sum(mask_rgb,dim=1,keepdim=True)
        #print(mask_rgb.shape)
        weight_diff = torch.mul(mask_rgb,diff_depth)
        #print(weight_diff.shape)
        loss_one_pair = 10*torch.mean(weight_diff)

        return loss_one_pair

'''
class flow_warping_loss_align_test672(nn.Module):
    def __init__(self, alpha=50):
        super(flow_warping_loss_align_test672, self).__init__()
        self.alpha = alpha

    def forward(self,warp_rgb,rgb,warp_depth,depth,device):#,mask_flow):
        
        #print(warp_depth.shape)
        #exit(1)
        warp_depth = normalize_prediction_robust(warp_depth.squeeze(1),torch.ones((warp_depth.shape[0],384,672)).to(device)).unsqueeze(1)
        depth = normalize_prediction_robust(depth.squeeze(1),torch.ones((warp_depth.shape[0],384,672)).to(device)).unsqueeze(1)
        
        #print(torch.mean(warp_depth),torch.mean(depth))
        diff_depth = torch.abs(warp_depth - depth)
        
        diff_rgb = (warp_rgb - rgb)**2
        
        #print(warp_depth.shape,depth.shape,diff_depth.shape)
        mask_rgb = torch.exp(-(self.alpha*diff_rgb))
        #print(mask_rgb.shape,diff_rgb.shape,warp_rgb.shape,rgb.shape)
        mask_rgb = torch.sum(mask_rgb,dim=1,keepdim=True)
        #print(mask_rgb.shape)
        weight_diff = torch.mul(mask_rgb,diff_depth)
        #print(weight_diff.shape)
        loss_one_pair = 10*torch.mean(weight_diff)

        return loss_one_pair
'''