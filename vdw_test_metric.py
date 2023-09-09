import os
import argparse
import cv2
from os.path import join as pjoin
import json
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import matplotlib.pyplot as plt
import random
import os
from torchvision import transforms, utils
import scipy.io as io
from Depth_decoder import Decoder
import torch.nn as nn
import random
from backbone import *
import torch 
from networks import *
from full_model import *
from utils import frame_utils
from utils.flow_viz import save_vis_flow_tofile
import argparse
from utils.utils import InputPadder, compute_out_of_boundary_mask
import glob
from gmflow.geometry import *
from gmflow.gmflow import GMFlow
from utils.flow_viz import save_vis_flow_tofile
from PIL import Image
from smooth_loss import *
from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet


def gt_png_loader(path):

    depth = cv2.imread(path, -1) / 65535.0
    return depth.astype(np.float32)


def mask_png_loader(path):

    mask = cv2.imread(path, -1)
    return mask

def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--vnum', default='1203', type=str)
    parser.add_argument('--result_dir', default='./nvds_vdwtest_results/', type=str)
    parser.add_argument('--gt_dir', default='./vdw_test/', type=str)

    return parser


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

def compute_errors(pred, gt, mask):
    
    pred = pred[mask==1]
    gt = gt[mask==1]

    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()

    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()
    
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    
    #rmse_log = (np.log(gt) - np.log(pred)) ** 2
    #rmse_log = np.sqrt(rmse_log.mean())
    
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    
    #err = np.log(pred) - np.log(gt)
    #silog = np.sqrt(np.mean(err ** 2) - 0.85*np.mean(err) ** 2) * 10
    
    #err = np.abs(np.log10(pred) - np.log10(gt))
    #log10 = np.mean(err)
    
    return rmse, abs_rel, sq_rel, d1, d2, d3


def compute_scale_and_shift(prediction, target, mask):

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))
    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]
    
    return x_0, x_1




if __name__ == '__main__':    

    parser = get_args_parser()
    args = parser.parse_args()
    
    threshold = 1.25
    
    ksh_base_dir = args.result_dir 
    
    pred_dir = ksh_base_dir + args.vnum+'/*/'
    
    prelen = len(ksh_base_dir + args.vnum+'/')
    
    pred_times = glob.glob(pred_dir)
    
    pred_times.sort()
    
    pred_times_nums = pred_times[:-2]
    
    pred_times_nums.sort(key=lambda x: int(x[prelen:-1]))
    
    gt_dir = args.gt_dir + args.vnum + '/left_gt/*.png' #'/data/lijiaqi/22w1/demo0301/gt/' + args.vnum + '/*.png'
    
    mask_dir = args.gt_dir + args.vnum + '/left_mask/*.png'
    
    gt_names = glob.glob(gt_dir)
    
    
    mask_names = glob.glob(mask_dir)
    
    gt_names.sort(key=lambda x: int(x[-10:-4]))
    
    mask_names.sort(key=lambda x: int(x[-10:-4]))
    
    all_d1 = []
    
    for i in range(len(pred_times)):
        
        if i == 0:
            pre_times_dir = pred_times[-2]
        elif i == len(pred_times)-1:
            pre_times_dir = pred_times[-1]
        else:
            pre_times_dir = pred_times_nums[i-1]
        
        print(i+1,'/',len(pred_times),pre_times_dir)
        
        pred_names = glob.glob(pre_times_dir+'/gray/*.png')
        
        pred_names.sort(key=lambda x: int(x[-10:-4]))
        
        
        assert len(pred_names)==len(gt_names)
        
        #assert len(pred_names)==len(mask_names)
        
        d1_time = 0
        abs_rel_time = 0
        d2_time = 0
        d3_time = 0

        for j in range(len(pred_names)):
            
            pred_name = pred_names[j]
            
            gt_name = gt_names[j]
            
            mask_name = mask_names[j]
            
            print(pred_name)
            print(gt_name)
            print(mask_name)
            
            
            
            gt = gt_png_loader(gt_name)
            gt = torch.Tensor(np.ascontiguousarray(gt.astype(np.float32))).unsqueeze(0)
            
            
            mask = mask_png_loader(mask_name)
            mask[mask!=255] = 1
            mask[mask==255] = 0
            mask = torch.Tensor(np.ascontiguousarray(mask)).unsqueeze(0)
            mask1 = gt>0
            mask = torch.mul(mask,mask1)
            
            #print(torch.min(normalize_prediction_robust(gt,mask)),torch.max(normalize_prediction_robust(gt,mask))) 
            
            pred = gt_png_loader(pred_name)
            pred = torch.Tensor(np.ascontiguousarray(cv2.resize(pred,(mask.shape[2], mask.shape[1]),interpolation=cv2.INTER_AREA))).unsqueeze(0)
            
            scale, shift = compute_scale_and_shift(pred, gt, mask) 
            
            pred_align = scale.view(-1, 1, 1) * pred + shift.view(-1, 1, 1)
            rmse, abs_rel, sq_rel, d1, d2, d3 = compute_errors(pred_align,gt,mask)


            
            
            err = torch.zeros_like(pred_align, dtype=torch.float)
            err[mask == 1] = torch.max(
            pred_align[mask == 1] / gt[mask == 1],
            gt[mask == 1] / pred_align[mask == 1],
            )
            err[mask == 1] = (err[mask == 1] < threshold).float()
            d1 = torch.sum(err, (1, 2)) / torch.sum(mask, (1, 2))
            #print(d1)
            d1_time = d1_time + d1.item()
            abs_rel_time = abs_rel_time + abs_rel.item()
            d2_time = d2_time + d2.item()
            d3_time = d3_time + d3.item()

        
        d1_time = d1_time / len(pred_names)
        d2_time = d2_time / len(pred_names)
        d3_time = d3_time / len(pred_names)
        abs_rel_time = abs_rel_time / len(pred_names)

        
        all_d1.append(d1_time)
        
        with open(ksh_base_dir + args.vnum+'/accuracy.txt','a') as f:
            if i == 0:
                f.write('\n'+'**********Initial**********'+'\n')
            elif i == len(pred_times)-1:
                f.write('\n'+'**********Mixing**********'+'\n')
            else:
                f.write('\n'+'**********Looping**********'+'\n')
                
            f.write('d1:'+ str(round(d1_time,3))+',d2:'+ str(round(d2_time,3))+',d3:'+ str(round(d3_time,3))+',absrel:'+ str(round(abs_rel_time,3))+'\n')

        
        #break ###
    
    
    #print(all_d1[0],all_d1[-1],max(all_d1[1:-1]))
    
    
    
            
            
            
            
            
            
            
            
            
    
    
    
    