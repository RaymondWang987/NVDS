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

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def get_args_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial_type",
        default= 'dpt',
        type=str,
        help="max epochs for training"
    )

    return parser


def gt_png_loader_nyu(path):

    depth = cv2.imread(path, -1)

    depth = depth / 1000.0

    return depth.astype(np.float32)
    
def img_loader(path):
    
    image = cv2.imread(path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

    return image

def gt_png_loader(path):

    depth = cv2.imread(path, -1) / 65535.0          
    return depth.astype(np.float32)
    
def compute_errors(pred, gt, mask):
    
    pred = pred[mask]
    gt = gt[mask]

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

parser = get_args_parser()
args = parser.parse_args()
print('Evaluations of NVDS on NYUDV2, initial model type:', args.initial_type)


datadir = './test_nyu_data/'
all_test = os.listdir(datadir)
all_test.sort(key=lambda x: int(x))

device = torch.device("cuda:0")
device_flow = torch.device('cuda:0')
__mean = [0.485, 0.456, 0.406]
__std = [0.229, 0.224, 0.225]
__mean_dpt = [0.5, 0.5, 0.5]
__std_dpt = [0.5, 0.5, 0.5]
seq_len = 4
clip_step = 1

# model and to_device 

checkpoint = torch.load('./NVDS_checkpoints/NVDS_Stabilizer_NYUDV2_Finetuned.pth', map_location = 'cpu')
model = NVDS()
model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

if args.initial_type == 'dpt':
    dpt = DPTDepthModel(
            path='./dpt/checkpoints/dpt_large-midas-2f21e586.pt',
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        ).to(device_flow)
elif args.initial_type == 'midas':
    dpt = MidasNet_large('./dpt/checkpoints/midas_v21-f6b98070.pt', non_negative=True).to(device_flow)


dpt.eval()

# Initial (Midas/DPT) Pred

for i in range(len(all_test)):

    print('Initial '+ args.initial_type + ' prediction:',i+1,'/',len(all_test))
    
    test_dir = datadir + all_test[i]
    save_disp_dir = test_dir + '/initial_'+args.initial_type+'/'
    os.makedirs(save_disp_dir, exist_ok=True)
    if len(os.listdir(test_dir+'/rgb/'))==4:
        
        test_frames = glob.glob(test_dir+'/rgb/*.png')
        test_frames.sort(key=lambda x: int(x[-10:-4]))
        
        #DPT input
        
        for j in range(seq_len):
            
            frame = test_frames[j]
            rgb = img_loader(frame)
            rgb = cv2.resize(
                      rgb,
                      (384, 384),
                      interpolation=cv2.INTER_CUBIC
                      )
            rgb = (rgb - __mean_dpt) / __std_dpt
            rgb = np.transpose(rgb, (2, 0, 1))
            rgb = torch.Tensor(np.ascontiguousarray(rgb).astype(np.float32)).unsqueeze(0)
        
            rgb = rgb.to(device)
            
            if j == 0:
                dpt_seq = rgb
            else:
                dpt_seq = torch.cat([dpt_seq,rgb],dim=0)
    
    else:
        test_frames = glob.glob(test_dir+'/rgb/*.png')
        test_frames.sort(key=lambda x: int(x[-10:-4]))
        frame = test_frames[0]
        dpt_seq = img_loader(frame)
        dpt_seq = cv2.resize(
                      dpt_seq,
                      (384, 384),
                      interpolation=cv2.INTER_CUBIC
                      )
        dpt_seq = (dpt_seq - __mean_dpt) / __std_dpt
        dpt_seq = np.transpose(dpt_seq, (2, 0, 1))
        dpt_seq = torch.Tensor(np.ascontiguousarray(dpt_seq).astype(np.float32)).unsqueeze(0)
        dpt_seq = dpt_seq.to(device)
        
    with torch.no_grad():
        outputs = dpt.forward(dpt_seq)
        outputs = outputs.cpu().numpy()#.squeeze()
        for u in range(outputs.shape[0]):
        
            depth_min = outputs[u].min()
            depth_max = outputs[u].max()
            outputs[u] = 65535.0 * (outputs[u] - depth_min) / (depth_max - depth_min)
            cv2.imwrite(save_disp_dir+'/%06d'%(u)+'.png', outputs[u].astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])


   

###NVDS pred


for i in range(len(all_test)):

    print('NVDS prediction:',i+1,'/',len(all_test))
    
    test_dir = datadir + all_test[i]
    save_disp_dir = test_dir + '/NVDS_'+args.initial_type+'/'
    os.makedirs(save_disp_dir, exist_ok=True)
    if len(os.listdir(test_dir+'/rgb/'))==4:
        
        test_frames = glob.glob(test_dir+'/rgb/*.png')
        test_frames.sort(key=lambda x: int(x[-10:-4]))
        current_disp = glob.glob(test_dir+'/initial_'+args.initial_type+'/*.png')
        current_disp.sort(key=lambda x: int(x[-10:-4]))
        
        #DPT input
        
        for j in range(seq_len):
            
            frame = test_frames[j]
            
            rgb = img_loader(frame)
            rgb = cv2.resize(
                      rgb,
                      (384, 384),
                      interpolation=cv2.INTER_CUBIC
                      )
            rgb = (rgb - __mean) / __std
            rgb = np.transpose(rgb, (2, 0, 1))
            rgb = torch.Tensor(np.ascontiguousarray(rgb).astype(np.float32)).unsqueeze(0)
        
            depth = gt_png_loader(current_disp[j])
            depth = cv2.resize(
                        depth,
                        (384, 384),
                        interpolation=cv2.INTER_NEAREST
                        )
            depth = torch.Tensor(np.ascontiguousarray(depth.astype(np.float32))).unsqueeze(0)
            depth = (depth-torch.min(depth))/(torch.max(depth)-torch.min(depth))
            depth = depth.unsqueeze(0)
            rgbd = torch.cat([rgb,depth],dim=1)
            
            if j==0:
                ref_seq = rgbd
            else:
                ref_seq = torch.cat([ref_seq,rgbd],dim=0)
                
        ref_seq = ref_seq.unsqueeze(0)
    
    else:
        test_frames = glob.glob(test_dir+'/rgb/*.png')
        test_frames.sort(key=lambda x: int(x[-10:-4]))
        current_disp = glob.glob(test_dir+'/initial_'+args.initial_type+'/*.png')
        current_disp.sort(key=lambda x: int(x[-10:-4]))
        
        frame = test_frames[0]
        rgb = img_loader(frame)
        rgb = cv2.resize(
                      rgb,
                      (384, 384),
                      interpolation=cv2.INTER_CUBIC
                      )
        rgb = (rgb - __mean) / __std
        rgb = np.transpose(rgb, (2, 0, 1))
        rgb = torch.Tensor(np.ascontiguousarray(rgb).astype(np.float32)).unsqueeze(0)
        
        depth = gt_png_loader(current_disp[0])
        depth = cv2.resize(
                        depth,
                        (384, 384),
                        interpolation=cv2.INTER_NEAREST
                        )
        depth = torch.Tensor(np.ascontiguousarray(depth.astype(np.float32))).unsqueeze(0)
        depth = (depth-torch.min(depth))/(torch.max(depth)-torch.min(depth))
        depth = depth.unsqueeze(0)
        rgbd = torch.cat([rgb,depth],dim=1)
        
        for j in range(seq_len):
            if j == 0:
                ref_seq = rgbd
            else:
                ref_seq = torch.cat([ref_seq,rgbd],dim=0)
        
        ref_seq = ref_seq.unsqueeze(0)
    
    
    ref_seq = ref_seq.to(device)
    
    
    
    with torch.no_grad():
    
        outputs = model(ref_seq)
        outputs = outputs.squeeze(1)
        outputs = outputs.cpu().numpy().squeeze()
        depth_min = outputs.min()
        depth_max = outputs.max()
        outputs = 65535.0 * (outputs - depth_min) / (depth_max - depth_min)
        cv2.imwrite(save_disp_dir+'000003.png', outputs.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])


### testing
all_dpt_d1 = 0
all_dpt_d2 = 0
all_dpt_d3 = 0
all_dpt_rel = 0
all_ours_d1 = 0
all_ours_d2 = 0
all_ours_d3 = 0
all_ours_rel = 0

for i in range(len(all_test)):

    print('Metrics Evaluations:',i+1,'/',len(all_test))
    
    test_dir = datadir + all_test[i]
    
    dpt_test = glob.glob(test_dir + '/initial_'+args.initial_type+'/*.png')
    dpt_test.sort(key=lambda x: int(x[-10:-4]))
    dpt_pred_name = dpt_test[-1]
    ours_pred_name = test_dir + '/NVDS_'+args.initial_type+'/000003.png'
    gt_name  = glob.glob(test_dir + '/gt/*.png')[0]
    gt = gt_png_loader_nyu(gt_name)
    gt = torch.Tensor(np.ascontiguousarray(gt.astype(np.float32))).unsqueeze(0)
    mask = gt>0
    
    dpt_pred = gt_png_loader(dpt_pred_name)
    dpt_pred = cv2.resize(
                        dpt_pred,
                        (gt.shape[2], gt.shape[1]),
                        interpolation=cv2.INTER_NEAREST
                        )
    dpt_pred = torch.Tensor(np.ascontiguousarray(dpt_pred.astype(np.float32))).unsqueeze(0)
    ours_pred = gt_png_loader(ours_pred_name)
    ours_pred = cv2.resize(
                        ours_pred,
                        (gt.shape[2], gt.shape[1]),
                        interpolation=cv2.INTER_NEAREST
                        )
    ours_pred = torch.Tensor(np.ascontiguousarray(ours_pred.astype(np.float32))).unsqueeze(0)
    
    target_disparity = torch.zeros_like(gt)
    target_disparity[mask == 1] = 1.0 / gt[mask == 1]
    scale, shift = compute_scale_and_shift(ours_pred, target_disparity, mask)
    ours_pred = scale.view(-1, 1, 1) * ours_pred + shift.view(-1, 1, 1)
    
    disparity_cap = 1.0 / 10
    ours_pred[ours_pred<disparity_cap] = disparity_cap
    ours_depth = 1.0 / ours_pred
    ours_depth = torch.clamp(ours_depth, min=0.7133, max=9.9955)[:,44:471, 40:601] 
    
    rmse, abs_rel, sq_rel, d1, d2, d3 = compute_errors(ours_depth,gt[:,44:471, 40:601],mask[:,44:471, 40:601])
    all_ours_d1 += d1
    all_ours_d2 += d2
    all_ours_d3 += d3
    all_ours_rel += abs_rel
    
    scaledpt, shiftdpt = compute_scale_and_shift(dpt_pred, target_disparity, mask)
    dpt_pred = scaledpt.view(-1, 1, 1) * dpt_pred + shiftdpt.view(-1, 1, 1)
    
    disparity_cap = 1.0 / 10
    dpt_pred[dpt_pred<disparity_cap] = disparity_cap
    dpt_depth = 1.0 / dpt_pred
    dpt_depth = torch.clamp(dpt_depth, min=0.7133, max=9.9955)[:,44:471, 40:601]
    
    rmsedpt, abs_reldpt, sq_reldpt, d1dpt, d2dpt, d3dpt = compute_errors(dpt_depth,gt[:,44:471, 40:601],mask[:,44:471, 40:601])
    all_dpt_d1 += d1dpt
    all_dpt_d2 += d2dpt
    all_dpt_d3 += d3dpt
    all_dpt_rel += abs_reldpt
    
print('Methods\t\t','d1\t\t','d2\t\t','d3\t\t','rel\t\t')
print('Initial_'+args.initial_type+':', all_dpt_d1/len(all_test),all_dpt_d2/len(all_test),all_dpt_d3/len(all_test),all_dpt_rel/len(all_test))
print('NVDS_'+args.initial_type+':',all_ours_d1/len(all_test),all_ours_d2/len(all_test),all_ours_d3/len(all_test),all_ours_rel/len(all_test))

    


        
    
    
    
    
    
    
    
    
    
    
    
    
    