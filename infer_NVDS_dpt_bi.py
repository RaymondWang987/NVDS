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

    # dataset
    parser.add_argument('--checkpoint_dir', default='tmp', type=str,
                        help='where to save the training log and models')
    parser.add_argument('--stage', default='chairs', type=str,
                        help='training stage')
    parser.add_argument('--image_size', default=[384, 512], type=int, nargs='+',
                        help='image size for training')
    parser.add_argument('--padding_factor', default=16, type=int,
                        help='the input should be divisible by padding_factor, otherwise do padding')

    parser.add_argument('--max_flow', default=400, type=int,
                        help='exclude very large motions during training')
    parser.add_argument('--val_dataset', default=['chairs'], type=str, nargs='+',
                        help='validation dataset')
    parser.add_argument('--with_speed_metric', action='store_true',
                        help='with speed metric when evaluation')

    # training
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--grad_clip', default=1.0, type=float)
    parser.add_argument('--num_steps', default=100000, type=int)
    parser.add_argument('--seed', default=326, type=int)
    parser.add_argument('--summary_freq', default=100, type=int)
    parser.add_argument('--val_freq', default=10000, type=int)
    parser.add_argument('--save_ckpt_freq', default=10000, type=int)
    parser.add_argument('--save_latest_ckpt_freq', default=1000, type=int)

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrain model for finetuing or resume from terminated training')
    parser.add_argument('--strict_resume', action='store_true')
    parser.add_argument('--no_resume_optimizer', action='store_true')

    # GMFlow model
    parser.add_argument('--num_scales', default=1, type=int,
                        help='basic gmflow model uses a single 1/8 feature, the refinement uses 1/4 feature')
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--attention_type', default='swin', type=str)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)

    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                        help='number of splits in attention')
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                        help='correlation radius for matching, -1 indicates global matching')
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                        help='self-attention radius for flow propagation, -1 indicates global attention')

    # loss
    parser.add_argument('--gamma', default=0.9, type=float,
                        help='loss weight')

    # evaluation
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--save_eval_to_file', action='store_true')
    parser.add_argument('--evaluate_matched_unmatched', action='store_true')

    # inference on a directory
    #parser.add_argument('--inference_dir', default=None, type=str)
    parser.add_argument('--inference_size', default=None, type=int, nargs='+',
                        help='can specify the inference size')
    parser.add_argument('--dir_paired_data', action='store_true',
                        help='Paired data in a dir instead of a sequence')
    parser.add_argument('--save_flo_flow', action='store_true')
    parser.add_argument('--pred_bidir_flow', action='store_true',
                        help='predict bidirectional flow')
    parser.add_argument('--fwd_bwd_consistency_check', action='store_true',
                        help='forward backward consistency check with bidirection flow')

    # predict on sintel and kitti test set for submission
    parser.add_argument('--submission', action='store_true',
                        help='submission to sintel or kitti test sets')
    parser.add_argument('--output_path', default='output', type=str,
                        help='where to save the prediction results')
    parser.add_argument('--save_vis_flow', action='store_true',
                        help='visualize flow prediction as .png image')
    parser.add_argument('--no_save_flo', action='store_true',
                        help='not save flow as .flo')

    # distributed training
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--launcher', default='none', type=str, choices=['none', 'pytorch'])
    parser.add_argument('--gpu_ids', default=0, type=int, nargs='+')

    parser.add_argument('--count_time', action='store_true',
                        help='measure the inference time on sintel')
    
    parser.add_argument(
        "--all_seq_len",
        type= int,
        default= 4 ,
        help="all sequence length for training"
    )
    parser.add_argument(
        "--max_epoch",
        default= 500,
        help="max epochs for training"
    )
    parser.add_argument(
        "--vnum",
        default= '000423',
        type=str
    )
    parser.add_argument(
        "--timesall",
        default= 2,
        type=int
    ) 
    parser.add_argument(
        "--base_dir",
        default= '/xxx/xxx/',
        type=str
    )
    parser.add_argument(
        "--infer_w",
        default= '896',
        type=int
    )
    parser.add_argument(
        "--infer_h",
        default= '384',
        type=int
    )
    
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    return parser


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def setup_for_distributed(is_master):

    import builtins as __builtin__
    builtin_print = __builtin__.print
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    setup_for_distributed(args.rank == 0)


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

def img_loader(path):
    
    image = cv2.imread(path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

    return image

def gt_png_loader(path):

    depth = cv2.imread(path, -1) #/ 65535.0          
    return depth.astype(np.float32)


if __name__ == '__main__':

    print('let us begin test NVDS(DPT) demo')
    device = torch.device("cuda:0")
    device_flow = torch.device('cuda:0')
    __mean = [0.485, 0.456, 0.406]
    __std = [0.229, 0.224, 0.225]
    __mean_dpt = [0.5, 0.5, 0.5]
    __std_dpt = [0.5, 0.5, 0.5]
    
    
    parser = get_args_parser()
    args = parser.parse_args()
    
    model_flow = GMFlow(feature_channels=args.feature_channels,
                   num_scales=args.num_scales,
                   upsample_factor=args.upsample_factor,
                   num_head=args.num_head,
                   attention_type=args.attention_type,
                   ffn_dim_expansion=args.ffn_dim_expansion,
                   num_transformer_layers=args.num_transformer_layers,
                   ).to(device_flow)
    model_flow = torch.nn.DataParallel(model_flow,device_ids=[0]) 
    model_flow = model_flow.module
    args.resume = './gmflow/checkpoints/gmflow_sintel-0c07dcb3.pth'
    print('Load checkpoint: %s' % args.resume)
    loc = 'cuda:{}'.format(args.local_rank)
    checkpoint = torch.load(args.resume, map_location = 'cpu')
    weights = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model_flow.load_state_dict(weights, strict=args.strict_resume)
    model_flow.to(device_flow)
    model_flow.eval()
    
    # model and to_device 
    checkpoint = torch.load('./NVDS_checkpoints/NVDS_Stabilizer.pth', map_location = 'cpu') 
    model = NVDS()
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    model.load_state_dict(checkpoint)
    model.to(device) 
    model.eval()
    
    
    dpt = DPTDepthModel(
            path='./dpt/checkpoints/dpt_large-midas-2f21e586.pt',
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        ).to(device_flow)
    dpt.eval()
    
    seq_len = 4
    clip_step = 1
    video_dir = './demo_videos/' +args.vnum+'/'
    infer_size = (int(args.infer_w),int(args.infer_h))
    temloss = flow_warping_loss_align_test(int(args.infer_h),int(args.infer_w))
    
    frames = glob.glob(video_dir+'/left/*.png')
    frames.sort(key=lambda x: int(x[-10:-4]))
    
    base_dir = args.base_dir
    save_disp_dir = base_dir + '/initial/gray/' 
    save_ksh_dir = base_dir + '/initial/color/'
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(save_disp_dir, exist_ok=True)
    os.makedirs(save_ksh_dir, exist_ok=True)
    
    all_tem = 0
    
    allinx = []
    for i in range(len(frames)):
        
        frame = frames[i]
        rgb = img_loader(frame)
        rgb = cv2.resize(
                      rgb,
                      infer_size,
                      interpolation=cv2.INTER_CUBIC
                      )
        rgb = (rgb - __mean_dpt) / __std_dpt
        rgb = np.transpose(rgb, (2, 0, 1))
        rgb = torch.Tensor(np.ascontiguousarray(rgb).astype(np.float32)).unsqueeze(0)
        
        rgb = rgb.to(device)
        with torch.no_grad():
            outputs = dpt.forward(rgb)
            #print(torch.min(outputs),torch.mean(outputs),torch.max(outputs))
            
            if i>=1:
                
                outputs_flow = outputs.clone().to(device_flow)
                rgb_flow = rgb.clone().to(device_flow).squeeze(1)
                
                results_dict = model_flow(rgb_flow, previous_rgb,
                             attn_splits_list=args.attn_splits_list,
                             corr_radius_list=args.corr_radius_list,
                             prop_radius_list=args.prop_radius_list,
                             pred_bidir_flow=args.pred_bidir_flow,
                             )
                flow2to1 = results_dict['flow_preds'][-1]
                
                #save_vis_flow_tofile(flow2to1[0].permute(1, 2, 0).cpu().numpy(), './ahhh/'+str(i)+'.png')
                
                previous_rgb[:,0,:,:] = previous_rgb[:,0,:,:]*0.5 + 0.5
                previous_rgb[:,1,:,:] = previous_rgb[:,1,:,:]*0.5 + 0.5
                previous_rgb[:,2,:,:] = previous_rgb[:,2,:,:]*0.5 + 0.5
                previous_rgb = previous_rgb*255
                rgb_flow[:,0,:,:] = rgb_flow[:,0,:,:]*0.5 + 0.5
                rgb_flow[:,1,:,:] = rgb_flow[:,1,:,:]*0.5 + 0.5
                rgb_flow[:,2,:,:] = rgb_flow[:,2,:,:]*0.5 + 0.5
                rgb_flow = rgb_flow*255
                
                img1to2_seq = flow_warp(previous_rgb,flow2to1,mask=False)
                
                #print(torch.mean(rgb_flow),torch.mean(previous_rgb),torch.mean(img1to2_seq),'llllllllllllll')
        
                previous_outputs = previous_outputs.unsqueeze(1).to(device_flow)
                outputs1to2 = flow_warp(previous_outputs,flow2to1,mask=False)
                
                tem_loss = temloss(img1to2_seq,rgb_flow,outputs1to2,outputs_flow,device_flow)
                
                print(tem_loss.item())
                all_tem = all_tem + tem_loss.item()
            previous_outputs = outputs.clone().to(device_flow)
            previous_rgb = rgb.clone().to(device_flow).squeeze(1)
            
            plt.imsave(save_ksh_dir+str(i)+'.png',outputs.cpu().numpy().squeeze(), cmap='inferno',vmin =np.min(outputs.cpu().numpy().squeeze()) , vmax = np.max(outputs.cpu().numpy().squeeze()))
            
            outputs = outputs.cpu().numpy().squeeze()
            depth_min = outputs.min()
            depth_max = outputs.max()
            outputs = 65535.0 * (outputs - depth_min) / (depth_max - depth_min)
            cv2.imwrite(save_disp_dir+'/frame_%06d'%(i)+'.png', outputs.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
                
    
    
    print('**********initial**********')
    print('all:',all_tem,'mean:',all_tem/len(frames),'frames:',len(frames))
    allinx.append(all_tem/len(frames))
    dpt_index = all_tem/len(frames)
    
    with open(base_dir+'/result.txt','a') as f:
        f.write('**********initial**********'+'\n')
        f.write('all:'+str(all_tem)+',mean:'+str(all_tem/len(frames))+',frames:'+str(len(frames)))
    
    min_fwd = 100000
    min_bwd = 100000
    best_fwd = []
    best_bwd = []
    no_change_times_fwd = 0
    no_change_times_bwd = 0
    
    for times in range(args.timesall):
        '''
        if no_change_times_bwd>=2 and no_change_times_fwd>=2:
            break
        print('No Changing times:',no_change_times_fwd,no_change_times_bwd)
        
        if times==0:
            disp_dir = base_dir+'/initial/gray/'
        else:
            disp_dir = base_dir+'/'+str(times)+'/gray/'
        '''
        disp_dir = base_dir+'/initial/gray/'
        if times%2==0:
            print('*********forward:'+str(times+1)+'*********')
            frames = glob.glob(video_dir+'/left/*.png')
            frames.sort(key=lambda x: int(x[-10:-4]))
            current_disp = glob.glob(disp_dir+'*.png')
            current_disp.sort(key=lambda x: int(x[-10:-4]))
        else:
            print('*********backward:'+str(times+1)+'*********')
            frames = glob.glob(video_dir+'/left/*.png')
            frames.sort(key=lambda x: int(x[-10:-4]),reverse=True)
            current_disp = glob.glob(disp_dir+'*.png')
            current_disp.sort(key=lambda x: int(x[-10:-4]),reverse=True)
    
        
            
        save_disp_dir = base_dir+'/'+str(times+1)+'/gray/'
        save_ksh_dir = base_dir+'/'+str(times+1)+'/color/'
        os.makedirs(save_disp_dir, exist_ok=True)
        os.makedirs(save_ksh_dir, exist_ok=True) 
    
        all_tem = 0
        temp_result = []
        for i in range(len(frames)):

            frame = frames[i]
            
            rgb = img_loader(frame)
            rgb = cv2.resize(
                      rgb,
                      infer_size,
                      interpolation=cv2.INTER_CUBIC
                      )
            rgb = (rgb - __mean) / __std
            rgb = np.transpose(rgb, (2, 0, 1))
            rgb = torch.Tensor(np.ascontiguousarray(rgb).astype(np.float32)).unsqueeze(0)
            rgb = rgb.unsqueeze(0)
        
            if i<=2:
            
                print(frame,'repeat')
                #print(current_disp[i])
            
                img = img_loader(frame)
                img = cv2.resize(
                      img,
                      infer_size,
                      interpolation=cv2.INTER_CUBIC
                      )
                img = (img - __mean) / __std
                img = np.transpose(img, (2, 0, 1))
                img = torch.Tensor(np.ascontiguousarray(img).astype(np.float32))
                img = img.unsqueeze(0)
             
                depth = gt_png_loader(current_disp[i])
                depth = cv2.resize(
                        depth,
                        infer_size,
                        interpolation=cv2.INTER_NEAREST
                        )
                depth = torch.Tensor(np.ascontiguousarray(depth.astype(np.float32))).unsqueeze(0)
                depth = (depth-torch.min(depth))/(torch.max(depth)-torch.min(depth))
                
                depth = depth.unsqueeze(0)
                #print(img.shape,depth.shape)
                #exit(1)
            
                rgbd = torch.cat([img,depth],dim=1)
            
                for j in range(seq_len):
                    if j == 0:
                        ref_seq = rgbd
                    else:
                        ref_seq = torch.cat([ref_seq,rgbd],dim=0)
                ref_seq = ref_seq.unsqueeze(0)

            elif i>=3 and i<=8:
            #elif i>=3 and i<=14:
        
                print(frame)
            
                for j in range(seq_len-1,-1,-1):
                
                    key_idx = int(frame[-10:-4])
                    if times%2==0:
                        frame_idx = key_idx - j*1
                    else:
                        frame_idx = key_idx + j*1
                
                    rgb_dir = video_dir + '/left/' + 'frame_%06d.png'%(frame_idx)
                    gt_dir = disp_dir + 'frame_%06d.png'%(frame_idx)
                
                    img = img_loader(rgb_dir)
                    img = cv2.resize(
                      img,
                      infer_size,
                      interpolation=cv2.INTER_CUBIC
                      )
                    img = (img - __mean) / __std
                    img = np.transpose(img, (2, 0, 1))
                    img = torch.Tensor(np.ascontiguousarray(img).astype(np.float32))
                    img = img.unsqueeze(0)
                    #print(gt_dir,'hhhhhh')
                    depth = gt_png_loader(gt_dir)
                    depth = cv2.resize(
                        depth,
                        infer_size,
                        interpolation=cv2.INTER_NEAREST
                        )
                    depth = torch.Tensor(np.ascontiguousarray(depth.astype(np.float32))).unsqueeze(0)
                    depth = (depth-torch.min(depth))/(torch.max(depth)-torch.min(depth))
                    depth = depth.unsqueeze(0)
                
                    rgbd = torch.cat([img,depth],dim=1)

                    if j==seq_len-1:
                        ref_seq = rgbd
                    else:
                        ref_seq = torch.cat([ref_seq,rgbd],dim=0)
                    
                ref_seq = ref_seq.unsqueeze(0)
            
            elif i>=9:
            #elif i>=15:
        
                print(frame)
            
                for j in range(seq_len-1,-1,-1):
                
                    key_idx = int(frame[-10:-4])
                    if times%2==0:
                        frame_idx = key_idx - j*clip_step
                    else:
                        frame_idx = key_idx + j*clip_step
                
                    rgb_dir = video_dir + '/left/' + 'frame_%06d.png'%(frame_idx)
                    gt_dir = disp_dir + 'frame_%06d.png'%(frame_idx)
                    
                    img = img_loader(rgb_dir)
                    img = cv2.resize(
                      img,
                      infer_size,
                      interpolation=cv2.INTER_CUBIC
                      )
                    img = (img - __mean) / __std
                    img = np.transpose(img, (2, 0, 1))
                    img = torch.Tensor(np.ascontiguousarray(img).astype(np.float32))
                    img = img.unsqueeze(0)
                    #print(gt_dir,'hhhhhh')
                    depth = gt_png_loader(gt_dir)
                    depth = cv2.resize(
                        depth,
                        infer_size,
                        interpolation=cv2.INTER_NEAREST
                        )
                    depth = torch.Tensor(np.ascontiguousarray(depth.astype(np.float32))).unsqueeze(0)
                    depth = depth.unsqueeze(0)
                    depth = (depth-torch.min(depth))/(torch.max(depth)-torch.min(depth))
                    rgbd = torch.cat([img,depth],dim=1)

                    if j==seq_len-1:
                        ref_seq = rgbd
                    else:
                        ref_seq = torch.cat([ref_seq,rgbd],dim=0)
                    
                ref_seq = ref_seq.unsqueeze(0)
        
        
            ref_seq = ref_seq.to(device)
            rgb = rgb.to(device)
            with torch.no_grad():
                outputs = model(ref_seq)
                #print(torch.min(outputs),torch.mean(outputs),torch.max(outputs))
                outputs = outputs.squeeze(1)
                temp_result.append(outputs)
            
                if i>=1:
                
                    outputs_flow = outputs.clone().to(device_flow)
                    rgb_flow = rgb.clone().to(device_flow).squeeze(1)
                
                    results_dict = model_flow(rgb_flow, previous_rgb,
                             attn_splits_list=args.attn_splits_list,
                             corr_radius_list=args.corr_radius_list,
                             prop_radius_list=args.prop_radius_list,
                             pred_bidir_flow=args.pred_bidir_flow,
                             )
                    flow2to1 = results_dict['flow_preds'][-1]
                
                    #save_vis_flow_tofile(flow2to1[0].permute(1, 2, 0).cpu().numpy(), './ahhh/'+str(i)+'.png')
                
                    previous_rgb[:,0,:,:] = previous_rgb[:,0,:,:]*0.229 + 0.485
                    previous_rgb[:,1,:,:] = previous_rgb[:,1,:,:]*0.224 + 0.456
                    previous_rgb[:,2,:,:] = previous_rgb[:,2,:,:]*0.225 + 0.406
                    previous_rgb = previous_rgb*255
                    rgb_flow[:,0,:,:] = rgb_flow[:,0,:,:]*0.229 + 0.485
                    rgb_flow[:,1,:,:] = rgb_flow[:,1,:,:]*0.224 + 0.456
                    rgb_flow[:,2,:,:] = rgb_flow[:,2,:,:]*0.225 + 0.406
                    rgb_flow = rgb_flow*255
                
                    img1to2_seq = flow_warp(previous_rgb,flow2to1,mask=False)
                
                    #print(torch.mean(rgb_flow),torch.mean(previous_rgb),torch.mean(img1to2_seq),'llllllllllllll')
        
                    previous_outputs = previous_outputs.unsqueeze(1).to(device_flow)
                    outputs1to2 = flow_warp(previous_outputs,flow2to1,mask=False)
                
                    tem_loss = temloss(img1to2_seq,rgb_flow,outputs1to2,outputs_flow,device_flow)
                
                    print(tem_loss.item())
                    all_tem = all_tem + tem_loss.item()
                
            
            
                previous_outputs = outputs.clone().to(device_flow)
                previous_rgb = rgb.clone().to(device_flow).squeeze(1)
            
                if times%2==0:
                    plt.imsave(save_ksh_dir+str(i)+'.png',outputs.cpu().numpy().squeeze(), cmap='inferno',vmin =np.min(outputs.cpu().numpy().squeeze()) , vmax = np.max(outputs.cpu().numpy().squeeze()))
                else:
                    plt.imsave(save_ksh_dir+str(len(frames)-i-1)+'.png',outputs.cpu().numpy().squeeze(), cmap='inferno',vmin =np.min(outputs.cpu().numpy().squeeze()) , vmax = np.max(outputs.cpu().numpy().squeeze()))
            
                outputs = outputs.cpu().numpy().squeeze()
                depth_min = outputs.min()
                depth_max = outputs.max()
                outputs = 65535.0 * (outputs - depth_min) / (depth_max - depth_min)
                if times%2==0:
                    cv2.imwrite(save_disp_dir+'/frame_%06d'%(i)+'.png', outputs.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
                else:
                    cv2.imwrite(save_disp_dir+'/frame_%06d'%(len(frames)-i-1)+'.png', outputs.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
                
        if times%2==0:
            print('**********forward**********')
            print('all:',all_tem,'mean:',all_tem/len(frames),'frames:',len(frames))
            allinx.append(all_tem/len(frames))
            with open(base_dir+'/result.txt','a') as f:
                f.write('\n'+'**********forward**********'+'\n')
                f.write('all:'+str(all_tem)+',mean:'+str(all_tem/len(frames))+',frames:'+str(len(frames)))
            if min_fwd - all_tem/len(frames) >= 1e-3:
                
                min_fwd = all_tem/len(frames)
                best_fwd = temp_result
                no_change_times_fwd = 0
            else:
                no_change_times_fwd = no_change_times_fwd + 1
                
        else:
            print('**********backward**********')
            print('all:',all_tem,'mean:',all_tem/len(frames),'frames:',len(frames))
            allinx.append(all_tem/len(frames))
            with open(base_dir+'/result.txt','a') as f:
                f.write('\n'+'**********backward**********'+'\n')
                f.write('all:'+str(all_tem)+',mean:'+str(all_tem/len(frames))+',frames:'+str(len(frames)))
            if min_bwd - all_tem/len(frames) >= 1e-3:
                
                min_bwd = all_tem/len(frames)
                best_bwd = temp_result
                no_change_times_bwd = 0
            else:
                no_change_times_bwd = no_change_times_bwd + 1
    
   
    
    
    # Mixing  forward_results backward_results
    
    all_tem_mix = 0
    save_disp_dir = base_dir+'/mix/gray/'
    save_ksh_dir = base_dir+'/mix/color/'
    os.makedirs(save_disp_dir, exist_ok=True)
    os.makedirs(save_ksh_dir, exist_ok=True)
    frames = glob.glob(video_dir+'/left/*.png')
    frames.sort(key=lambda x: int(x[-10:-4]))
    best_bwd.reverse()
    
    for i in range(len(frames)):
        
        fwpred = best_fwd[i]
        bwpred = best_bwd[i]
        frame = frames[i]
        print('Mixing:',frame)
        
        rgb = img_loader(frame)
        rgb = cv2.resize(
                      rgb,
                      infer_size,
                      interpolation=cv2.INTER_CUBIC
                      )
        rgb = (rgb - __mean) / __std
        rgb = np.transpose(rgb, (2, 0, 1))
        rgb = torch.Tensor(np.ascontiguousarray(rgb).astype(np.float32)).unsqueeze(0)
        rgb = rgb.unsqueeze(0)
        
        if min_fwd<dpt_index and min_bwd<dpt_index:
            outputs = (fwpred+bwpred)/2
        else:
            if min_fwd<min_bwd:
                outputs = fwpred
            else:
                outputs = bwpred
        
        if i>=1:
                
            outputs_flow = outputs.clone().to(device_flow)
            rgb_flow = rgb.clone().to(device_flow).squeeze(1)
                
            results_dict = model_flow(rgb_flow, previous_rgb,
                             attn_splits_list=args.attn_splits_list,
                             corr_radius_list=args.corr_radius_list,
                             prop_radius_list=args.prop_radius_list,
                             pred_bidir_flow=args.pred_bidir_flow,
                             )
            flow2to1 = results_dict['flow_preds'][-1]
                
            previous_rgb[:,0,:,:] = previous_rgb[:,0,:,:]*0.229 + 0.485
            previous_rgb[:,1,:,:] = previous_rgb[:,1,:,:]*0.224 + 0.456
            previous_rgb[:,2,:,:] = previous_rgb[:,2,:,:]*0.225 + 0.406
            previous_rgb = previous_rgb*255
            rgb_flow[:,0,:,:] = rgb_flow[:,0,:,:]*0.229 + 0.485
            rgb_flow[:,1,:,:] = rgb_flow[:,1,:,:]*0.224 + 0.456
            rgb_flow[:,2,:,:] = rgb_flow[:,2,:,:]*0.225 + 0.406
            rgb_flow = rgb_flow*255
                
            img1to2_seq = flow_warp(previous_rgb,flow2to1,mask=False)
        
            previous_outputs = previous_outputs.unsqueeze(1).to(device_flow)
            outputs1to2 = flow_warp(previous_outputs,flow2to1,mask=False)
                
            tem_loss = temloss(img1to2_seq,rgb_flow,outputs1to2,outputs_flow,device_flow)
            
            print(tem_loss.item())
            all_tem_mix = all_tem_mix + tem_loss.item() 
                
            
            
        previous_outputs = outputs.clone().to(device_flow)
        previous_rgb = rgb.clone().to(device_flow).squeeze(1)
        plt.imsave(save_ksh_dir+str(i)+'.png',outputs.cpu().numpy().squeeze(), cmap='inferno',vmin =np.min(outputs.cpu().numpy().squeeze()) , vmax = np.max(outputs.cpu().numpy().squeeze()))   
        outputs = outputs.cpu().numpy().squeeze()
        depth_min = outputs.min()
        depth_max = outputs.max()
        outputs = 65535.0 * (outputs - depth_min) / (depth_max - depth_min)
        cv2.imwrite(save_disp_dir+'/frame_%06d'%(i)+'.png', outputs.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
    print('**********Mixing**********')
    print('all:',all_tem_mix,'mean:',all_tem_mix/len(frames),'frames:',len(frames))
    print(all_tem_mix/len(frames),min_fwd,min_bwd)
    #allinx.append(all_tem_mix/len(frames))
    with open(base_dir+'/result.txt','a') as f:
        f.write('\n'+'**********Mixing**********'+'\n')
        f.write('all:'+str(all_tem_mix)+',mean:'+str(all_tem_mix/len(frames))+',frames:'+str(len(frames)))
    
    '''
    plt.figure(figsize=(30,11))
    ax = plt.axes()
    ax.set_facecolor([234/255,234/255,241/255])
    ax.grid(linewidth=4,color='white')

    plt.tick_params(labelsize=40)
    plt.xlabel("Times",fontsize=45)
    plt.ylabel("OPW",fontsize=45)
    plt.plot(allinx,color= 'skyblue',linestyle='-',label='ST-CLSTM',linewidth=7,alpha = 0.8)
    plt.savefig(base_dir+'/curve.png')
    print('FINISHED PROCESSING')
    '''
    
    
    
    
