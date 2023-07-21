import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict
from mmseg.ops import resize
from builder import HEADS
from decode_head import BaseDecodeHead, BaseDecodeHead_clips, BaseDecodeHead_clips_flow
from mmseg.models.utils import *
import attr
from IPython import embed
from stabilization_attention import BasicLayer3d3
import cv2
from networks import *
import warnings
from mmcv.utils import Registry, build_from_cfg
from torch import nn


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


#@HEADS.register_module()


class Stabilization_Network_Cross_Attention(BaseDecodeHead_clips_flow):
    
    def __init__(self, feature_strides, **kwargs):
        super(Stabilization_Network_Cross_Attention, self).__init__(input_transform='multiple_select', **kwargs)
        self.training = False
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = nn.Sequential(nn.Conv2d(embedding_dim*4, embedding_dim, kernel_size=(1, 1), stride=(1, 1), bias=False),\
                                         nn.ReLU(inplace=True))
        

        depths = decoder_params['depths']
        # self.decoder_swin=BasicLayer_focal(
        #         dim=embedding_dim,
        #         depth=depths,
        #         num_heads=8,
        #         window_size=(2,7,7),
        #         mlp_ratio=4.,
        #         qkv_bias=True,
        #         qk_scale=None,
        #         drop=0.,
        #         attn_drop=0.,
        #         drop_path=0.,
        #         norm_layer=nn.LayerNorm,
        #         downsample=None,
        #         use_checkpoint=False)

        self.decoder_focal=BasicLayer3d3(dim=embedding_dim,
               input_resolution=(96,
                                 96),
               depth=depths,
               num_heads=8,
               window_size=7,
               mlp_ratio=4.,
               qkv_bias=True, 
               qk_scale=None,
               drop=0., 
               attn_drop=0.,
               drop_path=0.,
               norm_layer=nn.LayerNorm, 
               pool_method='fc',
               downsample=None,
               focal_level=2, 
               focal_window=5, 
               expand_size=3, 
               expand_layer="all",                           
               use_conv_embed=False,
               use_shift=False, 
               use_pre_norm=False, 
               use_checkpoint=False, 
               use_layerscale=False, 
               layerscale_value=1e-4,
               focal_l_clips=[7,4,2],
               focal_kernel_clips=[7,5,3])
        
        self.ffm2 = FFM(inchannels= 256, midchannels= 256, outchannels = 128)
        self.ffm1 = FFM(inchannels= 128, midchannels= 128, outchannels = 64)
        self.ffm0 = FFM(inchannels= 64, midchannels= 64, outchannels = 32,upfactor=1)
        self.AO = AO(32, outchannels=1, upfactor=1)

    def forward(self, inputs,edge_feat,edge_feat1, num_clips=None, imgs=None):#,infermode=1):
        if self.training:
            assert self.num_clips==num_clips
        
        
        
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        


        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        batch_size = n // num_clips
        

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        
        
        _, _, h, w=_c.shape
        _c_further=_c.reshape(batch_size, num_clips, -1, h, w)  #h2w2
       
        _c2=self.decoder_focal(_c_further)
        assert _c_further.shape==_c2.shape
        
        
        # skip and head
        outframe = self.ffm2(_c_further[:,-1,:,:,:],_c2[:,-1,:,:,:])
        outframe = self.ffm1(edge_feat1,outframe)
        outframe = self.ffm0(edge_feat,outframe)
        outframe = self.AO(outframe)

        return outframe

