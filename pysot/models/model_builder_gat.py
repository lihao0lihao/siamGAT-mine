# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss_car import make_siamcar_loss_evaluator
from pysot.models.backbone import get_backbone
from pysot.models.head.car_head import CARHead
from ..utils.location_grid import compute_locations


class Graph_Attention_Union(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Graph_Attention_Union, self).__init__()

        # search region nodes linear transformation
        self.support = nn.Conv2d(in_channel, in_channel, 1, 1)

        # target template nodes linear transformation
        self.query = nn.Conv2d(in_channel, in_channel, 1, 1)

        # linear transformation for message passing
        self.g = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )

        # aggregated feature
        self.fi = nn.Sequential(
            nn.Conv2d(in_channel*2, out_channel, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, zf, xf):
        # linear transformation
        xf_trans = self.query(xf)
        zf_trans = self.support(zf)

        # linear transformation for message passing
        xf_g = self.g(xf)
        zf_g = self.g(zf)

        # calculate similarity
        shape_x = xf_trans.shape
        shape_z = zf_trans.shape

        zf_trans_plain = zf_trans.view(-1, shape_z[1], shape_z[2] * shape_z[3])
        zf_g_plain = zf_g.view(-1, shape_z[1], shape_z[2] * shape_z[3]).permute(0, 2, 1)
        xf_trans_plain = xf_trans.view(-1, shape_x[1], shape_x[2] * shape_x[3]).permute(0, 2, 1)

        similar = torch.matmul(xf_trans_plain, zf_trans_plain)
        similar = F.softmax(similar, dim=2)

        embedding = torch.matmul(similar, zf_g_plain).permute(0, 2, 1)
        embedding = embedding.view(-1, shape_x[1], shape_x[2], shape_x[3])

        # aggregated feature
        output = torch.cat([embedding, xf_g], 1)
        output = self.fi(output)
        return output


class Attention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build car head
        self.car_head = CARHead(cfg, 256)

        # build response map
        self.attention = Graph_Attention_Union(256, 256)
        self.attention_1 = Graph_Attention_Union(32, 32)
        self.attention_2 = Graph_Attention_Union(768,768)
        self.zf_new = None
        self.zf_0_new = None
        self.zf_1_new = None

        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)

        self.p0=nn.AdaptiveAvgPool2d(25)
        self.c0=nn.Conv2d(in_channels=32,out_channels=256,kernel_size=1,stride=1,padding=0)
        self.selfAttn=Attention(794)
        self.l1=nn.Linear(794,625)
    def template(self, z, roi):
        zf ,output= self.backbone(z, roi)
        a=output[0]
        d=output[3]
        self.zf = zf
        self.zf_0 = a
        self.zf_1 = d
    def get_template_feature(self,z,roi):
        zf,_ = self.backbone(z, roi)
        return zf
    def update_add_zf(self,zf_new):
        lamda=0.9
        self.zf=lamda*self.zf+(1-lamda)*zf_new
    def new_template(self,z,roi):
        zf ,output= self.backbone(z, roi)
        a=output[0]
        d=output[3]
        self.zf_new = zf
        self.zf_0_new = a
        self.zf_1_new = d
    def multi_scale(self,output):
        # p0=nn.AdaptiveAvgPool2d(25)
        # c0=nn.Conv2d(in_channels=output[0].shape[1],out_channels=256,kernel_size=1,stride=1,padding=0)
        # p1=nn.AdaptiveAvgPool2d(25)
        # c1=nn.Conv2d(in_channels=output[1].shape[1],out_channels=256,kernel_size=1,stride=1,padding=0)
        # p2=nn.AdaptiveAvgPool2d(25)
        # c2=nn.Conv2d(in_channels=output[2].shape[1],out_channels=256,kernel_size=1,stride=1,padding=0)
        # p3=nn.AdaptiveAvgPool2d(25)
        # c3=nn.Conv2d(in_channels=output[3].shape[1],out_channels=256,kernel_size=1,stride=1,padding=0)
        # p4=nn.AdaptiveAvgPool2d(25)
        # c4=nn.Conv2d(in_channels=output[4].shape[1],out_channels=256,kernel_size=1,stride=1,padding=0)
        # p5=nn.AdaptiveAvgPool2d(25)
        # c5=nn.Conv2d(in_channels=output[5].shape[1],out_channels=256,kernel_size=1,stride=1,padding=0)
        # 0 sss torch.Size([1, 32, 143, 143])
        # 1 sss torch.Size([1, 80, 70, 70])
        # 2 sss torch.Size([1, 288, 68, 68])
        # 3 sss torch.Size([1, 768, 33, 33])
        # 4 sss torch.Size([1, 768, 33, 33])
        # 5 sss torch.Size([1, 768, 33, 33])
        output[0]=self.c0(self.p0(output[0]))
        x=torch.zeros(1,256,25,25).cuda()
        x=x+output[0]
        # for o in output:
        #     x=x+o
        return x
    def track(self, x):
        xf,output= self.backbone(x)
        a=output[0]
        d=output[3]
     
        features = self.attention(self.zf, xf)
        features_1=self.attention_1(self.zf_0,a)
        features_2=self.attention_2(self.zf_1,d)
        if self.zf_new is not None:
            features_new = self.attention(self.zf_new, xf)
            features_1_new=self.attention_1(self.zf_0_new,a)
            features_2_new=self.attention_2(self.zf_1_new,d)
            features=self.fusion(feature=features,feature_1=features_1_new,feature_2=features_2_new)+0.01*features_new
            self.zf_new=None
            self.zf_0_new=None
            self.zf_1_new=None
        #print('shape',features.shape,features_1.shape,features_2.shape)
        #shape torch.Size([1, 32, 143, 143]) torch.Size([1, 768, 33, 33])
        #shape torch.Size([1, 256, 25, 25]) torch.Size([1, 32, 143, 143]) torch.Size([1, 768, 33, 33])
        features=self.fusion(feature=features,feature_1=features_1,feature_2=features_2)
        cls, loc, cen = self.car_head(features)
        return {
                'cls': cls,
                'loc': loc,
                'cen': cen
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls
    
    def fusion(self,feature,feature_1,feature_2):
        feature_1=feature_1.cuda()
        feature_2=feature_2.cuda()
        C,H,W=feature.shape[1],feature.shape[2],feature.shape[3]
        pool=nn.AdaptiveAvgPool2d((H,W))
        feature_1=pool(feature_1)
        feature_2=pool(feature_2)
        self.conv1=nn.Conv2d(in_channels=feature_1.shape[1],out_channels=C,kernel_size=1,stride=1,padding=0)
        self.conv2=nn.Conv2d(in_channels=feature_2.shape[1],out_channels=C,kernel_size=1,stride=1,padding=0)
        self.conv1=self.conv1.cuda()
        self.conv2=self.conv2.cuda()
        feature_1=self.conv1(feature_1)
        feature_2=self.conv2(feature_2)
        pool_1=nn.AdaptiveAvgPool2d(1)
        w_1=pool_1(feature_1)
        w_2=pool_1(feature_2)
        feature=0.9*feature+w_1*feature_1+w_2*feature_2
        return feature

    def getAttn(self,xf_features,zf_features):
        B,C,A,B=xf_features.shape
        xf_features=xf_features.view(xf_features.shape[0],xf_features.shape[1],xf_features.shape[2]*xf_features.shape[2])
        zf_features=zf_features.view(zf_features.shape[0],zf_features.shape[1],zf_features.shape[2]*zf_features.shape[2])
        mix=torch.cat([zf_features,xf_features],dim=2)
        
        #features shape is 1,256,25,25
        #attn=self.selfAttention(794)
        #x=torch.randn(20,32,256)
        output=self.selfAttn(mix)
        output=self.l1(output)
        output=output.view(xf_features.shape[0],xf_features.shape[1],A,B)
        # features=torch.randn(1,256,25,25)
        # output=output+features
        return output

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['bbox'].cuda()
        target_box = data['target_box'].cuda()
        neg = data['neg'].cuda()

        # get feature
        zf  , output_zf= self.backbone(template, target_box)
        zf_0=output_zf[0]
        zf_1=output_zf[3]
       
        xf , output  = self.backbone(search)
        a=output[0]
        d=output[3]
     
        features = self.attention(zf, xf)
        features_1=self.attention_1(zf_0,a)
        features_2=self.attention_2(zf_1,d)
        features=self.fusion(feature=features,feature_1=features_1,feature_2=features_2)
        
        #print('zf shape is ',zf.shape,'xf.shape is ',xf.shape,'feature shape ',features.shape)
        #zf shape is  torch.Size([32, 256, 13, 13]) xf.shape is  torch.Size([32, 256, 25, 25]) 
        #feature shape  torch.Size([32, 256, 25, 25])
        cls, loc, cen = self.car_head(features)
        locations = compute_locations(cls, cfg.TRACK.STRIDE, cfg.TRACK.OFFSET)
        cls = self.log_softmax(cls)
        cls_loss, loc_loss, cen_loss = self.loss_evaluator(
            locations,
            cls,
            loc,
            cen, label_cls, label_loc, neg
        )

        # get loss
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss
        return outputs
