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
    def multi_graph_scaled_dot_product_attention_topk(self,xf_trans_plain ,zf_trans_plain,zf_g_plain,top_k) :
        num_heads=8
        #xf_trans_plain,76，625，256;zf_trans_plain,76,256,169
        #B, Nt, E = xf_trans_plain.shape
        #q = q / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        #attn = torch.bmm(q, k.transpose(-2, -1))
        #mutli-head
        zf_trans_plain=zf_trans_plain.permute(0, 2, 1)#76,169,256
        num_units_1=zf_trans_plain.shape[-1]
        num_units_2=xf_trans_plain.shape[-1]
        num_units_3=zf_g_plain.shape[-1]
        
        split_size_1 = num_units_1 // num_heads
        split_size_2 = num_units_2 // num_heads
        split_size_3= num_units_3 // num_heads
        zf_trans_plain = torch.stack(torch.split(zf_trans_plain, split_size_1, dim=2), dim=0)  # [8, 76, 169, 32][h, N, T_q, num_units_1/h]
        xf_trans_plain = torch.stack(torch.split(xf_trans_plain, split_size_2, dim=2), dim=0)  # [8, 76, 625, 32][h, N, T_q, num_units_2/h]
        zf_g_plain=torch.stack(torch.split(zf_g_plain,split_size_3,dim=2),dim=0)
        attn = torch.matmul(xf_trans_plain, zf_trans_plain.permute(0,1,3,2))#f(hs,ht)
        
        #print("attn shape: ",attn.shape) 76,625,169
        attn = F.softmax(attn, dim=-1)#aij
        
        
        if top_k == 0:
            pass
            #print('top_k equals 0')#attn = F.softmax(attn, dim=-1)
        else:
            attn_topk, indices = torch.topk(attn, k=top_k, dim=-1)
            #max_vals, _ = torch.max(attn_topk, dim=-1, keepdim=True)
            # attn_topk_exp = torch.exp(attn_topk - max_vals)
            # attn_topk_exp_sum = torch.sum(attn_topk_exp, dim=-1, keepdim=True)
            # attn_topk_softmax /= (attn_topk_exp_sum + 1e-6)
            #attn_topk_softmax = torch.softmax(attn_topk - max_vals, dim=-1)
            new_attn = torch.zeros_like(attn, dtype=attn.dtype, device=attn.device, requires_grad=True)
            new_attn = torch.scatter(new_attn, -1, indices, attn_topk)
            attn = new_attn
            #print("new attn shape: ",attn.shape) 76,625,169
        similar=attn
        embedding = torch.matmul(similar, zf_g_plain)#.permute(0, 2, 1)#sum_aijWvht
        embedding = torch.cat(torch.split(embedding, 1, dim=0), dim=3).squeeze(0)
        return  embedding.permute(0,2,1)
    def graph_scaled_dot_product_attention_topk(self,xf_trans_plain ,zf_trans_plain ,top_k) :
        #B, Nt, E = xf_trans_plain.shape
        #q = q / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        #attn = torch.bmm(q, k.transpose(-2, -1))
        #print ('z',zf_trans_plain.shape,'x',xf_trans_plain.shape)
        attn = torch.matmul(xf_trans_plain, zf_trans_plain)#f(hs,ht)
        
        #print("attn shape: ",attn.shape) 76,625,169
        attn = F.softmax(attn, dim=2)#aij
        if top_k == 0:
            attn = F.softmax(attn, dim=-1)
        else:
            attn_topk, indices = torch.topk(attn, k=top_k, dim=-1)
            #max_vals, _ = torch.max(attn_topk, dim=-1, keepdim=True)
            # attn_topk_exp = torch.exp(attn_topk - max_vals)
            # attn_topk_exp_sum = torch.sum(attn_topk_exp, dim=-1, keepdim=True)
            # attn_topk_softmax /= (attn_topk_exp_sum + 1e-6)
            #attn_topk_softmax = torch.softmax(attn_topk - max_vals, dim=-1)
            new_attn = torch.zeros_like(attn, dtype=attn.dtype, device=attn.device, requires_grad=True)
            new_attn = torch.scatter(new_attn, -1, indices, attn_topk)
            attn = new_attn
            #print("new attn shape: ",attn.shape) 76,625,169
        return  attn
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
        ##similar=self.graph_scaled_dot_product_attention_topk(xf_trans_plain,zf_trans_plain,128)#32
        #similar = torch.matmul(xf_trans_plain, zf_trans_plain)
        #similar = F.softmax(similar, dim=2)
        ###
        ###

        ###
        ###
        ##embedding = torch.matmul(similar, zf_g_plain).permute(0, 2, 1)
        embedding=self.multi_graph_scaled_dot_product_attention_topk(xf_trans_plain,zf_trans_plain,
                                                                     zf_g_plain,0)
        embedding = embedding.view(-1, shape_x[1], shape_x[2], shape_x[3])

        # aggregated feature
        
        output = torch.cat([embedding, xf_g], 1)
        
        output = self.fi(output)
        return output


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

        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)

    def template(self, z, roi):
        zf = self.backbone(z, roi)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)

        features = self.attention(self.zf, xf)

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
        zf = self.backbone(template, target_box)
        xf = self.backbone(search)

        features = self.attention(zf, xf)

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
def graph_scaled_dot_product_attention_topk(xf_trans_plain ,zf_trans_plain ,top_k) :
        #B, Nt, E = xf_trans_plain.shape
        #q = q / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        #attn = torch.bmm(q, k.transpose(-2, -1))
        attn = torch.matmul(xf_trans_plain, zf_trans_plain)#f(hs,ht)
        print("attn shape: ",attn.shape)
        attn = F.softmax(attn, dim=2)#aij
        if top_k is None:
            attn = F.softmax(attn, dim=-1)
        else:
            attn_topk, indices = torch.topk(attn, k=top_k, dim=-1)
            #max_vals, _ = torch.max(attn_topk, dim=-1, keepdim=True)
            # attn_topk_exp = torch.exp(attn_topk - max_vals)
            # attn_topk_exp_sum = torch.sum(attn_topk_exp, dim=-1, keepdim=True)
            # attn_topk_softmax /= (attn_topk_exp_sum + 1e-6)
            #attn_topk_softmax = torch.softmax(attn_topk - max_vals, dim=-1)
            new_attn = torch.zeros_like(attn, dtype=attn.dtype, device=attn.device, requires_grad=True)
            new_attn = torch.scatter(new_attn, -1, indices, attn_topk)
            attn = new_attn
            print("new attn shape: ",attn.shape)
        return  attn