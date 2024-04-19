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
        self.support_0 = nn.Conv2d(in_channel, in_channel, 1, 1)
        self.support_1 = nn.Conv2d(in_channel, in_channel, 1, 1)
        self.support_2 = nn.Conv2d(in_channel, in_channel, 1, 1)
        self.support_3 = nn.Conv2d(in_channel, in_channel, 1, 1)
        # target template nodes linear transformation
        self.query_0 = nn.Conv2d(in_channel, in_channel, 1, 1)
        self.query_1 = nn.Conv2d(in_channel, in_channel, 1, 1)
        self.query_2 = nn.Conv2d(in_channel, in_channel, 1, 1)
        self.query_3 = nn.Conv2d(in_channel, in_channel, 1, 1)

        # linear transformation for message passing
        self.g_0 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )
        self.g_1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )
        self.g_2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )
        self.g_3 = nn.Sequential(
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
        
        #porcess catted tensor frm multihead 
        self.fc = nn.Linear(in_features=625*4, out_features=625, bias=False)
    def forward(self, zf, xf):
        # linear transformation
        xf_trans_0 = self.query_0(xf)
        xf_trans_1 = self.query_1(xf)
        xf_trans_2 = self.query_2(xf)
        xf_trans_3 = self.query_3(xf)
        zf_trans_0 = self.support_0(zf)
        zf_trans_1 = self.support_1(zf)
        zf_trans_2 = self.support_2(zf)
        zf_trans_3 = self.support_3(zf)

        # linear transformation for message passing
        xf_g_0 = self.g_0(xf)
        xf_g_1 = self.g_1(xf)
        xf_g_2 = self.g_2(xf)
        xf_g_3 = self.g_3(xf)
        xf_g=sum([xf_g_0,xf_g_1,xf_g_2,xf_g_3])/4
        zf_g_0 = self.g_0(zf)
        zf_g_1 = self.g_1(zf)
        zf_g_2 = self.g_2(zf)
        zf_g_3 = self.g_3(zf)
        # calculate similarity
        shape_x = xf_trans_0.shape
        shape_z = zf_trans_0.shape

        zf_trans_plain_0 = zf_trans_0.view(-1, shape_z[1], shape_z[2] * shape_z[3])
        zf_trans_plain_1 = zf_trans_0.view(-1, shape_z[1], shape_z[2] * shape_z[3])
        zf_trans_plain_2 = zf_trans_0.view(-1, shape_z[1], shape_z[2] * shape_z[3])
        zf_trans_plain_3 = zf_trans_0.view(-1, shape_z[1], shape_z[2] * shape_z[3])
        zf_g_plain_0 = zf_g_0.view(-1, shape_z[1], shape_z[2] * shape_z[3]).permute(0, 2, 1)
        zf_g_plain_1 = zf_g_1.view(-1, shape_z[1], shape_z[2] * shape_z[3]).permute(0, 2, 1)
        zf_g_plain_2 = zf_g_2.view(-1, shape_z[1], shape_z[2] * shape_z[3]).permute(0, 2, 1)
        zf_g_plain_3 = zf_g_3.view(-1, shape_z[1], shape_z[2] * shape_z[3]).permute(0, 2, 1)
        xf_trans_plain_0 = xf_trans_0.view(-1, shape_x[1], shape_x[2] * shape_x[3]).permute(0, 2, 1)
        xf_trans_plain_1 = xf_trans_1.view(-1, shape_x[1], shape_x[2] * shape_x[3]).permute(0, 2, 1)
        xf_trans_plain_2 = xf_trans_2.view(-1, shape_x[1], shape_x[2] * shape_x[3]).permute(0, 2, 1)
        xf_trans_plain_3 = xf_trans_3.view(-1, shape_x[1], shape_x[2] * shape_x[3]).permute(0, 2, 1)
        similar_0 = torch.matmul(xf_trans_plain_0, zf_trans_plain_0)
        similar_1 = torch.matmul(xf_trans_plain_1, zf_trans_plain_1)
        similar_2 = torch.matmul(xf_trans_plain_2, zf_trans_plain_2)
        similar_3 = torch.matmul(xf_trans_plain_3, zf_trans_plain_3)
        similar_0 = F.softmax(similar_0, dim=2)
        similar_1 = F.softmax(similar_1, dim=2)
        similar_2 = F.softmax(similar_2, dim=2)
        similar_3 = F.softmax(similar_3, dim=2)
        embedding_0 = torch.matmul(similar_0, zf_g_plain_0).permute(0, 2, 1)
        embedding_1 = torch.matmul(similar_1, zf_g_plain_1).permute(0, 2, 1)
        embedding_2 = torch.matmul(similar_2, zf_g_plain_2).permute(0, 2, 1)
        embedding_3 = torch.matmul(similar_3, zf_g_plain_3).permute(0, 2, 1)
        embedding_0 = embedding_0.view(-1, shape_x[1], shape_x[2], shape_x[3])
        embedding_1 = embedding_1.view(-1, shape_x[1], shape_x[2], shape_x[3])
        embedding_2 = embedding_2.view(-1, shape_x[1], shape_x[2], shape_x[3])
        embedding_3 = embedding_3.view(-1, shape_x[1], shape_x[2], shape_x[3])
        embedding=torch.cat([embedding_0,embedding_1,embedding_2,embedding_3],dim=-1)
        in_feature=embedding.shape[-1]
        out_feature=in_feature/4
        embedding=self.fc(embedding)##
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
