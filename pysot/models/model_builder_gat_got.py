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
        self.support = nn.Conv2d(in_channel, in_channel, 1, 1, bias=False)

        # target template nodes linear transformation
        self.query = nn.Conv2d(in_channel, in_channel, 1, 1, bias=False)

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
        #similar其实就是q*k，可以直接传入topk-sparesTT吗？
        similar=graph_scaled_dot_product_attention_topk(xf_trans_plain,zf_trans_plain,32)
        #similar = torch.matmul(xf_trans_plain, zf_trans_plain)#f(hs,ht)
        #similar = F.softmax(similar, dim=2)#aij
        #change
        #
        #

        #
        #
        #
        embedding = torch.matmul(similar, zf_g_plain).permute(0, 2, 1)
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
#
#new function:topK_scatter
#
def _scaled_dot_product_attention_topk(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    top_k: int = 32,#None
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    r"""
    Computes scaled dot product attention on query, key and value tensors, using
    an optional attention mask if passed, and applying dropout if a probability
    greater than 0.0 is specified.
    Returns a tensor pair containing attended values and attention weights.
    Args:
        q, k, v: query, key and value tensors. See Shape section for shape details.
        attn_mask: optional tensor containing mask values to be added to calculated
            attention. May be 2D or 3D; see Shape section for details.
        dropout_p: dropout probability. If greater than 0.0, dropout is applied.
    Shape:
        - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
            and E is embedding dimension.
        - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
            and E is embedding dimension.
        - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
            shape :math:`(Nt, Ns)`.
        - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
            have shape :math:`(B, Nt, Ns)`
    """
    B, Nt, E = q.shape
    q = q / math.sqrt(E)
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2, -1))
    if attn_mask is not None:
        attn += attn_mask
    if top_k is None:
        attn = F.softmax(attn, dim=-1)
    else:
        assert isinstance(top_k, int), "type of top_k ({}) must be int.".format(type(top_k))
        assert top_k > 0, "top_k ({}) must be a positive integer.".format(top_k)
        attn_topk, indices = torch.topk(attn, k=top_k, dim=-1)
        max_vals, _ = torch.max(attn_topk, dim=-1, keepdim=True)
        # attn_topk_exp = torch.exp(attn_topk - max_vals)
        # attn_topk_exp_sum = torch.sum(attn_topk_exp, dim=-1, keepdim=True)
        # attn_topk_softmax /= (attn_topk_exp_sum + 1e-6)
        attn_topk_softmax = torch.softmax(attn_topk - max_vals, dim=-1)
        new_attn = torch.zeros_like(attn, dtype=attn.dtype, device=attn.device, requires_grad=True)
        new_attn = torch.scatter(new_attn, -1, indices, attn_topk_softmax)
        attn = new_attn

    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn
def graph_scaled_dot_product_attention_topk(xf_trans_plain ,zf_trans_plain ,top_k=32) :
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

def topk_myself(similiar , topk:int=32):
    print(similiar.shape)
    similiar_topk,indices=torch.topk(similiar,k=topk,dim=-1)
    max_vals,_=torch.max(similiar_topk,dim=-1,keepdim=True)
    similiar_topk_softmax=torch.softmax(similiar_topk-max_vals,dim=-1)
    new_similiar=torch.zeros_like(similiar,dtype=attn.dtype, device=attn.device, requires_grad=True)
    new_similiar=torch.scatter(new_similiar,-1,indices,similiar_topk_softmax)
    return new_similiar


