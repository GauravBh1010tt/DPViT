# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mostly copy-paste from DINO and timm library:
https://github.com/facebookresearch/dino
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

import math
import torch
import torch.nn as nn
import pdb
from einops import rearrange, repeat, reduce
import torch.nn.functional as F
from torch import linalg as LA
from torch.autograd import Variable

from functools import partial
from utils import trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SimilarityMaps(nn.Module):
    def __init__(self, opt=None):
        super(SimilarityMaps, self).__init__()
        self.opt = opt

    def compute_dist(self, U, P):
        P = F.normalize(P, dim=-1)        
        UV_dist = U.mm(P.transpose(0, 1))
        return UV_dist

    def forward(self, x, P):
        nb, n_pat, e_dim = x.size()[0], x.size()[1], x.size()[2]
        U = rearrange(x, 'nb n_pat e_dim -> (nb n_pat) e_dim', e_dim=e_dim)
        assert len(U.shape) == 2
        U = F.normalize(U, dim=-1)
        dist_map = self.compute_dist(U, P)   
        return dist_map


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., 
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, init_values=0, args=None):
        super().__init__()

        self.args = args
        self.norm1 = norm_layer(dim)
        self.s_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        if args.use_parts:
            self.c_attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

            self.c_project = Mlp(in_features=args.K, out_features=dim, act_layer=act_layer, drop=drop)

            self.P = nn.Parameter(F.normalize(
                torch.nn.init.orthogonal_(torch.empty(args.K, dim)), 
                dim=-1))

            self.get_dmaps = SimilarityMaps(self.args)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            print('\n\n init gamma .. \n\n')
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, return_attention=False):
        y, attn = self.s_attn(self.norm1(x))

        if self.args.use_parts:
            dmap = self.get_dmaps(self.norm1(x), self.P).view(x.shape[0],x.shape[1],-1)
            z, c_attn = self.c_attn(self.c_project(dmap))
        else:
            dmap, c_attn = None, None

        if self.gamma_1 is None:
            if self.args.use_parts:
                x = x + self.drop_path(y) + self.drop_path(z)
            else:
                x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.args.use_parts:
                x = x + self.drop_path(self.gamma_1 * y) + self.drop_path(self.gamma_1 * z)
            else:
                x = x + self.drop_path(self.gamma_1 * y)

            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        if return_attention:
            return x, [attn,c_attn], dmap
        else:
            return x, None, dmap

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
            
    def forward(self, x):
        B, C, H, W = x.shape
        return self.proj(x)

class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), return_all_tokens=False, 
                 init_values=0, use_mean_pooling=False, masked_im_modeling=False, args=None):
        super().__init__()

        self.args = args

        if args.use_parts and args.lr_mix:
            self.alpha = nn.Parameter(torch.rand(1,args.num_fore))
            self.beta = nn.Parameter(torch.rand(1,args.K-args.num_fore))

        self.num_features = self.embed_dim = embed_dim
        self.return_all_tokens = return_all_tokens

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                init_values=init_values, args=args)
            for i in range(depth)])

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # masked image modeling
        self.masked_im_modeling = masked_im_modeling
        if masked_im_modeling:
            self.masked_embed = nn.Parameter(torch.zeros(1, embed_dim))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def get_last_crossattention(self, x_cls, x_patch):
        # x_patch = teacher_patch_features
        # x_cls = student_patch_features_aggregated
        x_cls_patch = torch.concat([x_cls.unsqueeze(1), x_patch], axis = 1) 
        blk = self.blocks[-1]
        x_cross, attn = blk(x_cls_patch, return_attention=True)
        return x_cross, attn

    def prepare_tokens(self, img, mask=None):
        B, nc, w, h = img.shape
        # patch linear embedding
        x_pat = self.patch_embed(img)
        # mask image modeling
        if mask is not None:
            x = self.mask_model(x_pat, mask)
        else:
            x = x_pat
        x = x.flatten(2).transpose(1, 2)

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)
  
    def ortho_loss_grahm(self):
        orth_loss = torch.tensor(0.0).cuda()
        param = self.blocks[-1].P
        param_flat = param.view(param.shape[0], -1)
        sym = torch.mm(param_flat, torch.t(param_flat))
        sym -= torch.eye(param_flat.shape[0]).cuda()
        orth_loss += sym.sum()
        #orth_loss = orth_loss.abs()
        orth_loss = (torch.norm(orth_loss,2))**2
        return orth_loss

    def SRIP_reg(self, W):

        row, col = W.shape[0], W.shape[1]
        w1 = W.view(-1,col)
        wt = torch.transpose(w1,0,1)

        if row > col:
            m = torch.matmul(wt,w1)
            indent = nn.Parameter(torch.eye(col,col))
        else:
            m = torch.matmul(w1,wt)
            indent = nn.Parameter(torch.eye(row,row))
        indent = indent.cuda()
        w_tmp = (m-indent)
        b_k = Variable(torch.rand(w_tmp.shape[1],1))
        b_k = b_k.cuda()

        v1 = torch.matmul(w_tmp, b_k)
        norm1 = torch.norm(v1,2)
        v2 = torch.div(v1,norm1)
        v3 = torch.matmul(w_tmp,v2)

        reg_o = (torch.norm(v3,2))**2

        return reg_o
        
    def ortho_loss(self):
        orth_loss = torch.tensor(0.0).cuda()
        param = self.blocks[-1].P
        fore_param, back_param = param[:,0:self.args.num_fore],  param[:,self.args.num_fore:]
        reg_fore = self.SRIP_reg(fore_param)
        reg_back = self.SRIP_reg(back_param)

        orth_loss += reg_fore + reg_back
        return orth_loss

    def compute_mix(self, dmap):

        n_patches, n_batch = dmap.shape[1] - 1, dmap.shape[0]   # removing the cls dmap
        fz = int((n_patches)**(1/2))
        resize = self.args.global_crops_size

        out = dmap.permute(0,2,1)[:,:,0:n_patches]
        out = out.view(n_batch,-1,fz,fz)

        if self.args.norm_mix:
            norm_alpha =  nn.functional.softmax(self.alpha, dim=-1)
            norm_beta =  nn.functional.softmax(self.beta, dim=-1)
        else:
            norm_alpha, norm_beta = self.alpha, self.beta

        fore_code = torch.sum(out[:,0:self.args.num_fore,:,:] * norm_alpha.view(1,-1,1,1),dim=1).unsqueeze(1)
        back_code = torch.sum(out[:,self.args.num_fore:,:,:] * norm_beta.view(1,-1,1,1),dim=1).unsqueeze(1)

        #nc_norm = LA.matrix_norm(fore_code+back_code, ord=1).mean()

        G_noise = torch.randn_like(fore_code)

        #pdb.set_trace()

        if self.args.lr_noise:
            fore_code = fore_code + self.args.lr_noise*G_noise
            back_code = back_code + self.args.lr_noise*G_noise

        fore_mix = F.interpolate(fore_code, size=(resize,resize), mode='bilinear', align_corners=False)
        back_mix = F.interpolate(back_code, size=(resize,resize), mode='bilinear', align_corners=False)
        
        fore_mix = fore_mix.repeat(1,3,1,1)
        back_mix = back_mix.repeat(1,3,1,1)

        return fore_mix, back_mix

    def forward(self, inp, return_all_tokens=None, mask=None, return_attention=False):
        # mim
        if self.masked_im_modeling:
            assert mask is not None
            x = self.prepare_tokens(inp, mask=mask)
        else:
            x = self.prepare_tokens(inp)

        #d_maps = []
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x, attn, dmap = blk(x)
            else:
                x, attn, dmap = blk(x, return_attention=return_attention)
            #d_maps.append(dmap)

        if self.args.use_parts and self.args.lr_mix:
            fore_mix, back_mix = self.compute_mix(dmap)
        else:
            fore_mix, back_mix = None, None, None

        x = self.norm(x)
        if self.fc_norm is not None:
            x[:, 0] = self.fc_norm(x[:, 1:, :].mean(1))
        
        return_all_tokens = self.return_all_tokens if \
            return_all_tokens is None else return_all_tokens
            
        attn = attn if return_attention else None
        x = x if return_all_tokens else x[:, 0]

        return x, attn, [fore_mix, back_mix, None]

        """
        if return_all_tokens and return_attention:
            return x, attn 
        elif return_all_tokens and not return_attention:
            return x, None
        elif not return_all_tokens and return_attention:
            return x[:, 0], attn
        else:
            return x[:, 0]
        """
        
    def get_last_selfattention(self, x):
        #print ('here')
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
                x = x[0] # return tokens 
            else:
                # return attention of the last block
                attns = blk(x, return_attention=True)[1]
                #pdb.set_trace()

                return attns # return attention matrix

    def get_mix(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
                x = x[0] # return tokens 
            else:
                # return dmap of the last block
                dmap_last = blk(x, return_attention=True)[-1]
                fore_mix, back_mix = self.compute_mix(dmap_last)
                return fore_mix, back_mix # return fore-back mix

    def get_intermediate_layers(self, x, n=1, return_attention=False):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        if return_attention:
            s_attns, c_attns = [], []
        for i, blk in enumerate(self.blocks):
            if len(self.blocks) - i <= n:
                x, attn, dmap = blk(x, return_attention=return_attention)
                output.append(self.norm(x))
                if return_attention:
                    s_attns.append(attn[0])
                    c_attns.append(attn[1])
            else:
                x, _, dmap = blk(x)
        if return_attention:
            return output, [s_attns, c_attns]
        return output, None
        
    def get_num_layers(self):
        return len(self.blocks)

    def mask_model(self, x, mask):
        x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed.to(x.dtype)
        return x

def vit_tiny(patch_size=16, args=None, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, args=args, **kwargs)
    return model

def vit_small(patch_size=16, args=None, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, args=args, **kwargs)
    return model

def vit_small_fixed_num_patches(num_patches=16, args=None, **kwargs):
    model = VisionTransformer(
        num_patches=num_patches, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), args=args, **kwargs)
    return model
    
def vit_base(patch_size=16, args=None, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, args=args, **kwargs)
    return model

def vit_large(patch_size=16, args=None, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, args=args, **kwargs)
    return model