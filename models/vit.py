"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

import torch
import torch.nn as nn
from functools import partial

import utils
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
        attn_wo_soft = attn
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn, attn_wo_soft


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False,return_soft_attn=False):
        y, attn, attn_wo_soft = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_soft_attn:
            return attn
        if return_attention:
            return x, attn_wo_soft, attn
        return x



class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224],num_patches=392, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, is_student=False,**kwargs):
        super().__init__()
        self.is_student = is_student
        self.num_features = self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x):
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        B,n,dim = x.shape
        N = self.num_patches
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, N, dim).permute(0, 2, 1),
            size=n-1,
            mode='linear',
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 1)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, n, d = x.shape

        if n == 392 or n == 196:
            # add positional encoding to each token
            x = x + self.pos_embed
        else:
            x = x + self.interpolate_pos_encoding(x)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                x, attn_wo_soft, attn = blk(x, return_attention=True)
        x = self.norm(x)
        return x, attn_wo_soft[:, :, 1:, 1:].squeeze(-2).sum(1), attn[:, :, 0, 1:].sum(1).squeeze(-2)

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_soft_attn=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output




def vit_small(num_patches=16, **kwargs):
    model = VisionTransformer(
        num_patches=num_patches, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class TeacherPooling(nn.Module):
    def __init__(self,patch_size=16,embed_dim=384,out_dim=8192,drop_path_rate=0.1,norm_last_layer=True,**kwargs):
        super().__init__()
        teacher_392 = vit_small(patch_size=patch_size)
        # multi-crop wrapper handles forward with inputs of different resolutions
        self.teacher_392 = utils.MultiCropWrapper(
            teacher_392,
            DINOHead(embed_dim, out_dim, use_bn=False), 'teacher'
        )

        teacher_196 = vit_small(patch_size=patch_size*2)
        # multi-crop wrapper handles forward with inputs of different resolutions
        self.teacher_196 = utils.MultiCropWrapper(
            teacher_196,
            DINOHead(embed_dim, out_dim, use_bn=False), 'teacher'
        )

    def forward(self,x):
        teacher_392 = self.teacher_392(x)  # only the 2 global views pass through the teacher
        teacher_196 = self.teacher_196(teacher_392)
        return teacher_392, teacher_196

# class StudentPooling(nn.Module):
#     def __init__(self,patch_size=16,embed_dim=384,out_dim=8192,drop_path_rate=0.1,norm_last_layer=True,**kwargs):
#         super().__init__()
#         student_392 = vit_small(
#             patch_size=patch_size,
#             drop_path_rate=drop_path_rate,  # stochastic depth
#             is_student=True
#         )
#         # multi-crop wrapper handles forward with inputs of different resolutions
#         self.student_392 = utils.MultiCropWrapper(student_392, DINOHead(
#             embed_dim,
#             out_dim,
#             use_bn=False,
#             norm_last_layer=norm_last_layer,
#         ), 'student')
#
#         student_196 = vit_small(
#             patch_size=patch_size*2,
#             drop_path_rate=drop_path_rate,  # stochastic depth
#             is_student=True
#         )
#         # multi-crop wrapper handles forward with inputs of different resolutions
#         self.student_196 = utils.MultiCropWrapper(student_196, DINOHead(
#             embed_dim,
#             out_dim,
#             use_bn=False,
#             norm_last_layer=norm_last_layer,
#         ), 'student')
#
#     def forward(self,x):
#         student_392, student_tokens_392 = self.student_392(x)
#         student_



class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
