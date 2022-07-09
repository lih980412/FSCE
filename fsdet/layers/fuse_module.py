import timm
from torch import nn
import torch
import math
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import numpy as np

import fvcore.nn.weight_init as weight_init
from fsdet.utils.GEM import GeM

class AttentionBlock(nn.Module):
    """
    ### Attention block
    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: torch.Tensor = None):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=1)
        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        #
        return res

class OrthogonalFusion(nn.Module):
    def __init__(self):
        super(OrthogonalFusion, self).__init__()
        self.gem = GeM()
        self.weight = nn.Conv2d(1024, 1, 1, padding=0)
        self.weight_support = nn.Conv2d(512, 256, 1, padding=0)
        # self.weight_support = nn.Sequential(
        #             nn.Linear(256, 128),
        #             nn.Linear(128, 1),
        # )
        # for layer in self.weight_support:
        #     if isinstance(layer, nn.Linear):
        #         weight_init.c2_xavier_fill(layer)

        self.sigmoid = nn.Sigmoid()

        # self.conv1 = nn.Conv2d(512, 256, 1, 1, 0)
        # self.relu = nn.ReLU()

        # self.gemQuery = GeM(name="Query")
        # self.gemSupport = GeM(name="Support")

    '2.1'
    # # [1024,256,7,7]
    # def forward(self, fl, fg):
    #
    #     fg = self.gem(fg).squeeze(-1).squeeze(-1)
    #
    #     bs, c, w, h = fl.shape
    #
    #     fl_dot_fg = torch.bmm(fg[:, None, :], fl.reshape(bs, c, -1))    #
    #     fl_dot_fg = fl_dot_fg.reshape(bs, 1, w, h)
    #     fg_norm = torch.norm(fg, dim=1)
    #
    #     fl_proj = (fl_dot_fg / fg_norm[:, None, None, None]) * fg[:, :, None, None]
    #     fl_orth = fl - fl_proj
    #
    #     f_fused = torch.cat([fl_orth, fg[:, :, None, None].repeat(1, 1, w, h)], dim=1)
    #     return f_fused

    '2.2'
    # def forward(self, support, query):
    #     # query_ = query.reshape(query.shape[0], query.shape[1], -1)      # 1024*256*49
    #     # support_ = support.reshape(support.shape[0], support.shape[1], -1)  # 1024*256*49
    #     # # simility = torch.bmm(query_, support_.permute(0, 2, 1))
    #     # simility = torch.bmm(F.normalize(query_, dim=1), F.normalize(support_, dim=1).permute(0, 2, 1))
    #     # (values, indices) = torch.topk(simility, 1, dim=2)
    #     # support_ = support_[:, indices[0].squeeze(1), :]
    #
    #     query_flatten = self.gemQuery(query).squeeze(-1).squeeze(-1)     # 1024 * 256
    #     support_flatten = self.gemSupport(support).squeeze(-1).squeeze(-1)
    #     # (256 * 1024) * (1024 * 256) 余弦相似度，计算与 query 特征最相关的 support 中的特征
    #     relevant_feature = F.normalize(query_flatten).t().mm(F.normalize(support_flatten))
    #     (values, indices) = torch.topk(relevant_feature, 1, dim=1)
    #     #  = torch.topk(F.normalize(fg).t().mm(F.normalize(fl_flatten)), 128, dim=0)
    #     # support_flatten = support_flatten[:, indices.squeeze(1).unique()]
    #     # most_valuable = query_flatten[:, indices.squeeze(1)]
    #
    #     bs, c, w, h = query.shape
    #     c = len(indices.squeeze(1).unique())
    #     fg = query_flatten[:, indices.squeeze(1).unique()]
    #     fl = support[:, indices.squeeze(1).unique(), :, :]
    #
    #     fl_dot_fg = torch.bmm(fg[:, None, :], fl.reshape(bs, c, -1))    #
    #     fl_dot_fg = fl_dot_fg.reshape(bs, 1, w, h)
    #     fg_norm = torch.norm(fg, dim=1)
    #
    #     fl_proj = (fl_dot_fg / fg_norm[:, None, None, None]) * fg[:, :, None, None]
    #     fl_orth = fl - fl_proj
    #
    #     f_fused = torch.cat([fl_orth, fg[:, :, None, None].repeat(1, 1, w, h)], dim=1)
    #     # query[:, indices.squeeze(1).unique(), :, :] = fl_orth
    #     repeat_time = 256 / f_fused.shape[1]
    #     if (repeat_time) % 1 == 0.0:                    # 如果能被整除
    #         f_fused = f_fused.repeat(1, int(repeat_time), 1, 1)
    #     else:                                           # 如果不能
    #         sub_time = 256 - int(repeat_time) * f_fused.shape[1]
    #         f_fused = f_fused.repeat(1, int(repeat_time), 1, 1)
    #         sub_fea = f_fused[:, :sub_time, :, :]
    #         f_fused = torch.cat([f_fused, sub_fea], dim=1)
    #     f_fused = torch.cat([f_fused, query_flatten[:, :, None, None].repeat(1, 1, w, h)], dim=1)
    #     # f_fused = torch.cat([f_fused, query], dim=1)
    #
    #     return f_fused




        # 如果256维太多
        # if c <= 16:
        #     repeat_time = 32 / f_fused.shape[1]
        #     if (repeat_time) % 1 == 0.0:                    # 如果能被整除
        #         f_fused = f_fused.repeat(1, int(repeat_time), 1, 1)
        #     else:                                           # 如果不能
        #         sub_time = 32 - int(repeat_time) * f_fused.shape[1]
        #         f_fused = f_fused.repeat(1, int(repeat_time), 1, 1)
        #         sub_fea = f_fused[:, :sub_time, :, :]
        #         f_fused = torch.cat([f_fused, sub_fea], dim=1)
        #     f_fused = torch.cat([f_fused, query_flatten[:, :, None, None].repeat(1, 1, w, h)], dim=1)
        # else:
        #     f_fused = torch.cat([f_fused[:, :32, :, :], query_flatten[:, :, None, None].repeat(1, 1, w, h)], dim=1)

    '2.3'
    # def forward(self, support, query):
    #     # query_ = query.reshape(query.shape[0], query.shape[1], -1)      # 1024*256*49
    #     # support_ = support.reshape(support.shape[0], support.shape[1], -1)  # 1024*256*49
    #     # # simility = torch.bmm(query_, support_.permute(0, 2, 1))
    #     # simility = torch.bmm(F.normalize(query_, dim=1), F.normalize(support_, dim=1).permute(0, 2, 1))
    #     # (values, indices) = torch.topk(simility, 1, dim=2)
    #     # support_ = support_[:, indices[0].squeeze(1), :]
    #     bs, c, w, h = query.shape
    #
    #     query_flatten = self.gemQuery(query).squeeze(-1).squeeze(-1)  # 1024 * 256
    #     support_flatten = self.gemSupport(support).squeeze(-1).squeeze(-1)
    #     # (256, 1024) * (1024, 256) 余弦相似度，计算与 query 特征最相关的 support 中的特征
    #     relevant_feature = F.normalize(query_flatten).t().mm(F.normalize(support_flatten))
    #     # similarity = torch.div(relevant_feature, 0.2)
    #
    #     # (values, indices) = torch.topk(relevant_feature, 1, dim=1)
    #     (values, indices) = torch.max(relevant_feature, dim=1, keepdim=True)
    #     similarity = relevant_feature - values.detach()
    #     # (256, 256)
    #     exp_sim = torch.exp(similarity)
    #
    #     # f_fused = query + support * exp_sim
    #     f_fused = support_flatten * (exp_sim.sum(1) / 256) + query_flatten
    #     f_fused = f_fused[:, :, None, None].repeat(1, 1, w, h)
    #
    #
    #
    #     # fg = query_flatten + values.detach()                                    # 1024 * 256
    #     # support_ = support.reshape(support.shape[0], support.shape[1], -1)      # 1024*256*49
    #     # #  = torch.topk(F.normalize(fg).t().mm(F.normalize(fl_flatten)), 128, dim=0)
    #     # # support_flatten = support_flatten[:, indices.squeeze(1).unique()]
    #     # # most_valuable = query_flatten[:, indices.squeeze(1)]
    #     #
    #     # bs, c, w, h = query.shape
    #     # c = len(indices.squeeze(1).unique())
    #     # fg = query_flatten[:, indices.squeeze(1).unique()]
    #     # fl = support[:, indices.squeeze(1).unique(), :, :]
    #     #
    #     # fl_dot_fg = torch.bmm(fg[:, None, :], fl.reshape(bs, c, -1))  #
    #     # fl_dot_fg = fl_dot_fg.reshape(bs, 1, w, h)
    #     # fg_norm = torch.norm(fg, dim=1)
    #     #
    #     # fl_proj = (fl_dot_fg / fg_norm[:, None, None, None]) * fg[:, :, None, None]
    #     # fl_orth = fl - fl_proj
    #     #
    #     # f_fused = torch.cat([fl_orth, fg[:, :, None, None].repeat(1, 1, w, h)], dim=1)
    #     # # query[:, indices.squeeze(1).unique(), :, :] = fl_orth
    #     # repeat_time = 256 / f_fused.shape[1]
    #     # if (repeat_time) % 1 == 0.0:  # 如果能被整除
    #     #     f_fused = f_fused.repeat(1, int(repeat_time), 1, 1)
    #     # else:  # 如果不能
    #     #     sub_time = 256 - int(repeat_time) * f_fused.shape[1]
    #     #     f_fused = f_fused.repeat(1, int(repeat_time), 1, 1)
    #     #     sub_fea = f_fused[:, :sub_time, :, :]
    #     #     f_fused = torch.cat([f_fused, sub_fea], dim=1)
    #     # f_fused = torch.cat([f_fused, query_flatten[:, :, None, None].repeat(1, 1, w, h)], dim=1)
    #
    #     return f_fused

    '2.6'
    # def forward(self, fl, fg):
    #
    #
    #     # fg = self.gem(fg)
    #     bs, c, w, h = fl.shape
    #
    #     fg = self.gem(fg).squeeze(-1).squeeze(-1).squeeze(0).repeat(bs, 1)
    #
    #
    #
    #
    #     fl_dot_fg = torch.bmm(fg[:, None, :], fl.reshape(bs, c, -1))    # [120, 1, 256] [120, 256, 49]
    #     fl_dot_fg = fl_dot_fg.reshape(bs, 1, w, h)  # [120, 1, 7, 7]
    #     fg_norm = torch.norm(fg, dim=1)
    #
    #     fl_proj = (fl_dot_fg / fg_norm[:, None, None, None]) * fg[:, :, None, None]
    #     fl_orth = fl - fl_proj
    #
    #     # f_fused = torch.cat([fl_orth, fg[:, :, None, None].repeat(1, 1, w, h)], dim=1)
    #     f_fused = torch.cat([fl_orth, fl], dim=1)
    #     # f_fused = self.relu(self.conv1(f_fused))
    #     return f_fused

    '2.8'
    def forward(self, fl, fg, mask=None):
        """
        :param fl:  [1024, 256, 7, 7] or [4, 256, 7, 7]
        :param fg:  [1024, 256, 7, 7] or [4, 256, h, w]
        :return:
        """
        fg_pooling = self.gem(fg)      # [4, 256, 1, 1]
        if mask is None:
            sigma = self.sigmoid(self.weight(fg_pooling.permute(1, 0, 2, 3)).flatten())
            return fl * (1 - sigma).unsqueeze(0).unsqueeze(2).unsqueeze(3) + fg * sigma.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        else:
            support_feature = torch.zeros(fl.shape, device=fg.device, dtype=fg.dtype)
            index = 0
            for mask_ in mask:
                if mask_ != 5:
                    # sigma = self.sigmoid(self.weight_support(torch.cat(fl[index], fg_pooling[mask_].unsqueeze(0))))
                    support_feature[index] = self.weight_support(torch.cat([fl[index].unsqueeze(0), fg[mask_].unsqueeze(0)], dim=1))
                    # support_feature[index] = fl[index] * sigma + fg_pooling[mask_].expand_as(fl[index]) * (1-sigma)
                else:
                    support_feature[index] = fl[index]
                index += 1
            # support_feature = support_feature[1:]

            # sigma = self.sigmoid(support_feature)
            # feature = (1-sigma) * fl + sigma * support_feature
            return support_feature


