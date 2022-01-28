# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from fsdet.layers import Conv2d, ShapeSpec, get_norm
from fsdet.utils.registry import Registry
from fsdet.modeling.backbone.resnet import BasicBlock, BottleneckBlock

import copy

ROI_BOX_HEAD_REGISTRY = Registry("ROI_BOX_HEAD")
ROI_BOX_HEAD_REGISTRY.__doc__ = """
Registry for box heads, which make box predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""

@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNRelationFCHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        num_conv   = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim   = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim     = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm       = cfg.MODEL.ROI_BOX_HEAD.NORM
        # fmt: on
        assert num_conv + num_fc > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

        # 这里与 relation net 的代码里有所不同，relation net RPN 中 train 和 test 的 RPN_POST_NMS_TOP_N 都为300。
        # self.nongt_dim = cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN if self.training else cfg.MODEL.RPN.POST_NMS_TOPK_TEST
        # 这里与其保持一致
        self.nongt_dim = 300

    def attention_module_multi_head(self, roi_feat, position_embedding,
                                    nongt_dim, fc_dim, feat_dim,
                                    dim=(1024, 1024, 1024),
                                    group=16, index=1):
        """ Attetion module with vectorized version

        Args:
            roi_feat: [num_rois, feat_dim]
            position_embedding: [num_rois, nongt_dim, emb_dim]
            nongt_dim:
            fc_dim: should be same as group
            feat_dim: dimension of roi_feat, should be same as dim[2]
            dim: a 3-tuple of (query, key, output)
            group:
            index:

        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)
        nongt_roi_feat = mx.symbol.slice_axis(data=roi_feat, axis=0, begin=0, end=nongt_dim)
        # [num_rois * nongt_dim, emb_dim]
        position_embedding_reshape = mx.sym.Reshape(position_embedding, shape=(-3, -2))
        # position_feat_1, [num_rois * nongt_dim, fc_dim]
        position_feat_1 = mx.sym.FullyConnected(name='pair_pos_fc1_' + str(index),
                                                data=position_embedding_reshape,
                                                num_hidden=fc_dim)
        position_feat_1_relu = mx.sym.Activation(data=position_feat_1, act_type='relu')
        # aff_weight, [num_rois, nongt_dim, fc_dim]
        aff_weight = mx.sym.Reshape(position_feat_1_relu, shape=(-1, nongt_dim, fc_dim))
        # aff_weight, [num_rois, fc_dim, nongt_dim]
        aff_weight = mx.sym.transpose(aff_weight, axes=(0, 2, 1))

        # multi head
        assert dim[0] == dim[1], 'Matrix multiply requires same dimensions!'
        q_data = mx.sym.FullyConnected(name='query_' + str(index),
                                       data=roi_feat,
                                       num_hidden=dim[0])
        q_data_batch = mx.sym.Reshape(q_data, shape=(-1, group, dim_group[0]))
        q_data_batch = mx.sym.transpose(q_data_batch, axes=(1, 0, 2))
        k_data = mx.symbol.FullyConnected(name='key_' + str(index),
                                          data=nongt_roi_feat,
                                          num_hidden=dim[1])
        k_data_batch = mx.sym.Reshape(k_data, shape=(-1, group, dim_group[1]))
        k_data_batch = mx.sym.transpose(k_data_batch, axes=(1, 0, 2))
        v_data = nongt_roi_feat
        # v_data =  mx.symbol.FullyConnected(name='value_'+str(index)+'_'+str(gid), data=roi_feat, num_hidden=dim_group[2])
        aff = mx.symbol.batch_dot(lhs=q_data_batch, rhs=k_data_batch, transpose_a=False, transpose_b=True)
        # aff_scale, [group, num_rois, nongt_dim]
        aff_scale = (1.0 / math.sqrt(float(dim_group[1]))) * aff
        aff_scale = mx.sym.transpose(aff_scale, axes=(1, 0, 2))

        assert fc_dim == group, 'fc_dim != group'
        # weighted_aff, [num_rois, fc_dim, nongt_dim]
        weighted_aff = mx.sym.log(mx.sym.maximum(left=aff_weight, right=1e-6)) + aff_scale
        aff_softmax = mx.symbol.softmax(data=weighted_aff, axis=2, name='softmax_' + str(index))
        # [num_rois * fc_dim, nongt_dim]
        aff_softmax_reshape = mx.sym.Reshape(aff_softmax, shape=(-3, -2))
        # output_t, [num_rois * fc_dim, feat_dim]
        'matrix multiplication'
        output_t = mx.symbol.dot(lhs=aff_softmax_reshape, rhs=v_data)
        # output_t, [num_rois, fc_dim * feat_dim, 1, 1]
        output_t = mx.sym.Reshape(output_t, shape=(-1, fc_dim * feat_dim, 1, 1))
        # linear_out, [num_rois, dim[2], 1, 1]
        linear_out = mx.symbol.Convolution(name='linear_out_' + str(index), data=output_t,
                                           kernel=(1, 1), num_filter=dim[2], num_group=fc_dim)
        output = mx.sym.Reshape(linear_out, shape=(0, 0))
        return output


    def extract_position_embedding(position_mat, feat_dim, wave_length=1000):
        # position_mat, [num_rois, nongt_dim, 4]
        feat_range = mx.sym.arange(0, feat_dim / 8)
        dim_mat = mx.sym.broadcast_power(lhs=mx.sym.full((1,), wave_length),
                                         rhs=(8. / feat_dim) * feat_range)
        dim_mat = mx.sym.Reshape(dim_mat, shape=(1, 1, 1, -1))
        position_mat = mx.sym.expand_dims(100.0 * position_mat, axis=3)
        div_mat = mx.sym.broadcast_div(lhs=position_mat, rhs=dim_mat)
        sin_mat = mx.sym.sin(data=div_mat)
        cos_mat = mx.sym.cos(data=div_mat)
        # embedding, [num_rois, nongt_dim, 4, feat_dim/4]
        embedding = mx.sym.concat(sin_mat, cos_mat, dim=3)
        # embedding, [num_rois, nongt_dim, feat_dim]
        embedding = mx.sym.Reshape(embedding, shape=(0, 0, feat_dim))
        return embedding


    def extract_position_matrix(bbox, nongt_dim):
        """ Extract position matrix

        Args:
            bbox: [num_boxes, 4]

        Returns:
            position_matrix: [num_boxes, nongt_dim, 4]
        """
        xmin, ymin, xmax, ymax = torch.split(bbox, 1, 1)
        # [num_fg_classes, num_boxes, 1]
        bbox_width = xmax - xmin + 1.
        bbox_height = ymax - ymin + 1.
        center_x = 0.5 * (xmin + xmax)
        center_y = 0.5 * (ymin + ymax)
        # [num_fg_classes, num_boxes, num_boxes]
        delta_x = torch.sub(center_x, center_x.T)
        delta_x = torch.div(delta_x, bbox_width)
        delta_x = torch.log(torch.maximum(torch.abs(delta_x), 1e-3))
        # delta_x = torch.log(torch.clip(torch.abs(delta_x), min=1e-3))
        delta_y = torch.sub(center_y, center_y.T)
        delta_y = torch.div(delta_y, bbox_height)
        delta_y = torch.log(torch.maximum(torch.abs(delta_y), 1e-3))
        # delta_y = torch.log(torch.clip(torch.abs(delta_y), min=1e-3))
        delta_width = torch.div(bbox_width, bbox_width.T)
        delta_width = torch.log(delta_width)
        delta_height = torch.div(bbox_height, bbox_height.T)
        delta_height = torch.log(delta_height)
        concat_list = [delta_x, delta_y, delta_width, delta_height]

        for idx, sym in enumerate(concat_list):
            sym = mx.sym.slice_axis(sym, axis=1, begin=0, end=nongt_dim)
            concat_list[idx] = mx.sym.expand_dims(sym, axis=2)
        position_matrix = mx.sym.concat(*concat_list, dim=2)
        return position_matrix


    def forward(self, x):

        # [num_rois, nongt_dim, 4]
        position_matrix = self.extract_position_matrix(x, nongt_dim=self.nongt_dim)
        # [num_rois, nongt_dim, 64]
        position_embedding = self.extract_position_embedding(position_matrix, feat_dim=64)
        '''to do'''
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            index = 1

            for layer in self.fcs:
                x = layer(x)
                fc_new = copy.deepcopy(x)
                attention = self.attention_module_multi_head(fc_new, position_embedding,
                                                       nongt_dim=self.nongt_dim, fc_dim=16, feat_dim=1024,
                                                       index=index, group=16,
                                                       dim=(1024, 1024, 1024))
                x = F.relu(x + attention)
                index += 1
                # x = F.relu(layer(x))
        return x

    @property
    def output_size(self):
        return self._output_size

@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNConvFCHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        num_conv   = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim   = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim     = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm       = cfg.MODEL.ROI_BOX_HEAD.NORM
        # fmt: on
        assert num_conv + num_fc > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        return x

    @property
    def output_size(self):
        return self._output_size


@ROI_BOX_HEAD_REGISTRY.register()
class FastRcnnNovelHead(nn.Module):
#     """
#     A head that has separate 1024 fc for regression and classification branch
#     """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        fc_dim        = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        sub_fc_dim    = cfg.MODEL.ROI_BOX_HEAD.SUB_FC_DIM
        norm          = cfg.MODEL.ROI_BOX_HEAD.NORM
        # fmt: on
        box_feat_shape = (input_shape.channels, input_shape.height, input_shape.width)

        self.fc_main = nn.Linear(np.prod(box_feat_shape), fc_dim)
        self.fc_reg = nn.Linear(fc_dim, sub_fc_dim)
        self.fc_cls = nn.Linear(fc_dim, sub_fc_dim)

        self._output_size = sub_fc_dim

        for layer in [self.fc_main, self.fc_reg, self.fc_cls]:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        main_feat = F.relu(self.fc_main(x))
        loc_feat = F.relu(self.fc_reg(main_feat))
        cls_feat = F.relu(self.fc_cls(main_feat))
        return loc_feat, cls_feat

    @property
    def output_size(self):
        return self._output_size


@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNDoubleHead(nn.Module):
    """
    Double Head as described in https://arxiv.org/pdf/1904.06493.pdf
    The Conv Head composed of 1 (BasicBlock) + x (BottleneckBlock) and average pooling
    for bbox regression. From config: num_conv = 1 + x
    The FC Head composed of 2 fc layers for classification.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        num_conv   = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim   = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim     = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm       = cfg.MODEL.ROI_BOX_HEAD.NORM
        # fmt: on
        assert num_conv + num_fc > 0

        self.convs = []
        for k in range(num_conv):
            if k == 0:
                # import pdb; pdb.set_trace()
                conv = BasicBlock(input_shape.channels, conv_dim, norm=norm)
                # for name, param in conv.named_parameters():
                #     print(name, param.requires_grad)

                # bottleneck_channels = conv_dim // 4
                # conv = BottleneckBlock(input_shape.channels, conv_dim,
                #                        bottleneck_channels=bottleneck_channels, norm=norm)
                # import pdb; pdb.set_trace()
                # for name, param in conv.named_parameters():
                #     print(name, param)
            else:
                bottleneck_channels = conv_dim // 4
                conv = BottleneckBlock(conv_dim, conv_dim,
                                       bottleneck_channels=bottleneck_channels, norm=norm)
            self.add_module("conv{}".format(k + 1), conv)
            self.convs.append(conv)
        # this is a @property, see line 153, will be used as input_size for box_predictor
        # here when this function return, self._output_size = fc_dim (=1024)
        self._output_size = input_shape.channels * input_shape.height * input_shape.width
        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(self._output_size, fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        # init has already been done in BasicBlock and BottleneckBlock
        # for layer in self.conv_norm_relus:
        #     weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        loc_feat = x
        for layer in self.convs:
            loc_feat = layer(loc_feat)

        loc_feat = F.adaptive_avg_pool2d(loc_feat, (1,1))
        loc_feat = torch.flatten(loc_feat, start_dim=1)

        cls_feat = torch.flatten(x, start_dim=1)
        for layer in self.fcs:
            cls_feat = F.relu(layer(cls_feat))
        return loc_feat, cls_feat

    @property
    def output_size(self):
        return self._output_size


def build_box_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.NAME
    return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape)
