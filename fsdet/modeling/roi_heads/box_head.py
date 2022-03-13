# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch, math
from torch import nn
from torch.nn import functional as F

from fsdet.layers import Conv2d, ShapeSpec, get_norm
from fsdet.utils.registry import Registry
from fsdet.modeling.backbone.resnet import BasicBlock, BottleneckBlock

import copy
from fsdet.utils.GEM import GeM

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
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm = cfg.MODEL.ROI_BOX_HEAD.NORM
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
        self.nongt_dim = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE * cfg.SOLVER.IMS_PER_BATCH

        self.conv_pair_pos_fc1 = nn.Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.relation_fcq1 = nn.Linear(1024, 1024)
        self.relation_fck1 = nn.Linear(1024, 1024)
        self.linear_out_fc1 = nn.Conv2d(16384, 1024, kernel_size=(1, 1), groups=16)
        self.linear_aggregation1 = nn.Linear(3072, 1024)

        self.conv_pair_pos_fc2 = nn.Conv2d(64, 16, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.relation_fcq2 = nn.Linear(1024, 1024)
        self.relation_fck2 = nn.Linear(1024, 1024)
        self.linear_out_fc2 = nn.Conv2d(16384, 1024, kernel_size=(1, 1), groups=16)
        self.linear_aggregation2 = nn.Linear(3072, 1024)
        # self.conv_pair_pos_fc1 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1),
        #         stride=(1, 1), padding=(0, 0), device="cuda",)
        # weight_init.c2_msra_fill(self.conv_pair_pos_fc1)
        # self.relation_fc1 = nn.Sequential(
        #     # nn.Conv2d(),
        #     nn.Linear(self.nongt_dim, 16),
        #     nn.ReLU(inplace=True),
        #     # nn.Linear(64, 1024),
        #     # nn.ReLU(inplace=True),
        #     # nn.LeakyReLU(inplace=True),
        #     # nn.ELU(inplace=True),
        #     # nn.PReLU(num_parameters=1, init=0.25),
        # )

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
        assert index == 1 or index == 2, "there is something wrong with attention module"
        dim_group = (dim[0] / group, dim[1] / group, dim[2] / group)
        if index == 1:
            # [1, emb_dim, num_rois, nongt_dim]
            # position_feat_1, [1, fc_dim, num_rois, nongt_dim]
            position_feat_1 = self.conv_pair_pos_fc1(position_embedding)
            position_feat_1_rule = F.relu(position_feat_1, inplace=True)
            # aff_weight, [num_rois, fc_dim, nongt_dim, 1]
            aff_weight = position_feat_1_rule.permute(2, 1, 3, 0)
            # aff_weight, [num_rois, fc_dim, nongt_dim]
            aff_weight = aff_weight.squeeze(3)

            # multi head
            assert dim[0] == dim[1], 'Matrix multiply requires same dimensions!'
            q_data = self.relation_fcq1(roi_feat)
            q_data_batch = q_data.reshape(-1, group, int(dim_group[0])).permute(1, 0, 2)
            nongt_roi_feat = roi_feat
            k_data = self.relation_fck1(nongt_roi_feat)
            k_data_batch = k_data.reshape(-1, group, int(dim_group[1])).permute(1, 0, 2)

            v_data = nongt_roi_feat
            # v_data =  mx.symbol.FullyConnected(name='value_'+str(index)+'_'+str(gid), data=roi_feat, num_hidden=dim_group[2])
            aff = torch.bmm(q_data_batch, k_data_batch.permute(0, 2, 1))
            # aff_scale, [group, num_rois, nongt_dim]->[num_rois, group, nongt_dim]
            aff_scale = ((1.0 / math.sqrt(float(dim_group[1]))) * aff).permute(1, 0, 2)

            assert fc_dim == group, 'fc_dim != group'
            # weighted_aff, [num_rois, fc_dim, nongt_dim]
            weighted_aff = torch.log(torch.maximum(aff_weight, torch.tensor(1e-6))) + aff_scale
            aff_softmax = F.softmax(weighted_aff, dim=2)
            # [num_rois * fc_dim, nongt_dim]
            aff_softmax_reshape = aff_softmax.reshape(-1, aff_softmax.shape[2])
            # output_t, [num_rois * fc_dim, feat_dim]
            output_t = torch.matmul(aff_softmax_reshape, v_data)
            # output_t, [num_rois, fc_dim * feat_dim, 1, 1]
            output_t = output_t.reshape(-1, fc_dim * feat_dim, 1, 1)
            # linear_out, [num_rois, dim[2], 1, 1]
            linear_out = self.linear_out_fc1(output_t)
            output = linear_out.reshape(linear_out.shape[0], linear_out.shape[1])
            return output
        else:
            # [1, emb_dim, num_rois, nongt_dim]
            # position_feat_1, [1, fc_dim, num_rois, nongt_dim]
            position_feat_2 = self.conv_pair_pos_fc2(position_embedding)
            position_feat_2_rule = F.relu(position_feat_2, inplace=True)
            # aff_weight, [num_rois, fc_dim, nongt_dim, 1]
            aff_weight = position_feat_2_rule.permute(2, 1, 3, 0)
            # aff_weight, [num_rois, fc_dim, nongt_dim]
            aff_weight = aff_weight.squeeze(3)                                                          # wGmn

            # multi head
            assert dim[0] == dim[1], 'Matrix multiply requires same dimensions!'
            q_data = self.relation_fcq2(roi_feat)
            q_data_batch = q_data.reshape(-1, group, int(dim_group[0])).permute(1, 0, 2)
            nongt_roi_feat = roi_feat
            k_data = self.relation_fck2(nongt_roi_feat)
            k_data_batch = k_data.reshape(-1, group, int(dim_group[1])).permute(1, 0, 2)

            v_data = nongt_roi_feat
            # v_data =  mx.symbol.FullyConnected(name='value_'+str(index)+'_'+str(gid), data=roi_feat, num_hidden=dim_group[2])
            aff = torch.bmm(q_data_batch, k_data_batch.permute(0, 2, 1))
            # aff_scale, [group, num_rois, nongt_dim]->[num_rois, group, nongt_dim]
            aff_scale = ((1.0 / math.sqrt(float(dim_group[1]))) * aff).permute(1, 0, 2)                 # wAmn

            assert fc_dim == group, 'fc_dim != group'
            # weighted_aff, [num_rois, fc_dim, nongt_dim]
            weighted_aff = torch.log(torch.maximum(aff_weight, torch.tensor(1e-6))) + aff_scale
            aff_softmax = F.softmax(weighted_aff, dim=2)                                                # wmn
            # [num_rois * fc_dim, nongt_dim]
            aff_softmax_reshape = aff_softmax.reshape(-1, aff_softmax.shape[2])
            # output_t, [num_rois * fc_dim, feat_dim]
            output_t = torch.matmul(aff_softmax_reshape, v_data)
            # output_t, [num_rois, fc_dim * feat_dim, 1, 1]
            output_t = output_t.reshape(-1, fc_dim * feat_dim, 1, 1)
            # linear_out, [num_rois, dim[2], 1, 1]
            linear_out = self.linear_out_fc2(output_t)                                                  # fR
            output = linear_out.reshape(linear_out.shape[0], linear_out.shape[1])
            return output

    def forward(self, x, position_embedding):
        position_embedding_reshape = torch.unsqueeze(position_embedding.permute(2, 0, 1), dim=0)
        '''to do'''
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            index = 1
            for layer in self.fcs:
                # x = F.relu(layer(x))
                '''Relation mudole'''
                'common relation module'
                # x = layer(x)
                # attention = self.attention_module_multi_head(x, position_embedding_reshape,
                #                                                  nongt_dim=self.nongt_dim, fc_dim=16, feat_dim=1024,
                #                                                  index=index, group=16,
                #                                                  dim=(1024, 1024, 1024))
                # x = F.relu(x + attention)
                # index += 1
                'AGG relation module'
                x = layer(x)
                attention = self.attention_module_multi_head(x, position_embedding_reshape,
                                                             nongt_dim=self.nongt_dim, fc_dim=16, feat_dim=1024,
                                                             index=index, group=16,
                                                             dim=(1024, 1024, 1024))
                if index == 1:
                    x = self.linear_aggregation1(torch.cat([x, attention, x + attention], dim=1))
                    index += 1
                else:
                    x = self.linear_aggregation2(torch.cat([x, attention, x + attention], dim=1))
                x = F.relu(x)
        return x

    @property
    def output_size(self):
        return self._output_size


@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNGEMFCHead(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm = cfg.MODEL.ROI_BOX_HEAD.NORM
        # fmt: on
        assert num_conv + num_fc > 0

        self._output_size = (input_shape.channels, 1, 1)

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

        self.gem = GeM()

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            x = self.gem(x)
            x = x.flatten(start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
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
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm = cfg.MODEL.ROI_BOX_HEAD.NORM
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
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        sub_fc_dim = cfg.MODEL.ROI_BOX_HEAD.SUB_FC_DIM
        norm = cfg.MODEL.ROI_BOX_HEAD.NORM
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
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm = cfg.MODEL.ROI_BOX_HEAD.NORM
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

        loc_feat = F.adaptive_avg_pool2d(loc_feat, (1, 1))
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
