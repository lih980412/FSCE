import shutil

from torch import nn
from fsdet.utils.GEM import GeM
from fsdet.layers.fuse_module import OrthogonalFusion
from fsdet.layers.swish import Swish
import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F
import torch
from sklearn.metrics import zero_one_loss
from fsdet.utils.events import get_event_storage


class query_support_module(nn.Module):
    """
    Region Proposal Network, introduced by the Faster R-CNN paper.
    """

    def __init__(self, dim_in, feat_dim, dim_out, which_p, which_pooling, output="", cross_entropy=False, temperature=0.5, iou_threshold=0.75):
        """

        :param which_p: choose which layer in the 'aux_features'
        :param which_pooling: use whay pooling layer
        """
        super().__init__()

        self.which_p = which_p
        # 消融一
        if which_pooling == "GeM":
            self.pooling = GeM()
            self.aux_pooling = GeM()
        elif which_pooling == "Avg":
            self.pooling = nn.AvgPool2d()
            self.aux_pooling = nn.AvgPool2d()
        elif which_pooling == "Max":
            self.pooling = nn.MaxPool2d()
            self.aux_pooling = nn.MaxPool2d()

        # 消融二
        self.fuse = OrthogonalFusion()

        self.head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.ReLU(inplace=True),
            # Swish(),
            nn.Linear(feat_dim, dim_out),
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)

        self.temperature = temperature

        self.softmax = nn.Softmax(dim=0)
        # self.loss = nn.MSELoss(reduction='sum')
        self.cross_entropy = cross_entropy
        self.iou_threshold = iou_threshold

        shutil.copy(r"D:/UserD/Li/FSCE-1/fsdet/modeling/query_support.py", output)

    def query_support_loss(self, aux_features_pooling, box_features_pooling, proposals, gt_instances_aux):
        aux_features_norm = self.head(aux_features_pooling)
        aux_features_norm = F.normalize(aux_features_norm, dim=1)
        box_features_norm = self.head(box_features_pooling)
        box_features_norm = F.normalize(box_features_norm, dim=1)


        # torch.cosine_similarity()
        similarity = torch.div(1 - torch.matmul(box_features_norm, aux_features_norm.T), self.temperature)
        # similarity = torch.matmul(box_features_norm, aux_features_norm.T)
        # for numerical stability
        # sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)         # 每个特征最不相似的特征
        # similarity = similarity - sim_row_max.detach()
        # similarity = torch.div(torch.matmul(box_features_norm, aux_features_norm.T), self.temperature)
        # # 1, for numerical stability
        # sim_row_max, sim_row_max_index = torch.max(similarity, dim=1, keepdim=True)  # 1024个 proposal 与第几个 aux 最像
        # gt_instances_aux = torch.tensor([i._fields['gt_classes'] for i in gt_instances_aux], device=similarity.device)
        # sim_row_max_pre = torch.gather(gt_instances_aux.unsqueeze(-1), dim=0, index=sim_row_max_index)  # 这两行找出最像的 aux 的类别
        #
        # sim_row_max_gt = torch.zeros([1, 1], device=similarity.device)
        # for gt_class in proposals:
        #     sim_row_max_gt = torch.cat([sim_row_max_gt, gt_class._fields['gt_classes'].unsqueeze(-1)], dim=0)
        # sim_row_max_gt = sim_row_max_gt[1:]
        #
        # mask = torch.ones(sim_row_max_gt.shape, device=sim_row_max_gt.device)
        # mask[torch.where(sim_row_max_gt == 3)] = 0  # support set 中没有 lop，不做惩罚
        # mask[torch.where(sim_row_max_gt == 5)] = 0  # bg 不做惩罚
        #
        # sim_row_max_pre = mask * sim_row_max_pre.float()
        # sim_row_max_gt *= mask
        # loss = self.loss(sim_row_max_pre, sim_row_max_gt)

        # 2
        # pre_row_max, pre_row_max_index = torch.max(similarity, dim=1, keepdim=True)

        gt_row_max_cls = torch.zeros([1, 1], device=similarity.device)
        iou_masks = torch.zeros([1, 1], device=similarity.device)
        for gt_class in proposals:
            gt_row_max_cls = torch.cat([gt_row_max_cls, gt_class._fields['gt_classes'].unsqueeze(-1)], dim=0)
            iou_masks = torch.cat([iou_masks, gt_class._fields['iou'].unsqueeze(-1)], dim=0)
        gt_row_max_cls = gt_row_max_cls[1:]
        iou_masks = iou_masks[1:]

        gt_instances_aux = torch.tensor([i._fields['gt_classes'] if len(i._fields['gt_classes']) == 1 else i._fields['gt_classes'][0] for i in gt_instances_aux ], device=similarity.device)

        mask = torch.zeros(similarity.shape, device=similarity.device)
        # mask_neg = torch.zeros(similarity.shape, device=similarity.device)
        # logprobs = F.log_softmax(similarity, dim=-1)


        list_index = 0
        feature_index = []
        count_use = 0
        for gt_query, iou_mask in zip(gt_row_max_cls, iou_masks):
            if gt_query[0] in gt_instances_aux and iou_mask >= self.iou_threshold:
                idx = torch.where(gt_instances_aux == gt_query[0])[0][0]
                mask[list_index][idx] = 1
                feature_index.append(int(idx))
                count_use += 1
            else:
                feature_index.append(5)
            # if gt_query[0] == 5:
                # idx = torch.where(gt_instances_aux == 5)[0][0]
                # mask_neg[list_index][idx] = 1
            list_index += 1

        # pre_row_max_cls = torch.gather(gt_instances_aux.unsqueeze(-1), dim=0, index=pre_row_max_index)

        # # gt到 index 的映射表
        # map_dict = []
        # for pre_cls, pre_index in zip(pre_row_max_cls, pre_row_max_index):
        #     map_dict.append(pre_index)

        # index = 0
        # for gt_query in gt_row_max_cls:
        #     if gt_query[0] in gt_instances_aux:
        #         mask[index][int(map_dict[index])] = 1
        #     index += 1

        # for pre_cls, gt_cls in zip(pre_row_max_cls, gt_row_max_cls):
        #     if pre_cls[0] == gt_cls[0]:
        #         # if pre_cls in map_dict:
        #         mask[index][int(map_dict[index])] = 1
        #     index += 1
        # exp_sim = torch.exp(similarity)
        # log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
        # per_label_log_prob = (log_prob * mask).sum(1) / mask.sum(1)
        # per_label_log_prob = (log_prob * mask).sum(1)
        storage = get_event_storage()
        storage.put_scalar("query_support_model/count_number1", count_use)
        if count_use == 0:
            return torch.tensor(0, device=mask.device, dtype=mask.dtype), feature_index

        if self.cross_entropy:
            similarity_log = -F.log_softmax(similarity / self.temperature, dim=1)
            loss = similarity_log * mask
            return loss.sum() / count_use, feature_index
        else:
            loss = similarity * mask
            return loss.sum(), feature_index

    def forward(self, box_features, aux_features, gt_instances, gt_instances_aux):
        """
        :param box_features:  [bs, 256, 7, 7]
        :param aux_features:  dict, "p2""p3""p4""p5""p6", [bs, 256, h, w]
        :return:
        """
        box_features_pooling = self.pooling(box_features)
        aux_features_pooling = self.aux_pooling(aux_features)  # [bs, 256]
        # fuse_features = self.fuse(aux_features_pooling, box_features_pooling, True)

        box_features_pooling = box_features_pooling.squeeze().squeeze()
        aux_features_pooling = aux_features_pooling.squeeze().squeeze()
        loss, mask = self.query_support_loss(aux_features_pooling, box_features_pooling, gt_instances, gt_instances_aux)

        # features = self.fuse(box_features, aux_features, mask)

        # return loss, features
        return loss, None



class query_support_module_no_gem(nn.Module):
    """
    Region Proposal Network, introduced by the Faster R-CNN paper.
    """

    def __init__(self, dim_in, feat_dim, dim_out, which_p, which_pooling, output="", cross_entropy=False, temperature=0.5, iou_threshold=0.75):
        """

        :param which_p: choose which layer in the 'aux_features'
        :param which_pooling: use whay pooling layer
        """
        super().__init__()

        self.conv1 = nn.Conv2d(7*7*256, 1024, kernel_size=1)
        self.conv2 = nn.Conv2d(7*7*256, 1024, kernel_size=1)


        self.head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.ReLU(inplace=True),
            # Swish(),
            nn.Linear(feat_dim, dim_out),
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)

        self.temperature = temperature

        self.softmax = nn.Softmax(dim=0)
        # self.loss = nn.MSELoss(reduction='sum')
        self.cross_entropy = cross_entropy
        self.iou_threshold = iou_threshold

        shutil.copy(r"D:/UserD/Li/FSCE-1/fsdet/modeling/query_support.py", output)

    def query_support_loss(self, aux_features_pooling, box_features_pooling, proposals, gt_instances_aux):
        # aux_features_norm = self.head(aux_features_pooling)
        # aux_features_norm = F.normalize(aux_features_norm, dim=1)
        # box_features_norm = self.head(box_features_pooling)
        # box_features_norm = F.normalize(box_features_norm, dim=1)


        # torch.cosine_similarity()
        similarity = torch.div(1 - torch.matmul(box_features_pooling, aux_features_pooling.T), self.temperature)
        # similarity = torch.div(1 - torch.matmul(box_features_norm, aux_features_norm.T), self.temperature)


        gt_row_max_cls = torch.zeros([1, 1], device=similarity.device)
        iou_masks = torch.zeros([1, 1], device=similarity.device)
        for gt_class in proposals:
            gt_row_max_cls = torch.cat([gt_row_max_cls, gt_class._fields['gt_classes'].unsqueeze(-1)], dim=0)
            iou_masks = torch.cat([iou_masks, gt_class._fields['iou'].unsqueeze(-1)], dim=0)
        gt_row_max_cls = gt_row_max_cls[1:]
        iou_masks = iou_masks[1:]

        gt_instances_aux = torch.tensor([i._fields['gt_classes'] if len(i._fields['gt_classes']) == 1 else i._fields['gt_classes'][0] for i in gt_instances_aux ], device=similarity.device)

        mask = torch.zeros(similarity.shape, device=similarity.device)
        # mask_neg = torch.zeros(similarity.shape, device=similarity.device)
        # logprobs = F.log_softmax(similarity, dim=-1)


        list_index = 0
        feature_index = []
        count_use = 0
        for gt_query, iou_mask in zip(gt_row_max_cls, iou_masks):
            if gt_query[0] in gt_instances_aux and iou_mask >= self.iou_threshold:
                idx = torch.where(gt_instances_aux == gt_query[0])[0][0]
                mask[list_index][idx] = 1
                feature_index.append(int(idx))
                count_use += 1
            else:
                feature_index.append(5)
            # if gt_query[0] == 5:
                # idx = torch.where(gt_instances_aux == 5)[0][0]
                # mask_neg[list_index][idx] = 1
            list_index += 1


        storage = get_event_storage()
        storage.put_scalar("query_support_model/count_number1", count_use)
        if count_use == 0:
            return torch.tensor(0, device=mask.device, dtype=mask.dtype), feature_index

        if self.cross_entropy:
            similarity_log = -F.log_softmax(similarity / self.temperature, dim=1)
            loss = similarity_log * mask
            return loss.sum() / count_use, feature_index
        else:
            loss = similarity * mask
            return loss.sum(), feature_index

    def forward(self, box_features, aux_features, gt_instances, gt_instances_aux):
        """
        :param box_features:  [bs, 256, 7, 7]
        :param aux_features:  dict, "p2""p3""p4""p5""p6", [bs, 256, h, w]
        :return:
        """
        box_features_flatten = box_features.flatten(1)
        box_features = self.conv1(box_features_flatten.reshape(box_features_flatten.shape[0], box_features_flatten.shape[1], 1, 1))
        aux_features_flatten = aux_features.flatten(1)
        aux_features = self.conv2(aux_features_flatten.reshape(aux_features_flatten.shape[0], aux_features_flatten.shape[1], 1, 1))

        # fuse_features = self.fuse(aux_features_pooling, box_features_pooling, True)

        loss, mask = self.query_support_loss(aux_features.reshape(aux_features.shape[0], aux_features.shape[1]), box_features.reshape(box_features.shape[0], box_features.shape[1]), gt_instances, gt_instances_aux)

        # features = self.fuse(box_features, aux_features, mask)

        # return loss, features
        return loss, None


class query_support_module_origin(nn.Module):
    """
    Region Proposal Network, introduced by the Faster R-CNN paper.
    """

    def __init__(self, dim_in, feat_dim, out_dim, which_p, which_pooling, output="", temperature=0.2):
        """

        :param which_p: choose which layer in the 'aux_features'
        :param which_pooling: use whay pooling layer
        """
        super().__init__()

        self.which_p = which_p
        # 消融一
        if which_pooling == "GeM":
            self.pooling = GeM()
            self.aux_pooling = GeM()
        elif which_pooling == "Avg":
            self.pooling = nn.AvgPool2d()
            self.aux_pooling = nn.AvgPool2d()
        elif which_pooling == "Max":
            self.pooling = nn.MaxPool2d()
            self.aux_pooling = nn.MaxPool2d()

        # 消融二
        self.fuse = OrthogonalFusion()

        self.head = nn.Sequential(
            nn.Linear(dim_in, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, out_dim),
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)

        self.temperature = temperature

        # self.loss = nn.MSELoss(reduction='sum')

        shutil.copy(r"D:/UserD/Li/FSCE-1/fsdet/modeling/query_support.py", output)

    def query_support_loss(self, aux_features_pooling, box_features_pooling, proposals, gt_instances_aux):
        aux_features_norm = self.head(aux_features_pooling)
        aux_features_norm = F.normalize(aux_features_norm, dim=1)
        box_features_norm = self.head(box_features_pooling)
        box_features_norm = F.normalize(box_features_norm, dim=1)


        # torch.cosine_similarity()
        similarity = torch.div(1 - torch.matmul(box_features_norm, aux_features_norm.T), self.temperature)
        # # for numerical stability
        # sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)         # 每个特征最不相似的特征
        # similarity = similarity - sim_row_max.detach()
        # similarity = torch.div(torch.matmul(box_features_norm, aux_features_norm.T), self.temperature)
        # # 1, for numerical stability
        # sim_row_max, sim_row_max_index = torch.max(similarity, dim=1, keepdim=True)  # 1024个 proposal 与第几个 aux 最像
        # gt_instances_aux = torch.tensor([i._fields['gt_classes'] for i in gt_instances_aux], device=similarity.device)
        # sim_row_max_pre = torch.gather(gt_instances_aux.unsqueeze(-1), dim=0, index=sim_row_max_index)  # 这两行找出最像的 aux 的类别
        #
        # sim_row_max_gt = torch.zeros([1, 1], device=similarity.device)
        # for gt_class in proposals:
        #     sim_row_max_gt = torch.cat([sim_row_max_gt, gt_class._fields['gt_classes'].unsqueeze(-1)], dim=0)
        # sim_row_max_gt = sim_row_max_gt[1:]
        #
        # mask = torch.ones(sim_row_max_gt.shape, device=sim_row_max_gt.device)
        # mask[torch.where(sim_row_max_gt == 3)] = 0  # support set 中没有 lop，不做惩罚
        # mask[torch.where(sim_row_max_gt == 5)] = 0  # bg 不做惩罚
        #
        # sim_row_max_pre = mask * sim_row_max_pre.float()
        # sim_row_max_gt *= mask
        # loss = self.loss(sim_row_max_pre, sim_row_max_gt)

        # 2
        # pre_row_max, pre_row_max_index = torch.max(similarity, dim=1, keepdim=True)

        gt_row_max_cls = torch.zeros([1, 1], device=similarity.device)
        for gt_class in proposals:
            gt_row_max_cls = torch.cat([gt_row_max_cls, gt_class._fields['gt_classes'].unsqueeze(-1)], dim=0)
        gt_row_max_cls = gt_row_max_cls[1:]

        gt_instances_aux = torch.tensor([i._fields['gt_classes'] if len(i._fields['gt_classes']) == 1 else i._fields['gt_classes'][0] for i in gt_instances_aux ], device=similarity.device)

        mask = torch.zeros(similarity.shape, device=similarity.device)
        # logprobs = F.log_softmax(similarity, dim=-1)

        list_index = 0
        feature_index = []
        count_use = 0
        for gt_query in gt_row_max_cls:
            if gt_query[0] in gt_instances_aux:
                idx = torch.where(gt_instances_aux == gt_query[0])[0][0]
                mask[list_index][idx] = 1
                feature_index.append(int(idx))
                count_use += 1
            else:
                feature_index.append(5)
            list_index += 1

        # pre_row_max_cls = torch.gather(gt_instances_aux.unsqueeze(-1), dim=0, index=pre_row_max_index)

        # # gt到 index 的映射表
        # map_dict = []
        # for pre_cls, pre_index in zip(pre_row_max_cls, pre_row_max_index):
        #     map_dict.append(pre_index)

        # index = 0
        # for gt_query in gt_row_max_cls:
        #     if gt_query[0] in gt_instances_aux:
        #         mask[index][int(map_dict[index])] = 1
        #     index += 1

        # for pre_cls, gt_cls in zip(pre_row_max_cls, gt_row_max_cls):
        #     if pre_cls[0] == gt_cls[0]:
        #         # if pre_cls in map_dict:
        #         mask[index][int(map_dict[index])] = 1
        #     index += 1
        # exp_sim = torch.exp(similarity)
        # log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))
        # per_label_log_prob = (log_prob * mask).sum(1) / mask.sum(1)
        # per_label_log_prob = (log_prob * mask).sum(1)

        loss = similarity * mask
        # loss = -per_label_log_prob
        storage = get_event_storage()
        storage.put_scalar("query_support_model/count_number1", count_use)
        if count_use == 0:
            return torch.tensor(0, device=loss.device, dtype=loss.dtype), feature_index
        return loss.sum(), feature_index

    def forward(self, box_features, aux_features, gt_instances, gt_instances_aux):
        """
        :param box_features:  [bs, 256, 7, 7]
        :param aux_features:  dict, "p2""p3""p4""p5""p6", [bs, 256, h, w]
        :return:
        """
        box_features_pooling = self.pooling(box_features)
        aux_features_pooling = self.aux_pooling(aux_features)  # [bs, 256]
        # fuse_features = self.fuse(aux_features_pooling, box_features_pooling, True)

        box_features_pooling = box_features_pooling.squeeze().squeeze()
        aux_features_pooling = aux_features_pooling.squeeze().squeeze()
        loss, mask = self.query_support_loss(aux_features_pooling, box_features_pooling, gt_instances, gt_instances_aux)

        # features = self.fuse(box_features, aux_features, mask)

        # return loss, features
        return loss, None