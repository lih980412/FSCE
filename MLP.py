# # import torch
# #
# # class MLP(torch.nn.Module):
# #     def __init__(self):
# #         super(MLP, self).__init__()
# #         self.fc1 = torch.nn.Linear(784, 512)
# #         self.fc2 = torch.nn.Linear(512, 128)
# #         self.fc3 = torch.nn.Linear(128, 10)
# #
# #     def forward(self, din):
# #         din = din.view(-1, 28 * 28)
# #         dout = torch.nn.functional.relu(self.fc1(din))
# #         dout = torch.nn.functional.relu(self.fc2(dout))
# #         return torch.nn.functional.softmax(self.fc3(dout))
# #
# #
# # model = MLP().cuda()
#
#
# # import time
# #
# # def func2(args):  # multiple parameters (arguments)
# #     # x, y = args
# #     x = args[0]  # write in this way, easier to locate errors
# #     y = args[1]  # write in this way, easier to locate errors
# #
# #     # time.sleep(1)  # pretend it is a time-consuming operation
# #     return x - y
# #
# #
# # def run__pool():  # main process
# #     from multiprocessing import Pool
# #
# #     cpu_worker_num = 3
# #     process_args = [(3, 1), (2, 9), (4, 4), (4, 3), (5, 1)]
# #
# #     print(f'| inputs:  {process_args}')
# #     start_time = time.time()
# #     with Pool(cpu_worker_num) as p:
# #         outputs = p.map(func2, process_args)
# #     print(f'| outputs: {outputs}    TimeUsed: {time.time() - start_time:.1f}    \n')
# #
# #     '''Another way (I don't recommend)
# #     Using 'functions.partial'. See https://stackoverflow.com/a/25553970/9293137
# #     from functools import partial
# #     # from functools import partial
# #     # pool.map(partial(f, a, b), iterable)
# #     '''
# #
# # if __name__ =='__main__':
# #     run__pool()
#
# from numba import jit
#
# @jit
# def add1(x):
#     return x + 1
#
# @jit
# def bar(fn, x):
#     return fn(x)
#
# @jit
# def foo(x):
#     return bar(add1, x)
#
# # Passing add1 within numba compiled code.
# print(foo(1))
# # Passing add1 into bar from interpreted code
# print(bar(add1, 1))
#
# import numpy as np
# def bboxes_giou(boxes1, boxes2):
#     '''
#     cal GIOU of two boxes or batch boxes
#     such as: (1)
#             boxes1 = np.asarray([[0,0,5,5],[0,0,10,10],[15,15,25,25]])
#             boxes2 = np.asarray([[5,5,10,10]])
#             and res is [-0.49999988  0.25       -0.68749988]
#             (2)
#             boxes1 = np.asarray([[0,0,5,5],[0,0,10,10],[0,0,10,10]])
#             boxes2 = np.asarray([[0,0,5,5],[0,0,10,10],[0,0,10,10]])
#             and res is [1. 1. 1.]
#     :param boxes1:[xmin,ymin,xmax,ymax] or
#                 [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
#     :param boxes2:[xmin,ymin,xmax,ymax]
#     :return:
#     '''
#
#     # cal the box's area of boxes1 and boxess
#     boxes1Area = (boxes1[...,2]-boxes1[...,0])*(boxes1[...,3]-boxes1[...,1])
#     boxes2Area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
#
#     # ===========cal IOU=============#
#     #cal Intersection
#     left_up = np.maximum(boxes1[...,:2],boxes2[...,:2])
#     right_down = np.minimum(boxes1[...,2:],boxes2[...,2:])
#
#     inter_section = np.maximum(right_down-left_up,0.0)
#     inter_area = inter_section[...,0] * inter_section[...,1]
#     union_area = boxes1Area+boxes2Area-inter_area
#     ious = np.maximum(1.0*inter_area/union_area,np.finfo(np.float32).eps) # np.float32类型的最小正数
#
#     # ===========cal enclose area for GIOU=============#
#     enclose_left_up = np.minimum(boxes1[..., :2], boxes2[..., :2])
#     enclose_right_down = np.maximum(boxes1[..., 2:], boxes2[..., 2:])
#     enclose = np.maximum(enclose_right_down - enclose_left_up, 0.0)
#     enclose_area = enclose[..., 0] * enclose[..., 1]
#
#     # cal GIOU
#     gious = ious - 1.0 * (enclose_area - union_area) / enclose_area
#
#     return gious
#
# def bboxes_diou(boxes1,boxes2):
#     '''
#     cal DIOU of two boxes or batch boxes
#     :param boxes1:[xmin,ymin,xmax,ymax] or
#                 [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
#     :param boxes2:[xmin,ymin,xmax,ymax]
#     :return:
#     '''
#
#     #cal the box's area of boxes1 and boxess
#     boxes1Area = (boxes1[...,2]-boxes1[...,0])*(boxes1[...,3]-boxes1[...,1])
#     boxes2Area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
#
#     #cal Intersection
#     left_up = np.maximum(boxes1[...,:2],boxes2[...,:2])
#     right_down = np.minimum(boxes1[...,2:],boxes2[...,2:])
#
#     inter_section = np.maximum(right_down-left_up,0.0)
#     inter_area = inter_section[...,0] * inter_section[...,1]
#     union_area = boxes1Area+boxes2Area-inter_area
#     ious = np.maximum(1.0*inter_area/union_area,np.finfo(np.float32).eps)
#
#     #cal outer boxes
#     outer_left_up = np.minimum(boxes1[..., :2], boxes2[..., :2])
#     outer_right_down = np.maximum(boxes1[..., 2:], boxes2[..., 2:])
#     outer = np.maximum(outer_right_down - outer_left_up, 0.0)
#     outer_diagonal_line = np.square(outer[...,0]) + np.square(outer[...,1])
#
#     #cal center distance
#     boxes1_center = (boxes1[..., :2] +  boxes1[...,2:]) * 0.5
#     boxes2_center = (boxes2[..., :2] +  boxes2[...,2:]) * 0.5
#     center_dis = np.square(boxes1_center[...,0]-boxes2_center[...,0]) +\
#                  np.square(boxes1_center[...,1]-boxes2_center[...,1])
#
#     #cal diou
#     dious = ious - center_dis / outer_diagonal_line
#
#     return dious
#
# def pairwise_diou(boxes1, boxes2):
#     '''
#     cal DIOU of two boxes or batch boxes
#     :param boxes1:[xmin,ymin,xmax,ymax] or
#                 [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
#     :param boxes2:[xmin,ymin,xmax,ymax]
#     :return:
#     '''
#
#     #cal the box's area of boxes1 and boxess
#     # boxes1 = Boxes(boxes1)
#     # boxes2 = Boxes(boxes2)
#     # area1 = boxes1.area()
#     # area2 = boxes2.area()
#     # boxes1, boxes2 = boxes1.tensor, boxes2.tensor
#     area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
#     area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
#
#     # cal Intersection
#     lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
#     rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
#
#     wh = (rb - lt).clamp(min=0)  # [N,M,2]
#     inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
#     union_area = area1[:, None] + area2 - inter
#     iou = torch.where(
#         inter > 0,
#         inter / union_area,
#         torch.zeros(1, dtype=inter.dtype, device=inter.device),
#     )
#
#     #cal outer boxes
#     left_up = torch.min(boxes1[:, None, :2], boxes2[:, :2])
#     right_down = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
#     outer = (right_down - left_up).clamp(min=0)
#     outer_diagonal_line = torch.square(outer[..., 0]) + torch.square(outer[..., 1])
#
#     #cal center distance
#     boxes1_center = (boxes1[..., :2] +  boxes1[..., 2:]) * 0.5
#     boxes2_center = (boxes2[..., :2] +  boxes2[..., 2:]) * 0.5
#     center_dis = torch.square(boxes1_center[..., 0]-boxes2_center[..., 0]) +\
#                     torch.square(boxes1_center[..., 1]-boxes2_center[..., 1])
#
#     #cal diou
#     dious = iou - center_dis / outer_diagonal_line
#
#     dious_flatten = (1 - dious).flatten()
#     localization_loss = dious_flatten.mean() * dious.shape[0]
#
#     return localization_loss
#
#
# def bboxes_ciou(boxes1,boxes2):
#     '''
#     cal CIOU of two boxes or batch boxes
#     :param boxes1:[xmin,ymin,xmax,ymax] or
#                 [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
#     :param boxes2:[xmin,ymin,xmax,ymax]
#     :return:
#     '''
#
#     #cal the box's area of boxes1 and boxess
#     boxes1Area = (boxes1[...,2]-boxes1[...,0])*(boxes1[...,3]-boxes1[...,1])
#     boxes2Area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
#
#     # cal Intersection
#     left_up = np.maximum(boxes1[...,:2],boxes2[...,:2])
#     right_down = np.minimum(boxes1[...,2:],boxes2[...,2:])
#
#     inter_section = np.maximum(right_down-left_up,0.0)
#     inter_area = inter_section[...,0] * inter_section[...,1]
#     union_area = boxes1Area+boxes2Area-inter_area
#     ious = np.maximum(1.0*inter_area/union_area,np.finfo(np.float32).eps)
#
#     # cal outer boxes
#     outer_left_up = np.minimum(boxes1[..., :2], boxes2[..., :2])
#     outer_right_down = np.maximum(boxes1[..., 2:], boxes2[..., 2:])
#     outer = np.maximum(outer_right_down - outer_left_up, 0.0)
#     outer_diagonal_line = np.square(outer[...,0]) + np.square(outer[...,1])
#
#     # cal center distance
#     boxes1_center = (boxes1[..., :2] +  boxes1[...,2:]) * 0.5
#     boxes2_center = (boxes2[..., :2] +  boxes2[...,2:]) * 0.5
#     center_dis = np.square(boxes1_center[...,0]-boxes2_center[...,0]) +\
#                  np.square(boxes1_center[...,1]-boxes2_center[...,1])
#
#     # cal penalty term
#     # cal width,height
#     boxes1_size = np.maximum(boxes1[...,2:]-boxes1[...,:2],0.0)
#     boxes2_size = np.maximum(boxes2[..., 2:] - boxes2[..., :2], 0.0)
#     v = (4.0/np.square(np.pi)) * np.square((
#             np.arctan((boxes1_size[...,0]/boxes1_size[...,1])) -
#             np.arctan((boxes2_size[..., 0] / boxes2_size[..., 1])) ))
#     alpha = v / (1-ious+v)
#
#
#     #cal ciou
#     cious = ious - (center_dis / outer_diagonal_line + alpha*v)
#
#     return cious
#
#     def _get_ground_truth(self):
#         """
#         Returns:
#             gt_objectness_logits: list of N tensors. Tensor i is a vector whose length is the
#                 total number of anchors in image i (i.e., len(anchors[i])). Label values are
#                 in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative class; 1 = positive class.
#             gt_anchor_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), 4).
#         """
#         gt_objectness_logits = []
#         gt_anchor_deltas = []
#         # Concatenate anchors from all feature maps into a single Boxes per image
#         anchors = [Boxes.cat(anchors_i) for anchors_i in self.anchors]
#         for image_size_i, anchors_i, gt_boxes_i in zip(self.image_sizes, anchors, self.gt_boxes):
#             """
#             image_size_i: (h, w) for the i-th image
#             anchors_i: anchors for i-th image
#             gt_boxes_i: ground-truth boxes for i-th image
#             """
#             match_quality_matrix = pairwise_iou(gt_boxes_i, anchors_i)
#             # matched_idxs is the ground-truth index in [0, M)
#             # gt_objectness_logits_i is [0, -1, 1] indicating proposal is true positive, ignored or false positive
#             matched_idxs, gt_objectness_logits_i = self.anchor_matcher(match_quality_matrix)
#
#             if self.boundary_threshold >= 0:
#                 # Discard anchors that go out of the boundaries of the image
#                 # NOTE: This is legacy functionality that is turned off by default in Detectron2
#                 anchors_inside_image = anchors_i.inside_box(image_size_i, self.boundary_threshold)
#                 gt_objectness_logits_i[~anchors_inside_image] = -1
#
#             if len(gt_boxes_i) == 0:
#                 # These values won't be used anyway since the anchor is labeled as background
#                 gt_anchor_deltas_i = torch.zeros_like(anchors_i.tensor)
#             else:
#                 # TODO wasted computation for ignored boxes
#                 matched_gt_boxes = gt_boxes_i[matched_idxs]
#                 gt_anchor_deltas_i = self.box2box_transform.get_deltas(
#                     anchors_i.tensor, matched_gt_boxes.tensor
#                 )
#
#             gt_objectness_logits.append(gt_objectness_logits_i)
#             gt_anchor_deltas.append(gt_anchor_deltas_i)
#
#         return gt_objectness_logits, gt_anchor_deltas
#
#
#
#     def losses(self):
#         """
#         Return the losses from a set of RPN predictions and their associated ground-truth.
#
#         Returns:
#             dict[loss name -> loss value]: A dict mapping from loss name to loss value.
#                 Loss names are: `loss_rpn_cls` for objectness classification and
#                 `loss_rpn_loc` for proposal localization.
#         """
#
#         def resample(label):
#             """
#             Randomly sample a subset of positive and negative examples by overwriting
#             the label vector to the ignore value (-1) for all elements that are not
#             included in the sample.
#             """
#             pos_idx, neg_idx = subsample_labels(
#                 label, self.batch_size_per_image, self.positive_fraction, 0
#             )
#             # Fill with the ignore label (-1), then set positive and negative labels
#             label.fill_(-1)
#             label.scatter_(0, pos_idx, 1)
#             label.scatter_(0, neg_idx, 0)
#             return label
#
#         """
#         gt_objectness_logits: list of N tensors. Tensor i is a vector whose length is the
#             total number of anchors in image i (i.e., len(anchors[i]))
#         gt_anchor_deltas: list of N tensors. Tensor i has shape (len(anchors[i]), B),
#             where B is the box dimension
#         """
#         # 这一步是对未经筛选所有 anchors，找到对应的 gt （MxN，N 是非常大的数目，所有 p_level 合并的）
#         # gt_objectness_logits in [0, -1, 1]
#         gt_objectness_logits, gt_anchor_deltas = self._get_ground_truth()
#
#         '显示 RPN 中正负 anchor 思路'
#         # image_anchor_per_featuremap_deltas -> image_anchor_per_featuremap -> * self.anchor_generator.strides -> upsample back to orginal image
#         # self.visPosNegAnchor(self.strides, self.images, gt_objectness_logits)
#
#         # Collect all objectness labels and delta targets over feature maps and images
#         # The final ordering is L, N, H, W, A from slowest to fastest axis.
#         num_anchors_per_map = [np.prod(x.shape[1:]) for x in self.pred_objectness_logits]
#         num_anchors_per_image = sum(num_anchors_per_map)
#
#         # Stack to: (N, num_anchors_per_image), e.g., torch.Size([2, 247086])
#         gt_objectness_logits = torch.stack(
#             # resample +1/-1 to fraction 0.5, inplace modify other laberls to -1
#             # -1 will be ingored in loss calculation function
#             # NOTE: in VOC, not enough positive sample pairs, 12-24 out of 256 are positive.
#             # NOTE: 负样本是从 247086 里面随机抽出来 512 - pos 的
#             [resample(label) for label in gt_objectness_logits], dim=0
#         )
#
#         # Log the number of positive/negative anchors per-image that's used in training
#         num_pos_anchors = (gt_objectness_logits == 1).sum().item()
#         num_neg_anchors = (gt_objectness_logits == 0).sum().item()
#         storage = get_event_storage()
#         storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / self.num_images)
#         storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / self.num_images)
#
#         assert gt_objectness_logits.shape[1] == num_anchors_per_image
#         # Split to tuple of L tensors, each with shape (N, num_anchors_per_map)
#         gt_objectness_logits = torch.split(gt_objectness_logits, num_anchors_per_map, dim=1)
#         # Concat from all feature maps
#         gt_objectness_logits = cat([x.flatten() for x in gt_objectness_logits], dim=0)
#
#         # Stack to: (N, num_anchors_per_image, B)
#         gt_anchor_deltas = torch.stack(gt_anchor_deltas, dim=0)
#         assert gt_anchor_deltas.shape[1] == num_anchors_per_image
#         B = gt_anchor_deltas.shape[2]  # box dimension (4 or 5)
#
#         # Split to tuple of L tensors, each with shape (N, num_anchors_per_image)
#         gt_anchor_deltas = torch.split(gt_anchor_deltas, num_anchors_per_map, dim=1)
#         # Concat from all feature maps
#         gt_anchor_deltas = cat([x.reshape(-1, B) for x in gt_anchor_deltas], dim=0)
#
#         # Collect all objectness logits and delta predictions over feature maps
#         # and images to arrive at the same shape as the labels and targets
#         # The final ordering is L, N, H, W, A from slowest to fastest axis.
#         pred_objectness_logits = cat(
#             [
#                 # Reshape: (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N*Hi*Wi*A, )
#                 x.permute(0, 2, 3, 1).flatten()
#                 for x in self.pred_objectness_logits
#             ],
#             dim=0,
#         )
#         pred_anchor_deltas = cat(
#             [
#                 # Reshape: (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B)
#                 #          -> (N*Hi*Wi*A, B)
#                 x.view(x.shape[0], -1, B, x.shape[-2], x.shape[-1])
#                     .permute(0, 3, 4, 1, 2)
#                     .reshape(-1, B)
#                 for x in self.pred_anchor_deltas
#             ],
#             dim=0,
#         )
#         objectness_loss, localization_loss = rpn_losses(
#             gt_objectness_logits,
#             gt_anchor_deltas,
#             pred_objectness_logits,
#             pred_anchor_deltas,
#             self.smooth_l1_beta,
#             self.rpn_focal_alpha,
#             self.rpn_focal_gamma
#         )
#
#         normalizer = 1.0 / (self.batch_size_per_image * self.num_images)
#         loss_cls = objectness_loss * normalizer  # cls: classification loss
#         loss_loc = localization_loss * normalizer  # loc: localization loss
#         losses = {"loss_rpn_cls": loss_cls, "loss_rpn_loc": loss_loc}
#
#         return losses
#
#
#
# class GeneralizedRCNN(nn.Module):
#     """
#     Generalized R-CNN. Any models that contains the following three components:
#     1. Per-image feature extraction (aka backbone)
#     2. Region proposal generation
#     3. Per-region feature extraction and prediction
#     """
#
#     def __init__(self, cfg):
#         super().__init__()
#         # fmt on #
#         self.input_format  = cfg.INPUT.FORMAT
#         self.vis_period    = cfg.INPUT.VIS_PERIOD
#         self.moco          = cfg.MODEL.MOCO.ENABLED
#         # fmt off #
#
#         self.device = torch.device(cfg.MODEL.DEVICE)
#
#         self.backbone = build_backbone(cfg)
#         self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
#         self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())  # specify roi_heads name in yaml
#         if self.moco:
#             if self.roi_heads.__class__ == 'MoCoROIHeadsV1':
#                 self.roi_heads._moco_encoder_init(cfg)
#
#             elif self.roi_heads.__class__ == 'MoCoROIHeadsV3':
#                 self.backbone_k = build_backbone(cfg)
#                 self.proposal_generator_k = build_proposal_generator(cfg, self.backbone_k.output_shape())
#                 self.roi_heads_k = build_roi_heads(cfg, self.backbone_k.output_shape())
#
#                 self.roi_heads._moco_encoder_init(cfg,
#                     self.backbone_k, self.proposal_generator_k, self.roi_heads_k)
#         else:
#             assert 'MoCo' not in cfg.MODEL.ROI_HEADS.NAME
#
#         assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
#         num_channels = len(cfg.MODEL.PIXEL_MEAN)
#         pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
#         pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
#         self.normalizer = lambda x: (x - pixel_mean) / pixel_std
#
#         self.to(self.device)
#
#         # s1 = 0
#         # s2 = 0
#         # for model in [self.backbone.fpn_lateral2, self.backbone.fpn_lateral3, self.backbone.fpn_lateral4, self.backbone.fpn_lateral5]:
#         #     s1 += sum(p.numel() for p in model.parameters())
#         # for model in [self.backbone.fpn_output2, self.backbone.fpn_output3, self.backbone.fpn_output4, self.backbone.fpn_output5]:
#         #     s2 += sum(p.numel() for p in model.parameters())
#         # print('FPN',s1, s2)
#         if cfg.INPUT.USE_MIXUP:
#             self.use_mixup = True
#         else:
#             self.use_mixup = False
#
#         if cfg.MODEL.BACKBONE.FREEZE:
#             for p in self.backbone.parameters():
#                 p.requires_grad = False
#             print('froze backbone parameters')
#
#         if cfg.MODEL.BACKBONE.FREEZE_P5:
#             for connection in [self.backbone.fpn_lateral5, self.backbone.fpn_output5]:
#                 for p in connection.parameters():
#                     p.requires_grad = False
#             print('frozen P5 in FPN')
#
#
#         if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
#             for p in self.proposal_generator.parameters():
#                 p.requires_grad = False
#             print('froze proposal generator parameters')
#
#         if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
#             for p in self.roi_heads.box_head.parameters():
#                 p.requires_grad = False
#             print('froze roi_box_head parameters')
#
#             if cfg.MODEL.ROI_HEADS.UNFREEZE_FC2:
#                 for p in self.roi_heads.box_head.fc2.parameters():
#                     p.requires_grad = True
#                 print('unfreeze fc2 in roi head')
#
#             # we do not ever need to use this in our works.
#             if cfg.MODEL.ROI_HEADS.UNFREEZE_FC1:
#                 for p in self.roi_heads.box_head.fc1.parameters():
#                     p.requires_grad = True
#                 print('unfreeze fc1 in roi head')
#         print('-------- Using Roi Head: {}---------\n'.format(cfg.MODEL.ROI_HEADS.NAME))
#
#
#     def forward(self, batched_inputs, lambda_=None):
#         """
#         Args:
#             batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
#                 Each item in the list contains the inputs for one image.
#                 For now, each item in the list is a dict that contains:
#
#                 * image: Tensor, image in (C, H, W) format.
#                 * instances (optional): groundtruth :class:`Instances`
#                 * proposals (optional): :class:`Instances`, precomputed proposals.
#
#                 Other information that's included in the original dicts, such as:
#
#                 * "height", "width" (int): the output resolution of the model, used in inference.
#                     See :meth:`postprocess` for details.
#
#         Returns:
#             list[dict]:
#                 Each dict is the output for one input image.
#                 The dict contains one key "instances" whose value is a :class:`Instances`.
#                 The :class:`Instances` object has the following keys:
#                     "pred_boxes", "pred_classes", "scores"
#         """
#         if not self.training:
#             return self.inference(batched_inputs)
#
#         # backbone FPN
#         images = self.preprocess_image(batched_inputs)
#         features = self.backbone(images.tensor)  # List of L, FPN features
#
#         # RPN
#         if "instances" in batched_inputs[0]:
#             gt_instances = [x["instances"].to(self.device) for x in batched_inputs]  # List of N
#         else:
#             gt_instances = None
#
#         if self.proposal_generator:
#             proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
#             # proposals is the output of find_top_rpn_proposals(), i.e., post_nms_top_K
#         else:
#             assert "proposals" in batched_inputs[0]
#             proposals = [x["proposals"].to(self.device) for x in batched_inputs]
#             proposal_losses = {}
#
#         # tensorboard visualize, visualize top-20 RPN proposals with largest objectness
#         if self.vis_period > 0:
#             storage = get_event_storage()
#             if storage.iter % self.vis_period == 0:
#                 self.visualize_training(batched_inputs, proposals)
#
#         if self.moco and self.roi_heads.__class__ == 'MoCoROIHeadsV2':
#             self.roi_heads.gt_instances = gt_instances
#         # RoI
#         # ROI inputs are post_nms_top_k proposals.
#         # detector_losses includes Contrast Loss, 和业务层的 cls loss, and reg loss
#         _, detector_losses = self.roi_heads(images, features, proposals, gt_instances, self.use_mixup, lambda_)
#
#         losses = {}
#         losses.update(detector_losses)
#         losses.update(proposal_losses)
#         return losses
#
#     def visualize_training(self, batched_inputs, proposals):
#         """
#         A function used to visualize images and proposals. It shows ground truth
#         bounding boxes on the original image and up to 20 top-scoring predicted
#         object proposals on the original image. Users can implement different
#         visualization functions for different models.
#         Args:
#             batched_inputs (list): a list that contains input to the model.
#             proposals (list): a list that contains predicted proposals. Both
#                 batched_inputs and proposals should have the same length.
#         """
#         storage = get_event_storage()
#         max_vis_prop = 20
#
#         for input, prop in zip(batched_inputs, proposals):
#             img = input["image"]
#             img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
#             v_gt = Visualizer(img, None)
#             v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
#             anno_img = v_gt.get_image()
#             box_size = min(len(prop.proposal_boxes), max_vis_prop)
#             v_pred = Visualizer(img, None)
#             v_pred = v_pred.overlay_instances(
#                 boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
#             )
#             prop_img = v_pred.get_image()
#             vis_img = np.concatenate((anno_img, prop_img), axis=1)
#             vis_img = vis_img.transpose(2, 0, 1)
#             vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
#             storage.put_image(vis_name, vis_img)
#             break  # only visualize one image in a batch
#
#     def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
#         """
#         Run inference on the given inputs.
#
#         Args:
#             batched_inputs (list[dict]): same as in :meth:`forward`
#             detected_instances (None or list[Instances]): if not None, it
#                 contains an `Instances` object per image. The `Instances`
#                 object contains "pred_boxes" and "pred_classes" which are
#                 known boxes in the image.
#                 The inference will then skip the detection of bounding boxes,
#                 and only predict other per-ROI outputs.
#             do_postprocess (bool): whether to apply post-processing on the outputs.
#
#         Returns:
#             same as in :meth:`forward`.
#         """
#         assert not self.training
#
#         images = self.preprocess_image(batched_inputs)
#         features = self.backbone(images.tensor)
#
#         if detected_instances is None:
#             if self.proposal_generator:
#                 proposals, _ = self.proposal_generator(images, features, None)
#             else:
#                 assert "proposals" in batched_inputs[0]
#                 proposals = [x["proposals"].to(self.device) for x in batched_inputs]
#
#             results, _ = self.roi_heads(images, features, proposals, None)
#         else:
#             detected_instances = [x.to(self.device) for x in detected_instances]
#             results = self.roi_heads.forward_with_given_boxes(features, detected_instances)
#
#         if do_postprocess:
#             processed_results = []
#             for results_per_image, input_per_image, image_size in zip(
#                 results, batched_inputs, images.image_sizes
#             ):
#                 height = input_per_image.get("height", image_size[0])
#                 width = input_per_image.get("width", image_size[1])
#                 r = detector_postprocess(results_per_image, height, width)
#                 processed_results.append({"instances": r})
#             return processed_results
#         else:
#             return results
#
#     def preprocess_image(self, batched_inputs):
#         """
#         Normalize, pad and batch the input images.
#         """
#         images = [x["image"].to(self.device) for x in batched_inputs]
#         # images = [self.normalizer(x) for x in images]
#         images = ImageList.from_tensors(images, self.backbone.size_divisibility)
#         return images


# import numpy as np  # linear algebra
# import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
#
# import seaborn as sns
#
# sns.set()
# import matplotlib.pyplot as plt
# from PIL import Image
# # from scipy.misc import imread
# import os, cv2
#
#
# def fill_targets(row):
#     row.loc[row.species] = 1
#     return row
#
# def show_images(images_paths: list, titles=None):
#     fig = plt.figure(figsize=(12, len(images_paths)))
#     columns = 3
#     rows = len(images_paths)//columns
#     rows += 1 if len(images_paths) % columns else 0
#     for i, image_path in enumerate(images_paths):
#         img = cv2.imread(image_path)
#         if img is not None:
#             img = img[...,::-1]
#             fig.add_subplot(rows, columns, i+1)
#             if titles is None:
#                 plt.title(image_path[-15:])
#             else:
#                 plt.title(str(titles[i]))
#             plt.imshow(img)
#
#
# if __name__ == "__main__":
#     img_dir = r"F:\Dataset\Kaggle\train_images"
#     train_labels = pd.read_csv(r"F:\Dataset\Kaggle\train.csv")
#     train_labels['path'] = train_labels.image.apply(lambda x: os.path.join(img_dir, x))    # 添加path列，用于可视化
#
#     label_names = {}
#     for i in range(len(train_labels['species'].unique())):
#         label_names[i] = train_labels['species'].unique()[i]        # 按species统计数量
#
#     show_images(train_labels[train_labels['species'] == 'beluga'].path[:12])    # 可视化
#
#     reverse_train_labels = dict((v, k) for k, v in label_names.items())
#     for key in label_names.keys():
#         train_labels[label_names[key]] = 0
#
#     train_labels = train_labels.apply(fill_targets, axis=1)
#
#     target_counts = train_labels.drop(["image", "species", 'individual_id'], axis=1).sum(axis=0).sort_values(ascending=False)
#     sns.barplot(y=target_counts.index.values, x=target_counts.values, order=target_counts.index)
#     print(len(target_counts))
#     # plt.xticks(rotation=45)
#     # sns.countplot(train_labels.drop(["image", 'individual_id'], axis=1)["species"])
#     # plt.figure(figsize=(30, 15))
#     # plt.savefig('squares_plot.png')

# import timm
# if __name__ == "__main__":
#     model_list = timm.list_models()
#     model_pretrain_list = timm.list_models(pretrained=True)
#     print(len(model_list), len(model_pretrain_list))


#
# if __name__ == "__main__":
#     a = input()
#     b = input()
#     print(type(b))
#     # b = [int(b[i]) for i in range(len(b))]
#     # print(b)


# import matplotlib.pyplot as plt
# import numpy as np
# import torch
#
# # 设置中文
# plt.rcParams['font.sans-serif'] = ['SimHei']
#
# plt.figure(figsize=(20,8),dpi=80)
# a = np.random.random(2000)
# print(a)
# plt.hist(a, # 绘图数据
#         bins = 200, # 指定直方图的条形数为20个
#         color = 'steelblue', # 指定填充色
#         edgecolor = 'k', # 设置直方图边界颜色
#         label = '直方图'
#         )# 为直方图呈现标签
# # 刻度设置
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
#
# # 添加描述信息
# plt.xlabel('年龄：岁',fontsize=20)
# plt.ylabel('人数：个',fontsize=20)
# plt.title('乘客年龄分布',fontsize=20)
#
# # 显示图形
# plt.show()


# import os
# from PIL import Image
# folder_path = r'D:\UserD\Li\FSCE-1\datasets\TianC\image'
# extensions = []
# index=0
# import time
#
# # sub_folder_path = os.path.join(folder_path, fldr)
# for filee in os.listdir(folder_path):
#     file_path = os.path.join(folder_path, filee)
#     print('** Path: {}  **'.format(file_path), end="\r", flush=True)
#     print(file_path)
#     im = Image.open(file_path)
#     rgb_im = im.convert('RGB')
#     time.sleep(0.1)

if __name__ == "__main__":
    a = input()
    matrix = []

    matrix.append(list(map(int, input().split())))
    matrix.append(list(map(int, input().split())))
    matrix.append(list(map(int, input().split())))
    matrix.append(list(map(int, input().split())))
    # index = list(map(int, input().strip().split()))
    print(matrix)