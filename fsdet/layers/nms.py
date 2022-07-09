# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torchvision.ops import boxes as box_ops
from torchvision.ops import nms  # BC-compat
from torchvision.ops import box_iou


def soft_nms(boxes, scores, iou_threshold=0.7, soft_threshold=0.01, weight_method=2, sigma=0.5):
    """
    :param boxes: [N, 4]， 此处传进来的框，是经过筛选（选取的得分TopK）之后的
    :param scores: [N]
    :param iou_threshold: 0.7
    :param soft_threshold soft nms 过滤掉得分太低的框 （手动设置）
    :param weight_method 权重方法 1. 线性 2. 高斯
    :return:
    """
    keep = []
    idxs = scores.argsort()
    while idxs.numel() > 0:  # 循环直到null； numel()： 数组元素个数
        # 由于scores得分会改变，所以每次都要重新排序，获取得分最大值
        idxs = scores.argsort()  # 评分排序
        if idxs.size(0) == 1:  # 就剩余一个框了；
            keep.append(idxs[-1])
            break
        keep_len = len(keep)
        max_score_index = idxs[-(keep_len + 1)]
        max_score_box = boxes[max_score_index][None, :]  # [1, 4]
        idxs = idxs[:-(keep_len + 1)]
        other_boxes = boxes[idxs]  # [?, 4]
        keep.append(max_score_index)  # 位置不能边
        ious = box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
        # Soft NMS 处理， 和 得分最大框 IOU大于阈值的框， 进行得分抑制
        if weight_method == 1:   # 线性抑制  # 整个过程 只修改分数
            ge_threshod_bool = ious[0] >= iou_threshold
            ge_threshod_idxs = idxs[ge_threshod_bool]
            scores[ge_threshod_idxs] *= (1. - ious[0][ge_threshod_bool])  # 小于IoU阈值的不变
            # idxs = idxs[scores[idxs] >= soft_threshold]  # 小于soft_threshold删除， 经过抑制后 阈值会越来越小；
        elif weight_method == 2:  # 高斯抑制， 不管大不大于阈值，都计算权重
            scores[idxs] *= torch.exp(-(ious[0] * ious[0]) / sigma) # 权重(0, 1]
            # idxs = idxs[scores[idxs] >= soft_threshold]
        # else:  # NMS
        #     idxs = idxs[ious[0] <= iou_threshold]

    # keep = scores[scores > soft_threshold].int()
    keep = idxs.new(keep)  # Tensor
    keep = keep[scores[keep] > soft_threshold]  # 最后处理阈值
    return keep
    # boxes = boxes[keep]  # 保留下来的框
    # scores = scores[keep]  # soft nms抑制后得分
    # return boxes, scores


def batched_nms(boxes, scores, idxs, iou_threshold, soft=False):
    """
    Same as torchvision.ops.boxes.batched_nms, but safer.
    """
    assert boxes.shape[-1] == 4
    # TODO may need better strategy.
    # Investigate after having a fully-cuda NMS op.
    if len(boxes) < 40000:
        return box_ops.batched_nms(boxes, scores, idxs, iou_threshold)

    result_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
    for id in torch.unique(idxs).cpu().tolist():
        mask = (idxs == id).nonzero().view(-1)
        if soft:
            keep = soft_nms(boxes[mask], scores[mask], iou_threshold)
        else:
            keep = nms(boxes[mask], scores[mask], iou_threshold)
        result_mask[mask[keep]] = True
    keep = result_mask.nonzero().view(-1)
    keep = keep[scores[keep].argsort(descending=True)]
    return keep


# Note: this function (nms_rotated) might be moved into
# torchvision/ops/boxes.py in the future
def nms_rotated(boxes, scores, iou_threshold):
    """
    Performs non-maximum suppression (NMS) on the rotated boxes according
    to their intersection-over-union (IoU).

    Rotated NMS iteratively removes lower scoring rotated boxes which have an
    IoU greater than iou_threshold with another (higher scoring) rotated box.

    Note that RotatedBox (5, 3, 4, 2, -90) covers exactly the same region as
    RotatedBox (5, 3, 4, 2, 90) does, and their IoU will be 1. However, they
    can be representing completely different objects in certain tasks, e.g., OCR.

    As for the question of whether rotated-NMS should treat them as faraway boxes
    even though their IOU is 1, it depends on the application and/or ground truth annotation.

    As an extreme example, consider a single character v and the square box around it.

    If the angle is 0 degree, the object (text) would be read as 'v';

    If the angle is 90 degrees, the object (text) would become '>';

    If the angle is 180 degrees, the object (text) would become '^';

    If the angle is 270/-90 degrees, the object (text) would become '<'

    All of these cases have IoU of 1 to each other, and rotated NMS that only
    uses IoU as criterion would only keep one of them with the highest score -
    which, practically, still makes sense in most cases because typically
    only one of theses orientations is the correct one. Also, it does not matter
    as much if the box is only used to classify the object (instead of transcribing
    them with a sequential OCR recognition model) later.

    On the other hand, when we use IoU to filter proposals that are close to the
    ground truth during training, we should definitely take the angle into account if
    we know the ground truth is labeled with the strictly correct orientation (as in,
    upside-down words are annotated with -180 degrees even though they can be covered
    with a 0/90/-90 degree box, etc.)

    The way the original dataset is annotated also matters. For example, if the dataset
    is a 4-point polygon dataset that does not enforce ordering of vertices/orientation,
    we can estimate a minimum rotated bounding box to this polygon, but there's no way
    we can tell the correct angle with 100% confidence (as shown above, there could be 4 different
    rotated boxes, with angles differed by 90 degrees to each other, covering the exactly
    same region). In that case we have to just use IoU to determine the box
    proximity (as many detection benchmarks (even for text) do) unless there're other
    assumptions we can make (like width is always larger than height, or the object is not
    rotated by more than 90 degrees CCW/CW, etc.)

    In summary, not considering angles in rotated NMS seems to be a good option for now,
    but we should be aware of its implications.

    Args:
        boxes (Tensor[N, 5]): Rotated boxes to perform NMS on. They are expected to be in
           (x_center, y_center, width, height, angle_degrees) format.
        scores (Tensor[N]): Scores for each one of the rotated boxes
        iou_threshold (float): Discards all overlapping rotated boxes with IoU < iou_threshold

    Returns:
        keep (Tensor): int64 tensor with the indices of the elements that have been kept
        by Rotated NMS, sorted in decreasing order of scores
    """
    from fsdet import _C

    return _C.nms_rotated(boxes, scores, iou_threshold)


# Note: this function (batched_nms_rotated) might be moved into
# torchvision/ops/boxes.py in the future
def batched_nms_rotated(boxes, scores, idxs, iou_threshold):
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 5]):
           boxes where NMS will be performed. They
           are expected to be in (x_ctr, y_ctr, width, height, angle_degrees) format
        scores (Tensor[N]):
           scores for each one of the boxes
        idxs (Tensor[N]):
           indices of the categories for each one of the boxes.
        iou_threshold (float):
           discards all overlapping boxes
           with IoU < iou_threshold

    Returns:
        Tensor:
            int64 tensor with the indices of the elements that have been kept
            by NMS, sorted in decreasing order of scores
    """
    assert boxes.shape[-1] == 5

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # Strategy: in order to perform NMS independently per class,
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap

    # Note that batched_nms in torchvision/ops/boxes.py only uses max_coordinate,
    # which won't handle negative coordinates correctly.
    # Here by using min_coordinate we can make sure the negative coordinates are
    # correctly handled.
    max_coordinate = (
        torch.max(boxes[:, 0], boxes[:, 1]) + torch.max(boxes[:, 2], boxes[:, 3]) / 2
    ).max()
    min_coordinate = (
        torch.min(boxes[:, 0], boxes[:, 1]) - torch.min(boxes[:, 2], boxes[:, 3]) / 2
    ).min()
    offsets = idxs.to(boxes) * (max_coordinate - min_coordinate + 1)
    boxes_for_nms = boxes.clone()  # avoid modifying the original values in boxes
    boxes_for_nms[:, :2] += offsets[:, None]
    keep = nms_rotated(boxes_for_nms, scores, iou_threshold)
    return keep
