# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import numpy as np
from enum import Enum, unique
from typing import Iterator, List, Tuple, Union
import torch, math

from fsdet.layers import cat

_RawBoxType = Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray]


@unique
class BoxMode(Enum):
    """
    Enum of different ways to represent a box.

    Attributes:

        XYXY_ABS: (x0, y0, x1, y1) in absolute floating points coordinates.
            The coordinates in range [0, width or height].
        XYWH_ABS: (x0, y0, w, h) in absolute floating points coordinates.
        XYXY_REL: (x0, y0, x1, y1) in range [0, 1]. They are relative to the size of the image.
        XYWH_REL: (x0, y0, w, h) in range [0, 1]. They are relative to the size of the image.
    """

    XYXY_ABS = 0
    XYWH_ABS = 1
    XYXY_REL = 2
    XYWH_REL = 3

    @staticmethod
    def convert(box: _RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode") -> _RawBoxType:
        """
        Args:
            box: can be a 4-tuple, 4-list or a Nx4 array/tensor.
            from_mode, to_mode (BoxMode)

        Returns:
            The converted box of the same type.
        """
        if from_mode == to_mode:
            return box

        original_type = type(box)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            arr = np.array(box)
            assert arr.shape == (
                4,
            ), "BoxMode.convert takes either a 4-tuple/list or a Nx4 array/tensor"
        else:
            arr = copy.deepcopy(box)  # avoid modifying the input box

        assert to_mode.value < 2 and from_mode.value < 2, "Relative mode not yet supported!"

        original_shape = arr.shape
        arr = arr.reshape(-1, 4)
        if to_mode == BoxMode.XYXY_ABS and from_mode == BoxMode.XYWH_ABS:
            arr[:, 2] += arr[:, 0]
            arr[:, 3] += arr[:, 1]
        elif from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS:
            arr[:, 2] -= arr[:, 0]
            arr[:, 3] -= arr[:, 1]
        else:
            raise RuntimeError("Cannot be here!")
        if single_box:
            return original_type(arr.flatten())
        return arr.reshape(*original_shape)


class Boxes:
    """
    This structure stores a list of boxes as a Nx4 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor: float matrix of Nx4.
    """

    BoxSizeType = Union[List[int], Tuple[int, int]]

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = torch.zeros(0, 4, dtype=torch.float32, device=device)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()

        self.tensor = tensor

    def set_tensor(self, tensors):
        self.tensor = torch.as_tensor(tensors, dtype=torch.float32, device="cpu")

    def clone(self) -> "Boxes":
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        return Boxes(self.tensor.clone())

    def to(self, device: str) -> "Boxes":
        return Boxes(self.tensor.to(device))

    def area(self) -> torch.Tensor:
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area

    def clip(self, box_size: BoxSizeType) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all()
        h, w = box_size
        self.tensor[:, 0].clamp_(min=0, max=w)
        self.tensor[:, 1].clamp_(min=0, max=h)
        self.tensor[:, 2].clamp_(min=0, max=w)
        self.tensor[:, 3].clamp_(min=0, max=h)

    def nonempty(self, threshold: int = 0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Boxes":
        """
        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.

        The following usage are allowed:
        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10_proposal2000]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Boxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Boxes(" + str(self.tensor) + ")"

    def inside_box(self, box_size: BoxSizeType, boundary_threshold: int = 0) -> torch.Tensor:
        """
        Args:
            box_size (height, width): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        height, width = box_size
        inds_inside = (
                (self.tensor[..., 0] >= -boundary_threshold)
                & (self.tensor[..., 1] >= -boundary_threshold)
                & (self.tensor[..., 2] < width + boundary_threshold)
                & (self.tensor[..., 3] < height + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The box centers in a Nx2 array of (x, y).
        """
        return (self.tensor[:, :2] + self.tensor[:, 2:]) / 2

    def scale(self, scale_x: float, scale_y: float) -> None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    @staticmethod
    def cat(boxes_list: List["Boxes"]) -> "Boxes":
        """
        Concatenates a list of Boxes into a single Boxes

        Arguments:
            boxes_list (list[Boxes])

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        assert len(boxes_list) > 0
        assert all(isinstance(box, Boxes) for box in boxes_list)

        cat_boxes = type(boxes_list[0])(cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) -> str:
        return self.tensor.device

    def __iter__(self) -> Iterator[torch.Tensor]:
        """
        Yield a box as a Tensor of shape (4,) at a time.
        """
        yield from self.tensor


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = boxes1.area()
    area2 = boxes2.area()

    boxes1, boxes2 = boxes1.tensor, boxes2.tensor

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    # handle empty boxes
    # inter > 0的话，更改 inter 为 inter / (area1[:, None] + area2 - inter)，否则更改 inter 为 torch.zeros(1, dtype=inter.dtype, device=inter.device)
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def matched_boxlist_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Compute pairwise intersection over union (IOU) of two sets of matched
    boxes. The box order must be (xmin, ymin, xmax, ymax).
    Similar to boxlist_iou, but computes only diagonal elements of the matrix
    Arguments:
        boxes1: (Boxes) bounding boxes, sized [N,4].
        boxes2: (Boxes) bounding boxes, sized [N,4].
    Returns:
        (tensor) iou, sized [N].
    """
    assert len(boxes1) == len(boxes2), (
        "boxlists should have the same"
        "number of entries, got {}, {}".format(len(boxes1), len(boxes2))
    )
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [N]
    box1, box2 = boxes1.tensor, boxes2.tensor
    lt = torch.max(box1[:, :2], box2[:, :2])  # [N,2]
    rb = torch.min(box1[:, 2:], box2[:, 2:])  # [N,2]
    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]
    iou = inter / (area1 + area2 - inter)  # [N]
    return iou


def pairwise_giou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    # cal the box's area of boxes1 and boxess
    boxes1 = Boxes(boxes1)
    boxes2 = Boxes(boxes2)
    area1 = boxes1.area()
    area2 = boxes2.area()

    boxes1, boxes2 = boxes1.tensor, boxes2.tensor

    # cal Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    union_area = area1[:, None] + area2 - inter
    # handle empty boxes
    # inter > 0的话，更改 inter 为 inter / (area1[:, None] + area2 - inter)，否则更改 inter 为 torch.zeros(1, dtype=inter.dtype, device=inter.device)
    iou = torch.where(
        inter > 0,
        inter / union_area,
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )

    # cal outer boxes
    left_up = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    right_down = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    enclose = (right_down - left_up).clamp(min=0)
    enclose_area = enclose[:, :, 0] * enclose[:, :, 1]

    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

    gious_flatten = (1 - giou).flatten()
    localization_loss = gious_flatten.mean() * giou.shape[0]
    return localization_loss


def pairwise_diou(boxes1, boxes2):
    '''
    cal DIOU of two boxes or batch boxes
    :param boxes1:[xmin,ymin,xmax,ymax] or
                [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
    :param boxes2:[xmin,ymin,xmax,ymax]
    :return:
    '''

    # cal the box's area of boxes1 and boxess
    # boxes1 = Boxes(boxes1)
    # boxes2 = Boxes(boxes2)
    # area1 = boxes1.area()
    # area2 = boxes2.area()
    # boxes1, boxes2 = boxes1.tensor, boxes2.tensor
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # cal Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    union_area = area1[:, None] + area2 - inter
    iou = torch.where(
        inter > 0,
        inter / union_area,
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )

    # cal outer boxes
    left_up = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    right_down = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    outer = (right_down - left_up).clamp(min=0)
    outer_diagonal_line = torch.square(outer[..., 0]) + torch.square(outer[..., 1])

    # cal center distance
    boxes1_center = (boxes1[..., :2] + boxes1[..., 2:]) * 0.5
    boxes2_center = (boxes2[..., :2] + boxes2[..., 2:]) * 0.5
    center_dis = torch.square(boxes1_center[..., 0] - boxes2_center[..., 0]) + \
                 torch.square(boxes1_center[..., 1] - boxes2_center[..., 1])

    # cal diou
    dious = iou - center_dis / outer_diagonal_line

    dious_flatten = (1 - dious).flatten()
    localization_loss = dious_flatten.mean() * dious.shape[0]

    return localization_loss


def pairwise_ciou(boxes1, boxes2):
    '''
    cal CIOU of two boxes or batch boxes
    :param boxes1:[xmin,ymin,xmax,ymax] or
                [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
    :param boxes2:[xmin,ymin,xmax,ymax]
    :return:
    '''

    # cal the box's area of boxes1 and boxess
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # cal Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    union_area = area1[:, None] + area2 - inter
    iou = torch.where(
        inter > 0,
        inter / union_area,
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )

    # cal outer boxes
    left_up = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    right_down = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    outer = (right_down - left_up).clamp(min=0)
    outer_diagonal_line = torch.square(outer[..., 0]) + torch.square(outer[..., 1])

    # cal center distance
    boxes1_center = (boxes1[..., :2] + boxes1[..., 2:]) * 0.5
    boxes2_center = (boxes2[..., :2] + boxes2[..., 2:]) * 0.5
    center_dis = torch.square(boxes1_center[..., 0] - boxes2_center[..., 0]) + \
                 torch.square(boxes1_center[..., 1] - boxes2_center[..., 1])

    # cal penalty term
    # cal width,height
    boxes1_wh = torch.maximum(boxes1[..., 2:] - boxes1[..., :2], torch.tensor(0.0))
    boxes2_wh = torch.maximum(boxes2[..., 2:] - boxes2[..., :2], torch.tensor(0.0))
    v = (4.0 / torch.square(torch.tensor(math.pi))) * torch.square(
        (torch.arctan(boxes1_wh[..., 0] / boxes1_wh[..., 1]) -
         torch.arctan(boxes2_wh[..., 0] / boxes2_wh[..., 1]))
    )
    alpha = v / (1 - iou + v)

    # cal ciou
    cious = iou - (center_dis / outer_diagonal_line + alpha * v)
    cious_flatten = (1 - cious).flatten()
    localization_loss = cious_flatten.mean() * cious.shape[0]

    return localization_loss
