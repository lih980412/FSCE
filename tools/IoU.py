import numpy as np


def IoU(pred_box, gt_box):
    ixmin = max(pred_box[0], gt_box[0])
    iymin = max(pred_box[1], gt_box[1])
    ixmax = min(pred_box[2], gt_box[2])
    iymax = min(pred_box[3], gt_box[3])
    inter_w = np.maximum(ixmax - ixmin + 1., 0)
    inter_h = np.maximum(iymax - iymin + 1., 0)

    inters = inter_w * inter_h

    uni = ((pred_box[2] - pred_box[0] + 1.) * (pred_box[3] - pred_box[1] + 1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) - inters)

    ious = inters / uni

    return ious


def maxIoU(pred_box, gt_box):
    ixmin = np.maximum(pred_box[:, 0], gt_box[0])
    iymin = np.maximum(pred_box[:, 1], gt_box[1])
    ixmax = np.minimum(pred_box[:, 2], gt_box[2])
    iymax = np.minimum(pred_box[:, 3], gt_box[3])
    inters_w = np.maximum(ixmax - ixmin + 1., 0)    # 逐元素求最大值和最小值 broadcasting
    inters_h = np.maximum(iymax - iymin + 1., 0)    # 逐元素求最大值和最小值 broadcasting

    inters = inters_w * inters_h

    uni = ((pred_box[:, 2] - pred_box[:, 0] + 1.) * (pred_box[:, 3] - pred_box[:, 1] + 1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) - inters)

    ious = inters / uni

    iou = np.max(ious)
    iou_id = np.argmax(ious)

    return iou, iou_id


if __name__ == "__main__":
    # test1
    pred_bbox = np.array([50, 50, 90, 100])  # top-left: <50, 50>, bottom-down: <90, 100>, <x-axis, y-axis>
    gt_bbox = np.array([70, 80, 120, 150])
    print(IoU(pred_bbox, gt_bbox))
    iou = IoU(pred_bbox, gt_bbox)

    # test2
    pred_bboxes = np.array([[15, 18, 47, 60],
                            [50, 50, 90, 100],
                            [70, 80, 120, 145],
                            [130, 160, 250, 280],
                            [25.6, 66.1, 113.3, 147.8]])
    gt_bbox = np.array([70, 80, 120, 150])
    print(maxIoU(pred_bboxes, gt_bbox))
    iou, iou_idx = maxIoU(pred_bboxes, gt_bbox)
