# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import functools
import logging

# import numba as numba
# from numba import types, typed

import numpy as np
import pickle, time
import torch
import torch.distributed as dist

_LOCAL_PROCESS_GROUP = None
"""
A torch process group which only includes processes that on the same machine as the current process.
This variable is set when processes are spawned by `launch()` in "engine/launch.py".
"""


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


def get_local_size() -> int:
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def is_main_process() -> bool:
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
            world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def gather(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group=group) == 1:
        return [data]
    rank = dist.get_rank(group=group)

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    # receiving Tensor from all ranks
    if rank == dst:
        max_size = max(size_list)
        tensor_list = [
            torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list
        ]
        dist.gather(tensor, tensor_list, dst=dst, group=group)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        return data_list
    else:
        dist.gather(tensor, [], dst=dst, group=group)
        return []


def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
            If workers need a shared RNG, they can use this shared seed to
            create one.

    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2 ** 31)
    all_ints = all_gather(ints)
    return all_ints[0]


def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.

    Args:
        input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
        average (bool): whether to do average or sum

    Returns:
        a dict with the same keys as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


'Mixup'


def padd(data, sub):
    import torch.nn.functional as F
    p2d = (0, 0, 0, sub)
    data = F.pad(data, p2d, 'constant', 0)
    return data


'Mosaic'
import copy
import os, cv2
import numpy as np
from PIL import ImageDraw
import PIL.Image as Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def merge_bboxes(bboxes, cutx, cuty):
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            if i == 0:
                if y1 > cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 1:
                if y2 < cuty or x1 > cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 2:
                if y2 < cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 3:
                if y1 > cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue

            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1])
            # tmp_box_tensor = torch.stack(tmp_box)
            merge_bbox.append(tmp_box)
    return merge_bbox


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a



from multiprocessing import Pool
def get_random_data1(p_images, p_ann, hue=.1, sat=1.5, val=1.5):
    '''random preprocessing for real-time data augmentation'''

    for i in range(len(p_ann)):
        # p_ann[i]["bbox"].append(p_ann[i]["category_id"])torch.cat([p_ann[i]["bbox"], p_ann[i]["category_id"]], dim=0)
        # p_ann[i]["bbox"] = list(map(int, p_ann[i]["bbox"]))
        p_ann[i]["bbox"] = torch.cat([p_ann[i]["bbox"], p_ann[i]["category_id"].unsqueeze(0).int()], dim=0)

    j = 0
    box = []
    for i in range(0, len(p_images)):
        temp = []

        # for j in range(len(p_ann)):
        #     if p_ann[j]["image_id"] == p_images[i]["image_id"]:
        #         temp.append(p_ann[j]["bbox"])
        #         temp_tensor = torch.stack(temp).cuda()

        while j < len(p_ann) and p_ann[j]["image_id"] == p_images[i]["image_id"]:
            temp.append(p_ann[j]["bbox"])
            temp_tensor = torch.stack(temp).cuda()
            # temp_tensor = numba.cuda.as_cuda_array(torch.stack(temp).cuda())
            j += 1

        box.append(temp_tensor)

        ' box_data = np.zeros((len(box[i]), 5))'
        ' box_data[:len(box[i])] = box[i]'
        ' 可以这样处理'
        # flag = 0
        # if flag == 0:
        #     box.append(temp_tensor)
        #     box_tensor = torch.stack(box)
        #     flag += 1
        # else:
        #     box = torch.cat([box_tensor, temp_tensor], dim=0)

    h, w = (p_images[0]["height"], p_images[0]["width"])
    min_offset_x = 0.4
    min_offset_y = 0.4
    scale_low = 1 - min(min_offset_x, min_offset_y)
    scale_high = scale_low + 0.2
    image_datas = []
    box_datas = []
    data = 0
    place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]
    place_y = [0, int(h * min_offset_y), int(w * min_offset_y), 0]

    '这里慢'
    # process_args = [[p_images[i], scale_low, scale_high, box[i], w, h, hue, sat, val, place_x, place_y, i] for i in range(4)]
    # with Pool(4) as p:
    #     output = p.map(process_data, process_args)
    # p = Pool(4)
    for i in range(0, 4):
        # image_data, box_data = p.apply_async(process_data, args=(i, p_images, scale_low, scale_high, box, w, h, hue, sat, val, place_x, place_y, data))
        # image_data, box_data = process_data(i, p_images, scale_low, scale_high, box, w, h, hue, sat, val, place_x, place_y, data)
        # data = data + 1
        # image_datas.append(image_data)
        # box_datas.append(box_data)

        # box[i][:, [2]] = box[i][:, [0]] + box[i][:, [2]]
        # box[i][:, [3]] = box[i][:, [1]] + box[i][:, [3]]
        image_name = p_images[i]["file_name"]
        image = Image.open(image_name)
        image = image.convert("RGB")
        '''检查标注'''
        # image1 = cv2.imread(image_name)
        '标注为(x,y,w,h)格式'
        # cv2.imshow("image"+str(i), image1)
        # image1 = cv2.rectangle(image1, (p_ann[i]["bbox"][0], p_ann[i]["bbox"][1]),
        #                        (p_ann[i]["bbox"][0] + p_ann[i]["bbox"][2], p_ann[i]["bbox"][1] + p_ann[i]["bbox"][3]),
        #                        (0, 255, 255), 2)
        '标注为(x,y,x,y)格式'
        # for j in range(len(box[i])):
        #     image1 = cv2.rectangle(image1, (box[i][j][0], box[i][j][1]), (box[i][j][2], box[i][j][3]), (255, 0, 255), 2)
        # cv2.imshow("draw" + str(i), image1)

        # 图片的大小
        iw, ih = image.size
        # 保存框的位置
        # image.save(str(data)+".jpg")
        # 是否翻转图片
        flip = rand() < .5
        if flip and len(box[i]) > 0:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            box[i][:, [0, 2]] = iw - box[i][:, [2, 0]]

        # 对输入进来的图片进行缩放
        new_ar = w / h
        scale = rand(scale_low, scale_high)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 进行色域变换
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = rgb_to_hsv(np.array(image) / 255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x > 1] = 1
        x[x < 0] = 0
        image = hsv_to_rgb(x)

        # image = np.array(image) / 255.

        image = Image.fromarray((image * 255).astype(np.uint8))
        # 将图片进行放置，分别对应四张分割图片的位置
        dx = place_x[data]
        dy = place_y[data]
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image) / 255
        # Image.fromarray((image_data*255).astype(np.uint8)).save(str(data)+"distort.jpg")
        data = data + 1
        box_data = []
        # 对box进行重新处理

        if len(box[i]) > 0:
            # np.random.shuffle(box[i])
            box[i][:, [0, 2]] = box[i][:, [0, 2]] * nw / iw + dx
            box[i][:, [1, 3]] = box[i][:, [1, 3]] * nh / ih + dy
            box[i][:, 0:2][box[i][:, 0:2] < 0] = 0
            box[i][:, 2][box[i][:, 2] > w] = w
            box[i][:, 3][box[i][:, 3] > h] = h
            box_w = abs(box[i][:, 2] - box[i][:, 0])
            box_h = abs(box[i][:, 3] - box[i][:, 1])
            box[i] = box[i][torch.logical_and(box_w > 1, box_h > 1)]
            box_data = torch.zeros((len(box[i]), 5))
            box_data[:len(box[i])] = box[i]
        image_datas.append(image_data)
        box_datas.append(box_data)

        '可视化 bbox'
        # img = Image.fromarray((image_data * 255).astype(np.uint8))
        # for j in range(len(box_data)):
        #     thickness = 3
        #     left, top, right, bottom = box_data[j][0:4]
        #     draw = ImageDraw.Draw(img)
        #     for i in range(thickness):
        #         draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(255, 255, 255))
        # img.show()

    # 将图片分割，放在一起
    # cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
    # cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))
    cutx = torch.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)), (1,))
    cuty = torch.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)), (1,))

    new_image = np.zeros([h, w, 3])
    new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
    new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
    new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
    new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]
    # Image.fromarray((new_image * 255).astype(np.uint8)).show()

    # 对框进行进一步的处理
    new_boxes = merge_bboxes(box_datas, cutx, cuty)

    return new_image, new_boxes


# @numba.jit(nopython=True)
# @torch.jit.script
def Mosaic(data):
    p_images = []
    # p_images = typed.List.empty_list(types.float64)
    p_ann = []
    for i in range(len(data)):
        temp_img = {}
        temp_img["file_name"] = data[i]["file_name"]
        temp_img["height"] = data[i]["height"]
        temp_img["width"] = data[i]["width"]
        temp_img["image_id"] = data[i]["image_id"]
        p_images.append(temp_img)

        for m in range(len(data[i]["instances"].get("gt_boxes").tensor)):
            temp = {}
            if len(data[i]["instances"].get("gt_boxes").tensor) == 1:
                # temp["bbox"] = (np.squeeze(np.asarray(data[i]["instances"].get("gt_boxes").tensor, dtype=int), 0)).tolist()
                # temp["category_id"] = data[i]["instances"].get('gt_classes').numpy()[0]
                temp["bbox"] = data[i]["instances"].get("gt_boxes").tensor.squeeze(0)
                temp["category_id"] = data[i]["instances"].get('gt_classes')[0]
            else:
                # temp["bbox"] = (np.asarray(data[i]["instances"].get("gt_boxes").tensor[m], dtype=int)).tolist()
                # temp["category_id"] = data[i]["instances"].get('gt_classes').numpy()[m]
                temp["bbox"] = data[i]["instances"].get("gt_boxes").tensor[m]
                temp["category_id"] = data[i]["instances"].get('gt_classes')[m]

            temp["image_id"] = data[i]["image_id"]
            p_ann.append(temp)

    mosaic_img, mosaic_ann = get_random_data1(p_images, p_ann)

    # cv2.waitKey(0)
    data_temp = copy.deepcopy(data[0])
    data_temp["image"] = torch.as_tensor(mosaic_img, dtype=torch.float32, device="cpu").permute(2, 0, 1)
    data_temp["width"] = 900
    data_temp["height"] = 700
    data_temp.pop('image_id')
    data_temp.pop('file_name')
    # mosaic_np = np.asarray(mosaic_ann, dtype=np.float32)
    ann = [mosaic_ann[i][0:4] for i in range(len(mosaic_ann))]
    # ann = torch.as_tensor(ann, dtype=torch.float32)
    data_temp["instances"].get("gt_boxes").set_tensor(ann)

    cls = torch.as_tensor([mosaic_ann[i][4] for i in range(len(mosaic_ann))], dtype=torch.int64)
    data_temp["instances"].set("gt_classes", cls)
    return data_temp


def vis_mosaic(data, id=-1):
    if id >= 0:
        pic = np.asarray(data[id]["image"].permute(1, 2, 0))
        # cv2.imshow("origin", pic)
        draw_1 = pic
        gt_box = data[id]["instances"].get("gt_boxes").tensor
        for i in range(len(gt_box)):
            draw_1 = cv2.rectangle(draw_1, (int(gt_box[i][0]), int(gt_box[i][1])),
                                   (int(gt_box[i][2]), int(gt_box[i][3])), (0, 255, 255), 2)

        cv2.imshow("mosaic", draw_1)

        cv2.waitKey(0)
    else:
        pic = np.asarray(data["image"].permute(1, 2, 0))
        # cv2.imshow("origin", pic)
        draw_1 = pic
        gt_box = data["instances"].get("gt_boxes").tensor
        for i in range(len(gt_box)):
            draw_1 = cv2.rectangle(draw_1, (int(gt_box[i][0]), int(gt_box[i][1])),
                                   (int(gt_box[i][2]), int(gt_box[i][3])), (0, 255, 255), 2)
        draw_1 = torch.as_tensor(draw_1)
        return draw_1.permute(2, 0, 1)
