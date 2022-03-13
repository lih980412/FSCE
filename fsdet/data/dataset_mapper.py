# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import torch

import json
import albumentations as A
from fsdet.structures import BoxMode

from . import detection_utils as utils
from . import transforms as T

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper", "AlbumentationMapper"]

# 如DatasetMapper文档所述，其__call__实现包括三个步骤。若用户自定义mapper，也可以参考这三个步骤：
#
# classdetectron2.data.DatasetMapper(****args,**kwargs***)
#
# 该callable对象当前执行下列操作：
#
# 1. 从"file_name"读取图像
# 2. 将裁剪/几何变换应用于图像和标注
# 3. 将图像和标注分别转换为Tensor类和Instance类

class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic.

    The callable currently does the following:
    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True, aux=None):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        if not aux:
            self.tfm_gens = utils.build_transform_gen(cfg, is_train)
        else:
            self.tfm_gens = utils.build_transform_gen(cfg, is_train, aux)

        # fmt: off
        self.img_format = cfg.INPUT.FORMAT
        # fmt: on

        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)   # apply image
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor((image.transpose(2, 0, 1).astype("float32")) / 255)
        # Can use uint8 if it turns out to be slow some day

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(  # apply box
                    obj, transforms, image_shape
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict


class AlbumentationMapper:
    debug_count = 5
    def __init__(self, cfg, is_train=True):
        # use the detectron2 crop_gen
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None
        # default detectron2 tfm_gens contains ResizeShortestEdge and RandomFlip(horizontal only by default)
        self.tfm_gens = utils.build_transform_gen(cfg, is_train)
        # load albumnentation json
        logging.getLogger(__name__).info("Albumentation json config used in training: "
                                         + cfg.INPUT.ALBUMENTATIONS_JSON)
        self.aug = self._get_aug(cfg.INPUT.ALBUMENTATIONS_JSON)
        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # will be modified by code below
        image = utils.read_image(dataset_dict['file_name'], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if 'annotations' not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            bboxes = [ann['bbox'] for ann in dataset_dict['annotations']]
            labels = [ann['category_id'] for ann in dataset_dict['annotations']]

            augm_annotation = self.aug(image=image, bboxes=bboxes, category_id=labels)
            image = augm_annotation['image']
            h, w = image.shape[:2]

            augm_boxes = np.array(augm_annotation['bboxes'], dtype=np.float32)
            # sometimes bbox annotations go beyond image
            augm_boxes[:, :] = augm_boxes[:, :].clip(min=[0,0,0,0], max=[w,h,w,h])

            # TODO BUG augm_boexes does not equalt to bboxes, when bboxes lost, augm_boxes becomre []
            # might be the ShiftScaleRotate bug ?
            # try:
            #     augm_boxes[:, :] = augm_boxes[:, :].clip(min=[0,0,0,0], max=[w,h,w,h])
            # except:
            #     print(augm_boxes, augm_annotation)
            #     print(dataset_dict['file_name'])
            #     print(dataset_dict)
            #     print(bboxes)

            augm_labels = np.array(augm_annotation['category_id'])

            try:
                box_mode = dataset_dict['annotations'][0]['bbox_mode']
            except:
                raise AttributeError('line 162 in dataset_mapper.py failed, please check your dataset/dataset_dict')

            dataset_dict['annotations'] = [
                {
                    'iscrowd': 0,
                    'bbox': augm_boxes[i].tolist(),
                    'category_id': augm_labels[i],
                    'bbox_mode': box_mode
                }
                for i in range(len(augm_boxes))
            ]
            if self.crop_gen:
                # detecton2 CROP, Generate a CropTransform so that the cropping region contains
                # the center of the given instance.
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"])  # TODO BUG: this line is buggy, choice from []
                )
                image = crop_tfm.apply_image(image)
            # detectron2 Resize transforms
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1).astype('float32'))
        )
        dataset_dict['height'] = image.shape[0]
        dataset_dict['width'] = image.shape[1]

        if not self.is_train:
            dataset_dict.pop('annotations', None)
            dataset_dict.pop('sem_seg_file_name', None)
            return dataset_dict

        if 'annotations' in dataset_dict:
            # we do not care about segmentation and keypoints
            for anno in dataset_dict["annotations"]:
                anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image.shape[:2]
                )
                for obj in dataset_dict.pop("annotations") if obj.get("iscrowd", 0) == 0
            ]
            # convert annotations to Iannotations_to_instancesnstances to be used by detectron2 models
            instances = utils.annotations_to_instances(annos, image.shape[:2])
            dataset_dict['instances'] = utils.filter_empty_instances(instances)
        # import pdb; pdb.set_trace
        # import pickle
        # with open('/data/tmp/album_dataset_dict.pkl', 'wb') as f:
        #     pickle.dump(dataset_dict, f)
        return dataset_dict

    def _get_aug(self, arg):
        with open(arg) as f:
            return A.from_dict(json.load(f))

# # 21.10.24 Mosaic增强
# from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
# from PIL import ImageDraw
# from PIL import Image
# import numpy as np
# import json, os
# import cv2
#
# def rand(a=0, b=1):
#     return np.random.rand() * (b - a) + a
# def merge_bboxes(bboxes, cutx, cuty):
#     merge_bbox = []
#     for i in range(len(bboxes)):
#         for box in bboxes[i]:
#             tmp_box = []
#             x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
#
#             if i == 0:
#                 if y1 > cuty or x1 > cutx:
#                     continue
#                 if y2 >= cuty and y1 <= cuty:
#                     y2 = cuty
#                     if y2 - y1 < 5:
#                         continue
#                 if x2 >= cutx and x1 <= cutx:
#                     x2 = cutx
#                     if x2 - x1 < 5:
#                         continue
#
#             if i == 1:
#                 if y2 < cuty or x1 > cutx:
#                     continue
#
#                 if y2 >= cuty and y1 <= cuty:
#                     y1 = cuty
#                     if y2 - y1 < 5:
#                         continue
#
#                 if x2 >= cutx and x1 <= cutx:
#                     x2 = cutx
#                     if x2 - x1 < 5:
#                         continue
#
#             if i == 2:
#                 if y2 < cuty or x2 < cutx:
#                     continue
#
#                 if y2 >= cuty and y1 <= cuty:
#                     y1 = cuty
#                     if y2 - y1 < 5:
#                         continue
#
#                 if x2 >= cutx and x1 <= cutx:
#                     x1 = cutx
#                     if x2 - x1 < 5:
#                         continue
#
#             if i == 3:
#                 if y1 > cuty or x2 < cutx:
#                     continue
#
#                 if y2 >= cuty and y1 <= cuty:
#                     y2 = cuty
#                     if y2 - y1 < 5:
#                         continue
#
#                 if x2 >= cutx and x1 <= cutx:
#                     x1 = cutx
#                     if x2 - x1 < 5:
#                         continue
#
#             tmp_box.append(x1)
#             tmp_box.append(y1)
#             tmp_box.append(x2)
#             tmp_box.append(y2)
#             tmp_box.append(box[-1])
#             merge_bbox.append(tmp_box)
#     return merge_bbox
# def cvtColor(image):
#     if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
#         return image
#     else:
#         image = image.convert('RGB')
#         return image
#
# '''not work'''
# def get_random_data_with_Mosaic(annotation_line, image_root, input_shape=None, max_boxes=100, hue=.1, sat=1.5, val=1.5):
#     with open(annotation_line, "r") as f:
#         p = json.load(f)
#         p_images = p["images"]
#         p_ann = p["annotations"]
#
#     for i in range(len(p_ann)):
#         p_ann[i]["bbox"].append(p_ann[i]["category_id"])
#         p_ann[i]["bbox"] = list(map(int, p_ann[i]["bbox"]))
#
#     j = 0
#     box = []
#     for i in range(0, len(p_images)):
#         temp = []
#         while j < len(p_ann) and p_ann[j]["image_id"] == p_images[i]["id"]:
#             temp.append(p_ann[j]["bbox"])
#             temp_np = np.array(temp)
#             j += 1
#         box.append(temp_np)
#
#     h, w = p_images[0]["height"], p_images[0]["width"]
#     min_offset_x = rand(0.25, 0.75)
#     min_offset_y = rand(0.25, 0.75)
#
#     nws = [int(w * rand(0.4, 1)), int(w * rand(0.4, 1)), int(w * rand(0.4, 1)),
#            int(w * rand(0.4, 1))]
#     nhs = [int(h * rand(0.4, 1)), int(h * rand(0.4, 1)), int(h * rand(0.4, 1)),
#            int(h * rand(0.4, 1))]
#
#     place_x = [int(w * min_offset_x) - nws[0], int(w * min_offset_x) - nws[1], int(w * min_offset_x),
#                int(w * min_offset_x)]
#     place_y = [int(h * min_offset_y) - nhs[0], int(h * min_offset_y), int(h * min_offset_y),
#                int(h * min_offset_y) - nhs[3]]
#
#     image_datas = []
#     box_datas = []
#     index = 0
#     for i in range(0, 4):
#         image_name = p_images[i]["file_name"]
#
#         image_path = os.path.join(image_root, image_name)
#         image = Image.open(image_path)
#         image = image.convert("RGB")
#         # image = image.resize()
#
#         image1 = cv2.imread(image_path)
#         # cv2.imshow("image"+str(i), image1)
#         image1 = cv2.rectangle(image1, (p_ann[i]["bbox"][0], p_ann[i]["bbox"][1]),
#                                (p_ann[i]["bbox"][0] + p_ann[i]["bbox"][2], p_ann[i]["bbox"][1] + p_ann[i]["bbox"][3]),
#                                (0, 255, 255), 2)
#         cv2.imshow("draw" + str(i), image1)
#
#         # 图片的大小
#         iw, ih = image.size
#         # 保存框的位置
#
#         # 是否翻转图片
#         flip = rand() < .5
#         if flip and len(box[i]) > 0:
#             image = image.transpose(Image.FLIP_LEFT_RIGHT)
#             box[i][:, [0, 2]] = iw - box[i][:, [2, 0]]
#
#         nw = nws[index]
#         nh = nhs[index]
#         image = image.resize((nw, nh), Image.BICUBIC)
#
#         # 将图片进行放置，分别对应四张分割图片的位置
#         dx = place_x[index]
#         dy = place_y[index]
#         new_image = Image.new('RGB', (w, h), (128, 128, 128))
#         new_image.paste(image, (dx, dy))
#         image_data = np.array(new_image)
#
#         index = index + 1
#         box_data = []
#         # 对box进行重新处理
#         if len(box[i]) > 0:
#             np.random.shuffle(box[i])
#             box[i][:, [0, 2]] = box[i][:, [0, 2]] * nw / iw + dx
#             box[i][:, [1, 3]] = box[i][:, [1, 3]] * nh / ih + dy
#             box[i][:, 0:2][box[i][:, 0:2] < 0] = 0
#             box[i][:, 2][box[i][:, 2] > w] = w
#             box[i][:, 3][box[i][:, 3] > h] = h
#             box_w = box[i][:, 2] - box[i][:, 0]
#             box_h = box[i][:, 3] - box[i][:, 1]
#             box[i] = box[i][np.logical_and(box_w > 1, box_h > 1)]
#             box_data = np.zeros((len(box[i]), 5))
#             box_data[:len(box[i])] = box[i]
#
#         image_datas.append(image_data)
#         box_datas.append(box_data)
#
#     # 将图片分割，放在一起
#     cutx = int(w * min_offset_x)
#     cuty = int(h * min_offset_y)
#
#     new_image = np.zeros([h, w, 3])
#     new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
#     new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
#     new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
#     new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]
#
#     # 进行色域变换
#     hue = rand(-hue, hue)
#     sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
#     val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
#     x = cv2.cvtColor(np.array(new_image / 255, np.float32), cv2.COLOR_RGB2HSV)
#     x[..., 0] += hue * 360
#     x[..., 0][x[..., 0] > 1] -= 1
#     x[..., 0][x[..., 0] < 0] += 1
#     x[..., 1] *= sat
#     x[..., 2] *= val
#     x[x[:, :, 0] > 360, 0] = 360
#     x[:, :, 1:][x[:, :, 1:] > 1] = 1
#     x[x < 0] = 0
#     new_image = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255
#
#     # 对框进行进一步的处理
#     new_boxes = merge_bboxes(box_datas, cutx, cuty)
#
#     return new_image, new_boxes
#
# '''worked'''
# def get_random_data1(annotation_line, input_shape=None, random=True, hue=.1, sat=1.5, val=1.5, proc_img=True, image_root=None):
#     '''random preprocessing for real-time data augmentation'''
#
#     with open(annotation_line, "r") as f:
#         p = json.load(f)
#         p_images = p["images"]
#         p_ann = p["annotations"]
#
#     for i in range(len(p_ann)):
#         p_ann[i]["bbox"].append(p_ann[i]["category_id"])
#         p_ann[i]["bbox"] = list(map(int, p_ann[i]["bbox"]))
#
#     j = 0
#     box = []
#     for i in range(0, len(p_images)):
#         temp = []
#         while j < len(p_ann) and p_ann[j]["image_id"] == p_images[i]["id"]:
#             temp.append(p_ann[j]["bbox"])
#             temp_np = np.array(temp)
#             j += 1
#         box.append(temp_np)
#
#     h, w = (p_images[0]["height"], p_images[0]["width"])
#     min_offset_x = 0.4
#     min_offset_y = 0.4
#     scale_low = 1 - min(min_offset_x, min_offset_y)
#     scale_high = scale_low + 0.2
#     image_datas = []
#     box_datas = []
#     index = 0
#     place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]
#     place_y = [0, int(h * min_offset_y), int(w * min_offset_y), 0]
#
#     for i in range(0, 4):
#         box[i][:, [2]] = box[i][:, [0]] + box[i][:, [2]]
#         box[i][:, [3]] = box[i][:, [1]] + box[i][:, [3]]
#         image_name = p_images[i]["file_name"]
#
#         image_path = os.path.join(image_root, image_name)
#         image = Image.open(image_path)
#         image = image.convert("RGB")
#         # image = image.resize()
#
#         image1 = cv2.imread(image_path)
#         # # cv2.imshow("image"+str(i), image1)
#         # image1 = cv2.rectangle(image1, (p_ann[i]["bbox"][0], p_ann[i]["bbox"][1]),
#         #                        (p_ann[i]["bbox"][0] + p_ann[i]["bbox"][2], p_ann[i]["bbox"][1] + p_ann[i]["bbox"][3]),
#         #                        (0, 255, 255), 2)
#         for j in range(len(box[i])):
#             image1 = cv2.rectangle(image1, (box[i][j][0], box[i][j][1]), (box[i][j][2], box[i][j][3]), (0, 255, 255), 2)
#         cv2.imshow("draw" + str(i), image1)
#
#         # 图片的大小
#         iw, ih = image.size
#         # 保存框的位置
#
#         # image.save(str(index)+".jpg")
#         # 是否翻转图片
#         flip = rand() < .5
#         if flip and len(box[i]) > 0:
#             image = image.transpose(Image.FLIP_LEFT_RIGHT)
#             box[i][:, [0, 2]] = iw - box[i][:, [2, 0]]
#
#         # 对输入进来的图片进行缩放
#         new_ar = w / h
#         scale = rand(scale_low, scale_high)
#         if new_ar < 1:
#             nh = int(scale * h)
#             nw = int(nh * new_ar)
#         else:
#             nw = int(scale * w)
#             nh = int(nw / new_ar)
#         image = image.resize((nw, nh), Image.BICUBIC)
#
#         # 进行色域变换
#         hue = rand(-hue, hue)
#         sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
#         val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
#         x = rgb_to_hsv(np.array(image) / 255.)
#         x[..., 0] += hue
#         x[..., 0][x[..., 0] > 1] -= 1
#         x[..., 0][x[..., 0] < 0] += 1
#         x[..., 1] *= sat
#         x[..., 2] *= val
#         x[x > 1] = 1
#         x[x < 0] = 0
#         image = hsv_to_rgb(x)
#
#         # image = np.array(image) / 255.
#
#         image = Image.fromarray((image * 255).astype(np.uint8))
#         # 将图片进行放置，分别对应四张分割图片的位置
#         dx = place_x[index]
#         dy = place_y[index]
#         new_image = Image.new('RGB', (w, h), (128, 128, 128))
#         new_image.paste(image, (dx, dy))
#         image_data = np.array(new_image) / 255
#         # Image.fromarray((image_data*255).astype(np.uint8)).save(str(index)+"distort.jpg")
#         index = index + 1
#         box_data = []
#         # 对box进行重新处理
#
#         if len(box[i]) > 0:
#             # np.random.shuffle(box[i])
#             box[i][:, [0, 2]] = box[i][:, [0, 2]] * nw / iw + dx
#             box[i][:, [1, 3]] = box[i][:, [1, 3]] * nh / ih + dy
#             box[i][:, 0:2][box[i][:, 0:2] < 0] = 0
#             box[i][:, 2][box[i][:, 2] > w] = w
#             box[i][:, 3][box[i][:, 3] > h] = h
#             box_w = abs(box[i][:, 2] - box[i][:, 0])
#             box_h = abs(box[i][:, 3] - box[i][:, 1])
#             box[i] = box[i][np.logical_and(box_w > 1, box_h > 1)]
#             box_data = np.zeros((len(box[i]), 5))
#             box_data[:len(box[i])] = box[i]
#         print(i)
#         image_datas.append(image_data)
#         box_datas.append(box_data)
#         print(box_datas)
#
#         img = Image.fromarray((image_data * 255).astype(np.uint8))
#         for j in range(len(box_data)):
#             thickness = 3
#             left, top, right, bottom = box_data[j][0:4]
#             draw = ImageDraw.Draw(img)
#             for i in range(thickness):
#                 draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(255, 255, 255))
#
#         # img.show()
#
#     # 将图片分割，放在一起
#     cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
#     cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))
#
#     new_image = np.zeros([h, w, 3])
#     new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
#     new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
#     new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
#     new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]
#
#     # 对框进行进一步的处理
#     new_boxes = merge_bboxes(box_datas, cutx, cuty)
#
#     return new_image, new_boxes
#
# '''Original'''
# def get_random_data(annotation_line, input_shape, random=True, hue=.1, sat=1.5, val=1.5, proc_img=True, image_root=None):
#     '''random preprocessing for real-time data augmentation'''
#     h, w = input_shape
#     min_offset_x = 0.4
#     min_offset_y = 0.4
#     scale_low = 1 - min(min_offset_x, min_offset_y)
#     scale_high = scale_low + 0.2
#
#     image_datas = []
#     box_datas = []
#     index = 0
#
#     place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]
#     place_y = [0, int(h * min_offset_y), int(w * min_offset_y), 0]
#
#     with open(annotation_line, 'r') as f:
#         lines = f.readlines()
#
#         for line in lines:
#             # 每一行进行分割
#             line_content = line.split()
#             # 打开图片
#             image = Image.open(line_content[0])
#             image = image.convert("RGB")
#             # 图片的大小
#             iw, ih = image.size
#             # 保存框的位置
#             box = np.array([np.array(list(map(int, map(float, (box.split(',')))))) for box in line_content[1:]])
#
#             # image.save(str(index)+".jpg")
#             # 是否翻转图片
#             flip = rand() < .5
#             if flip and len(box) > 0:
#                 image = image.transpose(Image.FLIP_LEFT_RIGHT)
#                 box[:, [0, 2]] = iw - box[:, [2, 0]]
#
#             # 对输入进来的图片进行缩放
#             new_ar = w / h
#             scale = rand(scale_low, scale_high)
#             if new_ar < 1:
#                 nh = int(scale * h)
#                 nw = int(nh * new_ar)
#             else:
#                 nw = int(scale * w)
#                 nh = int(nw / new_ar)
#             image = image.resize((nw, nh), Image.BICUBIC)
#
#             # 进行色域变换
#             hue = rand(-hue, hue)
#             sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
#             val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
#             x = rgb_to_hsv(np.array(image) / 255.)
#             x[..., 0] += hue
#             x[..., 0][x[..., 0] > 1] -= 1
#             x[..., 0][x[..., 0] < 0] += 1
#             x[..., 1] *= sat
#             x[..., 2] *= val
#             x[x > 1] = 1
#             x[x < 0] = 0
#             image = hsv_to_rgb(x)
#
#             image = Image.fromarray((image * 255).astype(np.uint8))
#             # 将图片进行放置，分别对应四张分割图片的位置
#             dx = place_x[index]
#             dy = place_y[index]
#             new_image = Image.new('RGB', (w, h), (128, 128, 128))
#             new_image.paste(image, (dx, dy))
#             image_data = np.array(new_image) / 255
#
#             # Image.fromarray((image_data*255).astype(np.uint8)).save(str(index)+"distort.jpg")
#
#             index = index + 1
#             box_data = []
#             # 对box进行重新处理
#             if len(box) > 0:
#                 np.random.shuffle(box)
#                 box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
#                 box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
#                 box[:, 0:2][box[:, 0:2] < 0] = 0
#                 box[:, 2][box[:, 2] > w] = w
#                 box[:, 3][box[:, 3] > h] = h
#                 box_w = abs(box[:, 2] - box[:, 0])
#                 box_h = abs(box[:, 3] - box[:, 1])
#                 box = box[np.logical_and(box_w > 1, box_h > 1)]
#                 box_data = np.zeros((len(box), 5))
#                 box_data[:len(box)] = box
#
#             image_datas.append(image_data)
#             box_datas.append(box_data)
#
#             img = Image.fromarray((image_data * 255).astype(np.uint8))
#             for j in range(len(box_data)):
#                 thickness = 3
#                 left, top, right, bottom = box_data[j][0:4]
#                 draw = ImageDraw.Draw(img)
#                 for i in range(thickness):
#                     draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(255, 255, 255))
#             img.show()
#
#     # 将图片分割，放在一起
#     cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
#     cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))
#
#     new_image = np.zeros([h, w, 3])
#     new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
#     new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
#     new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
#     new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]
#
#     # 对框进行进一步的处理
#     new_boxes = merge_bboxes(box_datas, cutx, cuty)
#
#     return new_image, new_boxes
#
#
# if __name__ == "__main__":
#     ann_path = r"D:\UserD\Li\FSCE-1\datasets\cocosplit\seed1\full_box_10shot_horse_trainval.json"
#     # ann_path = r"C:\Users\lenovo\Desktop\1.txt"
#     image_root = r"D:\UserD\Li\FSCE-1\datasets\coco\val2014"
#
#     mosaic_img, mosaic_ann = get_random_data1(ann_path, image_root=image_root)
#     # mosaic_img, mosaic_ann = get_random_data_with_Mosaic(ann_path, image_root)
#
#     # mosaic_img, mosaic_ann = get_random_data(ann_path, (480, 640), image_root=image_root)
#     mosaic_img *= 255.
#     mosaic_img = mosaic_img.astype(np.uint8)
#     print(mosaic_ann)
#     mosaic_img = cv2.cvtColor(mosaic_img, cv2.COLOR_RGB2BGR)
#     cv2.imshow("mosaic_img", mosaic_img)
#     draw_1 = mosaic_img
#     for i in range(len(mosaic_ann)):
#         draw_1 = cv2.rectangle(draw_1, (int(mosaic_ann[i][0]), int(mosaic_ann[i][1])),
#                                (int(mosaic_ann[i][2]), int(mosaic_ann[i][3])),
#                                (0, 255, 255), 2)
#
#     cv2.imshow("draw_1", draw_1)
#     cv2.waitKey(0)