#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 15:30:24 2020

@author: fanq15
"""

from pycocotools.coco import COCO
import cv2
import numpy as np
from os.path import join, isdir
from os import mkdir, makedirs
from concurrent import futures
import sys
import time
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import shutil
import sys


# def vis_image(im, bboxs, im_name):
#     dpi = 300
#     fig, ax = plt.subplots()
#     ax.imshow(im, aspect='equal')
#     plt.axis('off')
#     height, width, channels = im.shape
#     fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
#     plt.gca().xaxis.set_major_locator(plt.NullLocator())
#     plt.gca().yaxis.set_major_locator(plt.NullLocator())
#     plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
#     plt.margins(0, 0)
#     # Show box (off by default, box_alpha=0.0)
#     for bbox in bboxs:
#         ax.add_patch(
#             plt.Rectangle((bbox[0], bbox[1]),
#                           bbox[2] - bbox[0],
#                           bbox[3] - bbox[1],
#                           fill=False, edgecolor='r',
#                           linewidth=0.5, alpha=1))
#     output_name = os.path.basename(im_name)
#     plt.savefig(im_name, dpi=dpi, bbox_inches='tight', pad_inches=0)
#     plt.close('all')


def crop_support(img, bbox):
    img = img.transpose(2, 0, 1)

    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])

    area = (y2-y1) * (x2-x1)
    # if area < 64*64:
    #     return None
    if area < 32*32:
        return None

    crop_box = img[:, y1:y2, x1:x2]

    return crop_box.transpose(1, 2, 0)
    #
    #
    #
    # image_shape = img.shape[:2]  # h, w
    # data_height, data_width = image_shape
    #
    # img = img.transpose(2, 0, 1)
    #
    # x1 = int(bbox[0])
    # y1 = int(bbox[1])
    # x2 = int(bbox[2])
    # y2 = int(bbox[3])

    # width = x2 - x1
    # height = y2 - y1
    # context_pixel = 16 #int(16 * im_scale)
    #
    # new_x1 = 0
    # new_y1 = 0
    # new_x2 = width
    # new_y2 = height
    # target_size = (320, 320) #(384, 384)
    #
    # if width >= height:
    #     crop_x1 = x1 - context_pixel
    #     crop_x2 = x2 + context_pixel
    #
    #     # New_x1 and new_x2 will change when crop context or overflow
    #     new_x1 = new_x1 + context_pixel
    #     new_x2 = new_x1 + width
    #     if crop_x1 < 0:
    #         new_x1 = new_x1 + crop_x1
    #         new_x2 = new_x1 + width
    #         crop_x1 = 0
    #     if crop_x2 > data_width:
    #         crop_x2 = data_width
    #
    #     short_size = height
    #     long_size = crop_x2 - crop_x1
    #     y_center = int((y2+y1) / 2) #math.ceil((y2 + y1) / 2)
    #     crop_y1 = int(y_center - (long_size / 2)) #int(y_center - math.ceil(long_size / 2))
    #     crop_y2 = int(y_center + (long_size / 2)) #int(y_center + math.floor(long_size / 2))
    #
    #     # New_y1 and new_y2 will change when crop context or overflow
    #     new_y1 = new_y1 + math.ceil((long_size - short_size) / 2)
    #     new_y2 = new_y1 + height
    #     if crop_y1 < 0:
    #         new_y1 = new_y1 + crop_y1
    #         new_y2 = new_y1 + height
    #         crop_y1 = 0
    #     if crop_y2 > data_height:
    #         crop_y2 = data_height
    #
    #     crop_short_size = crop_y2 - crop_y1
    #     crop_long_size = crop_x2 - crop_x1
    #     square = np.zeros((3, crop_long_size, crop_long_size), dtype = np.uint8)
    #     delta = int((crop_long_size - crop_short_size) / 2) #int(math.ceil((crop_long_size - crop_short_size) / 2))
    #     square_y1 = delta
    #     square_y2 = delta + crop_short_size
    #
    #     new_y1 = new_y1 + delta
    #     new_y2 = new_y2 + delta
    #
    #     crop_box = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
    #     square[:, square_y1:square_y2, :] = crop_box
    #
    #     #show_square = np.zeros((crop_long_size, crop_long_size, 3))#, dtype=np.int16)
    #     #show_crop_box = original_img[crop_y1:crop_y2, crop_x1:crop_x2, :]
    #     #show_square[square_y1:square_y2, :, :] = show_crop_box
    #     #show_square = show_square.astype(np.int16)
    # else:
    #     crop_y1 = y1 - context_pixel
    #     crop_y2 = y2 + context_pixel
    #
    #     # New_y1 and new_y2 will change when crop context or overflow
    #     new_y1 = new_y1 + context_pixel
    #     new_y2 = new_y1 + height
    #     if crop_y1 < 0:
    #         new_y1 = new_y1 + crop_y1
    #         new_y2 = new_y1 + height
    #         crop_y1 = 0
    #     if crop_y2 > data_height:
    #         crop_y2 = data_height
    #
    #     short_size = width
    #     long_size = crop_y2 - crop_y1
    #     x_center = int((x2 + x1) / 2) #math.ceil((x2 + x1) / 2)
    #     crop_x1 = int(x_center - (long_size / 2)) #int(x_center - math.ceil(long_size / 2))
    #     crop_x2 = int(x_center + (long_size / 2)) #int(x_center + math.floor(long_size / 2))
    #
    #     # New_x1 and new_x2 will change when crop context or overflow
    #     new_x1 = new_x1 + math.ceil((long_size - short_size) / 2)
    #     new_x2 = new_x1 + width
    #     if crop_x1 < 0:
    #         new_x1 = new_x1 + crop_x1
    #         new_x2 = new_x1 + width
    #         crop_x1 = 0
    #     if crop_x2 > data_width:
    #         crop_x2 = data_width
    #
    #     crop_short_size = crop_x2 - crop_x1
    #     crop_long_size = crop_y2 - crop_y1
    #     square = np.zeros((3, crop_long_size, crop_long_size), dtype = np.uint8)
    #     delta = int((crop_long_size - crop_short_size) / 2) #int(math.ceil((crop_long_size - crop_short_size) / 2))
    #     square_x1 = delta
    #     square_x2 = delta + crop_short_size
    #
    #     new_x1 = new_x1 + delta
    #     new_x2 = new_x2 + delta
    #     crop_box = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
    #     square[:, :, square_x1:square_x2] = crop_box
    #
    #     #show_square = np.zeros((crop_long_size, crop_long_size, 3)) #, dtype=np.int16)
    #     #show_crop_box = original_img[crop_y1:crop_y2, crop_x1:crop_x2, :]
    #     #show_square[:, square_x1:square_x2, :] = show_crop_box
    #     #show_square = show_square.astype(np.int16)
    # #print(crop_y2 - crop_y1, crop_x2 - crop_x1, bbox, data_height, data_width)
    #
    # square = square.astype(np.float32, copy=False)
    # square_scale = float(target_size[0]) / long_size
    # square = square.transpose(1,2,0)
    # # square = cv2.resize(square, target_size, interpolation=cv2.INTER_LINEAR) # None, None, fx=square_scale, fy=square_scale, interpolation=cv2.INTER_LINEAR)
    # #square = square.transpose(2,0,1)
    # square = square.astype(np.uint8)
    #
    # new_x1 = int(new_x1 * square_scale)
    # new_y1 = int(new_y1 * square_scale)
    # new_x2 = int(new_x2 * square_scale)
    # new_y2 = int(new_y2 * square_scale)
    #
    # # For test
    # #show_square = cv2.resize(show_square, target_size, interpolation=cv2.INTER_LINEAR) # None, None, fx=square_scale, fy=square_scale, interpolation=cv2.INTER_LINEAR)
    # #self.vis_image(show_square, [new_x1, new_y1, new_x2, new_y2], img_path.split('/')[-1][:-4]+'_crop.jpg', './test')
    #
    # support_data = square
    # support_box = np.array([new_x1, new_y1, new_x2, new_y2]).astype(np.float32)
    # return support_data, support_box


# def main():
#     dataDir = r'K:\Dataset\MS COCO 2014\coco\train2014'
#     target_path = r'K:\Dataset\Data_crop'
#     annFile = r'K:\Dataset\MS COCO 2014\annotations\instances_train2014.json'
#     # root_path = sys.argv[1]
#     # support_path = os.path.join(root_path, 'support')
#     # if not isdir(support_path):
#     #     mkdir(support_path)
#     # else:
#     #    shutil.rmtree(support_path)
#
#     support_dict = {}
#
#     support_dict['support_box'] = []
#     support_dict['category_id'] = []
#     support_dict['image_id'] = []
#     support_dict['id'] = []
#     support_dict['file_path'] = []
#
#     for dataType in ['COCO_train2014']:  # , 'train2017']:
#         set_crop_base_path = join(target_path, dataType)
#         # img_dataDir = join(dataDir, dataType)
#
#         # with open(annFile,'r') as load_f:
#         #     dataset = json.load(load_f)
#         #     print(dataset.keys())
#         #     save_info = dataset['info']
#         #     save_licenses = dataset['licenses']
#         #     save_images = dataset['images']
#         #     save_categories = dataset['categories']
#
#         coco = COCO(annFile)
#
#         for img_id, id in enumerate(coco.imgs):
#             if img_id % 100 == 0:
#                 print(img_id)
#                 if img_id == 10000:
#                     print("Processed CoCo 10000 images")
#                     return
#             img = coco.loadImgs(id)[0]
#             anns = coco.loadAnns(coco.getAnnIds(imgIds=id, iscrowd=None))
#
#             if len(anns) == 0:
#                 continue
#
#             # frame_crop_base_path = join(set_crop_base_path, img['file_name'].split('/')[-1].split('.')[0])
#             # if not isdir(frame_crop_base_path):
#             #     makedirs(frame_crop_base_path)
#             im = cv2.imread('{}/{}'.format(dataDir, img['file_name']))
#             # print('{}/{}'.format(img_dataDir, img['file_name']))
#             for item_id, ann in enumerate(anns):
#                 rect = ann['bbox']
#                 bbox = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]
#                 # support_img, support_box = crop_support(im, bbox)
#                 support_img = crop_support(im, bbox)
#                 if support_img is not None:
#                     '每张图片只裁剪一个目标，排除小于64*64的目标'
#                     cv2.imwrite(os.path.join(set_crop_base_path, img['file_name']), support_img)
#                     break
#                 else:
#                     continue
#     #             #im_name = img['file_name'].split('.')[0] + '_' + str(item_id) + '.jpg'
#     #             #output_dir = './fig'
#     #             #vis_image(support_img[:, :, ::-1], support_box, join(frame_crop_base_path, '{:04d}.jpg'.format(item_id)))
#     #             if rect[2] <= 0 or rect[3] <=0:
#     #                 print(rect)
#     #                 continue
#     #             file_path = join(frame_crop_base_path, '{:04d}.jpg'.format(item_id))
#     #             # cv2.imwrite(file_path, support_img)
#     #             #print(file_path)
#     #             support_dict['support_box'].append(support_box.tolist())
#     #             support_dict['category_id'].append(ann['category_id'])
#     #             support_dict['image_id'].append(ann['image_id'])
#     #             support_dict['id'].append(ann['id'])
#     #             support_dict['file_path'].append(file_path)
#     #
#     #     support_df = pd.DataFrame.from_dict(support_dict)
#     #
#     # return support_df
#     print("coco done!")


def main_defect():
    json_dict = {"info": ['none'], "license": ['none'], "images": [], "annotations": [], "categories": []}
    images = []
    annotations = []
    ann_count = 1


    # coco_few = COCO(few_annFile)
    coco = COCO(img_annFile)
    for img_id, id in enumerate(coco.imgs):


        img = coco.imgs[id]
        # img = coco.loadImgs(id)[0]

        lists = coco.imgToAnns[id]
        ann_ = [ann['id'] for ann in lists]
        # if ann_[0] in coco_few.anns:
        #     continue

        anns = [coco.anns[id] for id in ann_]
        # anns = coco.loadAnns(coco.getAnnIds(imgIds=id, iscrowd=None))
        if len(anns) == 0:
            continue

        im = cv2.imread('{}/{}'.format(img_dataDir, img['file_name']))
        for item_id, ann in enumerate(anns):
            rect = ann['bbox']
            bbox = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]] # xywh
            # support_img, support_box = crop_support(im, bbox)
            support_img = crop_support(im, bbox)
            if support_img is None:
                continue


            img_dict = {}
            img_new_id = f"support{int((time.time() - int(time.time())) * 10000000)}"
            img_dict['file_name'] = img['file_name'].split(".")[0] + f'_{ann_count}.png'
            img_dict['height'] = support_img.shape[0]
            img_dict['width'] = support_img.shape[1]
            img_dict['id'] = img_new_id
            images.append(img_dict)
            cv2.imwrite(os.path.join(target_path_img, img['file_name'].split(".")[0] + f'_{ann_count}.png'), support_img)

            ann_dict = {}
            ann_dict['area'] = int(support_img.shape[0] * support_img.shape[1])
            ann_dict['iscrowd'] = 0
            ann_dict['image_id'] = img_new_id
            ann_dict['bbox'] = [0, 0, support_img.shape[1], support_img.shape[0]]
            ann_dict['category_id'] = ann['category_id']
            ann_dict['id'] = ann_count
            ann_count += 1
            ann_dict['ignore'] = 0
            ann_dict['segmentation'] = []
            annotations.append(ann_dict)



    json_dict['images'] = images
    json_dict['annotations'] = annotations
    json_dict['categories'] = [context for idx, context in coco.cats.items()]

    json_fp = open(target_path_ann, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    print("defect done!")


if __name__ == '__main__':
    img_dataDir = r'D:\UserD\Li\FSCE-1\datasets\DiBei\image'
    img_annFile = r'D:\UserD\Li\FSCE-1\datasets\DiBei\annotations\instances_train.json'        # 555 train
    few_annFile = img_annFile
    # few_annFile = r'D:\UserD\Li\FSCE-1\datasets\my_dataset_555\annotations\few-shot\5way50shot.json'

    target_path_img = r'D:\UserD\Li\FSCE-1\datasets\DiBei\image_support'
    target_path_ann = r'D:\UserD\Li\FSCE-1\datasets\DiBei\annotations\few-shot\support.json'

    since = time.time()
    main_defect()
    time_elapsed = time.time() - since

    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

