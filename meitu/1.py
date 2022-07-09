# import torch
# import copy
# import os, cv2
# import numpy as np
# from PIL import ImageDraw
# import PIL.Image as Image
# from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
#
#
# def rand(a=0, b=1):
#     return np.random.rand() * (b - a) + a
#
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
#             # tmp_box_tensor = torch.stack(tmp_box)
#             merge_bbox.append(tmp_box)
#     return merge_bbox
#
#
# def get_random_data1(p_images, p_ann, hue=.1, sat=1.5, val=1.5):
#     '''random preprocessing for real-time data augmentation'''
#
#     for i in range(len(p_ann)):
#         # p_ann[i]["bbox"].append(p_ann[i]["category_id"])torch.cat([p_ann[i]["bbox"], p_ann[i]["category_id"]], dim=0)
#         # p_ann[i]["bbox"] = list(map(int, p_ann[i]["bbox"]))
#         p_ann[i]["bbox"] = torch.cat([p_ann[i]["bbox"], p_ann[i]["category_id"].unsqueeze(0).int()], dim=0)
#
#     j = 0
#     box = []
#     for i in range(0, len(p_images)):
#         temp = []
#
#         # for j in range(len(p_ann)):
#         #     if p_ann[j]["image_id"] == p_images[i]["image_id"]:
#         #         temp.append(p_ann[j]["bbox"])
#         #         temp_tensor = torch.stack(temp).cuda()
#
#         while j < len(p_ann) and p_ann[j]["image_id"] == p_images[i]["image_id"]:
#             temp.append(p_ann[j]["bbox"])
#             temp_tensor = torch.stack(temp).cuda()
#             # temp_tensor = numba.cuda.as_cuda_array(torch.stack(temp).cuda())
#             j += 1
#
#         box.append(temp_tensor)
#
#         ' box_data = np.zeros((len(box[i]), 5))'
#         ' box_data[:len(box[i])] = box[i]'
#         ' 可以这样处理'
#         # flag = 0
#         # if flag == 0:
#         #     box.append(temp_tensor)
#         #     box_tensor = torch.stack(box)
#         #     flag += 1
#         # else:
#         #     box = torch.cat([box_tensor, temp_tensor], dim=0)
#
#     h, w = (p_images[0]["height"], p_images[0]["width"])
#     min_offset_x = 0.4
#     min_offset_y = 0.4
#     scale_low = 1 - min(min_offset_x, min_offset_y)
#     scale_high = scale_low + 0.2
#     image_datas = []
#     box_datas = []
#     data = 0
#     place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]
#     place_y = [0, int(h * min_offset_y), int(w * min_offset_y), 0]
#
#     '这里慢'
#     # process_args = [[p_images[i], scale_low, scale_high, box[i], w, h, hue, sat, val, place_x, place_y, i] for i in range(4)]
#     # with Pool(4) as p:
#     #     output = p.map(process_data, process_args)
#     # p = Pool(4)
#     for i in range(0, 4):
#         # image_data, box_data = p.apply_async(process_data, args=(i, p_images, scale_low, scale_high, box, w, h, hue, sat, val, place_x, place_y, data))
#         # image_data, box_data = process_data(i, p_images, scale_low, scale_high, box, w, h, hue, sat, val, place_x, place_y, data)
#         # data = data + 1
#         # image_datas.append(image_data)
#         # box_datas.append(box_data)
#
#         # box[i][:, [2]] = box[i][:, [0]] + box[i][:, [2]]
#         # box[i][:, [3]] = box[i][:, [1]] + box[i][:, [3]]
#         image_name = p_images[i]["file_name"]
#         image = Image.open(image_name)
#         image = image.convert("RGB")
#         '''检查标注'''
#         # image1 = cv2.imread(image_name)
#         '标注为(x,y,w,h)格式'
#         # cv2.imshow("image"+str(i), image1)
#         # image1 = cv2.rectangle(image1, (p_ann[i]["bbox"][0], p_ann[i]["bbox"][1]),
#         #                        (p_ann[i]["bbox"][0] + p_ann[i]["bbox"][2], p_ann[i]["bbox"][1] + p_ann[i]["bbox"][3]),
#         #                        (0, 255, 255), 2)
#         '标注为(x,y,x,y)格式'
#         # for j in range(len(box[i])):
#         #     image1 = cv2.rectangle(image1, (box[i][j][0], box[i][j][1]), (box[i][j][2], box[i][j][3]), (255, 0, 255), 2)
#         # cv2.imshow("draw" + str(i), image1)
#
#         # 图片的大小
#         iw, ih = image.size
#         # 保存框的位置
#         # image.save(str(data)+".jpg")
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
#         dx = place_x[data]
#         dy = place_y[data]
#         new_image = Image.new('RGB', (w, h), (128, 128, 128))
#         new_image.paste(image, (dx, dy))
#         image_data = np.array(new_image) / 255
#         # Image.fromarray((image_data*255).astype(np.uint8)).save(str(data)+"distort.jpg")
#         data = data + 1
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
#             box[i] = box[i][torch.logical_and(box_w > 1, box_h > 1)]
#             box_data = torch.zeros((len(box[i]), 5))
#             box_data[:len(box[i])] = box[i]
#         image_datas.append(image_data)
#         box_datas.append(box_data)
#
#         '可视化 bbox'
#         # img = Image.fromarray((image_data * 255).astype(np.uint8))
#         # for j in range(len(box_data)):
#         #     thickness = 3
#         #     left, top, right, bottom = box_data[j][0:4]
#         #     draw = ImageDraw.Draw(img)
#         #     for i in range(thickness):
#         #         draw.rectangle([left + i, top + i, right - i, bottom - i], outline=(255, 255, 255))
#         # img.show()
#
#     # 将图片分割，放在一起
#     # cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
#     # cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))
#     cutx = torch.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)), (1,))
#     cuty = torch.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)), (1,))
#
#     new_image = np.zeros([h, w, 3])
#     new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
#     new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
#     new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
#     new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]
#     # Image.fromarray((new_image * 255).astype(np.uint8)).show()
#
#     # 对框进行进一步的处理
#     new_boxes = merge_bboxes(box_datas, cutx, cuty)
#
#     return new_image, new_boxes
#
#
#
# def Mosaic(data):
#     p_images = []
#     # p_images = typed.List.empty_list(types.float64)
#     p_ann = []
#     for i in range(len(data)):
#         temp_img = {}
#         temp_img["file_name"] = data[i]["file_name"]
#         temp_img["height"] = data[i]["height"]
#         temp_img["width"] = data[i]["width"]
#         temp_img["image_id"] = data[i]["image_id"]
#         p_images.append(temp_img)
#
#         for m in range(len(data[i]["instances"].get("gt_boxes").tensor)):
#             temp = {}
#             if len(data[i]["instances"].get("gt_boxes").tensor) == 1:
#                 # temp["bbox"] = (np.squeeze(np.asarray(data[i]["instances"].get("gt_boxes").tensor, dtype=int), 0)).tolist()
#                 # temp["category_id"] = data[i]["instances"].get('gt_classes').numpy()[0]
#                 temp["bbox"] = data[i]["instances"].get("gt_boxes").tensor.squeeze(0)
#                 temp["category_id"] = data[i]["instances"].get('gt_classes')[0]
#             else:
#                 # temp["bbox"] = (np.asarray(data[i]["instances"].get("gt_boxes").tensor[m], dtype=int)).tolist()
#                 # temp["category_id"] = data[i]["instances"].get('gt_classes').numpy()[m]
#                 temp["bbox"] = data[i]["instances"].get("gt_boxes").tensor[m]
#                 temp["category_id"] = data[i]["instances"].get('gt_classes')[m]
#
#             temp["image_id"] = data[i]["image_id"]
#             p_ann.append(temp)
#
#     mosaic_img, mosaic_ann = get_random_data1(p_images, p_ann)
#
#     # cv2.waitKey(0)
#     data_temp = copy.deepcopy(data[0])
#     data_temp["image"] = torch.as_tensor(mosaic_img, dtype=torch.float32, device="cpu").permute(2, 0, 1)
#     data_temp["width"] = 900
#     data_temp["height"] = 700
#     data_temp.pop('image_id')
#     data_temp.pop('file_name')
#     # mosaic_np = np.asarray(mosaic_ann, dtype=np.float32)
#     ann = [mosaic_ann[i][0:4] for i in range(len(mosaic_ann))]
#     # ann = torch.as_tensor(ann, dtype=torch.float32)
#     data_temp["instances"].get("gt_boxes").set_tensor(ann)
#
#     cls = torch.as_tensor([mosaic_ann[i][4] for i in range(len(mosaic_ann))], dtype=torch.int64)
#     data_temp["instances"].set("gt_classes", cls)
#     return data_temp
#
# # data =
import json

import numpy as np
import torch
from torch import nn
import os, cv2


# def get_numpy_word_embed(word2ix):
#     row = 0
#     file = 'glove.42B.300d.txt'
#     path = r'F:/'
#     whole = os.path.join(path, file)
#     words_embed = {}
#     with open(whole, mode='r', encoding="utf-8") as f:
#         lines = f.readlines()
#         for line in lines:
#             # print(line)
#             # print(len(line.split()))
#             line_list = line.split()
#             word = line_list[0]
#             embed = line_list[1:]
#             embed = [float(num) for num in embed]
#             words_embed[word] = embed
#             # if row > 20000:
#             #     break
#             row += 1
#
#     import json
#     jsObj = json.dumps(words_embed)
#     fileObject = open(r'F:\embedding.json', 'w')
#     fileObject.write(jsObj)
#     fileObject.close()
#
#
#
#     print(1)
#     # word2ix = {}
#     ix2word = {ix: w for w, ix in word2ix.items()}
#     id2emb = {}
#     for ix in range(len(word2ix)):
#         if ix2word[ix] in words_embed:
#             id2emb[ix] = words_embed[ix2word[ix]]
#         else:
#             id2emb[ix] = [0.0] * 100
#     data = [id2emb[ix] for ix in range(len(word2ix))]
#
#     return data
#
# word2id = "whyme?"
# numpy_embed = get_numpy_word_embed(word2id)
# import torch
# embedding = nn.Embedding.from_pretrained(torch.FloatTensor(numpy_embed)).to('cuda')


def get_numpy_word_embed(dict_file):
    row = 0
    words_embed = {}
    with open(dict_file, mode='r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split()
            word = line_list[0]
            embed = line_list[1:]
            try:
                embed = [float(num) for num in embed]
            except:
                print(line_list)
            words_embed[word] = embed
            # if row > 20000:
            #     break
            row += 1

    return words_embed


def crop_rect(img, rect):
    # get the parameter of the small rectangle
    # print("rect!")
    # print(rect)
    center, size, angle = rect[0], rect[1], rect[2]
    if size[1] > size[0]:
        size = size[::-1]
        angle -= 90
    if (angle > -45):
        center, size = tuple(map(int, center)), tuple(map(int, size))
        # angle-=270
        height, width = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D(center, angle, 1)
        # size = tuple([int(rect[1][1]), int(rect[1][0])])
        img_rot = cv2.warpAffine(img, M, (width, height))
        # cv2.imwrite("F:/wangjiao/rotate/img_rot.jpg", img_rot)
        img_crop = cv2.getRectSubPix(img_rot, size, center)
    else:
        center = tuple(map(int, center))
        size = tuple([int(rect[1][1]), int(rect[1][0])])
        angle -= 270
        height, width = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D(center, angle, 1)
        img_rot = cv2.warpAffine(img, M, (width, height))
        # cv2.imwrite("F:/wangjiao/rotate/img_rot.jpg", img_rot)
        img_crop = cv2.getRectSubPix(img_rot, size, center)
    return img_crop, img_rot


def affine_trans(ann, img):
    cnt = np.array(ann['bbox']).reshape([4, 1, 2])
    rect = cv2.minAreaRect(cnt)
    img_crop, img_rot = crop_rect(img, rect)
    return img_crop

# if __name__ == "__main__":
#     # data_ann_dir = r"F:\TextSeg_Release\annotation"
#     dict_file = r"F:\glove.840B.300d.txt"
#
#     # ann_files = os.listdir(data_ann_dir)
#     # model_data = {}
#     #
#     embed_dict = get_numpy_word_embed(dict_file)
#     b = json.dumps(embed_dict)
#     f2 = open(r'F:\glove.840B.300d.json', 'w')
#     f2.write(b)
#     f2.close()


if __name__ == "__main__":
    data_ann_dir = r"F:\TextSeg_Release\annotation"
    save_path = r"F:\TextSeg_Release\model_input"
    mask_path = r"F:\TextSeg_Release\annotation"
    mask_files = os.listdir(mask_path)
    img_path = r"F:\TextSeg_Release\semantic_label"
    img_files = os.listdir(img_path)

    # dict_file = r"F:\glove.840B.300d.json"
    dict_file_1 = r"F:\glove.42B.300d.json"
    my_dict_file = r"F:\1.json"

    f1 = open(my_dict_file, mode='r', encoding="utf-8")
    embed_dict1 = json.load(f1)
    with open(dict_file_1, mode='r', encoding="utf-8") as f:
        # lines = f.readlines()
        embed_dict = json.load(f)

        ### '''将在第一个词向量表中不存在的key存入txt，再读取txt在第二个词向量表中查找，结果存入json'''
        # with open(dict_file, mode='r', encoding="utf-8") as f:
        #     embed_dict = json.load(f)
        #     json1 = {}
        #
        #     f2 = open(r"F:/1.txt", "r", encoding="utf-8")
        #     lines = f2.readlines()
        #     for line in lines:
        #         # print(line)
        #         if line in embed_dict:
        #             json1[line] = embed_dict[line]
        #         # except:
        #         #     print(1)
        #     b = json.dumps(json1)
        #     f2 = open(r'F:/1.json', 'w')
        #     f2.write(b)
        #     f2.close()
        ann_files = os.listdir(data_ann_dir)
        for file in ann_files:
            if not file.endswith(".json"):
                continue
            del_anns = []
            word_embedding = np.zeros([1, 300])
            with open(os.path.join(data_ann_dir, file), "r", encoding="utf-8") as f:
                anns = json.load(f)

                for idx, ann in anns.items():
                    if '#' in ann['text']:
                        temp = '#'
                    elif '...' in ann['text']:
                        temp = '...'
                    elif len(ann['text']) > 1:
                        temp = ann["text"].lower().replace('!', '').replace('?', '').replace('#', '').replace('&', '') \
                            .replace('+', '').replace('\"', '').replace('\'', '').replace(',', '').replace('.', '')
                    else:
                        temp = ann["text"].lower()

                    if temp in embed_dict:
                        embed = embed_dict[temp]
                    elif temp in embed_dict1:
                        embed = embed_dict1[temp]
                    else:
                        del_anns.append(ann["bbox"])
                        continue
                    # for debug
                    # try:
                    #     embed = embed_dict[temp]
                    #     embed = embed_dict1[temp]
                    # except:
                    #     print(ann["text"].lower())
                    # embed = torch.tensor(numpy_embed)
                    # word_embedding = torch.stack([word_embedding.squeeze(), numpy_embed], dim=0)

                    numpy_embed = np.array(embed)
                    while len(numpy_embed) > 300:
                        numpy_embed = numpy_embed[1:]
                    word_embedding = np.concatenate([word_embedding, numpy_embed[np.newaxis, :]])

                if np.all(word_embedding == 0):
                    break
                word_embedding = word_embedding[1:]
                embedding_len = len(word_embedding)
                while len(word_embedding) < 20:
                    word_embedding = np.concatenate([word_embedding, np.zeros([1, 300])])

                img_file = os.path.join(img_path, file.split('_')[0] + "_maskfg.png")
                mask_file = os.path.join(mask_path, file.split('_')[0] + "_mask.png")
                img = cv2.imread(img_file, 0)
                # img = img[:, :, ::-1]
                mask = cv2.imread(mask_file, 0)
                mask[mask >= 1] = 1
                img_mask = img * mask
                img_mask[img_mask > 0] = 255

                # if del_anns:
                #     for del_ann in del_anns:
                #         mask_del = np.ones(img.shape)
                #         # del_ann 中偶数位取一最大最小是height，奇数位取一最大最小是width
                #         height_min, height_max = min(del_ann["bbox"][1::2]), max(del_ann["bbox"][1::2])
                #         width_min, width_max = min(del_ann["bbox"][::2]), max(del_ann["bbox"][::2])
                #         mask_del[height_min:height_max, width_min:width_max] = 0
                #         img_mask *= mask_del

                height_min, height_max, width_min, width_max = 30000, 0, 30000, 0
                element_img = np.zeros([64, 1])
                coords_seg_centre = np.zeros([1, 4])            # 名字是 centre 但是存的是 left、top、right、bottom
                ratio = img_mask.shape[0] / img_mask.shape[1]
                lt_x, rb_x = 0, 0
                for idx, ann in anns.items():
                    # del_ann 中偶数位取一最大最小是height，奇数位取一最大最小是 width
                    height_min_element, height_max_element = min(ann["bbox"][1::2]), max(ann["bbox"][1::2])
                    width_min_element, width_max_element = min(ann["bbox"][::2]), max(ann["bbox"][::2])

                    if ann in del_anns:
                        mask_del = np.ones(img.shape)
                        mask_del[height_min_element:height_max_element, width_min_element:width_max_element] = 0
                        img_mask *= mask_del

                    if ann not in del_anns:
                        # element = img_mask[height_min_element:height_max_element, width_min_element:width_max_element]
                        element = affine_trans(ann, img_mask)
                        element = cv2.resize(element, [int(64 / ratio), 64])
                        element_position = [lt_x, 64, rb_x+int(64 / ratio), 64]
                        lt_x = rb_x+int(64 / ratio)
                        rb_x = rb_x+int(64 / ratio)
                        coords = [lt_x, 0, rb_x, 64]
                        coords = np.array(coords)[np.newaxis, :]
                        coords_seg_centre = np.concatenate([coords_seg_centre, coords])
                        # element_positions.append(element_position)
                        element_img = np.concatenate([element_img, element], axis=1)
                        height_min = min(height_min, height_min_element)
                        height_max = max(height_max, height_max_element)
                        width_min = min(width_min, width_min_element)
                        width_max = max(width_max, width_max_element)
                # logo = img_mask[height_min:min(height_max+300, img_mask.shape[0]), width_min:width_max]
                logo = img_mask[height_min:min(int(height_max * 1.5), img_mask.shape[0]), width_min:width_max]

                logo_resized = cv2.resize(logo, [128, 128])
                element_img = element_img[:, 1:]
                coords_seg_centre = coords_seg_centre[1:]

                if element_img.shape[1] <= 1280:
                    sub = 1280 - element_img.shape[1]
                    element_img = np.concatenate([element_img, np.zeros([64, sub])], axis=1)
                elif element_img.shape[1] > 1280:
                    element_img = cv2.resize(element_img, [1280, 64])
                    ratio_element = int(element_img.shape[1] / 1280)
                    coords_seg_centre = coords_seg_centre[:, ::2] / (ratio_element + np.array(1e-7))
                element_img[element_img > 0] = 255

                if not os.path.exists(os.path.join(save_path, file.split('_')[0])):
                    os.mkdir(os.path.join(save_path, file.split('_')[0]))
                np.save(os.path.join(save_path, file.split('_')[0], "coords_seg_centre.npy"), coords_seg_centre)
                np.save(os.path.join(save_path, file.split('_')[0], "word_embeds.npy"), word_embedding)
                np.save(os.path.join(save_path, file.split('_')[0], "len.npy"), np.array(embedding_len))
                cv2.imwrite(os.path.join(save_path, file.split('_')[0], "logo_resized.png"), logo_resized)
                cv2.imwrite(os.path.join(save_path, file.split('_')[0], "elements.png"), element_img)

                print(f"done {file}!")
