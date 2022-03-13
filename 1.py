import torch
import copy
import os, cv2
import numpy as np
from PIL import ImageDraw
import PIL.Image as Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a

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

# data =