import os
import cv2
import json



def find_id_img(json_images):
    dict = {}
    for i in range(len(json_images)):
        dict[json_images[i]["file_name"]] = json_images[i]["id"]
    return dict

# view outlier
if __name__ == "__main__":
    jpg_dir = [r'D:\UserD\Li\FSCE-1\datasets\my_dataset_before\image\004425.jpg',
               r'D:\UserD\Li\FSCE-1\datasets\my_dataset_before\image\004705.jpg',
               r'D:\UserD\Li\FSCE-1\datasets\my_dataset_before\image\004546.jpg',
               r'D:\UserD\Li\FSCE-1\datasets\my_dataset_before\image\004809.jpg',
               r'D:\UserD\Li\FSCE-1\datasets\my_dataset_before\image\004036.jpg',
               r'D:\UserD\Li\FSCE-1\datasets\my_dataset_before\image\004548.jpg',
               r'D:\UserD\Li\FSCE-1\datasets\my_dataset_before\image\004452.jpg',
               r'D:\UserD\Li\FSCE-1\datasets\my_dataset_before\image\005023.jpg',
               r'D:\UserD\Li\FSCE-1\datasets\my_dataset_before\image\004949.jpg',
               r'D:\UserD\Li\FSCE-1\datasets\my_dataset_before\image\004998.jpg',
               r'D:\UserD\Li\FSCE-1\datasets\my_dataset_before\image\006781.jpg',
               r'D:\UserD\Li\FSCE-1\datasets\my_dataset_before\image\004295.jpg',
               r'D:\UserD\Li\FSCE-1\datasets\my_dataset_before\image\004426.jpg',
               # r'D:\UserD\Li\FSCE-1\datasets\my_dataset_before\image\004295.jpg',
               r'D:\UserD\Li\FSCE-1\datasets\my_dataset_before\image\004677.jpg',
               r'D:\UserD\Li\FSCE-1\datasets\my_dataset_before\image\006809.jpg',
               r'D:\UserD\Li\FSCE-1\datasets\my_dataset_before\image\004323.jpg',
               r'D:\UserD\Li\FSCE-1\datasets\my_dataset_before\image\004675.jpg',
               # r'D:\UserD\Li\FSCE-1\datasets\my_dataset_before\image\004323.jpg',
               ]
    ann_file = r'D:\UserD\Li\FSCE-1\datasets\my_dataset_before\annotations\instances_train.json'

    json_file = json.load(open(ann_file))
    img2id = find_id_img(json_file["images"])
    for jpg in jpg_dir:
        jpg_name = jpg.split("\\")[-1]
        jpg_id = img2id[str(jpg_name)]
        img = cv2.imread(jpg)
        for index in range(len(json_file["annotations"])):
            if json_file["annotations"][index]["image_id"] == jpg_id:
                while json_file["annotations"][index]["image_id"] == jpg_id:
                    bbox = json_file["annotations"][index]["bbox"]
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                                  (0, 0, 255), 1)
                    print(json_file["annotations"][index]["category_id"])
                    print(str(bbox[2] * bbox[3]))
                    break
                cv2.imshow("img", img)
                cv2.waitKey(0)

# # view anno
# if __name__ == "__main__":
#     ann_file = r'D:\UserD\Li\FSCE-1\datasets\my_dataset_before\annotations\instances_val.json'
#     jpg_dir = r'D:\UserD\Li\FSCE-1\datasets\my_dataset_before\image'
#
#
#     with open(ann_file, "r") as f:
#         content = f.read()
#         a = json.loads(content)
#         index = 0
#         for id in range(0, 555):
#             jpg_name = a["images"][id]["file_name"]
#             img = cv2.imread(os.path.join(jpg_dir, jpg_name))
#             jpg_id = a["images"][id]["id"]
#
#             while a["annotations"][index]["image_id"] == jpg_id:
#                 bbox = a["annotations"][index]["bbox"]
#                 cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0, 0, 255), 1)
#                 print(a["annotations"][index]["category_id"])
#                 print(str(bbox[2]*bbox[3]))
#                 index += 1
#
#             print(jpg_name)
#
#             cv2.imshow("img", img)
#             cv2.waitKey(0)


# import cv2
# import xml.dom.minidom
#
# if __name__ == "__main__":
#     for id in range(0, 753):
#         xml_path = r'F:\Dataset\work field\BeforeAugXml\\' + str(id).zfill(6) + ".xml"
#         jpg_path = r'F:\Dataset\work field\BeforeAugJpg\\' + str(id).zfill(6) + ".jpg"
#         # 打开xml文档
#         dom = xml.dom.minidom.parse(xml_path)
#         # 得到文档元素对象
#         root = dom.documentElement
#
#         xmin = root.getElementsByTagName('xmin')
#         xmax = root.getElementsByTagName('xmax')
#         ymin = root.getElementsByTagName('ymin')
#         ymax = root.getElementsByTagName('ymax')
#         print(len(xmin))
#         img = cv2.imread(jpg_path)
#         for j in range(len(xmin)):
#             xmin_value = xmin[j].childNodes[0].nodeValue
#             xmax_value = xmax[j].childNodes[0].nodeValue
#             ymin_value = ymin[j].childNodes[0].nodeValue
#             ymax_value = ymax[j].childNodes[0].nodeValue
#             cv2.rectangle(img, (int(xmin_value), int(ymin_value)), (int(xmax_value), int(ymax_value)), (0, 0, 255), 1)
#         cv2.imshow("img", img)
#         cv2.waitKey(0)

# import torch
# ckpt = torch.load(r"D:\UserD\Li\FSCE-1\checkpoints\mydataset\baseline\R_101_FPN_baseline\model.pth")
# ckpt["iteration"] = 0


