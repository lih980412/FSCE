import os
import shutil
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET

# ID = 0
#
# '''数字底片'''
# if __name__ == "__main__":
#     xml_dirs = [r'F:\Dataset\数字底片\imagesWebsiteA\1round', r'F:\Dataset\数字底片\imagesWebsiteA\2crack',
#                 r'F:\Dataset\数字底片\imagesWebsiteA\3icf', r'F:\Dataset\数字底片\imagesWebsiteA\4lop',
#                 r'F:\Dataset\数字底片\imagesWebsiteA\5bar',
#                 r'F:\Dataset\数字底片\imagesWebsite\1round', r'F:\Dataset\数字底片\imagesWebsite\2crack',
#                 r'F:\Dataset\数字底片\imagesWebsite\3icf',r'F:\Dataset\数字底片\imagesWebsite\4lop',
#                 r'F:\Dataset\数字底片\imagesWebsite\5bar']
#     taget_dir = r"D:\UserD\Li\FSCE-1\datasets\my_dataset\image"
#
#     # 删除无效标注
#     for xml_dir in xml_dirs:
#         filename = os.listdir(xml_dir)
#         for file in filename:
#             if file.endswith(".xml"):
#                 tree = ET.parse(os.path.join(xml_dir, file))
#                 root = tree.getroot()
#                 if root.tag == "Edit_Img":
#                     os.remove(os.path.join(xml_dir, file))
#                     print("remove ann")
#
#     # 删除没有标注的图片
#     for xml_dir in xml_dirs:
#         filename = os.listdir(xml_dir)
#         for file in filename:
#             if file.endswith(".jpg"):
#                 name = file.split('.')[0] + ".xml"
#                 if not os.path.exists(os.path.join(xml_dir, name)):
#                     os.remove(os.path.join(xml_dir, file))
#                     print("remove pic")
#
#
#     # 删除没有图片的标注以及重命名
#     for xml_dir in xml_dirs:
#         filename = os.listdir(xml_dir)
#         for file in filename:
#             if file.endswith(".xml"):
#                 pic_name = file.split(".")[0] + ".jpg"
#                 pic_path = os.path.join(xml_dir, pic_name)
#                 if os.path.exists(pic_path):
#                     pic_newname = str(ID).zfill(6) + ".jpg"
#                     xml_newname = str(ID).zfill(6) + ".xml"
#                     os.rename(os.path.join(xml_dir, pic_name), os.path.join(xml_dir, pic_newname))
#                     os.rename(os.path.join(xml_dir, file), os.path.join(xml_dir, xml_newname))
#                     pic_path = os.path.join(xml_dir, pic_newname)
#                     shutil.copy(pic_path, taget_dir)
#                     ID += 1
#                 else:
#                     os.remove(os.path.join(xml_dir, file))
#                     print(111)
#     print(ID)





'''现场底片'''
if __name__ == "__main__":
    xml_dir = r'F:\Dataset\Weld Defects\work field\Annotations'

    # xml_dirs = [r'F:\Dataset\Weld Defects\web\noisy_aug\gauss&saltpepper_aug\Annotations',
    #             r'F:\Dataset\Weld Defects\web\geometric_aug\Rotate\Annotations',
    #             r'F:\Dataset\Weld Defects\web\geometric_aug\Flip\Annotations',
    #             r'F:\Dataset\Weld Defects\web\Original\Annotations']
    # jpg_dir = r'D:\UserD\Li\FSCE-1\datasets\my_dataset\image'
    # import glob
    # xml_list = []
    # for xml_dir in xml_dirs:
    #     temp = []
    #     temp = glob.glob(xml_dir + "/*.xml")
    #     xml_list += temp
    # xml_list = [xml.split("\\")[-1] for xml in xml_list]
    # j = 0
    # # 删除没有标注的图片
    # filename = os.listdir(jpg_dir)
    # for file in filename:
    #     if file.endswith(".png"):
    #         name = file.split('.')[0] + ".xml"
    #
    #         if name not in xml_list:
    #             os.remove(os.path.join(jpg_dir, file))
    #             j += 1
    #             print("remove pic " + str(j))

    jpg_dir = r'F:\Dataset\Weld Defects\work field\JPEGImages'
    taget_dir = r"D:\UserD\Li\FSCE-1\datasets\my_dataset_workfield\image"

    i = 0
    j = 0
    k = 0
    ID = 0
    dec_list = ["bar", "round", "icf", "crack", "lop"]

    # 删除没有标注的图片
    filename = os.listdir(jpg_dir)
    for file in filename:
        if file.endswith(".jpg"):
            name = file.split('.')[0] + ".xml"
            if not os.path.exists(os.path.join(xml_dir, name)):
                os.remove(os.path.join(xml_dir, file))
                j += 1
                print("remove pic " + str(j))


    filename = os.listdir(xml_dir)
    for file in filename:
        if file.endswith(".xml"):
            # 删除无效标注
            doc = minidom.parse(os.path.join(xml_dir, file))
            root_node = doc.documentElement
            if root_node.nodeName != "annotation":
                os.remove(os.path.join(xml_dir, file))
                i += 1
                print("remove ann " + str(i))
                continue
            flag = True
            for idx in range(len(root_node.getElementsByTagName('name'))):
                filename_node = root_node.getElementsByTagName('name')[idx]
                if filename_node.childNodes[0].data in dec_list:
                    flag = False
                    break
            if flag is True:
                os.remove(os.path.join(xml_dir, file))
                i += 1
                print("remove other ann " + str(i))
                continue

            # 删除没有图片的标注以及重命名
            pic_name = file.split(".")[0] + ".jpg"
            pic_path = os.path.join(jpg_dir, pic_name)
            if os.path.exists(pic_path):
                pic_newname = str(ID).zfill(6) + ".jpg"
                xml_newname = str(ID).zfill(6) + ".xml"
                os.rename(os.path.join(jpg_dir, pic_name), os.path.join(jpg_dir, pic_newname))
                os.rename(os.path.join(xml_dir, file), os.path.join(xml_dir, xml_newname))
                pic_path = os.path.join(jpg_dir, pic_newname)
                shutil.copy(pic_path, taget_dir)
                ID += 1
            else:
                k += 1
                print("remove xml " + str(k))
                os.remove(os.path.join(xml_dir, file))







# if __name__ == "__main__":
#     dir = r'F:\Dataset\Weld Defects\digital\imagesWebsiteA\4lop'
#     target_img = r'F:\Dataset\Weld Defects\web\add_lop\JPEGImages'
#     target_xml = r'F:\Dataset\Weld Defects\web\add_lop\Annotations'
#     files = os.listdir(dir)
#     count = 0
#     for file in files:
#         if file.endswith(".xml"):
#             if count == 200:
#                 break
#             else:
#                 xml_path = os.path.join(dir, file)
#                 jpg_path = os.path.join(dir, file.split(".")[0] + ".jpg")
#
#                 count += 1
#                 shutil.copy(xml_path, target_xml)
#                 shutil.copy(jpg_path, target_img)


'''
.jpg 转 .png
'''
# if __name__ == "__main__":
#     dir = r'D:\UserD\Li\FSCE-1\datasets\my_dataset\image'
#     files = os.listdir(dir)
#     count = 0
#     for file in files:
#         if file.endswith(".jpg"):
#             count += 1
#             os.rename(os.path.join(dir, file), os.path.join(dir, file.split(".")[0] + ".png"))
#
#     print("rename " + str(count) + " images")