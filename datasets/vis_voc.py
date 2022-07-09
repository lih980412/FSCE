import os
import cv2
import re

pattens = ['name', 'xmin', 'ymin', 'xmax', 'ymax']


def get_annotations(xml_path):
    bbox = []

    with open(xml_path, 'r', encoding='utf-8') as f:
        text = f.read().replace('\n', 'return')
        p1 = re.compile(r'(?<=<object>)(.*?)(?=</object>)')
        result = p1.findall(text)
        for obj in result:
            tmp = []
            for patten in pattens:
                p = re.compile(r'(?<=<{}>)(.*?)(?=</{}>)'.format(patten, patten))
                if patten == 'name':
                    tmp.append(p.findall(obj)[0])
                else:
                    tmp.append(int(float(p.findall(obj)[0])))
            bbox.append(tmp)
    return bbox


def save_viz_image(image_path, xml_path, cat_num, save_path=None):
    bbox = get_annotations(xml_path)
    if bbox[0][0] in cat_num:
        cat_num[bbox[0][0]] += 1
    else:
        cat_num[bbox[0][0]] = 0
    'vis ann'
    # image = cv2.imread(image_path)
    # for info in bbox:
    #     print(info)
    #     cv2.rectangle(image, (info[1], info[2]), (info[3], info[4]), (255, 0, 0), thickness=2)
    #     cv2.putText(image, info[0], (info[1], info[2]), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255), 1)
    # cv2.imshow("im", image)
    'analyze ann'
    print(cat_num)
    # cv2.waitKey(0)
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # cv2.imwrite(os.path.join(save_path, os.path.split(image_path)[-1]), image)


if __name__ == '__main__':
    image_dir = r'F:\Dataset\ReZha\image'
    xml_dir = r'F:\Dataset\ReZha\annotations'
    # save_dir = 'viz_images'
    # viz_num = 10
    image_list = os.listdir(image_dir)
    cat_num = {}
    cnt = 0
    for i in image_list:
            # cnt += 1

        image_path = os.path.join(image_dir, i)
        xml_path = os.path.join(xml_dir, os.path.splitext(i)[0] + '.xml')
        save_viz_image(image_path, xml_path, cat_num)