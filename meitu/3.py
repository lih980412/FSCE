
import cv2, json
import numpy as np


def main():
    img = cv2.imread(r"F:\TextSeg_Release\semantic_label\a00017_maskfg.png")
    cnt = np.array([
        [[215, 381]],
        [[1109, 452]],
        [[1102, 532]],
        [[209, 461]]
    ])
    # cnt = np.array([
    #         [[281, 573]],
    #         [[1054, 498]],
    #         [[1061, 566]],
    #         [[287, 641]]
    #     ])

    file = r"F:\TextSeg_Release\annotation\a00017_anno.json"
    with open(file, "r", encoding="utf-8") as f:
        anns = json.load(f)
        for idx, ann in anns.items():
            cnt = np.array(ann['bbox']).reshape([4, 1, 2])


            # cv2.drawContours(img, [cnt], 0, (0, 255, 255), 2)
            rect = cv2.minAreaRect(cnt)
            # print("rect: {}".format(rect))

            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

            # img_crop will the cropped rectangle, img_rot is the rotated image
            img_crop, img_rot = crop_rect(img, rect)
            # cv2.imwrite("F:\wangjiao/rotate/img.jpg", img)
            # cv2.imwrite("F:\wangjiao/rotate/cropped_img.jpg", img_crop)
            # cv2.imshow("origin", img)
            cv2.imshow("img_crop", img_crop)
            cv2.waitKey(0)


def crop_rect(img, rect):
    # get the parameter of the small rectangle
    print("rect!")
    print(rect)
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


if __name__ == "__main__":
    main()