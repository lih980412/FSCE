import copy

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math, os
import pandas as pd
import torch


def nju(img, n):
    ave = img.mean()
    s = 0
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])  # 直方图
    N = sum(hist)
    hist = hist / N[0]
    '图像关于灰度均值ave的n阶矩'
    for i in range(len(hist)):
        temp = cv2.pow(i - ave, n)
        s += hist[i][0] * temp[0][0]
        # print('s=%f' % (s))
    return s


def others(img):
    # ave = img.mean()
    su = 0
    se = 0
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])  # 直方图
    N = sum(hist)
    hist = hist / N[0]
    for i in range(len(hist)):
        su += hist[i][0] * hist[i][0]
        if hist[i][0] != 0:
            se += hist[i][0] * math.log(hist[i][0], 2)
    return su, -se


def maxGrayLevel(img):
    max_gray_level = 0
    (height, width) = img.shape
    # print(height, width)
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    return max_gray_level + 1

def feature_computer(p, gray_level):
    Con = 0.0
    Ent = 0.0
    Asm = 0.0
    Idm = 0.0
    for i in range(gray_level):
        for j in range(gray_level):
            '对比度'
            Con += (i - j) * (i - j) * p[i][j]
            '能量'
            Asm += p[i][j] * p[i][j]
            '逆差距'
            Idm += p[i][j] / (1 + (i - j) * (i - j))
            '熵'
            if p[i][j] > 0.0:
                Ent += p[i][j] * math.log(p[i][j])
            '相关性'
            # ui += i * p[i][j]
    return Asm, Con, -Ent, Idm

def getGlcm(input, d_x, d_y ,gray_level):
    srcdata = copy.deepcopy(input)
    ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height, width) = input.shape

    '最大像素值'
    max_gray_level = maxGrayLevel(input)

    # 若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
    '正则化至gray_level'
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i] * gray_level / max_gray_level

    '灰度共生矩阵'
    for j in range(height - d_y):
        for i in range(width - d_x):
            rows = srcdata[j][i]
            cols = srcdata[j + d_y][i + d_x]
            ret[rows][cols] += 1.0

    '灰度共生矩阵正则化'
    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j] /= float(height * width)

    return ret


def cedu(img_path, name):

    print(f"统计总体 {name} 的测度值--------------------------------------------------")
    ave, std, ju3, u_total, e_total = 0, 0, 0, 0, 0
    number = len(os.listdir(img_path))
    time = 0
    for file_name in os.listdir(img_path):
        if time % 100 == 0:
            print(time)
        img = cv2.imread(os.path.join(img_path, file_name), 0)
        '均值'
        ave += img.mean()  # 均值
        '标准差'
        std += math.sqrt(nju(img, 2))  # 标准差
        # r = 1 - 1 / (1 + std * std)
        # r = r / (255 * 255)  # 归一化
        '三阶矩'
        ju3 += nju(img, 3) / (255 * 255)  # 归一化
        '一致性、图像熵'
        u, e = others(img)
        u_total += u
        e_total += e
        time += 1
    return format(ave / number, '.3f'), format(std / number, '.3f'), format(ju3 / number, '.3f'), format(
        u_total / number, '.3f'), format(e_total / number, '.3f')
    # return format(ave/number, '.3f'), std/number, ju3/number, u_total/number, e_total/number

def cedu_instance(img_path):
    print(f"按目标统计 {img_path} 的测度值--------------------------------------------------")
    img = cv2.imread(img_path, 0)

    '均值'
    '标准差'
    '三阶矩'
    '一致性、图像熵'
    u, e = others(img)

    return img.mean(), math.sqrt(nju(img, 2)), nju(img, 3) / (255 * 255), u, e


def compute_instance(img_path, gray_level):
    print(f"按目标统计 {img_path} 的灰度共生矩阵相关值--------------------------------------------------")
    img = cv2.imread(img_path, 0)

    glcm_0 = getGlcm(img, 1, 0, gray_level)
    # glcm_1 = getGlcm(img, 0, 1, gray_level)
    # glcm_2 = getGlcm(img, 1, 1, gray_level)
    # # glcm_3 = getGlcm(img, -1, 1)
    # TODO 相关性
    asm_0, con_0, ent_0, idm_0 = feature_computer(glcm_0, gray_level)
    # asm_1, con_1, ent_1, idm_1 = feature_computer(glcm_1, gray_level)
    # asm_2, con_2, ent_2, idm_2 = feature_computer(glcm_2, gray_level)
    # # asm_3, con_3, ent_3, idm_3 = feature_computer(glcm_3)

    return asm_0, con_0, ent_0, idm_0
    # return [format(asm_0_total / number, '.3f'), format(con_0_total / number, '.3f'), format(ent_0_total / number, '.3f'), format(idm_0_total / number, '.3f')], \
    #        [asm_1_total, con_1_total, ent_1_total, idm_1_total], [asm_2_total, con_2_total, ent_2_total, idm_2_total]
    # # print(asm, con, eng, idm)
    # print(f"灰度共生矩阵_00的能量为：{asm_0:.5f}，对比度为{con_0:.5f}，熵为{ent_0:.5f}，相关性为_，逆差距为{idm_0:.5f}")
    # print(f"灰度共生矩阵_45的能量为：{asm_1:.5f}，对比度为{con_1:.5f}，熵为{ent_1:.5f}，相关性为_，逆差距为{idm_1:.5f}")
    # print(f"灰度共生矩阵_90的能量为：{asm_2:.5f}，对比度为{con_2:.5f}，熵为{ent_2:.5f}，相关性为_，逆差距为{idm_2:.5f}")
    # # print(f"{name} glcm_3的能量为：{asm_3:.5f}，对比度为{con_3:.5f}，熵为{ent_3:.5f}，相关性为_，逆差距为{idm_3:.5f}")

def compute(img_path, gray_level, name):
    print(f"统计总体 {name} 的灰度共生矩阵相关值--------------------------------------------------")
    number = len(os.listdir(img_path))
    asm_0_total, con_0_total, ent_0_total, idm_0_total = 0, 0, 0, 0
    asm_1_total, con_1_total, ent_1_total, idm_1_total = 0, 0, 0, 0
    asm_2_total, con_2_total, ent_2_total, idm_2_total = 0, 0, 0, 0

    time = 0
    for file_name in os.listdir(img_path):

        if time % 100 == 0:
            print(time)
        # img = cv2.imread(path, 0)
        # try:
        #     img_shape = img.shape
        # except:
        #     print('imread error')
        #     return -1
        img = cv2.imread(os.path.join(img_path, file_name), 0)
        # img = cv2.resize(img, (img_shape[1] / 2, img_shape[0] / 2), interpolation=cv2.INTER_CUBIC)
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        glcm_0 = getGlcm(img, 1, 0, gray_level)
        # glcm_1 = getGlcm(img, 0, 1, gray_level)
        # glcm_2 = getGlcm(img, 1, 1, gray_level)
        # # glcm_3 = getGlcm(img, -1, 1)
        # TODO 相关性
        asm_0, con_0, ent_0, idm_0 = feature_computer(glcm_0, gray_level)
        # asm_1, con_1, ent_1, idm_1 = feature_computer(glcm_1, gray_level)
        # asm_2, con_2, ent_2, idm_2 = feature_computer(glcm_2, gray_level)
        # # asm_3, con_3, ent_3, idm_3 = feature_computer(glcm_3)

        asm_0_total += asm_0
        con_0_total += con_0
        ent_0_total += ent_0
        idm_0_total += idm_0

        time += 1
    asm_0_total /= number
    con_0_total /= number
    ent_0_total /= number
    idm_0_total /= number

    return [asm_0_total, con_0_total, ent_0_total, idm_0_total], 0, 0
    # return [format(asm_0_total / number, '.3f'), format(con_0_total / number, '.3f'), format(ent_0_total / number, '.3f'), format(idm_0_total / number, '.3f')], \
    #        [asm_1_total, con_1_total, ent_1_total, idm_1_total], [asm_2_total, con_2_total, ent_2_total, idm_2_total]
    # # print(asm, con, eng, idm)
    # print(f"灰度共生矩阵_00的能量为：{asm_0:.5f}，对比度为{con_0:.5f}，熵为{ent_0:.5f}，相关性为_，逆差距为{idm_0:.5f}")
    # print(f"灰度共生矩阵_45的能量为：{asm_1:.5f}，对比度为{con_1:.5f}，熵为{ent_1:.5f}，相关性为_，逆差距为{idm_1:.5f}")
    # print(f"灰度共生矩阵_90的能量为：{asm_2:.5f}，对比度为{con_2:.5f}，熵为{ent_2:.5f}，相关性为_，逆差距为{idm_2:.5f}")
    # # print(f"{name} glcm_3的能量为：{asm_3:.5f}，对比度为{con_3:.5f}，熵为{ent_3:.5f}，相关性为_，逆差距为{idm_3:.5f}")



if __name__ == "__main__":
    coco_path = r"D:\UserD\Li\FSCE-1\tools\Static\Croped\COCO2014_train"
    defect_path = r"D:\UserD\Li\FSCE-1\tools\Static\Croped\DiPian_tarin"
    moli_path = r"D:\UserD\Li\FSCE-1\tools\Static\Croped\MoLi_train"
    csv_path = r"D:\UserD\Li\FSCE-1\tools\Static\Croped\result.csv"
    name_coco = coco_path.split("\\")[-1]
    name_defect = defect_path.split("\\")[-1]
    name_moli = moli_path.split("\\")[-1]

    '总体平均'
    # 均值，标准差，三阶矩，一致性，图像熵
    ave_coco, std_coco, ju3_coco, u_coco, e_coco = cedu(coco_path, name_coco)
    ave_defect, std_defect, ju3_defect, u_defect, e_defect = cedu(defect_path, name_defect)
    ave_moli, std_moli, ju3_moli, u_moli, e_moli = cedu(moli_path, name_moli)

    # 灰度共生矩阵结果
    gray_level = 32
    ans_coco_00, ans_coco_90, ans_coco_45 = compute(coco_path, gray_level, name_coco)
    ans_defect_00, ans_defect_90, ans_defect_45 = compute(defect_path, gray_level, name_defect)
    ans_moli_00, ans_moli_90, ans_moli_45 = compute(moli_path, gray_level, name_moli)


    df = pd.DataFrame([[name_coco, ave_coco, std_coco, ju3_coco, u_coco, e_coco, 0, ans_coco_00[0], ans_coco_00[1], ans_coco_00[2], ans_coco_00[3]],
                       [name_defect, ave_defect, std_defect, ju3_defect, u_defect, e_defect, 0, ans_defect_00[0], ans_defect_00[1], ans_defect_00[2], ans_defect_00[3]],
                       [name_moli, ave_moli, std_moli, ju3_moli, u_moli, e_moli, 0, ans_moli_00[0], ans_moli_00[1], ans_moli_00[2], ans_moli_00[3]]],
                      columns=['name', '均值', '标准差', '三阶矩', '一致性', '图像熵', '灰度共生矩阵结果', '能量', '对比度', '熵', '逆差距'])
    df.to_csv(csv_path)

    '一行一行添加'
    # gray_level = 32
    # df = pd.DataFrame(columns=['name', '均值', '标准差', '三阶矩', '一致性', '图像熵', '灰度共生矩阵结果', '能量', '对比度', '熵', '逆差距'])
    #
    # row = 0
    # coco_files = os.listdir(coco_path)
    # for file in coco_files:
    #
    #     file_path = os.path.join(coco_path, file)
    #     ave_coco, std_coco, ju3_coco, u_coco, e_coco = cedu_instance(file_path)
    #     asm_coco, con_coco, ent_coco, idm_coco = compute_instance(file_path, gray_level)
    #
    #     # 输出csv
    #     df.loc[row] = [name_coco, ave_coco, std_coco, ju3_coco, u_coco, e_coco, 0, asm_coco, con_coco, ent_coco, idm_coco]  # 其中loc[]中需要加入的是插入地方dataframe的索引，默认是整数型
    #     row += 1
    #
    # defect_files = os.listdir(defect_path)
    # for file in defect_files:
    #
    #     file_path = os.path.join(defect_path, file)
    #     ave_defect, std_defect, ju3_defect, u_defect, e_defect = cedu_instance(file_path)
    #     asm_defect, con_defect, ent_defect, idm_defect = compute_instance(file_path, gray_level)
    #
    #     # 输出csv
    #     df.loc[row] = [name_defect, ave_defect, std_defect, ju3_defect, u_defect, e_defect, 0, asm_defect, con_defect, ent_defect, idm_defect]
    #     row += 1


    # df.to_csv(csv_path)