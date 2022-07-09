# https://blog.csdn.net/lk3030/article/details/84034963
import cv2
import numpy as np
import matplotlib.pyplot as plt


def origin_LBP(img):
    dst = np.zeros(img.shape, dtype=img.dtype)
    h, w = img.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center = img[i][j]
            code = 0

            code |= (img[i - 1][j - 1] >= center) << (np.uint8)(7)
            code |= (img[i - 1][j] >= center) << (np.uint8)(6)
            code |= (img[i - 1][j + 1] >= center) << (np.uint8)(5)
            code |= (img[i][j + 1] >= center) << (np.uint8)(4)
            code |= (img[i + 1][j + 1] >= center) << (np.uint8)(3)
            code |= (img[i + 1][j] >= center) << (np.uint8)(2)
            code |= (img[i + 1][j - 1] >= center) << (np.uint8)(1)
            code |= (img[i][j - 1] >= center) << (np.uint8)(0)

            dst[i - 1][j - 1] = code
    return dst


def circular_LBP(img, radius=3, neighbors=8):
    h, w = img.shape
    dst = np.zeros((h - 2 * radius, w - 2 * radius), dtype=np.float32)
    for k in range(neighbors):
        # 计算采样点对于中心点坐标的偏移量rx，ry
        rx = radius * np.cos(2.0 * np.pi * k / neighbors)
        ry = -(radius * np.sin(2.0 * np.pi * k / neighbors))
        # 为双线性插值做准备
        # 对采样点偏移量分别进行上下取整
        x1 = int(np.floor(rx))
        x2 = int(np.ceil(rx))
        y1 = int(np.floor(ry))
        y2 = int(np.ceil(ry))
        # 将坐标偏移量映射到0-1之间
        tx = rx - x1
        ty = ry - y1
        # 根据0-1之间的x，y的权重计算公式计算权重，权重与坐标具体位置无关，与坐标间的差值有关
        w1 = (1 - tx) * (1 - ty)
        w2 = tx * (1 - ty)
        w3 = (1 - tx) * ty
        w4 = tx * ty
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                # 获得中心像素点的灰度值
                center = img[i, j]
                # 根据双线性插值公式计算第k个采样点的灰度值
                neighbor = img[i + y1, j + x1] * w1 + img[i + y2, j + x1] * w2 + img[i + y1, j + x2] * w3 + img[
                    i + y2, j + x2] * w4
                # LBP特征图像的每个邻居的LBP值累加，累加通过与操作完成，对应的LBP值通过移位取得
                dst[i - radius, j - radius] |= (neighbor > center) << (np.uint8)(neighbors - k - 1)
    return dst


def rotation_invariant_LBP(img, radius=3, neighbors=8):
    h, w = img.shape
    dst = np.zeros((h - 2 * radius, w - 2 * radius), dtype=img.dtype)
    for k in range(neighbors):
        # 计算采样点对于中心点坐标的偏移量rx，ry
        rx = radius * np.cos(2.0 * np.pi * k / neighbors)
        ry = -(radius * np.sin(2.0 * np.pi * k / neighbors))
        # 为双线性插值做准备
        # 对采样点偏移量分别进行上下取整
        x1 = int(np.floor(rx))
        x2 = int(np.ceil(rx))
        y1 = int(np.floor(ry))
        y2 = int(np.ceil(ry))
        # 将坐标偏移量映射到0-1之间
        tx = rx - x1
        ty = ry - y1
        # 根据0-1之间的x，y的权重计算公式计算权重，权重与坐标具体位置无关，与坐标间的差值有关
        w1 = (1 - tx) * (1 - ty)
        w2 = tx * (1 - ty)
        w3 = (1 - tx) * ty
        w4 = tx * ty
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                # 获得中心像素点的灰度值
                center = img[i, j]
                # 根据双线性插值公式计算第k个采样点的灰度值
                neighbor = img[i + y1, j + x1] * w1 + img[i + y2, j + x1] * w2 + img[i + y1, j + x2] * w3 + img[
                    i + y2, j + x2] * w4
                # LBP特征图像的每个邻居的LBP值累加，累加通过与操作完成，对应的LBP值通过移位取得
                dst[i - radius, j - radius] |= (neighbor > center) << (np.uint8)(neighbors - k - 1)
    # 进行旋转不变处理
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            currentValue = dst[i, j]
            minValue = currentValue
            for k in range(1, neighbors):
                # 循环左移
                temp = (np.uint8)(currentValue >> (neighbors - k)) | (np.uint8)(currentValue << k)
                if temp < minValue:
                    minValue = temp

            dst[i, j] = minValue

    return dst


def uniform_pattern_LBP(img, radius=3, neighbors=8):
    h, w = img.shape
    dst = np.zeros((h - 2 * radius, w - 2 * radius), dtype=img.dtype)
    # LBP特征值对应图像灰度编码表，直接默认采样点为8位
    temp = 1
    table = np.zeros((256), dtype=img.dtype)
    for i in range(256):
        if getHopTimes(i) < 3:
            table[i] = temp
            temp += 1
    # 是否进行UniformPattern编码的标志
    flag = False
    # 计算LBP特征图
    for k in range(neighbors):
        if k == neighbors - 1:
            flag = True

        # 计算采样点对于中心点坐标的偏移量rx，ry
        rx = radius * np.cos(2.0 * np.pi * k / neighbors)
        ry = -(radius * np.sin(2.0 * np.pi * k / neighbors))
        # 为双线性插值做准备
        # 对采样点偏移量分别进行上下取整
        x1 = int(np.floor(rx))
        x2 = int(np.ceil(rx))
        y1 = int(np.floor(ry))
        y2 = int(np.ceil(ry))
        # 将坐标偏移量映射到0-1之间
        tx = rx - x1
        ty = ry - y1
        # 根据0-1之间的x，y的权重计算公式计算权重，权重与坐标具体位置无关，与坐标间的差值有关
        w1 = (1 - tx) * (1 - ty)
        w2 = tx * (1 - ty)
        w3 = (1 - tx) * ty
        w4 = tx * ty
        # 循环处理每个像素
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                # 获得中心像素点的灰度值
                center = img[i, j]
                # 根据双线性插值公式计算第k个采样点的灰度值
                neighbor = img[i + y1, j + x1] * w1 + img[i + y2, j + x1] * w2 + img[i + y1, j + x2] * w3 + img[
                    i + y2, j + x2] * w4
                # LBP特征图像的每个邻居的LBP值累加，累加通过与操作完成，对应的LBP值通过移位取得
                dst[i - radius, j - radius] |= (neighbor > center) << (np.uint8)(neighbors - k - 1)
                # 进行LBP特征的UniformPattern编码
                if flag:
                    dst[i - radius, j - radius] = table[dst[i - radius, j - radius]]
    return dst


def getHopTimes(data):
    '''
    计算跳变次数
    '''
    count = 0
    binaryCode = "{0:0>8b}".format(data)

    for i in range(1, len(binaryCode)):
        if binaryCode[i] != binaryCode[(i - 1)]:
            count += 1
    return count


def multi_scale_block_LBP(img, scale):
    h, w = img.shape

    # 定义并计算积分图像
    cellSize = int(scale / 3)
    offset = int(cellSize / 2)
    cellImage = np.zeros((h - 2 * offset, w - 2 * offset), dtype=img.dtype)

    for i in range(offset, h - offset):
        for j in range(offset, w - offset):
            temp = 0
            for m in range(-offset, offset + 1):
                for n in range(-offset, offset + 1):
                    temp += img[i + n, j + m]

            temp /= (cellSize * cellSize)
            cellImage[i - int(cellSize / 2), j - int(cellSize / 2)] = np.uint8(temp)
    cv2.imshow("cellImage", cellImage)
    dst = origin_LBP(cellImage)
    return dst


def getLBPH(img_lbp, numPatterns, grid_x, grid_y, normed):
    '''
    计算LBP特征图像的直方图LBPH
    '''
    h, w = img_lbp.shape
    width = int(w / grid_x)
    height = int(h / grid_y)
    # 定义LBPH的行和列，grid_x*grid_y表示将图像分割的块数，numPatterns表示LBP值的模式种类
    result = np.zeros((grid_x * grid_y, numPatterns), dtype=float)
    resultRowIndex = 0
    # 对图像进行分割，分割成grid_x*grid_y块，grid_x，grid_y默认为8
    for i in range(grid_x):
        for j in range(grid_y):
            # 图像分块
            src_cell = img_lbp[i * height:(i + 1) * height, j * width:(j + 1) * width]
            # 计算直方图
            hist_cell = getLocalRegionLBPH(src_cell, 0, (numPatterns - 1), True)
            # 将直方图放到result中
            result[resultRowIndex] = hist_cell
            resultRowIndex += 1
    return np.reshape(result, (-1))


def getLocalRegionLBPH(src, minValue, maxValue, normed):
    '''
    计算一个LBP特征图像块的直方图
    '''
    data = np.reshape(src, (-1))
    # 计算得到直方图bin的数目，直方图数组的大小
    bins = maxValue - minValue + 1;
    # 定义直方图每一维的bin的变化范围
    ranges = (float(minValue), float(maxValue + 1))
    hist, bin_edges = np.histogram(src, bins=bins, range=ranges, normed=normed)
    return hist


if __name__ == "__main__":
    gray = cv2.imread(r"K:\Li\A-1.png", cv2.IMREAD_GRAYSCALE)
    # gray = cv2.resize(gray, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)
    # cv2.imshow('img', gray)

    # img = cv2.medianBlur(gray, 9)
    '原始LBP'
    org_lbp = origin_LBP(gray)
    # cv2.imshow('org_lbp', org_lbp)
    '圆型'
    circul_3_8 = circular_LBP(gray, 3, 8)
    circul_3_8_invariant = rotation_invariant_LBP(gray, 3, 8)
    # cv2.imshow('cir_lbp', circul_3_8)
    '计算跳变的LBP'
    uniform_pattern = uniform_pattern_LBP(gray, 3, 8)
    # cv2.imshow('uniform_lbp', uniform_pattern)
    '类似于HoG的Multiscale Block LBP'
    mb_9 = multi_scale_block_LBP(gray, 9)
    # cv2.imshow('mb_lbp', mb_9)
    'LBPH，图像的LBP特征向量'
    uniform_pattern = uniform_pattern_LBP(gray, 3, 8)
    lbph = getLBPH(uniform_pattern, 59, 8, 8, True)

    contours, heridency = cv2.findContours(org_lbp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(gray, contours, -1, (228, 0, 0), 2)
    cv2.imshow("drawContours", gray)
    cv2.waitKey(0)
