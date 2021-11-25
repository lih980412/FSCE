import json
import numpy as np
from numpy import *
import matplotlib.pyplot as plt


# https://www.cnblogs.com/eczhou/p/7860424.html


# 随机生成聚类中心
def randCenters(dataSet, k):
    n = shape(dataSet)[1]  # 列数
    clustercents = mat(zeros((k, n)))  # 初始化聚类中心矩阵：k*n
    for col in range(n):
        mincol = min(dataSet[:, col])
        maxcol = max(dataSet[:, col])
        # random.rand(k, 1):产生一个0~1之间的随机数向量（k,1表示产生k行1列的随机数）
        clustercents[:, col] = mat(mincol + float(maxcol - mincol) * random.rand(k, 1))  # 按列赋值
    return clustercents


# 欧式距离计算公式
def distEclud(vecA, vecB):
    return linalg.norm(vecA - vecB)


# 绘制散点图
def drawScatter(plt, mydata, size=20, color='blue', mrkr='o'):
    plt.scatter(mydata.T[0].tolist(), mydata.T[1].tolist(), s=size, c=color, marker=mrkr)

# 绘制散点图
def drawScatter_bbox(plt, mydata, size=20, color='blue', mrkr='o'):
    plt.scatter(mydata.T[0].tolist(), mydata.T[0].tolist(), s=size, c=color, marker=mrkr)


# 以不同颜色绘制数据集里的点
def color_cluster_bbox(dataindx, dataSet, plt,):
    datalen = len(dataindx)
    for indx in range(datalen):
        if int(dataindx[indx]) == 0:
            plt.scatter(dataSet[indx, 0], dataSet[indx, 0], s=10, c='blue', marker='o')
        elif int(dataindx[indx]) == 1:
            plt.scatter(dataSet[indx, 0], dataSet[indx, 0], s=10, c='green', marker='o')
        elif int(dataindx[indx]) == 2:
            plt.scatter(dataSet[indx, 0], dataSet[indx, 0], s=10, c='red', marker='o')
        elif int(dataindx[indx]) == 3:
            plt.scatter(dataSet[indx, 0], dataSet[indx, 0], s=10, c='cyan', marker='o')
        elif int(dataindx[indx]) == 4:
            plt.scatter(dataSet[indx, 0], dataSet[indx, 0], s=10, c='orange', marker='o')

# 以不同颜色绘制数据集里的点
def color_cluster(dataindx, dataSet, plt, id2img):
    datalen = len(dataindx)
    for indx in range(datalen):
        # if dataSet[indx, 1] > 420:
        #     print("outlier img: " + id2img[str(dataSet[indx, 2])])
        if int(dataindx[indx]) == 0:
            plt.scatter(dataSet[indx, 0], dataSet[indx, 1], s=10, c='blue', marker='o')
        elif int(dataindx[indx]) == 1:
            plt.scatter(dataSet[indx, 0], dataSet[indx, 1], s=10, c='green', marker='o')
        elif int(dataindx[indx]) == 2:
            plt.scatter(dataSet[indx, 0], dataSet[indx, 1], s=10, c='red', marker='o')
        elif int(dataindx[indx]) == 3:
            plt.scatter(dataSet[indx, 0], dataSet[indx, 1], s=10, c='cyan', marker='o')
        elif int(dataindx[indx]) == 4:
            plt.scatter(dataSet[indx, 0], dataSet[indx, 1], s=10, c='orange', marker='o')


def kMeans(dataSet, k):
    m = shape(dataSet)[0]  # 返回矩阵的行数

    # 本算法核心数据结构:行数与数据集相同
    # 列1：数据集对应的聚类中心,列2:数据集行向量到聚类中心的距离
    ClustDist = mat(zeros((m, 2)))

    # 随机生成一个数据集的聚类中心:本例为4*2的矩阵
    # 确保该聚类中心位于min(dataSet[:,j]),max(dataSet[:,j])之间
    clustercents = randCenters(dataSet, k)  # 随机生成聚类中心

    flag = True  # 初始化标志位,迭代开始
    counter = []  # 计数器

    # 循环迭代直至终止条件为False
    # 算法停止的条件：dataSet的所有向量都能找到某个聚类中心,到此中心的距离均小于其他k-1个中心的距离
    while flag:
        flag = False  # 预置标志位为False

        # ---- 1. 构建ClustDist： 遍历DataSet数据集,计算DataSet每行与聚类的最小欧式距离 ----#
        # 将此结果赋值ClustDist=[minIndex,minDist]
        for i in range(m):

            # 遍历k个聚类中心,获取最短距离
            distlist = [distEclud(clustercents[j, :], dataSet[i, :]) for j in range(k)]
            minDist = min(distlist)
            minIndex = distlist.index(minDist)

            if ClustDist[i, 0] != minIndex:  # 找到了一个新聚类中心
                flag = True  # 重置标志位为True，继续迭代

            # 将minIndex和minDist**2赋予ClustDist第i行
            # 含义是数据集i行对应的聚类中心为minIndex,最短距离为minDist
            ClustDist[i, :] = minIndex, minDist

        # ---- 2.如果执行到此处，说明还有需要更新clustercents值: 循环变量为cent(0~k-1)----#
        # 1.用聚类中心cent切分为ClustDist，返回dataSet的行索引
        # 并以此从dataSet中提取对应的行向量构成新的ptsInClust
        # 计算分隔后ptsInClust各列的均值，以此更新聚类中心clustercents的各项值
        for cent in range(k):
            # 从ClustDist的第一列中筛选出等于cent值的行下标
            dInx = nonzero(ClustDist[:, 0].A == cent)[0]
            # 从dataSet中提取行下标==dInx构成一个新数据集
            ptsInClust = dataSet[dInx]
            # 计算ptsInClust各列的均值: mean(ptsInClust, axis=0):axis=0 按列计算
            clustercents[cent, :] = mean(ptsInClust, axis=0)
    return clustercents, ClustDist


def find_img_id(json_images):
    dict = {}
    for i in range(len(json_images)):
        dict[json_images[i]["id"]] = json_images[i]["file_name"]
    return dict


if __name__ == "__main__":
    json_file = json.load(open(r"D:\UserD\Li\FSCE-1\datasets\my_dataset_before\annotations\instances_train.json"))
    annotations = json_file['annotations']
    id2img = find_img_id(json_file['images'])
    data = []
    del json_file
    for anno in annotations:
        id = anno["image_id"]
        anno["bbox"].append(int(id))
        data.append(anno["bbox"])
    data = np.array(data)
    data_wight_height = data[:][:, 2:]
    data_area = data_wight_height[:, 0] * data_wight_height[:, 1]
    data_area = data_area[:, np.newaxis]

    # clustercents, ClustDist = kMeans(data_wight_height[:, :2], 3)
    # # 返回计算完成的聚类中心
    # print("clustercents:\n", clustercents)

    bboxarea, ClustDist_bbox = kMeans(data_area, 5)
    # 返回计算完成的聚类中心
    print("bboxarea:\n", bboxarea)

    fig = plt.figure()
    # # 输出生成的ClustDist：对应的聚类中心(列1),到聚类中心的距离(列2),行与dataSet一一对应
    # color_cluster(ClustDist[:, 0:1], data_wight_height, plt, id2img)
    # # 绘制聚类中心
    # drawScatter(plt, clustercents, size=20, color='black', mrkr='D')

    # 输出生成的ClustDist：对应的聚类中心(列1),到聚类中心的距离(列2),行与dataSet一一对应
    color_cluster_bbox(ClustDist_bbox[:, 0:1], data_area, plt)
    # 绘制聚类中心
    drawScatter_bbox(plt, bboxarea, size=20, color='black', mrkr='D')
    plt.show()
