import numpy as np
import cv2
import os
# https://blog.csdn.net/dcrmg/article/details/102467434?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-2.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-2.no_search_link

# img_h, img_w = 32, 32
img_h, img_w = 640, 480  # 根据自己数据集适当调整，影响不大
means, stdevs = [], []
img_list = []

imgs_path = r'F:\Dataset\Weld Defects\web\Original\JPEGImages'
imgs_path_list = os.listdir(imgs_path)

len_ = len(imgs_path_list)
i = 0
for item in imgs_path_list:
    img = cv2.imread(os.path.join(imgs_path, item))
    img = cv2.resize(img, (img_w, img_h))
    img = img[:, :, :, np.newaxis]
    img_list.append(img)
    i += 1
    print(i, '/', len_)

imgs = np.concatenate(img_list, axis=3)
imgs = imgs.astype(np.float32) / 255.

for i in range(3):
    pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

# BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
means.reverse()
stdevs.reverse()

means = [means[i]*255 for i in range(len(means))]
stdevs = [stdevs[i]*255 for i in range(len(stdevs))]

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))