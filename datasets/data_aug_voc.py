# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 10:54:45 2019

@author: Lenovo

为文件夹中的jpg图片批量添加噪声

添加5%的椒盐噪声：util.random_noise(img,mode='s&p',amount=0.05)
添加（0，0.01）的高斯噪声：util.random_noise(img,mode='gaussian')
添加(0,0.1)的斑点噪声：util.random_noise(img,mode='speckle')
"""

import cv2
import random
# from numpy import *
import os
import shutil
import xml.etree.ElementTree as ET
import skimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import util

# 批处理
noise = 'gaussian'  # (or speckle or s&p or gaussian)

# path='C:/Users/dell/Desktop/1/'
# path='E:/ano/JPEGImages/ori/'
# image_names=os.listdir(path)  #列举path下所有文件和文件夹名称（返回一个list）
# save_dir=os.path.join(path,noise)
# save_dir = 'E:/ano/JPEGImages/'
# os.mkdir(save_dir)
take_path = r"F:\Dataset\DiBei\preprocess_0410_train_aug"
save_path = r"F:\Dataset\DiBei\preprocess_0410_train_aug"
xml_path = r"F:\Dataset\DiBei\preprocess_0410_train_aug\\"
xml_save_path = r"F:\Dataset\DiBei\preprocess_0410_train_aug\\"
file_names = os.listdir(take_path)
# 转化所有图片
for img in file_names:
    if img.endswith("remap.png"):
        suffix = f"_{noise}0001"
        fname, ext = os.path.splitext(img)
        t_path = os.path.join(take_path, img)
        s_path = os.path.join(save_path, fname + suffix + ext)
        img = Image.open(t_path)
        img = np.array(img)
        noise_img = util.random_noise(img, mode=noise, mean=0, var=0.001)  # mean为均值 var为方差
        noise_img = noise_img * 255
        noise_img = noise_img.astype(np.int64)
        # newName=image_name.replace('.','_'+noise+'00001'+'.')
        cv2.imwrite(s_path, noise_img)
        img = cv2.imread(s_path)
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(s_path, img1)
        # os.rename(os.path.join(xml_path, frame + '.xml'), os.path.join(xml_path, frame + suffix + '.xml')
        doc = ET.parse(xml_path + fname + '.xml')
        # objects = doc.findall('object')
        root = doc.getroot()
        filename = root.find('filename')
        filename.text = fname + suffix + ext
        doc.write(xml_save_path + fname + suffix + '.xml')
'''    
for image_name in image_names:
    if(image_name.endswith('.jpg')): 
        img=Image.open(os.path.join(path,image_name))
        img=np.array(img)

        # 生成噪声图像noise_image
		#S&P
        #noise_img=util.random_noise(img,mode=noise,amount=0.0001)
		#高斯
        noise_img=util.random_noise(img,mode=noise,mean=0,var=0.0001)#mean为均值 var为方差
        noise_img=noise_img*255
        noise_img=noise_img.astype(np.int)

        # 保存图像并重写打开，将BGR转换成RGB，再重新保存
        #newName=image_name.replace('.','_'+noise+'.')
        newName=image_name.replace('.','_'+noise+'00001'+'.')
        cv2.imwrite(os.path.join(save_dir,newName),noise_img)       
        img = cv2.imread(os.path.join(save_dir,newName))
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(save_dir,newName),img1)
'''

