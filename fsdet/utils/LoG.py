import torch
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from math import exp
from PIL import Image
from torchvision.utils import save_image, make_grid
import cv2


# 将最后的矩阵中的元素归一化到0~1之间
def minmaxscaler(data):
    min = torch.min(data)
    max = torch.max(data)
    return (data - min) / (max - min)


# LoG变换
def LoG(img1_tensor, window, window_size, mode="RGB"):
    # img1_array = np.array(img, dtype=np.float32)  # Image -> array
    # img1_tensor = torch.from_numpy(img1_array)  # array -> tensor
    # 处理不同通道数的数据
    if mode == 'L':
        img1_tensor = img1_tensor.unsqueeze(0).unsqueeze(0)  # h,w -> n,c,h,w
    else:  # RGB or RGBA
        img1_tensor = img1_tensor.permute(2, 0, 1)  # h,w,c -> c,h,w
        img1_tensor = img1_tensor.unsqueeze(0)  # c,h,w -> n,c,h,w
    channel = img1_tensor.size()[1]
    window = Variable(window.expand(channel, 1, window_size, window_size).contiguous())
    output = F.conv2d(img1_tensor, window, padding=window_size // 2, groups=channel)
    output = minmaxscaler(output)  # 归一化到0~1之间
    # if (channel == 4):
    #     save_image(output, "output.png", normalize=False)
    # else:
    #     save_image(output, "output.jpg", normalize=False)
    output = np.array(output)
    output = np.squeeze(output, 0)
    return output


if __name__ == "__main__":
    # 近似卷积核
    window = torch.Tensor([[[0, 1, 1, 2, 2, 2, 1, 1, 0],
                            [1, 2, 4, 5, 5, 5, 4, 2, 1],
                            [1, 4, 5, 3, 0, 3, 5, 4, 1],
                            [2, 5, 3, -12, -24, -12, 3, 5, 2],
                            [2, 5, 0, -24, -40, -24, 0, 5, 2],
                            [2, 5, 3, -12, -24, -12, 3, 5, 2],
                            [1, 4, 5, 3, 0, 3, 4, 4, 1],
                            [1, 2, 4, 5, 5, 5, 4, 2, 1],
                            [0, 1, 1, 2, 2, 2, 1, 1, 0]]])
    window_size = 9
    img = Image.open(r"K:\Li\A-3.png")
    # img = img.convert('L')
    import time
    start = time.time()
    cv2.imshow('LoG', LoG(img, window, window_size, img.mode))
    print(time.time() - start)
    cv2.waitKey(0)
