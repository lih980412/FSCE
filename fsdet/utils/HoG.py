from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 读取图片
    img = imread(r"K:\Li\A-3.png")
    # 改变图片尺寸
    # img = resize(img, (128, 64))

    # 产生HOG特征
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True,
                        multichannel=True)
    print('\n\nShape of Image Features\n\n')
    print(fd.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

    ax1.imshow(img, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # 缩放直方图以便更好地显示
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')

    plt.show()
