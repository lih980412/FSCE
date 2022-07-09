import cv2, os
import numpy as np
import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(25, 16))
import pandas as pd
import posixpath

'Ben'
def Ben(image):
    cv2.imshow("origin", image)
    image = crop_image_from_gray(image)
    image2 = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 10), -4, 128)
    cv2.imshow("Ben_10", image2)
    image3 = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 40), -4, 128)
    cv2.imshow("Ben_30", image3)
    cv2.waitKey(0)

'auto-cropping'
def crop_image_from_gray(img, tol=20):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
        # cv2.imshow("Auto-cropping", img[np.ix_(mask.any(1), mask.any(0))])
        # cv2.waitKey(0)
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1, img2, img3], axis=-1)
        #         print(img.shape)
        return img
        # cv2.imshow("Auto-crop", img)
        # cv2.waitKey(0)


def circle_crop(img, sigmaX=10):
    """
    Create circular crop around image centre
    """

    img = crop_image_from_gray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, depth = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigmaX), -4, 128)
    # return img
    cv2.imshow("Circle_crop", img)
    cv2.waitKey(0)

def remap_channel(image):

    ids_sorted = np.argsort(((image) + np.random.random(image.shape) - 0.5).ravel())
    # ids_sorted = np.argsort(((image//255.0).astype(np.float64) + np.random.random(image.shape) - 0.5).ravel())
    values = np.floor(np.linspace(0.0, 256.0, num=len(ids_sorted), endpoint=False)).astype(np.uint8)
    s = image.shape
    image = image.ravel()
    image[ids_sorted] = values
    image = image.reshape(s)

    return image

def remap_image(image):
    plt.subplot(221)
    plt.title("Original image")
    plt.imshow(image)
    plt.subplot(222)
    plt.title("Original image histogram")
    plt.hist(image.ravel(), bins=256)


    if len(image.shape) == 2:
        image_rep = remap_channel(image)
    else:
        import copy
        image_rep = copy.deepcopy(image)
        image_rep[:, :, 0] = remap_channel(image_rep[:, :, 0])
        image_rep[:, :, 1] = remap_channel(image_rep[:, :, 1])
        image_rep[:, :, 2] = remap_channel(image_rep[:, :, 2])



    plt.subplot(223)
    plt.title("Transformed image")
    plt.imshow(image_rep)
    plt.subplot(224)
    plt.title("Original image histogram")
    plt.hist(image_rep.ravel(), bins=256)
    plt.show()


if __name__ == "__main__":
    # img_path = r"K:\Li\A-1.png"
    # BASE_PATH = r"F:\Dataset\Kaggle\train_images"
    # train = pd.read_csv(r"F:\Dataset\Kaggle\fold.csv")
    # names = train["image"]
    # for name in names[:200]:
    #     image = cv2.imread(os.path.join(BASE_PATH, name), 1)
    #     image = cv2.resize(image, [512, 512])
    #     # Ben(image)
    #     remap_image(image)
    # print(1)
    BASEPATH = r"F:\Dataset\ReZha\1"
    fires = os.listdir(BASEPATH)
    for file in fires:
        img_path = os.path.join(BASEPATH, file)
        image = cv2.imread(img_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Ben(image)
        crop_image_from_gray(image, 40)
        # circle_crop(image, 30)
        remap_image(image)





'提取焊缝区域想法：求出每一行和每一列像素和的最小值的下标，再比较这两个最小值。'
'如果是行小，就把这一行以上或以下的部分切掉；如果是列小，就把这一列以左或以右的部分切掉。'

