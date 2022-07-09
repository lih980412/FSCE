import cv2, os



if __name__ == "__main__":
    mask_path = r"F:\TextSeg_Release\annotation"
    mask_files = os.listdir(mask_path)

    img_path = r"F:\TextSeg_Release\semantic_label"
    img_files = os.listdir(img_path)

    for img_file in img_files:
        img_name = img_file.split("_")[0]
        mask_name = img_name + "_mask.png"

        img = cv2.imread(os.path.join(img_path, img_file), 1)
        # img = img[:, :, ::-1]
        mask = cv2.imread(os.path.join(mask_path, mask_name), 1)
        mask[mask >= 1] = 1

        img_mask = img*mask
        logo_resized = cv2.resize(img_mask, [128, 128])


        cv2.imshow("1", logo_resized)
        cv2.waitKey(0)