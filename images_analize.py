import os
import glob

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm



def crop_image(img, mask, tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = mask > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def crop_by_mask(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = np.bincount(image_gray.flatten())
    ind = np.argmax(hist)
    thresh = ind + 10
    thresh_image = (image_gray > thresh).astype(np.uint8) * 255
    if thresh < 25:
        thresh_image = crop_image(image, thresh_image, 0)
    else:
        thresh_image = image_gray
    return thresh_image


def calc_stat(globs):
    shapes = list()
    targets_frame = pd.DataFrame(columns=['name', 'r_mean', 'g_mean', 'b_mean', 'r_var', 'g_var', 'b_var', 'mean', 'var'])
    for image_path in tqdm(globs):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        h = image.shape[0]

        if h > 1024:
            dif = (h - 1024)
            if dif % 2 == 0:
                start = int(dif/2)
                stop = -int(dif/2)
            elif dif == 1:
                start = 0
                stop = -1
            else:
                start = int(dif/2)
                stop = -int(dif/2)-1
            image = image[start:stop, ...]
        elif h < 1024:
            image_zero = np.zeros(shape=(1024, 1024, 3), dtype=np.uint8)

            dif = 1024 - h
            if dif == 1:
                image_zero[1:, ...] = image
                image = image_zero
            elif dif % 2 == 0:
                start = int(dif/2)
                stop = -int(dif/2)
                image_zero[start:stop, ...] = image
                image = image_zero
            else:
                start = int(dif/2)
                stop = -int(dif/2)-1
                image_zero[start:stop, ...] = image
                image = image_zero

        # thresh_image = crop_by_mask(image)
        image_name = os.path.basename(image_path)
        # print(image_name)
        # thresh_image = cv2.resize(thresh_image, (1024, 1024))
        image = cv2.resize(image, dsize=(256, 256))
        cv2.imwrite(os.path.join('/mnt/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/aptos2019-blindness-detection/train_cropped_images_256_256', image_name), image)
        # cv2.namedWindow('f', 0)
        # cv2.imshow('f', thresh_image)
        # cv2.namedWindow('image', 0)
        # cv2.imshow('image', image)
        # cv2.waitKey()

        shapes.append(image.shape)
    return shapes


def main():
    path = '/mnt/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/aptos2019-blindness-detection/train_croped_images_1024'
    globs = glob.glob(os.path.join(path, '*.png'))

    # globs = ['/mnt/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/aptos2019-blindness-detection/train_images/65dda202653d.png']
    shapes = calc_stat(globs)

    print()
    for shape in set(shapes):
        print(shape)


if __name__ == '__main__':
    main()

