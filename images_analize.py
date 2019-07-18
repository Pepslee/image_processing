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
        thresh_image = crop_by_mask(image)
        image_name = os.path.basename(image_path)
        print(image_name)
        # thresh_image = cv2.resize(thresh_image, (1024, 1024))
        cv2.imwrite(os.path.join('/mnt/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/aptos2019-blindness-detection/train_thresh_images', image_name), thresh_image)
        # cv2.namedWindow('f', 0)
        # cv2.imshow('f', thresh_image)
        # cv2.namedWindow('image_gray', 0)
        # cv2.imshow('image_gray', image)
        # cv2.waitKey()

        shapes.append(image.shape)
    return shapes


def main():
    path = '/mnt/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/aptos2019-blindness-detection/train_images'
    globs = glob.glob(os.path.join(path, '*.png'))

    # globs = ['/mnt/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/aptos2019-blindness-detection/train_images/65dda202653d.png']
    shapes = calc_stat(globs)

    print()
    for shape in set(shapes):
        print(shape)


if __name__ == '__main__':
    main()

