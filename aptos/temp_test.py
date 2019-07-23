import cv2

import numpy as np


def pad_or_crop_image(image):
    size = image.shape[1]
    h = image.shape[0]
    if h > size:
        dif = (h - size)
        if dif % 2 == 0:
            start = int(dif / 2)
            stop = -int(dif / 2)
        elif dif == 1:
            start = 0
            stop = -1
        else:
            start = int(dif / 2)
            stop = -int(dif / 2) - 1
        image = image[start:stop, ...]
    elif h < size:
        image_zero = np.zeros(shape=(size, size, 3), dtype=np.uint8)

        dif = size - h
        if dif == 1:
            image_zero[1:, ...] = image
            image = image_zero
        elif dif % 2 == 0:
            start = int(dif / 2)
            stop = -int(dif / 2)
            image_zero[start:stop, ...] = image
            image = image_zero
        else:
            start = int(dif / 2)
            stop = -int(dif / 2) - 1
            image_zero[start:stop, ...] = image
            image = image_zero
    return image


image = np.zeros((512, 511, 3), dtype=np.uint8)


print(image.shape)
image = pad_or_crop_image(image)
print(image.shape)