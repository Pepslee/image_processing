import cv2
import os
import numpy as np


def eq(crop_img):
    mean = np.mean(crop_img, axis=(0, 1))
    shifted = np.float32(crop_img) - mean
    norm = cv2.normalize(shifted, 0, 0, 255, cv2.NORM_MINMAX)

    return norm.astype(np.uint8)


path_0 = '/home/panchenko/Downloads/obstacle_detection/video/frames/12.jpg'
path_1 = '/home/panchenko/Downloads/obstacle_detection/video/frames/16.jpg'
path_241 = '/home/panchenko/Downloads/obstacle_detection/video/frames/256.jpg'

image_0 = cv2.imread(path_0, cv2.IMREAD_COLOR)[20:, :]
other = cv2.imread(path_1, cv2.IMREAD_COLOR)[20:, :]
image_241 = cv2.imread(path_241, cv2.IMREAD_COLOR)[20:, :]
other = np.clip(np.float32(other) * 0.5, 0, 255).astype(np.uint8)
image_241 = np.clip(np.float32(image_241)*1.3, 0, 255).astype(np.uint8)

image_0 = eq(image_0)
other = eq(other)
image_241 = eq(image_241)


hsv_0 = cv2.cvtColor(image_0, cv2.COLOR_BGR2HSV)
hsv_1 = cv2.cvtColor(other, cv2.COLOR_BGR2HSV)
hsv_241 = cv2.cvtColor(image_241, cv2.COLOR_BGR2HSV)

diff_d = cv2.absdiff(image_0, other)
diff_s = cv2.absdiff(image_0, image_241)
diff_diferent = cv2.absdiff(hsv_0, hsv_1)
diff_same = cv2.absdiff(hsv_0, hsv_241)

print('dif')
print(cv2.medianBlur(diff_diferent[..., 1], 15).mean())
# print(cv2.medianBlur(diff_diferent[..., 1], 15).var())
print(cv2.medianBlur(diff_diferent[..., 1], 15).std())
print('same')
print(cv2.medianBlur(diff_same[..., 1], 15).mean())
# print(cv2.medianBlur(diff_same[..., 1], 15).var())
print(cv2.medianBlur(diff_same[..., 1], 15).std())



# cv2.namedWindow('image_0', 0)
# cv2.imshow('image_0', image_0.astype(np.uint8))
# cv2.namedWindow('other', 0)
# cv2.imshow('other', other.astype(np.uint8))
# cv2.namedWindow('image_241', 0)
# cv2.imshow('image_241', image_241.astype(np.uint8))
#
# cv2.namedWindow('s_dif', 0)
# cv2.imshow('s_dif', diff_diferent[..., 1].astype(np.uint8))
# cv2.namedWindow('s_same', 0)
# cv2.imshow('s_same', diff_same[..., 1].astype(np.uint8))
#
# cv2.waitKey()

pass