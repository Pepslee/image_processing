#!/usr/bin/env python
from os.path import join

import numpy as np
import random
import cv2
from scipy.ndimage import rotate
import pandas as pd
from skimage.transform import rescale
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.python.keras.utils import to_categorical
from imgaug import augmenters as iaa
import imgaug as ia

class Augmentations:
    """ Augmentation functional set """

    @staticmethod
    def draw_hist(image):
        """ Draw image histogram. """
        bit_size = 2 ** (image.dtype.itemsize*8) - 1
        from matplotlib import pyplot as plt
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([image], [i], None, [bit_size], [0, bit_size])
            plt.plot(histr, color=col)
            plt.xlim([0, bit_size])
        plt.show()

    @staticmethod
    def brightness_change(result_img, brightness_range):
        """ Changing brightness augmentation method. """
        def inner_brightness_change(result_img, brightness_range):
            k_brightness = random.uniform(brightness_range[0], brightness_range[1])
            ret = (result_img.astype(np.uint32) * k_brightness)
            ret = np.clip(ret, a_min=0, a_max=255).astype(result_img.dtype)
            return ret
        return inner_brightness_change(result_img, brightness_range)

    @staticmethod
    def hist_stretching(result_img, brightness_range):
        """ Histogram stretching augmentation method."""
        def inner_hist_stretching(result_img, stretching_range):
            before_normalize_max = 2 ** (result_img.dtype.itemsize * 8) - 1
            prc_min = before_normalize_max * stretching_range[0] / 100
            prc_max = before_normalize_max * stretching_range[1] / 100
            k_min = random.randint(0, prc_min)
            k_max = random.randint(prc_max, before_normalize_max)
            ret = cv2.normalize(result_img, result_img, k_min, k_max, cv2.NORM_MINMAX)
            return ret
        return inner_hist_stretching(result_img, brightness_range)

    @staticmethod
    def rgb2gray(img):
        shape = img.shape
        if len(shape) > 2:
            channels = shape[-1]
        else:
            return img
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret = np.stack([gray]*channels, axis=-1)
        return ret

    def rgb_random_contrast(self, image):
        shape = image.shape
        if len(shape) > 2:
            channels = shape[-1]
        else:
            return self.random_contrast(image)
        bands = np.split(image, channels, axis=-1)
        eq_bands = list()
        for band in bands:
            if random.randint(0, 2):
                band = self.random_contrast(band)
            band = np.squeeze(band)
            eq_bands.append(band)
        return np.stack(eq_bands, axis=-1)

    @staticmethod
    def random_contrast(img, scale_down=0.5, scale_up=1.5):
        alpha = random.uniform(scale_down, scale_up)
        gray = np.mean(img, axis=-1)
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        ret = alpha * img + gray
        ret = np.clip(ret, a_min=0, a_max=255).astype(img.dtype)
        return ret


def flip_8_side(data):
    switch = random.randint(0, 6)
    if switch == 0:
        data = np.flip(data, axis=0)
    elif switch == 1:
        data = np.flip(data, axis=1)
    elif switch == 2:
        data = np.flip(np.flip(data, axis=0), axis=1)
    elif switch == 3:
        data = np.transpose(data, axes=[1, 0, 2])
    elif switch == 4:
        data = np.flip(data, axis=0)
    elif switch == 5:
        data = np.flip(data, axis=1)
    elif switch == 6:
        data = np.flip(np.flip(data, axis=0), axis=1)
    return data


def preproc(image):
    image = cv2.resize(image, (224, 224))
    image = (image.astype(np.float32) - np.mean(image, axis=(0, 1))) / np.std(image, axis=(0, 1))
    return image


def sometimes(aug):
    return iaa.Sometimes(0.5, aug)


def crop(image):
    crop_pad = 60
    image_ = image[crop_pad:-crop_pad, crop_pad:-crop_pad, ...]

    return image_


def augment(crop_img):
    """ Set of augmentation algorithms. It is a custom function, be attentive."""

    shape = crop_img.shape
    augmentation = Augmentations()
    is_brightness = 10
    brightness_range = [0.8, 1.2]

    is_stretching = 50
    scale = 0.5

    is_equalization = 10

    adaptive_equalization = 10

    if random.randint(0, 100) < is_equalization:
        for i in range(crop_img.shape[-1]):
            crop_img[:, :, i] = cv2.equalizeHist(crop_img[:, :, i])

    if random.randint(0, 100) < adaptive_equalization:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for i in range(crop_img.shape[-1]):
            crop_img[:, :, i] = clahe.apply(crop_img[:, :, i])

    if random.randint(0, 100) < is_brightness:
        crop_img = augmentation.brightness_change(crop_img, brightness_range)

    if random.randint(0, 100) < 10:
        crop_img = augmentation.rgb_random_contrast(crop_img)

    # if is_fastai_augmentation:
    #     crop_img = augmentation(crop_img)

    if random.randint(0, 100) < 10:
        crop_img = augmentation.random_contrast(crop_img)

    if random.randint(0, 100) < 10:
        crop_img = cv2.bitwise_not(crop_img)

    if random.randint(0, 100) < 30:
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)

    if random.randint(0, 100) < 10:
        data_ = cv2.cvtColor(crop_img, cv2.cv2.COLOR_BGRA2GRAY)
        crop_img = np.stack([data_]*shape[-1], axis=-1)

    if random.randint(0, 100) < 5:
        crop_img = cv2.medianBlur(crop_img, 5)



    # cv2.namedWindow('crop_img', 0)
    # cv2.namedWindow('crop_mask', 0)
    # cv2.imshow('crop_img', crop_img)
    # cv2.imshow('crop_mask', crop_mask*255)
    # cv2.waitKey()

    return crop_img


class DataGenerator:
    def __init__(self, data_table, image_dir, batch_size, phase, ext):
        self.data_table = data_table
        self.df = shuffle(self.data_table)
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.length = self.df.shape[0]
        self.phase = phase
        self.image_mode ='color'
        self.ext = ext

    def generator(self):
        i = 0
        count = self.df.shape[0]
        while(True):
            x_batch = list()
            y_batch = list()
            for b in range(self.batch_size):
                ind = i % count
                image_path = join(self.image_dir, self.df['id_code'].iloc[ind] + self.ext)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if self.image_mode == 'gray':
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    gray_image_norm = (gray_image.astype(np.float32) - np.mean(gray_image)) / (
                                np.std(gray_image) + 0.001)
                    image = cv2.normalize(gray_image_norm, gray_image_norm, alpha=0, beta=255,
                                                    norm_type=cv2.NORM_MINMAX).astype(np.uint8)
                # if self.phase == 'train':
                #     image = seq.augment_image(image)
                if self.phase == 'train':
                    image = crop(image)
                    image = augment(image)
                image = preproc(image)
                if self.image_mode == 'gray':
                    image = np.stack([image]*3, axis=-1)
                if random.randint(0, 1):
                    image = flip_8_side(image)

                diagnosis = self.df['diagnosis'].iloc[ind]
                diagnosis = to_categorical(diagnosis, 5)
                x_batch.append(image)
                y_batch.append(diagnosis)
                i += 1
                if ind == 0:
                    self.df = shuffle(self.df)
            x_batch = np.stack(x_batch)
            y_batch = np.stack(y_batch)
            yield x_batch, y_batch

    def __len__(self):
        return int(self.length/self.batch_size)

    def crop_after_rotation_and_scaling(self, image, mask, crop_params, index_mask):
        """ Cut some crop from image and mask by angle and size.

            Args:
                image: input image from data set
                mask: input mask from data set
                crop_params:
                    scale_prob: scale probability
                    rotation_prob: rotation probability
                    rotation_type: type of rotation (contours, objects)
                        contours - use cv2.find_contours to vectorize contours to avoid broking during rotation.
                        objects - rotate mask with NEAREST_NEIGHBOR interpolation
                    crop_size: crop size

            Returns: image and mask crop of crop size

            """
        crop_size, scale_prob, rotation_prob, rotation_type = \
            crop_params['crop_size'], crop_params['scale_prob'], crop_params['rot_prob'], crop_params['rot_type']

        rot_angle = 0
        if random.randint(0, 100) <= rotation_prob:
            rot_angle = random.randint(1, 89)

        scale = 1
        if rot_angle:
            if random.randint(0, 100) <= scale_prob:
                scale = random.uniform(0.5, 1.5)
        rotated_size = self.get_rotated_box_size((crop_size[0], crop_size[1]),
                                                 rot_angle) * (1 / (scale if scale <= 1 else 1))
        rotated_size = rotated_size.astype(np.uint32)
        mask_sum = 0
        img_ = None
        mask_ = None
        while mask_sum == 0:
            y_min = rotated_size[0] // 2 + 1
            y_max = image.shape[0] - (rotated_size[0] // 2 + 1)
            x_min = rotated_size[1] // 2 + 1
            x_max = image.shape[1] - (rotated_size[1] // 2 + 1)
            if x_min > x_max or y_min > y_max:
                center_y = image.shape[0] // 2
                center_x = image.shape[1] // 2
                rot_angle = 0
                scale = 1
                rotated_size = self.get_rotated_box_size((crop_size[0], crop_size[1]),
                                                         rot_angle) * (1 / (scale if scale <= 1 else 1))
                rotated_size = rotated_size.astype(np.uint32)
            else:

                center_ind = random.randint(0, index_mask.shape[0]-1)
                center = index_mask[center_ind]
                center_y = center[0]
                center_x = center[1]
                if center_y < y_min or center_y > y_max:
                    continue
                if center_x < x_min or center_x > x_max:
                    continue
                # center_y = random.randint(y_min, y_max)
                # center_x = random.randint(x_min, x_max)
            center = [center_y, center_x]
            rotated_rect_img = self.crop(image, center, rotated_size)
            rotated_rect_mask = self.crop(mask, center, rotated_size)
            if scale != 1:
                if rot_angle > 0:
                    rotated_rect_img, rotated_rect_mask = self.zoom(rotated_rect_img, rotated_rect_mask, scale)

            img_ = self.rotate_image_and_crop(rotated_rect_img, rot_angle, crop_size, cv2.INTER_CUBIC)
            if rotation_type == 'contours':
                mask_ = self.rotate_contours(rotated_rect_mask, rot_angle, crop_size)
            elif rotation_type == 'objects':
                mask_ = self.rotate_image_and_crop(rotated_rect_mask, rot_angle, crop_size, cv2.INTER_NEAREST)
            else:
                print('Unknown rotation_type. Must be one of: contours/objects.')
                exit()
            mask_sum = np.sum(mask_ + 1)
        if mask_.shape[:2] != crop_params['crop_size']:
            raise ValueError(mask_.shape[:-1], crop_params['crop_size'])
        return img_, np.squeeze(mask_)

    @staticmethod
    def zoom(image, mask, scale):
        if scale != 1:
            # image_ = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            image_ = rescale(image, scale, order=1, preserve_range=True, multichannel=True)
            mask_ = rescale(mask, scale, order=0, preserve_range=True, multichannel=True)
            # mask_ = cv2.resize(mask, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            if len(mask.shape) == 2:
                mask_ = np.expand_dims(mask_, axis=-1)
            return image_, mask_
        else:
            return image, mask

    @staticmethod
    def rotate_image_and_crop(source, rot_angle, size, interpolation=cv2.INTER_CUBIC):
        """ Rotate image and crop by size

        Args:
            source: input image
            rot_angle: random angle to rotate
            size: crop size
            interpolation: interpolation type (opencv)

        Returns: rotated crop

        """
        if rot_angle != 0:
            # rot_mat = cv2.getRotationMatrix2D((source.shape[0] // 2, source.shape[1] // 2), rot_angle, 1.0)
            # rotated = cv2.warpAffine(source, rot_mat, (source.shape[1], source.shape[0]), flags=interpolation)
            rotated = rotate(source, rot_angle, mode='nearest', order=0)
            # rotated = imrotate(source, rot_angle, interp='nearest')
        else:
            return source.copy()
        sx = rotated.shape[1] // 2 - size[0] // 2
        sy = rotated.shape[0] // 2 - size[1] // 2
        crop_and_rot = rotated[sy:sy + size[0], sx:sx + size[1]].copy()
        if len(source.shape) > 2 and len(crop_and_rot.shape) < 3:
            crop_and_rot = np.expand_dims(crop_and_rot, axis=2)
        return crop_and_rot
