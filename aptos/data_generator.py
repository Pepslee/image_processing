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


class DataGenerator:
    def __init__(self, data_table, image_dir, batch_size):
        self.data_table = data_table
        self.df = shuffle(self.data_table)
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.length = self.df.shape[0]

    def generator(self):
        i = 0
        count = self.df.shape[0]
        while(True):
            x_batch = list()
            y_batch = list()
            for b in range(self.batch_size):
                ind = i % count
                image_path = join(self.image_dir, self.df['id_code'].iloc[ind])
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if random.randint(0, 1):
                    image = flip_8_side(image)
                image = (image.astype(np.float32) - 128)/128.0
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
