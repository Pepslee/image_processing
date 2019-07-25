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
    image = (image.astype(np.float32) - np.mean(image, axis=-1)) / np.std(image, axis=-1)
    return image


def sometimes(aug):
    return iaa.Sometimes(0.5, aug)


def augment():
    # sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # translate by -20 to +20 percent (per axis)
                rotate=(-10, 10),  # rotate by -45 to +45 degrees
                shear=(-5, 5),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                       [
                           sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                           # convert images into their superpixel representation
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 1.0)),  # blur images with a sigma between 0 and 3.0
                               iaa.AverageBlur(k=(3, 5)),
                               # blur image using local means with kernel sizes between 2 and 7
                               iaa.MedianBlur(k=(3, 5)),
                               # blur image using local medians with kernel sizes between 2 and 7
                           ]),
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)),  # sharpen images
                           iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                           # search either for all edges or for directed edges,
                           # blend the result with the original image using a blobby mask
                           iaa.SimplexNoiseAlpha(iaa.OneOf([
                               iaa.EdgeDetect(alpha=(0.5, 1.0)),
                               iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                           ])),
                           iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01 * 255), per_channel=0.5),
                           # add gaussian noise to images
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.05), per_channel=0.5),  # randomly remove up to 10% of the pixels
                               iaa.CoarseDropout((0.01, 0.03), size_percent=(0.01, 0.02), per_channel=0.2),
                           ]),
                           iaa.Invert(0.01, per_channel=True),  # invert color channels
                           iaa.Add((-2, 2), per_channel=0.5),
                           # change brightness of images (by -10 to 10 of original value)
                           iaa.AddToHueAndSaturation((-1, 1)),  # change hue and saturation
                           # either change the brightness of the whole image (sometimes
                           # per channel) or change the brightness of subareas
                           iaa.OneOf([
                               iaa.Multiply((0.9, 1.1), per_channel=0.5),
                               iaa.FrequencyNoiseAlpha(
                                   exponent=(-1, 0),
                                   first=iaa.Multiply((0.9, 1.1), per_channel=True),
                                   second=iaa.ContrastNormalization((0.9, 1.1))
                               )
                           ]),
                           sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                           # move pixels locally around (with random strengths)
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                           # sometimes move parts of the image around
                           sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                       ],
                       random_order=True
                       )
        ],
        random_order=True)
    return seq


class DataGenerator:
    def __init__(self, data_table, image_dir, batch_size, phase):
        self.data_table = data_table
        self.df = shuffle(self.data_table)
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.length = self.df.shape[0]
        self.phase = phase

    def generator(self):
        i = 0
        count = self.df.shape[0]
        seq = augment()
        while(True):
            x_batch = list()
            y_batch = list()
            for b in range(self.batch_size):
                ind = i % count
                image_path = join(self.image_dir, self.df['id_code'].iloc[ind])
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if self.phase == 'train':
                    image = seq.augment_image(image)
                image = preproc(image)
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
