import argparse
import os
import shutil

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.models import load_model
from tqdm import tqdm

from aptos.data_generator import DataGenerator
from aptos.model import model_keras, optimizer
from aptos.loss import loss
from aptos.metrics import metrics
from aptos.callbacks import callbacks


def pad_or_crop_image(image, size):
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


def crop_image(img, mask, tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = mask > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def preproc(image):
    image = cv2.resize(image, (299, 299))
    image = (image.astype(np.float32) - 128) / 128.0
    return image


def crop_by_mask(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = np.bincount(image_gray.flatten())
    ind = np.argmax(hist)
    thresh = ind + 10
    thresh_image = (image_gray > thresh).astype(np.uint8) * 255
    if thresh < 25:
        thresh_image = crop_image(image, thresh_image, 0)
    else:
        thresh_image = image
    return thresh_image


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', type=str, help='Path to csv file.')
    parser.add_argument('image_dir', type=str, help='Path to image directory.')
    parser.add_argument('ckpts_path', type=str, help='Path to output ckpts.')
    parser.add_argument('-bs', '--batch_size', type=int, required=False, default=5, help='Batch size')

    return parser.parse_args()


def main(csv_path, image_dir, ckpts_path, batch_size):
    SIZE = 299
    csv_path = os.path.abspath(csv_path)
    image_dir = os.path.abspath(image_dir)
    ckpts_path = os.path.abspath(ckpts_path)

    model = load_model(ckpts_path, compile=False)

    data_frame = pd.read_csv(csv_path)
    diagnosis = []
    batch = []
    for i, row in tqdm(data_frame.iterrows()):
        image_path = os.path.join(image_dir, row['id_code'] + '.png')
        image = cv2.imread(image_path)

        image = crop_by_mask(image)

        y_size, x_size = image.shape[:2]
        k = float(y_size) / x_size
        image = cv2.resize(image, (SIZE, int(k * SIZE)))

        image = pad_or_crop_image(image, SIZE)

        image = preproc(image)

        batch.append(image)
        image = np.expand_dims(image, axis=0)
        pred = model.predict(image)
        diagnosis.append(np.argmax(pred, axis=-1))
        pass
    data_frame['diagnosis'] = diagnosis
    data_frame.to_csv('/mnt/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/aptos2019-blindness-detection/best_model_0_test.csv', index=False)


if __name__ == '__main__':
    args = create_parser()
    main(args.csv_path, args.image_dir, args.ckpts_path, args.batch_size)
