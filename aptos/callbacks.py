import os
import warnings
from functools import reduce
from os.path import join


import cv2
import numpy as np
import h5py
import tensorflow.keras
from tensorflow.python.keras.callbacks import LearningRateScheduler, Callback, TensorBoard
import tensorflow.keras.backend as K
import tensorflow as tf
from skimage.morphology import dilation, watershed, square, erosion, star, diamond, disk, label
from skimage.color import label2rgb
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python import keras
from sklearn.metrics import classification_report, precision_recall_fscore_support, cohen_kappa_score

from dsel import geo_io, colorize
from aptos.tb_writers import TensorboardWriter
from sklearn.metrics import cohen_kappa_score

from tensorflow.python.keras.losses import categorical_crossentropy

from dsel.colorize import Qml
from dsel.rendering import rendering
from tensorflow.python.keras.utils import to_categorical


def IoU(y, x, thresh):
    x = x > thresh
    y = y > 0

    y = to_categorical(y, 5).astype(np.bool)
    x = to_categorical(x, 5).astype(np.bool)
    intersection = np.sum(np.bitwise_and(x, y))
    # if intersection == 0:
    #     return 0
    union = np.sum(np.bitwise_or(x, y))
    # if union == 0:
    #     return 0
    return intersection/(union + 0.0001)


def dice_coef(y_true, y_pred, smooth=1e-3):
    smooth = 1e-3
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return np.mean((2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth))


def mse_metric(y_true, y_pred, smooth=1e-3):
    err = (np.square(y_true - y_pred)).mean()
    return 1. - err


class ModelCheckpoint(Callback):
    """ Save the model after every epoch. """

    def __init__(self, args):
        self.save_path = '{}/best_model.h5'.format(args['checkpoints_path'])
        self.best = -np.Inf
        self.test_df = args['test_df']
        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        self.save_by_metric(epoch)

    def save_by_metric(self, iteration):

        y_pred = []
        y_true = []
        for row in self.test_df.iterrows():
            image_path = join(self.image_dir, self.row['id_code'])
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = (image.astype(np.float32)-128)/128.0
            image = np.expand_dims(image, axis=0)
            y_pred.append(np.argmax(self.model.predict(image)[0], axis=-1))
            y_true.append(self.row['id_code'])

        ck = cohen_kappa_score(y_pred, y_true)
        self.tb_writer.log_scalar(self.log_path, 'kappa', [ck], iteration, 'Train')
        if ck > self.best:
            print('\nIteration %05d: %s improved from %0.5f to %0.5f, saving model to %s'
                  % (iteration, 'iou', float(self.best), float(ck), self.save_path))
            self.best = ck
            self.model.save(self.save_path, overwrite=True)
        else:
            print('\nIteration %05d: %s did not improve from %0.5f' %
                  (iteration, 'iou', float(self.best)))


class Metrics(Callback):
    def __init__(self, args):
        Callback.__init__(self)
        self.tb_writer = TensorboardWriter()
        self.log_path = args['log_path']
        self.best = -np.Inf

    def on_batch_end(self, batch, logs=None):
        output_dist = logs.get('loss')
        iteration = K.eval(self.model.optimizer.iterations)
        self.tb_writer.log_scalar(self.log_path, 'loss', [output_dist], iteration, 'Train')

        #input, input_mask = next(self.validation_data)
        #val_pred = np.asarray(self.model.predict(input))
        #y_pred = np.argmax(val_pred, axis=-1)+1
        #y_true = np.argmax(input_mask, axis=-1)+1
        #iou = IoU(y_pred, y_true, 0)
        #self.tb_writer.log_scalar(self.log_path, 'kappa', [iou], iteration, 'Train')


def callbacks(args):
    # best_model = ModelCheckpoint(args)

    args['log_path'] = os.path.join(args['checkpoints_path'], 'log')
    if not os.path.exists(args['checkpoints_path']):
        os.makedirs(args['checkpoints_path'])
    if not os.path.exists(args['log_path']):
        os.makedirs(args['log_path'])
    best_model = keras.callbacks.ModelCheckpoint('{}/best_model.h5'.format(args['checkpoints_path']), monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    metrics = Metrics(args)

    lr_callback = keras.callbacks.ReduceLROnPlateau(mode='min', min_delta=0.0001, cooldown=0, min_lr=args['start_lr'],
                                                    patience=3)
    tb = keras.callbacks.TensorBoard(log_dir=args['log_path'])
    return [best_model, metrics, lr_callback, tb]
