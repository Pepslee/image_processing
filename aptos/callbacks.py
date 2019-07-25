import os
import warnings
from functools import reduce
from os.path import join
import shutil


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
from sklearn.metrics import classification_report, precision_recall_fscore_support, cohen_kappa_score, f1_score

from dsel import geo_io, colorize
from aptos.tb_writers import TensorboardWriter
from sklearn.metrics import cohen_kappa_score

from tensorflow.python.keras.losses import categorical_crossentropy

from dsel.colorize import Qml
from dsel.rendering import rendering
from tensorflow.python.keras.utils import to_categorical
from aptos.data_generator import preproc


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


def absdiff(y_true, pred):
    y_true = np.array(y_true)
    pred = np.array(pred)
    diff = np.abs(y_true.astype(np.float32)-pred.astype(np.float32))
    return np.mean(diff)


class ModelCheckpoint(Callback):
    """ Save the model after every epoch. """

    def __init__(self, args):
        self.ckpts_path = args['checkpoints_path']
        k = args['fold']
        self.k = k

        self.best = -np.Inf
        self.test_df = args['test_df']
        self.epoch = 0
        self.image_dir = args['image_dir']
        self.tb_writer = TensorboardWriter()
        self.log_path = args['log_path']

    def on_batch_end(self, batch, logs=None):
        loss = logs.get('loss')
        iteration = K.eval(self.model.optimizer.iterations)
        self.tb_writer.log_scalar(self.log_path, 'loss', [loss], iteration, str(self.k))

    def on_epoch_end(self, epoch, logs=None):
        self.save_path = f'{self.ckpts_path}/best_model_{self.k}_e_{epoch}.h5'
        self.epoch = epoch
        self.save_by_metric(epoch)
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        if loss is not None:
            self.tb_writer.log_scalar(self.log_path, 'train_loss', [loss], self.epoch, str(self.k) + '_loss')
        if val_loss is not None:
            self.tb_writer.log_scalar(self.log_path, 'val_loss', [val_loss], self.epoch, str(self.k) + '_loss')



        accuracy = logs.get('accuracy')
        val_accuracy = logs.get('val_accuracy')
        if accuracy is not None:
            self.tb_writer.log_scalar(self.log_path, 'train_accuracy', [accuracy], self.epoch, str(self.k) + '_accuracy')
        if val_accuracy is not None:
            self.tb_writer.log_scalar(self.log_path, 'val_accuracy', [val_accuracy], self.epoch, str(self.k) + '_accuracy')

    def save_by_metric(self, iteration):

        y_pred = []
        y_true = []
        for i, row in self.test_df.iterrows():
            image_path = join(self.image_dir, row['id_code'])
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = preproc(image)
            image = np.expand_dims(image, axis=0)
            y_pred.append(np.argmax(self.model.predict(image)[0], axis=-1))
            y_true.append(int(row['diagnosis']))


        diff = absdiff(y_pred, y_true)
        self.tb_writer.log_scalar(self.log_path, 'diff', [diff], iteration, str(self.k))

        f1 = f1_score(to_categorical(y_pred, 5), to_categorical(y_true, 5))
        self.tb_writer.log_scalar(self.log_path, 'f1', [f1], iteration, str(self.k))



        ck = cohen_kappa_score(y_pred, y_true)
        self.tb_writer.log_scalar(self.log_path, 'kappa', [ck], iteration, str(self.k))

        if ck > self.best:
            print('\nIteration %05d: %s improved from %0.5f to %0.5f, saving model to %s'
                  % (iteration, 'iou', float(self.best), float(ck), self.save_path))
            self.best = ck
            self.model.save(self.save_path, overwrite=True)
        else:
            print('\nIteration %05d: %s did not improve from %0.5f' %
                  (iteration, 'iou', float(self.best)))


def callbacks(args):
    # best_model = ModelCheckpoint(args)

    # best_model = keras.callbacks.ModelCheckpoint('{}/best_model.h5'.format(args['checkpoints_path']), monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    best_model = ModelCheckpoint(args)

    lr_callback = keras.callbacks.ReduceLROnPlateau(mode='min', min_delta=0.0001, cooldown=0, min_lr=args['start_lr'],
                                                    patience=3)
    # tb = keras.callbacks.TensorBoard(log_dir=args['log_path'])
    return [best_model, lr_callback]
