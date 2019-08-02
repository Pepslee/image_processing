import tensorflow as tf
import tensorflow.python.keras
from tensorflow.python.keras import Model, Input, Sequential, layers, regularizers
from tensorflow.python.keras.backend import mean
from tensorflow.python.keras.layers import UpSampling2D, Conv2D, BatchNormalization, Activation, concatenate, Add, Dropout, Lambda, MaxPooling2D
from tensorflow.python.keras.utils import get_file
from tensorflow.python.keras.applications.densenet import DenseNet121
from tensorflow.python.keras.applications.densenet import DenseNet169
from tensorflow.python.keras.applications.densenet import DenseNet201
from tensorflow.python.keras.applications.imagenet_utils import decode_predictions
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.mobilenet import MobileNet
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.applications.nasnet import NASNetLarge
from tensorflow.python.keras.applications.nasnet import NASNetMobile
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.applications import ResNet50V2
from tensorflow.python.keras.applications import VGG16



import os
import random
import time
from abc import abstractmethod


import cv2
import numpy as np
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.preprocessing.image import Iterator
from tensorflow.python.keras import backend as K

#regularizer = tf.keras.regularizers.l2(0.01)
regularizer = None

def model_keras(k):
    with tf.name_scope(str(k)):
        input_shape = (224, 224, 3)
        # input_shape = None
        #img_input = Input(input_shape)
        channels = 5
        # img_input = Input(input_shape)
        # ret = ResNet50V2(input_shape=input_shape, include_top=False, weights='imagenet', classes=channels)
        # ret = InceptionResNetV2(input_shape=input_shape, include_top=False, weights='imagenet', classes=channels)
        # ret.trainable = False
        #ret = VGG16(input_shape=input_shape, include_top=False, weights='imagenet', classes=channels)
        ret = DenseNet121(input_shape=input_shape, include_top=False, weights='imagenet')
        # for layer in ret.layers:
        #     if hasattr(layer, 'kernel_regularizer'):
        #         layer.kernel_regularizer = regularizer

        model = Sequential()
        model.add(ret)
        model.add(layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                            beta_initializer='zeros', gamma_initializer='ones',
                                            moving_mean_initializer='zeros',
                                            moving_variance_initializer='ones', beta_regularizer=None,
                                            gamma_regularizer=None,
                                            beta_constraint=None, gamma_constraint=None, trainable=True))
        model.add(layers.GlobalAveragePooling2D())

        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1024, activation='relu', kernel_regularizer=regularizer, trainable=True))
        model.add(layers.Dropout(0.8))
        model.add(layers.Dense(channels, activation='softmax', kernel_regularizer=regularizer, trainable=True))

        for layer in model.layers:
            layer.trainable = False

        for i in range(-5, 0):
            model.layers[i].trainable = True

    return model


def optimizer(optimizer_type, start_lr):
    if optimizer_type == 'RMSProp':
        grad_optimizer = tf.keras.optimizers.RMSprop(lr=start_lr, decay=float(start_lr))
    elif optimizer_type == 'Adam':
        grad_optimizer = tf.keras.optimizers.Adam(lr=start_lr, decay=float(start_lr))
    elif optimizer_type == 'AMSgrad':
        grad_optimizer = tf.keras.optimizers.Adam(lr=start_lr, decay=float(start_lr), amsgrad=True)
    elif optimizer_type == 'SGD':
        grad_optimizer = tf.keras.optimizers.SGD(lr=start_lr, momentum=0.9, nesterov=True, decay=float(start_lr))
    else:
        raise RuntimeError('Unknown optimizer type: ' + optimizer_type)
    return grad_optimizer

