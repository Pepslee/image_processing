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

def conv_bn_relu(input, num_channel, kernel_size, stride, name, padding='same', bn_axis=-1, bn_momentum=0.99,
                 bn_scale=True, use_bias=True):
    x = Conv2D(filters=num_channel, kernel_size=(kernel_size, kernel_size),
               strides=stride, padding=padding,
               kernel_initializer="he_normal",
               kernel_regularizer=regularizer,
               use_bias=use_bias,
               name=name + "_conv")(input)
    x = BatchNormalization(name=name + '_bn', scale=bn_scale, axis=bn_axis, momentum=bn_momentum, epsilon=1.001e-5, )(x)
    x = Activation('relu', name=name + '_relu')(x)
    return x


def conv_bn(input, num_channel, kernel_size, stride, name, padding='same', bn_axis=-1, bn_momentum=0.99, bn_scale=True,
            use_bias=True):
    x = Conv2D(filters=num_channel, kernel_size=(kernel_size, kernel_size),
               strides=stride, padding=padding,
               kernel_initializer="he_normal",
               kernel_regularizer=regularizer,
               use_bias=use_bias,
               name=name + "_conv")(input)
    x = BatchNormalization(name=name + '_bn', scale=bn_scale, axis=bn_axis, momentum=bn_momentum, epsilon=1.001e-5, )(x)
    return x


def conv_relu(input, num_channel, kernel_size, stride, name, padding='same', use_bias=True, activation='relu'):
    x = Conv2D(filters=num_channel, kernel_size=(kernel_size, kernel_size),
               strides=stride, padding=padding,
               kernel_initializer="he_normal",
               kernel_regularizer=regularizer,
               use_bias=use_bias,
               name=name + "_conv")(input)
    x = Activation(activation, name=name + '_relu')(x)
    return x


def decoder_block(input, filters, skip, block_name):
    x = UpSampling2D()(input)
    x = conv_bn_relu(x, filters, 3, stride=1, padding='same', name=block_name + '_conv1')
    x = concatenate([x, skip], axis=-1, name=block_name + '_concat')
    x = conv_bn_relu(x, filters, 3, stride=1, padding='same', name=block_name + '_conv2')
    return x


def decoder_block_no_bn(input, filters, skip, block_name, activation='relu'):
    x = UpSampling2D()(input)
    x = conv_bn_relu(x, filters, 3, stride=1, padding='same', name=block_name + '_conv1')
    x = concatenate([x, skip], axis=-1, name=block_name + '_concat')
    x = conv_bn_relu(x, filters, 3, stride=1, padding='same', name=block_name + '_conv2')
    return x


def create_pyramid_features(C1, C2, C3, C4, C5, feature_size=256):
    P5 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='P5', kernel_initializer="he_normal", kernel_regularizer=regularizer)(C5)
    P5_upsampled = UpSampling2D(name='P5_upsampled')(P5)

    P4 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced',
                kernel_initializer="he_normal")(C4)
    P4 = Add(name='P4_merged')([P5_upsampled, P4])
    P4 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4', kernel_initializer="he_normal", kernel_regularizer=regularizer)(P4)
    P4_upsampled = UpSampling2D(name='P4_upsampled')(P4)

    P3 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced',
                kernel_initializer="he_normal")(C3)
    P3 = Add(name='P3_merged')([P4_upsampled, P3])
    P3 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3', kernel_initializer="he_normal", kernel_regularizer=regularizer)(P3)
    P3_upsampled = UpSampling2D(name='P3_upsampled')(P3)

    P2 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C2_reduced',
                kernel_initializer="he_normal")(C2)
    P2 = Add(name='P2_merged')([P3_upsampled, P2])
    P2 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P2', kernel_initializer="he_normal", kernel_regularizer=regularizer)(P2)
    P2_upsampled = UpSampling2D(size=(2, 2), name='P2_upsampled')(P2)

    P1 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C1_reduced',
                kernel_initializer="he_normal")(C1)
    P1 = Add(name='P1_merged')([P2_upsampled, P1])
    P1 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P1', kernel_initializer="he_normal", kernel_regularizer=regularizer)(P1)

    return P1, P2, P3, P4, P5


def prediction_fpn_block(x, name, upsample=None):
    x = conv_bn_relu(x, 128, 3, stride=1, name="predcition_" + name + "_1")
    x = conv_bn_relu(x, 128, 3, stride=1, name="prediction_" + name + "_2")
    if upsample:
        x = UpSampling2D(upsample)(x)
    return x


def model_(data_shape, label_shape, train_params):
    input_shape = (1024, 1024, 3)
    img_input = Input(input_shape)
    channels = 5

    num = 8

    conv1 = conv_bn_relu(img_input, num*2, 3, stride=1, padding='same', name='conv1_1')
    conv1 = conv_bn_relu(conv1, num*2, 3, stride=1, padding='same', name='conv1_2')
    conv1 = conv_bn_relu(conv1, num*2, 3, stride=1, padding='same', name='conv1_3')
    pool1 = MaxPooling2D(pool_size=(2, 2), name='max_pool_1')(conv1)
    conv2 = conv_bn_relu(pool1, num*4, 3, stride=1, padding='same', name='conv2_1')
    conv2 = conv_bn_relu(conv2, num*4, 3, stride=1, padding='same', name='conv2_2')
    conv2 = conv_bn_relu(conv2, num*4, 3, stride=1, padding='same', name='conv2_3')
    pool2 = MaxPooling2D(pool_size=(2, 2), name='max_pool_2')(conv2)
    conv3 = conv_bn_relu(pool2, num*8, 3, stride=1, padding='same', name='conv3_1')
    conv3 = conv_bn_relu(conv3, num*8, 3, stride=1, padding='same', name='conv3_2')
    pool3 = MaxPooling2D(pool_size=(2, 2), name='max_pool_3')(conv3)
    conv4 = conv_bn_relu(pool3, num*16, 3, stride=1, padding='same', name='conv4_1')
    conv4 = conv_bn_relu(conv4, num*16, 3, stride=1, padding='same', name='conv4_2')
    pool4 = MaxPooling2D(pool_size=(2, 2), name='max_pool_4')(conv4)

    conv5 = conv_bn_relu(pool4, num*32, 3, stride=1, padding='same', name='conv5_1')

    conv5 = Dropout(0.5)(conv5)

    x = Conv2D(channels, (1, 1), activation=None, kernel_regularizer=regularizer, name="mask")(conv5)
    x = mean(x, axis=[1, 2])

    x_softmax = Activation('softmax')(x)

    ret = Model(img_input, x_softmax)

    return ret


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
        model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizer, trainable=True))
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

