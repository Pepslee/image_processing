import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.losses import categorical_crossentropy, mean_squared_error

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def loss(y_true, y_pred):

    return categorical_crossentropy(y_true, y_pred)*(1 - K.sum(y_true*y_pred, axis=-1))**2
