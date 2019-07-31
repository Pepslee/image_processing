from sklearn.metrics import cohen_kappa_score
from tensorflow.python import is_nan
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics
from tensorflow.python.ops import metrics_impl
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.distributions.normal import Normal
from tensorflow.python.util.deprecation import deprecated
from tensorflow_core.python.ops.confusion_matrix import confusion_matrix


def cohen_kappa(labels,
                predictions_idx,
                num_classes,
                weights=None,
                metrics_collections=None,
                updates_collections=None,
                name=None):


    # Table of the counts of agreement:
    stat_dtype = (dtypes.int64)
    confusion = confusion_matrix(
        labels,
        predictions_idx,
        num_classes=num_classes,
        dtype=stat_dtype)

    sum0 = K.sum(confusion, axis=0)
    sum1 = K.sum(confusion, axis=1)
    expected = K.expand_dims(sum0)*sum1 / K.sum(sum0)
    w_mat = K.zeros([num_classes, num_classes])
    w_mat = K.cast(w_mat, dtype='float32') + K.cast(K.arange(num_classes), dtype='float32')

    w_mat = K.pow(w_mat - K.transpose(w_mat), 2)

    print(w_mat)
    k = K.sum(w_mat * K.cast(confusion, dtype='float32')) / K.sum(w_mat * K.cast(expected, dtype='float32'))
    return 1 - k


def accuracy(y_true, y_pred):
    y_pred.get_shape().assert_is_compatible_with(y_true.get_shape())
    if y_true.dtype != y_pred.dtype:
        y_pred = math_ops.cast(y_pred, y_true.dtype)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)
    return math_ops.cast(math_ops.equal(y_true, y_pred), K.floatx())


def cappa_coen(y_true, y_pred):

    cappa = cohen_kappa(K.argmax(y_true, axis=-1), K.argmax(y_pred), 5)
    return cappa


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)




def mean_dif(y_true, y_pred):
    return K.mean(K.abs(K.argmax(y_true, axis=-1) - K.argmax(y_pred)))


metrics = [cappa_coen, accuracy, f1, mean_dif]
