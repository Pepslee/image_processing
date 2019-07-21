from sklearn.metrics import cohen_kappa_score
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
from tensorflow.python.keras import backend as K
import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import confusion_matrix
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


def cohen_kappa(labels,
                predictions_idx,
                num_classes,
                weights=None,
                metrics_collections=None,
                updates_collections=None,
                name=None, weight_matrix=None):
    """Calculates Cohen's kappa.

    [Cohen's kappa](https://en.wikipedia.org/wiki/Cohen's_kappa) is a statistic
    that measures inter-annotator agreement.

    The `cohen_kappa` function calculates the confusion matrix, and creates three
    local variables to compute the Cohen's kappa: `po`, `pe_row`, and `pe_col`,
    which refer to the diagonal part, rows and columns totals of the confusion
    matrix, respectively. This value is ultimately returned as `kappa`, an
    idempotent operation that is calculated by

        pe = (pe_row * pe_col) / N
        k = (sum(po) - sum(pe)) / (N - sum(pe))

    For estimation of the metric over a stream of data, the function creates an
    `update_op` operation that updates these variables and returns the
    `kappa`. `update_op` weights each prediction by the corresponding value in
    `weights`.

    Class labels are expected to start at 0. E.g., if `num_classes`
    was three, then the possible labels would be [0, 1, 2].

    If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

    NOTE: Equivalent to `sklearn.metrics.cohen_kappa_score`, but the method
    doesn't support weighted matrix yet.

    Args:
      labels: 1-D `Tensor` of real labels for the classification task. Must be
        one of the following types: int16, int32, int64.
      predictions_idx: 1-D `Tensor` of predicted class indices for a given
        classification. Must have the same type as `labels`.
      num_classes: The possible number of labels.
      weights: Optional `Tensor` whose shape matches `predictions`.
      metrics_collections: An optional list of collections that `kappa` should
        be added to.
      updates_collections: An optional list of collections that `update_op` should
        be added to.
      name: An optional variable_scope name.

    Returns:
      kappa: Scalar float `Tensor` representing the current Cohen's kappa.
      update_op: `Operation` that increments `po`, `pe_row` and `pe_col`
        variables appropriately and whose value matches `kappa`.

    Raises:
      ValueError: If `num_classes` is less than 2, or `predictions` and `labels`
        have mismatched shapes, or if `weights` is not `None` and its shape
        doesn't match `predictions`, or if either `metrics_collections` or
        `updates_collections` are not a list or tuple.
      RuntimeError: If eager execution is enabled.
    """
    if context.executing_eagerly():
        raise RuntimeError('tf.contrib.metrics.cohen_kappa is not supported '
                           'when eager execution is enabled.')
    if num_classes < 2:
        raise ValueError('`num_classes` must be >= 2.'
                         'Found: {}'.format(num_classes))
    with variable_scope.variable_scope(name, 'cohen_kappa',
                                       (labels, predictions_idx, weights)):
        # Convert 2-dim (num, 1) to 1-dim (num,)
        labels.get_shape().with_rank_at_most(2)
        if labels.get_shape().ndims == 2:
            labels = array_ops.squeeze(labels, axis=[-1])
        predictions_idx, labels, weights = (
            metrics_impl._remove_squeezable_dimensions(  # pylint: disable=protected-access
                predictions=predictions_idx,
                labels=labels,
                weights=weights))
        predictions_idx.get_shape().assert_is_compatible_with(labels.get_shape())

        stat_dtype = (
            dtypes.int64
            if weights is None or weights.dtype.is_integer else dtypes.float32)
        po = metrics_impl.metric_variable((num_classes,), stat_dtype, name='po')
        pe_row = metrics_impl.metric_variable(
            (num_classes,), stat_dtype, name='pe_row')
        pe_col = metrics_impl.metric_variable(
            (num_classes,), stat_dtype, name='pe_col')

        # Table of the counts of agreement:
        counts_in_table = confusion_matrix.confusion_matrix(
            labels,
            predictions_idx,
            num_classes=num_classes,
            weights=weights,
            dtype=stat_dtype,
            name='counts_in_table')

        po_t = array_ops.diag_part(counts_in_table)
        pe_row_t = math_ops.reduce_sum(counts_in_table, axis=0)
        pe_col_t = math_ops.reduce_sum(counts_in_table, axis=1)
        update_po = state_ops.assign_add(po, po_t)
        update_pe_row = state_ops.assign_add(pe_row, pe_row_t)
        update_pe_col = state_ops.assign_add(pe_col, pe_col_t)

        def _calculate_k(po, pe_row, pe_col, name):
            po_sum = math_ops.reduce_sum(po)
            total = math_ops.reduce_sum(pe_row)
            pe_sum = math_ops.reduce_sum(
                math_ops.div(  # pylint: disable=protected-access
                    pe_row * pe_col, total, None)*weight_matrix)
            po_sum, pe_sum, total = (math_ops.to_double(po_sum),
                                     math_ops.to_double(pe_sum),
                                     math_ops.to_double(total))
            # kappa = (po - pe) / (N - pe)
            k = metrics_impl._safe_scalar_div(  # pylint: disable=protected-access
                po_sum - pe_sum,
                total - pe_sum,
                name=name)
            k_ = tf.keras.backend.cast(math_ops.reduce_sum(counts_in_table*weight_matrix), dtype='float64') / tf.keras.backend.cast(pe_sum, dtype='float64')
            return k
            # expected = np.outer(sum0, sum1) / np.sum(sum0)
            # confusion = confusion_matrix(y1, y2, labels=labels,
            #                              sample_weight=sample_weight)
            # k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
            # cohen_kappa_score()

        kappa = _calculate_k(po, pe_row, pe_col, name='value')
        update_op = _calculate_k(
            update_po, update_pe_row, update_pe_col, name='update_op')

        if metrics_collections:
            ops.add_to_collections(metrics_collections, kappa)

        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)

        return kappa


def cohen_kappa_my(labels,
                predictions_idx,
                num_classes,
                weights=None,
                metrics_collections=None,
                updates_collections=None,
                name=None, weight_matrix=None):
    """Calculates Cohen's kappa.

    [Cohen's kappa](https://en.wikipedia.org/wiki/Cohen's_kappa) is a statistic
    that measures inter-annotator agreement.

    The `cohen_kappa` function calculates the confusion matrix, and creates three
    local variables to compute the Cohen's kappa: `po`, `pe_row`, and `pe_col`,
    which refer to the diagonal part, rows and columns totals of the confusion
    matrix, respectively. This value is ultimately returned as `kappa`, an
    idempotent operation that is calculated by

        pe = (pe_row * pe_col) / N
        k = (sum(po) - sum(pe)) / (N - sum(pe))

    For estimation of the metric over a stream of data, the function creates an
    `update_op` operation that updates these variables and returns the
    `kappa`. `update_op` weights each prediction by the corresponding value in
    `weights`.

    Class labels are expected to start at 0. E.g., if `num_classes`
    was three, then the possible labels would be [0, 1, 2].

    If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

    NOTE: Equivalent to `sklearn.metrics.cohen_kappa_score`, but the method
    doesn't support weighted matrix yet.

    Args:
      labels: 1-D `Tensor` of real labels for the classification task. Must be
        one of the following types: int16, int32, int64.
      predictions_idx: 1-D `Tensor` of predicted class indices for a given
        classification. Must have the same type as `labels`.
      num_classes: The possible number of labels.
      weights: Optional `Tensor` whose shape matches `predictions`.
      metrics_collections: An optional list of collections that `kappa` should
        be added to.
      updates_collections: An optional list of collections that `update_op` should
        be added to.
      name: An optional variable_scope name.

    Returns:
      kappa: Scalar float `Tensor` representing the current Cohen's kappa.
      update_op: `Operation` that increments `po`, `pe_row` and `pe_col`
        variables appropriately and whose value matches `kappa`.

    Raises:
      ValueError: If `num_classes` is less than 2, or `predictions` and `labels`
        have mismatched shapes, or if `weights` is not `None` and its shape
        doesn't match `predictions`, or if either `metrics_collections` or
        `updates_collections` are not a list or tuple.
      RuntimeError: If eager execution is enabled.
    """
    if context.executing_eagerly():
        raise RuntimeError('tf.contrib.metrics.cohen_kappa is not supported '
                           'when eager execution is enabled.')
    if num_classes < 2:
        raise ValueError('`num_classes` must be >= 2.'
                         'Found: {}'.format(num_classes))
    with variable_scope.variable_scope(name, 'cohen_kappa',
                                       (labels, predictions_idx, weights)):
        # Convert 2-dim (num, 1) to 1-dim (num,)
        labels.get_shape().with_rank_at_most(2)
        if labels.get_shape().ndims == 2:
            labels = array_ops.squeeze(labels, axis=[-1])
        predictions_idx, labels, weights = (
            metrics_impl._remove_squeezable_dimensions(  # pylint: disable=protected-access
                predictions=predictions_idx,
                labels=labels,
                weights=weights))
        predictions_idx.get_shape().assert_is_compatible_with(labels.get_shape())

        stat_dtype = (
            dtypes.int64
            if weights is None or weights.dtype.is_integer else dtypes.float32)
        po = metrics_impl.metric_variable((num_classes,), stat_dtype, name='po')
        pe_row = metrics_impl.metric_variable(
            (num_classes,), stat_dtype, name='pe_row')
        pe_col = metrics_impl.metric_variable(
            (num_classes,), stat_dtype, name='pe_col')

        # Table of the counts of agreement:
        counts_in_table = confusion_matrix.confusion_matrix(
            labels,
            predictions_idx,
            num_classes=num_classes,
            weights=weights,
            dtype=stat_dtype,
            name='counts_in_table')

        po_t = array_ops.diag_part(counts_in_table)
        pe_row_t = math_ops.reduce_sum(counts_in_table, axis=0)
        pe_col_t = math_ops.reduce_sum(counts_in_table, axis=1)
        update_po = state_ops.assign_add(po, po_t)
        update_pe_row = state_ops.assign_add(pe_row, pe_row_t)
        update_pe_col = state_ops.assign_add(pe_col, pe_col_t)

        def _calculate_k(po, pe_row, pe_col, name):
            po_sum = math_ops.reduce_sum(po)
            total = math_ops.reduce_sum(pe_row)
            pe_sum = math_ops.reduce_sum(
                math_ops.div(  # pylint: disable=protected-access
                    pe_row * pe_col, total, None)*weight_matrix)
            po_sum, pe_sum, total = (math_ops.to_double(po_sum),
                                     math_ops.to_double(pe_sum),
                                     math_ops.to_double(total))
            # kappa = (po - pe) / (N - pe)
            # k = metrics_impl._safe_scalar_div(  # pylint: disable=protected-access
            #     po_sum - pe_sum,
            #     total - pe_sum,
            #     name=name)
            k_ = tf.keras.backend.cast(math_ops.reduce_sum(counts_in_table*weight_matrix), dtype='float64') / tf.keras.backend.cast(pe_sum, dtype='float64')
            return k_
            # expected = np.outer(sum0, sum1) / np.sum(sum0)
            # confusion = confusion_matrix(y1, y2, labels=labels,
            #                              sample_weight=sample_weight)
            # k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
            # cohen_kappa_score()

        kappa = _calculate_k(po, pe_row, pe_col, name='value')
        update_op = _calculate_k(
            update_po, update_pe_row, update_pe_col, name='update_op')

        if metrics_collections:
            ops.add_to_collections(metrics_collections, kappa)

        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)

        return kappa

def accuracy(y_true, y_pred):
    y_pred.get_shape().assert_is_compatible_with(y_true.get_shape())
    if y_true.dtype != y_pred.dtype:
        y_pred = math_ops.cast(y_pred, y_true.dtype)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)
    return math_ops.cast(math_ops.equal(y_true, y_pred), K.floatx())


def cappa_coen(y_true, y_pred):
    n_classes = 5
    w_mat = K.zeros([n_classes, n_classes], dtype='int64')
    w_mat = w_mat + K.arange(n_classes, dtype='int64')

    w_mat = (w_mat - K.transpose(w_mat)) ** 2
    cappa = cohen_kappa(K.argmax(y_true, axis=-1), K.argmax(y_pred), 5, weight_matrix=w_mat)
    return cappa


def cappa_coen_my(y_true, y_pred):
    n_classes = 5
    w_mat = K.zeros([n_classes, n_classes], dtype='int64')
    w_mat = w_mat + K.arange(n_classes, dtype='int64')

    w_mat = (w_mat - K.transpose(w_mat)) ** 2
    cappa = cohen_kappa_my(K.argmax(y_true, axis=-1), K.argmax(y_pred), 5, weight_matrix=w_mat)
    return cappa


metrics = [accuracy]