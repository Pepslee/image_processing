import os
import warnings
from functools import reduce
from os.path import join
import shutil

import six

import cv2
import numpy as np
import h5py
import tensorflow.keras
from tensorflow.python.keras.callbacks import LearningRateScheduler, Callback
import tensorflow.keras.backend as K
import tensorflow as tf
from skimage.morphology import dilation, watershed, square, erosion, star, diamond, disk, label
from skimage.color import label2rgb
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python import keras
from sklearn.metrics import classification_report, precision_recall_fscore_support, cohen_kappa_score, f1_score

from aptos.tb_writers import TensorboardWriter
from sklearn.metrics import cohen_kappa_score

from tensorflow.python.keras.losses import categorical_crossentropy

from tensorflow.python.keras.utils import to_categorical
from aptos.data_generator import preproc



from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.distribute import multi_worker_training_state
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import checkpoint_management
from tensorflow.python.util.tf_export import keras_export



class TensorBoard(Callback):
  # pylint: disable=line-too-long
  """Enable visualizations for TensorBoard.

  TensorBoard is a visualization tool provided with TensorFlow.

  This callback logs events for TensorBoard, including:
  * Metrics summary plots
  * Training graph visualization
  * Activation histograms
  * Sampled profiling

  If you have installed TensorFlow with pip, you should be able
  to launch TensorBoard from the command line:

  ```sh
  tensorboard --logdir=path_to_your_logs
  ```

  You can find more information about TensorBoard
  [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

  Arguments:
      log_dir: the path of the directory where to save the log files to be
        parsed by TensorBoard.
      histogram_freq: frequency (in epochs) at which to compute activation and
        weight histograms for the layers of the model. If set to 0, histograms
        won't be computed. Validation data (or split) must be specified for
        histogram visualizations.
      write_graph: whether to visualize the graph in TensorBoard. The log file
        can become quite large when write_graph is set to True.
      write_images: whether to write model weights to visualize as image in
        TensorBoard.
      update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`,
        writes the losses and metrics to TensorBoard after each batch. The same
        applies for `'epoch'`. If using an integer, let's say `1000`, the
        callback will write the metrics and losses to TensorBoard every 1000
        samples. Note that writing too frequently to TensorBoard can slow down
        your training.
      profile_batch: Profile the batch to sample compute characteristics. By
        default, it will profile the second batch. Set profile_batch=0 to
        disable profiling. Must run in TensorFlow eager mode.
      embeddings_freq: frequency (in epochs) at which embedding layers will
        be visualized. If set to 0, embeddings won't be visualized.
      embeddings_metadata: a dictionary which maps layer name to a file name in
        which metadata for this embedding layer is saved. See the
        [details](
          https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
        about metadata files format. In case if the same metadata file is
        used for all embedding layers, string can be passed.

  Raises:
      ValueError: If histogram_freq is set and no validation data is provided.
  """

  # pylint: enable=line-too-long

  def __init__(self,
               log_dir='logs',
               fold=0,
               histogram_freq=0,
               write_graph=True,
               write_images=False,
               update_freq='epoch',
               profile_batch=2,
               embeddings_freq=0,
               embeddings_metadata=None,
               **kwargs):
    super(TensorBoard, self).__init__()
    self._validate_kwargs(kwargs)

    self.fold = fold
    self.log_dir = log_dir
    self.histogram_freq = histogram_freq
    self.write_graph = write_graph
    self.write_images = write_images
    if update_freq == 'batch':
      self.update_freq = 1
    else:
      self.update_freq = update_freq
    self.embeddings_freq = embeddings_freq
    self.embeddings_metadata = embeddings_metadata

    self._samples_seen = 0
    self._samples_seen_at_last_write = 0
    self._current_batch = 0
    self._total_batches_seen = 0
    self._total_val_batches_seen = 0

    # A collection of file writers currently in use, to be closed when
    # training ends for this callback. Writers are keyed by the
    # directory name under the root logdir: e.g., "train" or
    # "validation".
    self._writers = {}
    self._train_run_name = 'train'
    self._validation_run_name = 'validation'

    self._profile_batch = profile_batch
    # True when a trace is running.
    self._is_tracing = False

    # TensorBoard should only write summaries on the chief when in a
    # Multi-Worker setting.
    self._chief_worker_only = True

  def _validate_kwargs(self, kwargs):
    """Handle arguments were supported in V1."""
    if kwargs.get('write_grads', False):
      logging.warning('`write_grads` will be ignored in TensorFlow 2.0 '
                      'for the `TensorBoard` Callback.')
    if kwargs.get('batch_size', False):
      logging.warning('`batch_size` is no longer needed in the '
                      '`TensorBoard` Callback and will be ignored '
                      'in TensorFlow 2.0.')
    if kwargs.get('embeddings_layer_names', False):
      logging.warning('`embeddings_layer_names` is not supported in '
                      'TensorFlow 2.0. Instead, all `Embedding` layers '
                      'will be visualized.')
    if kwargs.get('embeddings_data', False):
      logging.warning('`embeddings_data` is not supported in TensorFlow '
                      '2.0. Instead, all `Embedding` variables will be '
                      'visualized.')

    unrecognized_kwargs = set(kwargs.keys()) - {
        'write_grads', 'embeddings_layer_names', 'embeddings_data', 'batch_size'
    }

    # Only allow kwargs that were supported in V1.
    if unrecognized_kwargs:
      raise ValueError('Unrecognized arguments in `TensorBoard` '
                       'Callback: ' + str(unrecognized_kwargs))

  def set_model(self, model):
    """Sets Keras model and writes graph if specified."""
    self.model = model
    with context.eager_mode():
      self._close_writers()
      if self.write_graph:
        with self._get_writer(self._train_run_name, self.fold).as_default():
          with summary_ops_v2.always_record_summaries():
            if not model.run_eagerly:
              summary_ops_v2.graph(K.get_graph(), step=0)

            summary_writable = (
                self.model._is_graph_network or  # pylint: disable=protected-access
                self.model.__class__.__name__ == 'Sequential')  # pylint: disable=protected-access
            if summary_writable:
              summary_ops_v2.keras_model('keras', self.model, step=0)

    if self.embeddings_freq:
      self._configure_embeddings()

  def _configure_embeddings(self):
    """Configure the Projector for embeddings."""
    # TODO(omalleyt): Add integration tests.
    from tensorflow.python.keras.layers import embeddings
    try:
      from tensorboard.plugins import projector
    except ImportError:
      raise ImportError('Failed to import TensorBoard. Please make sure that '
                        'TensorBoard integration is complete."')
    config = projector.ProjectorConfig()
    for layer in self.model.layers:
      if isinstance(layer, embeddings.Embedding):
        embedding = config.embeddings.add()
        embedding.tensor_name = layer.embeddings.name

        if self.embeddings_metadata is not None:
          if isinstance(self.embeddings_metadata, str):
            embedding.metadata_path = self.embeddings_metadata
          else:
            if layer.name in embedding.metadata_path:
              embedding.metadata_path = self.embeddings_metadata.pop(layer.name)

    if self.embeddings_metadata:
      raise ValueError('Unrecognized `Embedding` layer names passed to '
                       '`keras.callbacks.TensorBoard` `embeddings_metadata` '
                       'argument: ' + str(self.embeddings_metadata.keys()))

    class DummyWriter(object):
      """Dummy writer to conform to `Projector` API."""

      def __init__(self, logdir):
        self.logdir = logdir

      def get_logdir(self):
        return self.logdir

    writer = DummyWriter(self.log_dir)
    projector.visualize_embeddings(writer, config)

  def _close_writers(self):
    """Close all remaining open file writers owned by this callback.

    If there are no such file writers, this is a no-op.
    """
    with context.eager_mode():
      for writer in six.itervalues(self._writers):
        writer.close()
      self._writers.clear()

  def _get_writer(self, writer_name, fold):
    """Get a summary writer for the given subdirectory under the logdir.

    A writer will be created if it does not yet exist.

    Arguments:
      writer_name: The name of the directory for which to create or
        retrieve a writer. Should be either `self._train_run_name` or
        `self._validation_run_name`.

    Returns:
      A `SummaryWriter` object.
    """
    if writer_name not in self._writers:
      path = os.path.join(self.log_dir, str(fold), writer_name)
      writer = summary_ops_v2.create_file_writer_v2(path)
      self._writers[writer_name] = writer
    return self._writers[writer_name]

  def on_train_begin(self, logs=None):
    if self._profile_batch == 1:
      summary_ops_v2.trace_on(graph=True, profiler=True)
      self._is_tracing = True

  def on_batch_end(self, batch, logs=None):
    """Writes scalar summaries for metrics on every training batch.

    Performs profiling if current batch is in profiler_batches.

    Arguments:
      batch: Integer, index of batch within the current epoch.
      logs: Dict. Metric results for this batch.
    """
    # Don't output batch_size and batch number as TensorBoard summaries
    logs = logs or {}
    self._samples_seen += logs.get('size', 1)
    samples_seen_since = self._samples_seen - self._samples_seen_at_last_write
    if self.update_freq != 'epoch' and samples_seen_since >= self.update_freq:
      self._log_metrics(logs, prefix='batch_', step=self._total_batches_seen)
      self._samples_seen_at_last_write = self._samples_seen
    self._total_batches_seen += 1
    if self._is_tracing:
      self._log_trace()
    elif (not self._is_tracing and
          self._total_batches_seen == self._profile_batch - 1):
      self._enable_trace()

  def on_epoch_end(self, epoch, logs=None):
    """Runs metrics and histogram summaries at epoch end."""
    step = epoch if self.update_freq == 'epoch' else self._samples_seen
    self._log_metrics(logs, prefix=f'epoch_', step=step)

    if self.histogram_freq and epoch % self.histogram_freq == 0:
      self._log_weights(epoch)

    if self.embeddings_freq and epoch % self.embeddings_freq == 0:
      self._log_embeddings(epoch)

  def on_train_end(self, logs=None):
    if self._is_tracing:
      self._log_trace()
    self._close_writers()

  def _enable_trace(self):
    if context.executing_eagerly():
      summary_ops_v2.trace_on(graph=True, profiler=True)
      self._is_tracing = True

  def _log_trace(self):
    if context.executing_eagerly():
      with self._get_writer(self._train_run_name, self.fold).as_default(), \
          summary_ops_v2.always_record_summaries():
        # TODO(b/126388999): Remove step info in the summary name.
        summary_ops_v2.trace_export(
            name='batch_%d' % self._total_batches_seen,
            step=self._total_batches_seen,
            profiler_outdir=os.path.join(self.log_dir, 'train'))
      self._is_tracing = False

  def _log_metrics(self, logs, prefix, step):
    """Writes metrics out as custom scalar summaries.

    Arguments:
        logs: Dict. Keys are scalar summary names, values are NumPy scalars.
        prefix: String. The prefix to apply to the scalar summary names.
        step: Int. The global step to use for TensorBoard.
    """
    if logs is None:
      logs = {}

    # Group metrics by the name of their associated file writer. Values
    # are lists of metrics, as (name, scalar_value) pairs.
    logs_by_writer = {
        self._train_run_name: [],
        self._validation_run_name: [],
    }
    validation_prefix = 'val_'
    for (name, value) in logs.items():
      if name in ('batch', 'size', 'num_steps'):
        # Scrub non-metric items.
        continue
      if name.startswith(validation_prefix):
        name = name[len(validation_prefix):]
        writer_name = self._validation_run_name
      else:
        writer_name = self._train_run_name
      name = prefix + name  # assign batch or epoch prefix
      logs_by_writer[writer_name].append((name, value))

    with context.eager_mode():
      with summary_ops_v2.always_record_summaries():
        for writer_name in logs_by_writer:
          these_logs = logs_by_writer[writer_name]
          if not these_logs:
            # Don't create a "validation" events file if we don't
            # actually have any validation data.
            continue
          writer = self._get_writer(writer_name, self.fold)
          with writer.as_default():
            for (name, value) in these_logs:
              summary_ops_v2.scalar(name, value, step=step)

  def _log_weights(self, epoch):
    """Logs the weights of the Model to TensorBoard."""
    writer = self._get_writer(self._train_run_name, self.fold)
    with context.eager_mode(), \
          writer.as_default(), \
          summary_ops_v2.always_record_summaries():
      for layer in self.model.layers:
        for weight in layer.weights:
          weight_name = weight.name.replace(':', '_')
          with ops.init_scope():
            weight = K.get_value(weight)
          summary_ops_v2.histogram(weight_name, weight, step=epoch)
          if self.write_images:
            self._log_weight_as_image(weight, weight_name, epoch)
      writer.flush()

  def _log_weight_as_image(self, weight, weight_name, epoch):
    """Logs a weight as a TensorBoard image."""
    w_img = array_ops.squeeze(weight)
    shape = K.int_shape(w_img)
    if len(shape) == 1:  # Bias case
      w_img = array_ops.reshape(w_img, [1, shape[0], 1, 1])
    elif len(shape) == 2:  # Dense layer kernel case
      if shape[0] > shape[1]:
        w_img = array_ops.transpose(w_img)
        shape = K.int_shape(w_img)
      w_img = array_ops.reshape(w_img, [1, shape[0], shape[1], 1])
    elif len(shape) == 3:  # ConvNet case
      if K.image_data_format() == 'channels_last':
        # Switch to channels_first to display every kernel as a separate
        # image.
        w_img = array_ops.transpose(w_img, perm=[2, 0, 1])
        shape = K.int_shape(w_img)
      w_img = array_ops.reshape(w_img, [shape[0], shape[1], shape[2], 1])

    shape = K.int_shape(w_img)
    # Not possible to handle 3D convnets etc.
    if len(shape) == 4 and shape[-1] in [1, 3, 4]:
      summary_ops_v2.image(weight_name, w_img, step=epoch)

  def _log_embeddings(self, epoch):
    embeddings_ckpt = os.path.join(self.log_dir, 'train',
                                   'keras_embedding.ckpt-{}'.format(epoch))
    self.model.save_weights(embeddings_ckpt)



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
        self.epoch = epoch
        self.save_path = f'{self.ckpts_path}/best_model_{self.k}_e_{self.epoch}.h5'
        self.save_by_metric(self.epoch, logs)
        # loss = logs.get('loss')
        # val_loss = logs.get('val_loss')
        # if loss is not None:
        #     self.tb_writer.log_scalar(self.log_path, 'train_loss', [loss], self.epoch, str(self.k) + '_loss')
        # if val_loss is not None:
        #     self.tb_writer.log_scalar(self.log_path, 'val_loss', [val_loss], self.epoch, str(self.k) + '_loss')
        #
        #
        # accuracy = logs.get('accuracy')
        # val_accuracy = logs.get('val_accuracy')
        # if accuracy is not None:
        #     self.tb_writer.log_scalar(self.log_path, 'train_accuracy', [accuracy], self.epoch, str(self.k) + '_accuracy')
        # if val_accuracy is not None:
        #     self.tb_writer.log_scalar(self.log_path, 'val_accuracy', [val_accuracy], self.epoch, str(self.k) + '_accuracy')

    def save_by_metric(self, iteration, logs):

        # y_pred = []
        # y_true = []
        # for i, row in self.test_df.iterrows():
        #     image_path = join(self.image_dir, row['id_code'] + '.png')
        #     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        #     image = preproc(image)
        #     image = np.expand_dims(image, axis=0)
        #     y_pred.append(np.argmax(self.model.predict(image)[0], axis=-1))
        #     y_true.append(int(row['diagnosis']))
        #
        #
        # diff = absdiff(y_pred, y_true)
        # self.tb_writer.log_scalar(self.log_path, 'diff', [diff], iteration, str(self.k))
        #
        # f1 = f1_score(to_categorical(y_pred, 5), to_categorical(y_true, 5), average='macro')
        # self.tb_writer.log_scalar(self.log_path, 'f1', [f1], iteration, str(self.k))
        #
        #
        #
        # ck = cohen_kappa_score(y_pred, y_true)
        # self.tb_writer.log_scalar(self.log_path, 'kappa', [ck], iteration, str(self.k))

        ck = logs.get('cappa_coen')
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
    tb = TensorBoard(log_dir=args['log_path'], fold=args['fold'])
    return [best_model, lr_callback, tb]
