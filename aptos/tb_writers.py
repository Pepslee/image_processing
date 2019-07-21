import os
from io import BytesIO

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class TensorboardWriter:
    def __init__(self):
        self.writers = dict()

    # def log_image(self, log_dir, tag, images, iteration, phase_name):
    #     """Log an image to tensorboard."""
    #     for nr, img in enumerate(images):
    #         save_dir = os.path.join(log_dir, phase_name if len(images) == 1 else str(nr), tag)
    #         # Write the image to a string
    #         s = BytesIO()
    #         if len(img.shape) > 2:
    #             plt.imsave(s, img, format='png')
    #         else:
    #             plt.imsave(s, img, format='png', vmax=np.max(img), cmap=plt.get_cmap('gray'))
    #         plt.close()
    #         # Create an Image object
    #         img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(), height=img.shape[0], width=img.shape[1])
    #         # Create a Summary value
    #         im_summaries = tf.Summary.Value(tag=tag, image=img_sum)
    #         # Create and write Summary
    #         summary = tf.Summary(value=[im_summaries])
    #         if save_dir in self.writers:
    #             writer = self.writers[save_dir]
    #         else:
    #             writer = tf.summary.create_file_writer(save_dir)
    #             self.writers[save_dir] = writer
    #         writer.add_summary(summary, iteration)
    #         writer.flush()

    def log_image(self, log_dir, tag, images, iteration, phase_name):
        """Log an image to tensorboard."""
        for nr, img in enumerate(images):
            save_dir = os.path.join(log_dir, phase_name if len(images) == 1 else str(nr), tag)
            # Write the image to a string
            # summary = tf.summary.image(name=tag, data=img)
            if save_dir in self.writers:
                writer = self.writers[save_dir]
            else:
                writer = tf.summary.create_file_writer(save_dir)
                self.writers[save_dir] = writer
            with writer.as_default():
                tf.summary.image(name=tag, data=np.expand_dims(img, axis=0), step=iteration)

    def log_scalar(self, log_dir, tag, values, iteration, phase_name):
        """Log a scalar to tensorboard."""
        values = np.array(values)
        for i, value in enumerate(values):
            save_dir = os.path.join(log_dir, phase_name if values.shape[0] == 1 else str(i), tag)
            # summary = tf.Summary(value=[
            #     tf.Summary.Value(tag='%s%s' % ('' if values.shape[0] == 1 else phase_name.lower() + '_', tag),
            #                      simple_value=value)])

            if save_dir in self.writers:
                writer = self.writers[save_dir]
            else:
                writer = tf.summary.create_file_writer(save_dir)
                self.writers[save_dir] = writer
            with writer.as_default():
                tf.summary.scalar(name='%s%s' % ('' if values.shape[0] == 1 else phase_name.lower() + '_', tag), data=value, step=iteration)

    def log_graph(self, log_dir, graph):
        writer = tf.summary.create_file_writer(logdir=log_dir, graph=graph)
        writer.flush()

