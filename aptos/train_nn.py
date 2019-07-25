import argparse
import os
import shutil

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.framework.ops import reset_default_graph
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K
import tensorflow as tf

from aptos.data_generator import DataGenerator
from aptos.model import model_keras, optimizer
from aptos.loss import loss
from aptos.metrics import metrics
from aptos.callbacks import callbacks


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', type=str, help='Path to csv file.')
    parser.add_argument('image_dir', type=str, help='Path to image directory.')
    parser.add_argument('ckpts_path', type=str, help='Path to output ckpts.')
    parser.add_argument('-bs', '--batch_size', type=int, required=False, default=5, help='Batch size')

    return parser.parse_args()


def main(csv_path, image_dir, ckpts_path, batch_size):
    csv_path = os.path.abspath(csv_path)
    image_dir = os.path.abspath(image_dir)
    ckpts_path = os.path.abspath(ckpts_path)

    data_frame = pd.read_csv(csv_path)

    callbacks_params = {'checkpoints_path': ckpts_path,
                        'start_lr': 0.0001,
                        'image_dir': image_dir}

    callbacks_params['log_path'] = os.path.join(callbacks_params['checkpoints_path'], 'log')
    if not os.path.exists(callbacks_params['checkpoints_path']):
        os.makedirs(callbacks_params['checkpoints_path'])
    else:
        shutil.rmtree(callbacks_params['checkpoints_path'])
        os.makedirs(callbacks_params['checkpoints_path'])
    if not os.path.exists(callbacks_params['log_path']):
        os.makedirs(callbacks_params['log_path'])
    else:
        shutil.rmtree(callbacks_params['log_path'])
        os.makedirs(callbacks_params['log_path'])

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=1)
    for k, (train_ind, test_ind) in enumerate(skf.split(data_frame, data_frame['diagnosis'], )):
        g = tf.Graph()
        with g.as_default():
            train_df = data_frame.iloc[train_ind]
            test_df = data_frame.iloc[train_ind]

            train_generator = DataGenerator(train_df, image_dir, batch_size, 'train')
            test_generator = DataGenerator(test_df, image_dir, 1, 'test')

            model = model_keras(k)
            optimizer_type = optimizer('Adam', 0.00005)

            model.compile(loss=loss, optimizer=optimizer_type, metrics=metrics)
            model.summary()

            # add test_data and fold number to callbacks params dict
            callbacks_params['test_df'] = test_df
            callbacks_params['fold'] = k

            callbacks_list = callbacks(callbacks_params)

            print('Start ...')

            model.fit_generator(generator=iter(train_generator.generator()),
                                # steps_per_epoch=10,
                                steps_per_epoch=len(train_generator),
                                epochs=50,
                                validation_data=iter(test_generator.generator()),
                                validation_steps=len(test_generator),
                                # validation_steps=10,
                                callbacks=callbacks_list,
                                max_queue_size=1,
                                verbose=1,
                                workers=0)


if __name__ == '__main__':
    args = create_parser()
    main(args.csv_path, args.image_dir, args.ckpts_path, args.batch_size)
