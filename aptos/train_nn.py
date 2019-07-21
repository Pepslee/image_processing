import argparse
import os

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.models import load_model

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
    data_frame = pd.read_csv(csv_path)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    for train_ind, test_ind in skf.split(data_frame, data_frame['diagnosis'], ):
        train_df = data_frame.iloc[train_ind]
        test_df = data_frame.iloc[train_ind]

        train_generator = DataGenerator(train_df, image_dir, batch_size)
        test_generator = DataGenerator(test_df, image_dir, 1)

        model = model_keras((256, 256, 3), 5)
        optimizer_type = optimizer('Adam', 0.001)

        model.compile(loss=loss, optimizer=optimizer_type, metrics=metrics)

        callbacks_params = {'checkpoints_path': ckpts_path,
                            'start_lr': 0.001,
                            'test_df': test_df,
                            'image_dir': image_dir}

        callbacks_list = callbacks(callbacks_params)

        print('Start ...')

        model.fit_generator(generator=iter(train_generator.generator()),
                            # steps_per_epoch=10,
                            steps_per_epoch=len(train_generator),
                            epochs=10,
                            validation_data=iter(test_generator.generator()),
                            validation_steps=len(test_generator),
                            callbacks=callbacks_list,
                            max_queue_size=1,
                            verbose=1,
                            workers=0)


if __name__ == '__main__':
    args = create_parser()
    main(args.csv_path, args.image_dir, args.ckpts_path, args.batch_size)
