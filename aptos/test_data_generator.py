from aptos.data_generator import DataGenerator
import pandas as pd
import numpy as np
import cv2

csv_path = '/mnt/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/aptos2019-blindness-detection/train.csv'

image_dir = '/mnt/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/aptos2019-blindness-detection/train_images'


train_df = pd.read_csv(csv_path)

data = DataGenerator(train_df, image_dir, 1, 'train')

it = iter(data.generator())

for i in range(100):
    image, di = next(it)
    image = np.squeeze(image)


    cv2.namedWindow('image', 0)
    cv2.imshow('image', image)
    cv2.waitKey()
