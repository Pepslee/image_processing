import os
import glob
import cv2
from tqdm import tqdm

load_path = '/mnt/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/aptos2019-blindness-detection/train_croped_images'
save_path = '/mnt/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/aptos2019-blindness-detection/train_croped_images_1024'

globs = glob.glob(os.path.join(load_path, '*.png'))

SIZE = 1024

for image_path in tqdm(globs):
    image_name = os.path.basename(image_path)
    image = cv2.imread(image_path)
    y_size, x_size = image.shape[:2]
    k = float(y_size)/x_size
    resized_image = cv2.resize(image, (SIZE, int(k*SIZE)))
    cv2.imwrite(os.path.join(save_path, image_name), resized_image)
