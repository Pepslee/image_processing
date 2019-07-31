import pandas as pd
import matplotlib.pyplot as plt


path_test = '/mnt/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/aptos2019-blindness-detection/best_model_0.csv'
path_train = '/mnt/75b9aae6-291e-4314-90c2-b27cf3e3f5cd/Kaggle/aptos2019-blindness-detection/train.csv'

train_frame = pd.read_csv(path_train)
test_frame = pd.read_csv(path_test)


train_hist = train_frame.hist(bins=5)
test_hist = test_frame.hist(bins=5)

plt.show()
