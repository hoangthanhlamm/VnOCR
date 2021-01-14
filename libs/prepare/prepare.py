import gdown
import pandas as pd
from sklearn.model_selection import train_test_split

import logging
import os
import shutil

from config import csv_path, data_path, checkpoint_path, download_data_url, download_model_url


def download_data():
    data_dest = os.path.join(data_path, 'data.tar.gz')

    model_path = os.path.join(data_path, 'models')
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_dest = os.path.join(model_path, 'vn_model.h5')

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    try:
        gdown.download(download_data_url, data_dest)
        gdown.download(download_model_url, model_dest)
        shutil.unpack_archive(data_dest, data_path)
    except Exception as err:
        logging.exception(err)


def split_dataset(test_size=0.2, val_size=0.16):
    data = pd.read_csv(os.path.join(csv_path, 'words.csv'), sep=';')
    tmp, test = train_test_split(data, test_size=test_size, random_state=12)
    test.to_csv(os.path.join(csv_path, 'test.csv'), index=False, sep=';')

    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(tmp, test_size=val_ratio, random_state=12)


    train.to_csv(os.path.join(csv_path, 'train.csv'), index=False, sep=';')
    val.to_csv(os.path.join(csv_path, 'val.csv'), index=False, sep=';')


if __name__ == '__main__':
    download_data()
    split_dataset()
