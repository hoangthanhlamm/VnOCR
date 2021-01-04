import numpy as np
import pandas as pd
import cv2
from keras.callbacks import Callback
from tqdm import tqdm

import random
from datetime import datetime

from libs.utils.utils import word_to_label
from config import *


class DataGenerator(Callback):

    def __init__(self, img_dirpath, img_width_, img_height_, batch_size_, n, output_labels, max_text_len=15):
        self.img_width = img_width_
        self.img_height = img_height_
        self.batch_size = batch_size_
        self.max_text_len = max_text_len

        self.n = n
        self.img_dir = img_dirpath[:self.n]
        self.indexes = list(range(self.n))
        self.cur_index = 0
        self.imgs = np.zeros((self.n, self.img_height, self.img_width))
        self.texts = output_labels[:self.n]

    def build_data(self):
        print("Image Loading start with {} images...".format(self.n))
        start = datetime.now()
        cnt = 0
        for img_file in tqdm(self.img_dir):
            img = cv2.imread(os.path.join(data_path, img_file))
            img = img[:, :, 1]
            img = cv2.resize(img, (self.img_width, self.img_height))
            img = img / 255
            self.imgs[cnt, :, :] = img
            cnt += 1
        end = datetime.now()
        print("Number of Texts matches with Total Number of Images :", len(self.texts) == self.n)
        print(self.n, "\tImage Loading finish...")
        print("Total Time: ", end - start)

    def next_data(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def next_batch(self):
        while True:
            X_data = np.ones([self.batch_size, self.img_width, self.img_height, 1])  # Single channel Gray Size Scale images for input
            # Initializing with -1 to aid for padding labels of different lengths
            Y_data = np.ones([self.batch_size, self.max_text_len]) * -1  # Text labels for input
            # Input_length for CTC which is the number of time-steps of the RNN output
            input_length = np.ones((self.batch_size, 1)) * pred_length  # Model predicted output length ignore 2 first letter
            label_length = np.zeros((self.batch_size, 1))  # Label length for CTC
            source_str = []  # List to store Ground Truth Labels

            for i in range(self.batch_size):
                img, text = self.next_data()  # Getting the image and text data pointed by current index
                # Taking transpose of image
                img = img.T
                img = np.expand_dims(img, -1)  # Expanding image to have a single channel
                X_data[i] = img
                label = word_to_label(text)  # Encoding label text to integer list and storing in temp label variable
                lbl_len = len(label)
                Y_data[i, 0:lbl_len] = label  # Storing the label till its length and padding others
                label_length[i] = len(label)
                source_str.append(text)  # Storing Ground Truth Labels which will be accessed as reference for calculating metrics

            # Preparing the input for the Model
            inputs = {
                'img_input': X_data,
                'ground_truth_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length,
                'source_str': np.array(source_str)  # Used for visualization only
            }
            # Preparing output for the Model and initializing to zeros
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield inputs, outputs  # Return the Prepared input and output to the Model


def get_generator(mode):
    filename = mode + '_final.csv'
    data = pd.read_csv(os.path.join(csv_path, filename), sep=';')
    paths = data['Image'].values.tolist()
    labels = data['Label'].values.tolist()
    if mode == 'val':
        gene = DataGenerator(paths, img_width, img_height, batch_size, val_size, labels, max_text_len=max_length)
    else:
        gene = DataGenerator(paths, img_width, img_height, batch_size, train_size, labels, max_text_len=max_length)
    gene.build_data()
    n_batches = int(gene.n / gene.batch_size)
    return gene, n_batches
