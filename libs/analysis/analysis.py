import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import os

from libs.utils.utils import plot_cdf
from config import letters, data_path


class DataAnalysis:
    def __init__(self, path):
        data = pd.read_csv(path, sep=';')
        self.img_paths = data['Image'].values.tolist()
        self.labels = data['Label'].values.tolist()
        self.letters = letters

        self.data = self.analysis()
        self.out_path = path.replace('.csv', '_final.csv')
        self.to_csv()

    def analysis(self):
        data = []
        cnt = 0
        idx = 0
        for img_path in tqdm(self.img_paths):
            label = self.labels[idx]
            idx += 1
            try:
                length = len(label)
            except TypeError:
                continue

            img = cv2.imread(os.path.join(data_path, img_path))
            if img is None:
                continue
            data.append([img_path, img.shape[0], img.shape[1], length, label])
            # size = [img.shape[0], img.shape[1]]
            # sizes.append(size)
            # lengths.append(len(label))
            cnt += 1

        print("Total images: {:} / {:}".format(cnt, len(self.img_paths)))
        return pd.DataFrame(data, columns=['Image', 'Height', 'Width', 'Length', 'Label'])

    def height_analysis(self, percent=90):
        heights = self.data['Height']

        for i in range(percent, 100):
            print("Images Height " + str(i) + " percentile :", np.round(np.percentile(heights, i)))
        print("Max Images Height: ", np.max(heights))
        plot_cdf(heights, title='Images Height CDF Plot', xlabel='Images Height')

    def width_analysis(self, percent=90):
        widths = self.data['Width']

        for i in range(percent, 100):
            print("Images Width " + str(i) + " percentile :", np.round(np.percentile(widths, i)))
        print("Max Images Width: ", np.max(widths))
        plot_cdf(widths, title='Images Width CDF Plot', xlabel='Images Width')

    def length_analysis(self, percent=90):
        lengths = self.data['Length']
        for i in range(percent, 100):
            print("Labels Length " + str(i) + " percentile :", np.round(np.percentile(lengths, i)))
        print("Max Labels Length: ", np.max(lengths))
        plot_cdf(lengths, title='Labels Length CDF Plot', xlabel='Labels Length')

    def to_csv(self):
        self.data.to_csv(self.out_path, sep=';', index=False)

    def character_analysis(self):
        cnts = [0] * len(self.letters)
        for word in self.labels:
            if type(word) != str:
                continue
            for c in word:
                idx = self.letters.index(c)
                cnts[idx] += 1

        plt.figure(figsize=(20, 6))
        plt.plot(cnts)
        plt.xticks(range(len(self.letters)), self.letters)
        plt.show()

        letters_ = []
        for idx, c in enumerate(self.letters):
            if cnts[idx] < 50:
                # counts.append(cnts[idx])
                letters_.append(c)
                # indicates.append(idx)

        return letters_
