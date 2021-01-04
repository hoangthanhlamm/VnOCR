import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras import backend as K
import itertools
from tqdm import tqdm

from config import *
from libs.word_beam_search.word_beam_search import word_beam_search

import logging
from datetime import datetime


def word_to_label(word):
    label_lst = []
    for char in word:
        label_lst.append(letters.find(char))
    return label_lst


def label_to_word(label):
    txt = []
    for c in label:
        if c < len(letters):
            txt.append(letters[c])
        else:
            txt.append('')
    return "".join(txt)


def ctc_loss_function(args):
    y_pred, y_true, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)


def decode_label(lm, out, word=True):
    mat = out[0, 2:, :]
    if word:
        out_str = word_beam_search(mat, beam_width, lm, False)
    else:
        out_best = list(np.argmax(mat, axis=1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        out_str = label_to_word(out_best)
    return out_str


def decode_batch(test_function, word_batch):
    """
    Decode batch labels to words
    """
    out = test_function([word_batch])[0]
    ret = []
    for i in range(out.shape[0]):
        out_best = list(np.argmax(out[i, 2:], axis=1))
        out_best = [k for k, _ in itertools.groupby(out_best)]
        out_str = label_to_word(out_best)
        ret.append(out_str)
    return ret


def accuracies(actual_labels, predicted_labels):
    """
    Calculate the accuracy and letter accuracy on a batch
    """
    acc = 0
    letter_acc = 0
    letter_cnt = 0
    cnt = 0
    for i in range(len(actual_labels)):
        predicted_output = predicted_labels[i]
        actual_output = actual_labels[i]
        cnt += 1
        for j in range(min(len(predicted_output), len(actual_output))):
            if predicted_output[j] == actual_output[j]:
                letter_acc += 1
        letter_cnt += max(len(predicted_output), len(actual_output))
        if actual_output == predicted_output:
            acc += 1
    final_accuracy = np.round((acc / len(actual_labels)) * 100, 2)
    final_letter_accuracy = np.round((letter_acc / letter_cnt) * 100, 2)
    return final_accuracy, final_letter_accuracy


def predict_label(model, image, lm):
    try:
        img = cv2.imread(image)
        if img is None:
            logging.warning('Image not found')
            return None
        img = cv2.resize(img, (img_width, img_height))
        img = img[:, :, 1]
        img = img.T
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        img = img / 255
        out = model.predict(img)
        predicted = decode_label(lm, out)
        return predicted
    except Exception as e:
        logging.exception(e)


def predict_data_output(model, images, labels, lm, n=None):
    if n is None or n > len(images):
        n = len(images)
    start = datetime.now()
    acc = 0
    letter_acc = 0
    letter_cnt = 0
    cnt = 0
    letter_mis = []  # Count miss match letter for each images
    predicteds = []
    for i in tqdm(range(n)):
        img_path = os.path.join(data_path, images[i])
        predicted = predict_label(model, img_path, lm)
        if predicted is None:
            continue
        predicteds.append(predicted)
        actual = labels[i]

        cnt += 1
        mis = 0
        for j in range(min(len(predicted), len(actual))):
            if predicted[j] == actual[j]:
                letter_acc += 1
            else:
                mis += 1
        letter_cnt += max(len(predicted), len(actual))
        letter_mis.append(mis)

        if predicted == actual:
            acc += 1
        # if cnt % 100 == 0:
        #     print("Processed {} images".format(cnt))
    end = datetime.now()
    print("Total Times: ", end - start)
    return acc, letter_acc, letter_cnt, letter_mis, cnt


def plot_cdf(x, title='CDF Plot', xlabel='Values', ylabel='CDF'):
    counts, bin_edges = np.histogram(x, bins=12, density=True)
    pdf = counts / np.sum(counts)
    cdf = np.cumsum(pdf)

    plt.figure(figsize=(8, 6))
    plt.plot(bin_edges[1:], cdf)
    plt.title(title, fontsize=24)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.show()
