import numpy as np
from keras.callbacks import Callback

from libs.utils.utils import decode_batch, accuracies


class VizCallback(Callback):
    """
    The Custom Callback created for printing the Accuracy and Letter Accuracy Metrics at the End of Each Epoch
    """

    def __init__(self, test_func, text_img_gen, is_train, acc_compute_batches):
        self.test_func = test_func
        self.text_img_gen = text_img_gen
        self.is_train = is_train                # Used to indicate whether the callback is called to for Train or Validation Data
        self.acc_batches = acc_compute_batches  # Number of Batches for which the metrics are computed typically equal to steps/epoch

    def show_accuracy_metrics(self, num_batches):
        """
        Calculates the accuracy and letter accuracy for each batch of inputs,
        and prints the avarage accuracy and letter accuracy across all the batches
        """
        accuracy = 0
        letter_accuracy = 0
        batches_cnt = num_batches
        while batches_cnt > 0:
            word_batch = next(self.text_img_gen)[0]   # Gets the next batch from the Data generator
            decoded_res = decode_batch(self.test_func, word_batch['img_input'])
            actual_res = word_batch['source_str']
            acc, let_acc = accuracies(actual_res, decoded_res)
            accuracy += acc
            letter_accuracy += let_acc
            batches_cnt -= 1
        accuracy = accuracy / num_batches
        letter_accuracy = letter_accuracy / num_batches
        if self.is_train:
            print("\nTrain Average Accuracy: \t", np.round(accuracy, 2), " %")
            print("Train Average Letter Accuracy: \t", np.round(letter_accuracy, 2), " %")
        else:
            print("Validation Average Accuracy: \t\t", np.round(accuracy, 2), " %")
            print("Validation Average Letter Accuracy: \t", np.round(letter_accuracy, 2), " %")

    def on_epoch_end(self, epoch, logs=None):
        self.show_accuracy_metrics(self.acc_batches)
