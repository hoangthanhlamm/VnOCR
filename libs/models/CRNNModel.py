import os
import logging
import codecs
from datetime import datetime

import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

from libs.nets.CRNN import CRNN
from libs.prepare.generator import get_generator
from libs.utils.callbacks import VizCallback
from libs.utils.utils import predict_label, ctc_loss_function, predict_data_output
from libs.word_beam_search.language_model import LanguageModel
from config import data_path, letters, word_chars, checkpoint_path


class CRNNModel(object):
    def __init__(self, model_path, initial_state=True):
        self.model_path = model_path

        self.chars = letters
        self.lm = self.get_language_model(data_path, word_chars)

        if initial_state:
            self.build_model('train')
        else:
            self.build_model('predict')

    def build_model(self, mode='train'):
        crnn = CRNN(stage=mode, loss_fn=ctc_loss_function)
        if mode == 'train':
            model_input, y_pred, self.model = crnn()
            self.test_func = K.function([model_input], [y_pred])

            self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
        else:
            self.model = crnn()
            self.load_model()

    def get_language_model(self, path, word_characters):
        corpus = codecs.open(os.path.join(path, 'corpus.txt'), 'r', 'utf8').read()
        lm = LanguageModel(corpus, self.chars, word_characters)
        return lm

    def load_model(self, model_path=None):
        if model_path is None:
            self.model.load_weights(self.model_path)
        else:
            self.model.load_weights(model_path)

    def save_model(self, model_save_path=None):
        if model_save_path is None:
            self.model.save(self.model_path)
        else:
            self.model.save(model_save_path)

    def get_model_description(self, info_path):
        try:
            self.model.summary()
            plot_model(self.model, info_path, show_shapes=True)
        except Exception as err:
            logging.exception(err)

    def plot_learning_curve(self):
        try:
            plt.title('Learning Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.plot(self.history.history['loss'], label='train')
            plt.plot(self.history.history['val_loss'], label='val')
            plt.legend()
            plt.show()
        except Exception as err:
            logging.exception(err)

    def fit(self, epochs=20, early_stopping=False):
        callbacks = []

        train_gene, train_n_batches = get_generator(mode='train')
        val_gene, val_n_batches = get_generator(mode='val')
        callbacks.extend([train_gene, val_gene])

        train_viz_cb = VizCallback(self.test_func, train_gene.next_batch(), True, train_n_batches)
        val_viz_cb = VizCallback(self.test_func, val_gene.next_batch(), False, val_n_batches)
        callbacks.extend([train_viz_cb, val_viz_cb])

        if early_stopping:
            early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            callbacks.append(early_stop)

        model_checkpoint = ModelCheckpoint(
            os.path.join(checkpoint_path, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
            monitor='val_loss', save_best_only=False,
            save_weights_only=True, verbose=0, mode='auto'
        )
        callbacks.append(model_checkpoint)

        start = datetime.now()
        self.history = self.model.fit_generator(
            generator=train_gene.next_batch(),
            steps_per_epoch=train_n_batches,
            epochs=epochs,
            callbacks=[train_viz_cb, val_viz_cb, train_gene, val_gene, model_checkpoint],
            validation_data=val_gene.next_batch(),
            validation_steps=val_n_batches
        )
        end = datetime.now()
        print("Time to train: ", end - start)

        self.save_model('models/model.h5')

    def evaluate(self, X, y):
        acc, letter_acc, letter_cnt, mis_match, n_predicteds = predict_data_output(self.model, X, y, self.lm)

        accuracy = round((acc / n_predicteds) * 100, 2)
        letter_accuracy = round((letter_acc / letter_cnt) * 100, 2)

        print("Validation Accuracy: ", accuracy, " %")
        print("Validation Letter Accuracy: ", letter_accuracy, " %")

        return accuracy, letter_accuracy

    def predict(self, x):
        predicted = predict_label(self.model, x, self.lm)
        return predicted
