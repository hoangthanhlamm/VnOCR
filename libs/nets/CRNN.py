from keras import backend as K
from keras.layers import Activation, Bidirectional
from keras.layers import BatchNormalization, Dropout
from keras.layers import Input, Conv2D, Dense, MaxPooling2D
from keras.layers import Lambda
from keras.layers import Reshape
from keras.layers.recurrent import LSTM
from keras.models import Model

from config import *


class CRNN(object):
    def __init__(self, stage, loss_fn, dropout=0.35):
        self.stage = stage
        self.loss_function = loss_fn
        self.dropout = dropout

    def __call__(self, *args, **kwargs):
        if K.image_data_format == 'channels_first':
            input_shape = (1, img_width, img_height)
        else:
            input_shape = (img_width, img_height, 1)

        model_input = Input(shape=input_shape, name='img_input', dtype='float32')

        # Convolution Layer
        model = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(model_input)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = MaxPooling2D(pool_size=(2, 2), name='max1')(model)

        model = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = MaxPooling2D(pool_size=(2, 2), name='max2')(model)

        model = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(model)
        model = Dropout(self.dropout)(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = MaxPooling2D(pool_size=(1, 2), name='max3')(model)

        model = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = Conv2D(512, (3, 3), padding='same', name='conv6')(model)
        model = Dropout(self.dropout)(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)
        model = MaxPooling2D(pool_size=(1, 2), name='max4')(model)

        model = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(model)
        model = Dropout(0.25)(model)
        model = BatchNormalization()(model)
        model = Activation('relu')(model)

        # CNN to RNN
        model = Reshape(target_shape=(42, 2048), name='reshape')(model)
        model = Dense(128, activation='relu', kernel_initializer='he_normal', name='dense1')(model)

        # Recurrent Layer
        model = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='he_normal'), merge_mode='sum')(model)
        model = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='he_normal'), merge_mode='concat')(model)

        # Transforms RNN output to character activations:
        model = Dense(n_classes, kernel_initializer='he_normal', name='dense2')(model)
        y_pred = Activation('softmax', name='softmax')(model)

        labels = Input(name='ground_truth_labels', shape=[max_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        # CTC loss function
        loss_out = Lambda(self.loss_function, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        if self.stage == 'train':
            return model_input, y_pred, Model(inputs=[model_input, labels, input_length, label_length], outputs=loss_out)
        else:
            return Model(inputs=[model_input], outputs=y_pred)
