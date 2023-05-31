import pandas as pd
import numpy as np
import tensorflow as tf
import os
import json

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from gensim.models import KeyedVectors


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def build_lstm(input_layer):

    layer = tf.keras.layers.Dropout(0.2)(input_layer)
    layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=True))(layer)
    layer = tf.keras.layers.Dropout(0.2)(layer)
    layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, return_sequences=True))(layer)
    layer = tf.keras.layers.Dropout(0.2)(layer)
    layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=False))(layer)

    return layer


def build_cnn(input_layer, start_neurons=8, kernel_size=4, dropout=0.25):

    conv1 = tf.keras.layers.Conv1D(start_neurons * 1, kernel_size, activation="relu", padding="same")(input_layer)
    pool1 = tf.keras.layers.MaxPooling1D(2)(conv1)

    conv2 = tf.keras.layers.Conv1D(start_neurons * 2, kernel_size, activation="relu", padding="same")(pool1)
    pool2 = tf.keras.layers.MaxPooling1D(2)(conv2)

    # Middle
    convm = tf.keras.layers.Conv1D(start_neurons * 16, kernel_size, activation="relu", padding="same")(pool2)

    deconv2 = tf.keras.layers.Conv1DTranspose(start_neurons * 2, kernel_size, strides=2, padding="same")(convm)
    uconv2 = tf.keras.layers.concatenate([deconv2, conv2])
    uconv2 = tf.keras.layers.Conv1D(start_neurons * 2, kernel_size, activation="relu", padding="same")(uconv2)

    deconv1 = tf.keras.layers.Conv1DTranspose(start_neurons * 1, kernel_size, strides=2, padding="same")(uconv2)
    uconv1 = tf.keras.layers.concatenate([deconv1, conv1])

    uconv1 = tf.keras.layers.Conv1D(start_neurons * 1, kernel_size, activation="relu", padding="same")(uconv1)

    output_layer = tf.keras.layers.GlobalMaxPool1D()(uconv1)

    # output_layer = layers.Dense(128, activation='relu')(output_layer)
    # output_layer = layers.Conv1D(1, 1, padding="same", activation="sigmoid")(uconv1)

    return output_layer


def build_model(input_shape, num_words, class_num, gensim_weight_matrix, LABEL, MODEL, EMBEDDING_DIM=100):

    model_input = tf.keras.Input(shape=input_shape, dtype="float64")
    x = tf.keras.layers.Embedding(
        input_dim=num_words,
        output_dim=EMBEDDING_DIM,
        weights=[gensim_weight_matrix],
        trainable=False
    )(model_input)

    if MODEL == 'cnn':
        x = build_cnn(x)
    if MODEL == 'lstm':
        x = build_lstm(x)

    if LABEL == '_timeconsumption':
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.Model(model_input, x)
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
    else:
        x = tf.keras.layers.Dense(class_num, activation='softmax')(x)
        model = tf.keras.Model(model_input, x)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



TEXTS = ['_html_tags']
LABELS = ['_placement']
WORDVECS = ['self']
MODELS = ['cnn', 'lstm']

gensim = {}

for WORDVEC in WORDVECS:
    for TEXT in TEXTS:
        if WORDVEC == 'self':
            vector_size = 100
            gensim[WORDVEC] = KeyedVectors.load_word2vec_format(f'data/word2vec{TEXT}_100d.txt')
        else:
            vector_size = 300
            gensim[WORDVEC] = KeyedVectors.load_word2vec_format(f'data/{WORDVEC}')

print('<--------------------------------------------------------------->')
print('LOADED')
print('<--------------------------------------------------------------->')

for MODEL in MODELS:
    for WORDVEC in WORDVECS:
        for TEXT in TEXTS:
            for LABEL in LABELS:

                df_train = pd.read_csv(f'data/cached_train{TEXT}{LABEL}.csv')
                df_test = pd.read_csv(f'data/cached_test{TEXT}{LABEL}.csv')

                num_classes = np.max(np.concatenate([df_train.label.values, df_test.label.values])) + 1

                if LABEL == '_timeconsumption':
                    y_train = df_train.label.values
                    y_test = df_test.label.values
                else:
                    y_train = to_categorical(
                        df_train.label.values,
                        num_classes=num_classes)
                    y_test = to_categorical(
                        df_test.label.values,
                        num_classes=num_classes)

                num_words = 20000

                tokenizer = Tokenizer(num_words, lower=True)
                df_total = pd.concat([df_train.text, df_test.text], axis=0)
                tokenizer.fit_on_texts(df_total)

                X_train = tokenizer.texts_to_sequences(df_train.text)
                X_train_pad = pad_sequences(X_train, maxlen=304, padding='post')
                X_test = tokenizer.texts_to_sequences(df_test.text)
                X_test_pad = pad_sequences(X_test, maxlen=304, padding='post')

                if WORDVEC == 'self':
                    vector_size = 100
                else:
                    vector_size = 300

                gensim_weight_matrix = np.zeros((num_words, vector_size))

                for word, index in tokenizer.word_index.items():
                    if index < num_words:
                        if word in gensim[WORDVEC].key_to_index:
                            gensim_weight_matrix[index] = gensim[WORDVEC][word]
                        else:
                            gensim_weight_matrix[index] = np.zeros(vector_size)

                # Build and train model
                model = build_model((None,), num_words, num_classes, gensim_weight_matrix, LABEL, MODEL, EMBEDDING_DIM=vector_size)

                tf.keras.utils.plot_model(
                    model,
                    to_file=f'{MODEL}_model_plot.png',
                    show_shapes=False,
                    show_layer_names=False
                )