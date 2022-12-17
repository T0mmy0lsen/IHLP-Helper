
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from gensim.models import Word2Vec

import re
import os
from keras.utils import pad_sequences
from sklearn import preprocessing
import random

from gensim.models.keyedvectors import KeyedVectors

from pipeline.models.loader import Loader

DATA_PATH = 'data/3_cnn_word_embedding/data.csv'
DATA_PATH_X_TRAIN = 'data/3_cnn_word_embedding/data_x_train.csv'
DATA_PATH_Y_TRAIN = 'data/3_cnn_word_embedding/data_y_train.csv'
DATA_PATH_X_TEST = 'data/3_cnn_word_embedding/data_x_test.csv'
DATA_PATH_Y_TEST = 'data/3_cnn_word_embedding/data_y_test.csv'

PATH_TO_WORD2VEC = 'word2vec/word2vec.model'


class ModelCNN_WordEmbedding:

    def verify_data(self):
        pass

    def __init__(self):

        LEARNING_RATE = 0.0005

        inputs, embedding, x_train, y_train, x_test, y_test = Loader.get_word_embedding_and_keep_else(
            dim=128,
            nrows=10000,
            word_vectors=Word2Vec.load(PATH_TO_WORD2VEC).wv
        )

        HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([4, 6, 8, 12, 16, 24, 32]))
        HP_FILTERS_1 = hp.HParam('filters_1', hp.Discrete([4, 6, 8, 12, 16, 24, 32, 48, 64]))
        HP_FILTERS_2 = hp.HParam('filters_2', hp.Discrete([4, 6, 8, 12, 16, 24, 32, 48, 64]))
        HP_KERNEL_SIZE_1 = hp.HParam('kernel_size_1', hp.Discrete([2, 4, 6, 8, 12, 16, 24, 32]))
        HP_KERNEL_SIZE_2 = hp.HParam('kernel_size_2', hp.Discrete([2, 4, 6, 8, 12, 16, 24, 32]))
        HP_DROPOUT_1 = hp.HParam('dropout_1', hp.Discrete([0.0, 0.1, 0.2, 0.3, 0.4]))
        HP_DROPOUT_2 = hp.HParam('dropout_2', hp.Discrete([0.0, 0.1, 0.2, 0.3, 0.4]))
        HP_DROPOUT_3 = hp.HParam('dropout_3', hp.Discrete([0.0, 0.1, 0.2, 0.3, 0.4]))
        HP_DROPOUT_4 = hp.HParam('dropout_4', hp.Discrete([0.0, 0.1, 0.2, 0.3, 0.4]))
        HP_DROPOUT_5 = hp.HParam('dropout_5', hp.Discrete([0.0, 0.1, 0.2, 0.3, 0.4]))
        HP_DROPOUT_6 = hp.HParam('dropout_6', hp.Discrete([0.0, 0.1, 0.2, 0.3, 0.4]))
        HP_DENSE_1 = hp.HParam('dense_1', hp.Discrete([8, 12, 16, 24]))
        HP_DENSE_2 = hp.HParam('dense_2', hp.Discrete([8, 12, 16, 24]))
        HP_DENSE_3 = hp.HParam('dense_3', hp.Discrete([8, 12, 16, 24]))
        HP_DENSE_4 = hp.HParam('dense_4', hp.Discrete([8, 12, 16, 24]))
        HP_INITIALIZE = hp.HParam('initialize', hp.Discrete([
            'GlorotUniform',
            'HeNormal',
            'HeUniform',
            'LecunNormal',
            'RandomNormal',
            'RandomUniform',
        ]))
        HP_REGULARIZE = hp.HParam('regularize', hp.Discrete([
            'L1L2',
            'L2',
        ]))
        HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete([
            'Adadelta',
            'Adamax',
            'Nadam',
            'RMSprop',
        ]))

        METRIC_ACCURACY = 'accuracy'

        with tf.summary.create_file_writer('logs/tensorboard').as_default():
            hp.hparams_config(
                hparams=[
                    HP_FILTERS_1,
                    HP_FILTERS_2,
                    HP_KERNEL_SIZE_1,
                    HP_KERNEL_SIZE_2,
                    HP_DROPOUT_1,
                    HP_DROPOUT_2,
                    HP_DROPOUT_3,
                    HP_DROPOUT_4,
                    HP_DROPOUT_5,
                    HP_DROPOUT_6,
                    HP_OPTIMIZER,
                    HP_BATCH_SIZE,
                    HP_DENSE_1,
                    HP_DENSE_2,
                    HP_DENSE_3,
                    HP_DENSE_4,
                    HP_INITIALIZE,
                    HP_REGULARIZE,
                ],
                metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
            )

        def build_model(hparams):

            x = tf.keras.layers.Conv1D(
                filters=hparams[HP_FILTERS_1],
                kernel_size=hparams[HP_KERNEL_SIZE_1],
                padding='same',
                activation='relu',
                kernel_regularizer=hparams[HP_REGULARIZE],
                kernel_initializer=hparams[HP_INITIALIZE],
            )(embedding)
            x = tf.keras.layers.Dropout(hparams[HP_DROPOUT_1])(x)
            x = tf.keras.layers.Conv1D(
                filters=hparams[HP_FILTERS_2],
                kernel_size=hparams[HP_KERNEL_SIZE_2],
                padding='same',
                activation='relu',
                kernel_regularizer=hparams[HP_REGULARIZE],
                kernel_initializer=hparams[HP_INITIALIZE],
            )(x)
            x = tf.keras.layers.GlobalMaxPool1D()(x)
            x = tf.keras.layers.Dropout(hparams[HP_DROPOUT_2])(x)

            x = tf.keras.layers.Dense(hparams[HP_DENSE_1], activation='relu')(x)
            x = tf.keras.layers.Dropout(hparams[HP_DROPOUT_3])(x)
            x = tf.keras.layers.Dense(hparams[HP_DENSE_2], activation='relu')(x)
            x = tf.keras.layers.Dropout(hparams[HP_DROPOUT_4])(x)
            x = tf.keras.layers.Dense(hparams[HP_DENSE_3], activation='relu')(x)
            x = tf.keras.layers.Dropout(hparams[HP_DROPOUT_5])(x)
            x = tf.keras.layers.Dense(hparams[HP_DENSE_4], activation='relu')(x)
            x = tf.keras.layers.Dropout(hparams[HP_DROPOUT_6])(x)

            preds = tf.keras.layers.Dense(len(np.unique(np.concatenate((y_train, y_test)))), activation="softmax")(x)
            model = tf.keras.Model(inputs, preds)

            model.compile(
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                optimizer=hparams[HP_OPTIMIZER],
                metrics=['accuracy'],
            )

            model.fit(
                x_train,
                y_train,
                batch_size=hparams[HP_BATCH_SIZE],
                epochs=1,
                validation_data=(x_test, y_test),
                callbacks=[
                    tf.keras.callbacks.TensorBoard('logs/tensorboard'),
                    hp.KerasCallback('logs/tensorboard', hparams),
                ],
            )

            _, accuracy = model.evaluate(x_test, y_test)
            return accuracy

        def run(run_dir, hparams):
            with tf.summary.create_file_writer(run_dir).as_default():
                hp.hparams(hparams)
                accuracy = build_model(hparams)
                tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

        session_num = 0

        for _ in list(range(0, 1000)):
            hp_filters_1 = random.choice(HP_FILTERS_1.domain.values)
            hp_filters_2 = random.choice(HP_FILTERS_2.domain.values)
            hp_kernel_size_1 = random.choice(HP_KERNEL_SIZE_1.domain.values)
            hp_kernel_size_2 = random.choice(HP_KERNEL_SIZE_2.domain.values)
            hp_dropout_1 = random.choice(HP_DROPOUT_1.domain.values)
            hp_dropout_2 = random.choice(HP_DROPOUT_2.domain.values)
            hp_dropout_3 = random.choice(HP_DROPOUT_3.domain.values)
            hp_dropout_4 = random.choice(HP_DROPOUT_4.domain.values)
            hp_dropout_5 = random.choice(HP_DROPOUT_5.domain.values)
            hp_dropout_6 = random.choice(HP_DROPOUT_6.domain.values)
            hp_optimizer = random.choice(HP_OPTIMIZER.domain.values)
            hp_batch_size = random.choice(HP_BATCH_SIZE.domain.values)
            hp_dense_1 = random.choice(HP_DENSE_1.domain.values)
            hp_dense_2 = random.choice(HP_DENSE_2.domain.values)
            hp_dense_3 = random.choice(HP_DENSE_3.domain.values)
            hp_dense_4 = random.choice(HP_DENSE_4.domain.values)
            hp_initialize = random.choice(HP_INITIALIZE.domain.values)
            hp_regularize = random.choice(HP_REGULARIZE.domain.values)

            hparams = {
                HP_FILTERS_1: hp_filters_1,
                HP_FILTERS_2: hp_filters_2,
                HP_KERNEL_SIZE_1: hp_kernel_size_1,
                HP_KERNEL_SIZE_2: hp_kernel_size_2,
                HP_DROPOUT_1: hp_dropout_1,
                HP_DROPOUT_2: hp_dropout_2,
                HP_DROPOUT_3: hp_dropout_3,
                HP_DROPOUT_4: hp_dropout_4,
                HP_DROPOUT_5: hp_dropout_5,
                HP_DROPOUT_6: hp_dropout_6,
                HP_OPTIMIZER: hp_optimizer,
                HP_BATCH_SIZE: hp_batch_size,
                HP_DENSE_1: hp_dense_1,
                HP_DENSE_2: hp_dense_2,
                HP_DENSE_3: hp_dense_3,
                HP_DENSE_4: hp_dense_4,
                HP_INITIALIZE: hp_initialize,
                HP_REGULARIZE: hp_regularize,
            }

            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run('logs/tensorboard/' + run_name, hparams)
            session_num += 1


ModelCNN_WordEmbedding()