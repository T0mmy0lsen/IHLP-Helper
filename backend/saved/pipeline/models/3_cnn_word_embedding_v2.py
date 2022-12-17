
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Reshape, Conv2D, MaxPooling2D, concatenate, Flatten, Dropout, Dense, SpatialDropout1D, \
    Concatenate, MaxPool2D
from keras import regularizers, Model
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

PATH_TO_IHLP_WORD2VEC = 'word2vec/ihlp_word2vec.model'
PATH_TO_STACKOVERFLOW_WORD2VEC = 'word2vec/stackoverflow_word2vec.model'

PATH_TO_WORD2VEC_GOOGLE_NEGATIVE = 'word2vec/downloaded/google_negative/model.bin'
PATH_TO_WORD2VEC_GOOGLE_NEWS = 'word2vec/downloaded/google_news/model.bin'
PATH_TO_WORD2VEC_WIKIPEDIA = 'word2vec/downloaded/wikipedia/model.bin'


class ModelCNN_WordEmbedding:

    def verify_data(self):
        pass

    def __init__(self):

        LEARNING_RATE = 0.0005
        EMBEDDING_DIM = 300
        MAX_LEN = 1024
        MULTIPLIER = 1  # Size of word_vectors

        inputs, embedding, x_train, y_train, x_test, y_test = Loader.get_word_embedding_and_labels(
            labels='top_all',
            nrows=10000,
            dim=EMBEDDING_DIM,
            max_len=MAX_LEN,
            word_vectors=[
                Word2Vec.load(PATH_TO_IHLP_WORD2VEC).wv,
                # KeyedVectors.load_word2vec_format(PATH_TO_WORD2VEC_GOOGLE_NEGATIVE, binary=True),
                # KeyedVectors.load_word2vec_format(PATH_TO_WORD2VEC_GOOGLE_NEWS, binary=True),
                # KeyedVectors.load_word2vec_format(PATH_TO_WORD2VEC_WIKIPEDIA, binary=True),
            ]
        )

        HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([4, 6, 8, 12, 16, 24, 32]))
        HP_DROPOUT_1 = hp.HParam('dropout_1', hp.Discrete([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]))
        HP_DROPOUT_2 = hp.HParam('dropout_2', hp.Discrete([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]))
        HP_FILTERS_1 = hp.HParam('filters_1', hp.Discrete([4, 6, 8, 12, 16, 24, 32, 48, 64]))
        HP_FILTERS_2 = hp.HParam('filters_2', hp.Discrete([4, 6, 8, 12, 16, 24, 32, 48, 64]))
        HP_FILTERS_3 = hp.HParam('filters_3', hp.Discrete([4, 6, 8, 12, 16, 24, 32, 48, 64]))
        HP_FILTERS_4 = hp.HParam('filters_4', hp.Discrete([4, 6, 8, 12, 16, 24, 32, 48, 64]))
        HP_KERNEL_SIZE_1 = hp.HParam('kernel_size_1', hp.Discrete([2, 4, 6, 8, 12, 16, 24, 32]))
        HP_KERNEL_SIZE_2 = hp.HParam('kernel_size_2', hp.Discrete([2, 4, 6, 8, 12, 16, 24, 32]))
        HP_KERNEL_SIZE_3 = hp.HParam('kernel_size_3', hp.Discrete([2, 4, 6, 8, 12, 16, 24, 32]))
        HP_KERNEL_SIZE_4 = hp.HParam('kernel_size_4', hp.Discrete([2, 4, 6, 8, 12, 16, 24, 32]))
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
            'Adamax',
            'Nadam',
            'RMSprop',
        ]))

        METRIC_ACCURACY = 'accuracy'

        with tf.summary.create_file_writer('logs/tensorboard').as_default():
            hp.hparams_config(
                hparams=[
                    HP_DROPOUT_1,
                    HP_DROPOUT_2,
                    HP_FILTERS_1,
                    HP_KERNEL_SIZE_1,
                    HP_KERNEL_SIZE_2,
                    HP_KERNEL_SIZE_3,
                    HP_KERNEL_SIZE_4,
                    HP_OPTIMIZER,
                    HP_BATCH_SIZE,
                    HP_INITIALIZE,
                    HP_REGULARIZE,
                ],
                metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
            )

        def build_model(hparams):

            DROPOUT = 0.1

            sequence_length = MAX_LEN
            filter_sizes = [2, 3, 4, 8]
            num_filters = 36

            x = SpatialDropout1D(hparams[HP_DROPOUT_1])(embedding)
            x = Reshape((sequence_length, EMBEDDING_DIM * MULTIPLIER, 1))(x)

            maxpool_pool = []

            # Needs to match MULTIPLIER
            conv = Conv2D(hparams[HP_FILTERS_1], kernel_size=(hparams[HP_KERNEL_SIZE_1], EMBEDDING_DIM * MULTIPLIER), kernel_initializer=hparams[HP_INITIALIZE], activation='elu')(x)
            maxpool_pool.append(MaxPool2D(pool_size=(sequence_length - hparams[HP_KERNEL_SIZE_1] + 1, 1))(conv))
            # conv = Conv2D(hparams[HP_FILTERS_1], kernel_size=(hparams[HP_KERNEL_SIZE_2], EMBEDDING_DIM * MULTIPLIER), kernel_initializer=hparams[HP_INITIALIZE], activation='elu')(x)
            # maxpool_pool.append(MaxPool2D(pool_size=(sequence_length - hparams[HP_KERNEL_SIZE_2] + 1, 1))(conv))
            # conv = Conv2D(hparams[HP_FILTERS_1], kernel_size=(hparams[HP_KERNEL_SIZE_3], EMBEDDING_DIM * MULTIPLIER), kernel_initializer=hparams[HP_INITIALIZE], activation='elu')(x)
            # maxpool_pool.append(MaxPool2D(pool_size=(sequence_length - hparams[HP_KERNEL_SIZE_3] + 1, 1))(conv))
            # conv = Conv2D(hparams[HP_FILTERS_1], kernel_size=(hparams[HP_KERNEL_SIZE_4], EMBEDDING_DIM * MULTIPLIER), kernel_initializer=hparams[HP_INITIALIZE], activation='elu')(x)
            # maxpool_pool.append(MaxPool2D(pool_size=(sequence_length - hparams[HP_KERNEL_SIZE_4] + 1, 1))(conv))

            z = Concatenate(axis=1)(maxpool_pool)
            z = Flatten()(z)
            z = Dropout(hparams[HP_DROPOUT_2])(z)

            preds = tf.keras.layers.Dense(len(np.unique(np.concatenate((y_train, y_test)))), activation="softmax")(z)

            model = Model(inputs=inputs, outputs=preds)

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
            hp_dropout_1 = random.choice(HP_DROPOUT_1.domain.values)
            hp_dropout_2 = random.choice(HP_DROPOUT_2.domain.values)
            hp_filter_1 = random.choice(HP_FILTERS_1.domain.values)
            hp_kernel_size_1 = random.choice(HP_KERNEL_SIZE_1.domain.values)
            hp_kernel_size_2 = random.choice(HP_KERNEL_SIZE_2.domain.values)
            hp_kernel_size_3 = random.choice(HP_KERNEL_SIZE_3.domain.values)
            hp_kernel_size_4 = random.choice(HP_KERNEL_SIZE_4.domain.values)
            hp_optimizer = random.choice(HP_OPTIMIZER.domain.values)
            hp_batch_size = random.choice(HP_BATCH_SIZE.domain.values)
            hp_initialize = random.choice(HP_INITIALIZE.domain.values)
            hp_regularize = random.choice(HP_REGULARIZE.domain.values)

            hparams = {
                HP_DROPOUT_1: hp_dropout_1,
                HP_DROPOUT_2: hp_dropout_2,
                HP_FILTERS_1: hp_filter_1,
                HP_KERNEL_SIZE_1: hp_kernel_size_1,
                HP_KERNEL_SIZE_2: hp_kernel_size_2,
                HP_KERNEL_SIZE_3: hp_kernel_size_3,
                HP_KERNEL_SIZE_4: hp_kernel_size_4,
                HP_OPTIMIZER: hp_optimizer,
                HP_BATCH_SIZE: hp_batch_size,
                HP_INITIALIZE: hp_initialize,
                HP_REGULARIZE: hp_regularize,
            }

            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run('logs/tensorboard/' + run_name, hparams)
            session_num += 1


ModelCNN_WordEmbedding()