from collections import defaultdict

import numpy as np
import pandas as pd
import os
import random
import pickle
import tensorflow as tf
from gensim.models import KeyedVectors
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split

PATH_TO_TD_IDF_FOLDER = '../data/output/5_data_encode/texts_td_idf/'
PATH_TO_LABEL_TOP_100 = '../data/output/5_data_encode/label_top_100/output.csv'
PATH_TO_LABEL_TOP_ALL = '../data/output/5_data_encode/label_top_all/output.csv'
PATH_TO_LABEL_KEEP_ELSE = '../data/output/5_data_encode/label_keep_else/output.csv'
PATH_TO_TEXT_NO_STOPWORDS = '../data/output/5_data_encode/texts_no_stopwords/output.csv'

PATH_TO_LABEL_TIME = 'notebooks/data/output_time.csv'
PATH_TO_LABEL_TOP_100 = 'notebooks/data/output.csv'

PATH_TO_FEATURE_VECTOR = 'notebooks/data/features/output.csv'
PATH_TO_FEATURE_RECEIVED_BY = 'notebooks/data/features/output_received_by.csv'
PATH_TO_FEATURE_TITLE = 'notebooks/data/features/output_title.csv'
PATH_TO_FEATURE_OMK = 'notebooks/data/features/output_omk.csv'
PATH_TO_TEXT = 'notebooks/data/output_heavy.csv'


class Loader:

    def __init__(self):
        pass

    @staticmethod
    def data():

        idx = None
        nrows = None

        df_time = pd.read_csv(PATH_TO_LABEL_TIME, dtype={'id': int, 'label': str, 'label_encoded': int}, sep=',', na_values='', nrows=10000)
        df_responsible = pd.read_csv(PATH_TO_LABEL_TOP_100, dtype={'id': int, 'label': str, 'label_encoded': int}, sep=',', na_values='')

        # Load input features
        df_feature_vector = pd.read_csv(PATH_TO_FEATURE_VECTOR, sep=',', na_values='')
        df_feature_received_by = pd.read_csv(PATH_TO_FEATURE_RECEIVED_BY, sep=',', na_values='')
        df_feature_title = pd.read_csv(PATH_TO_FEATURE_TITLE, sep=',', na_values='')
        df_feature_omk = pd.read_csv(PATH_TO_FEATURE_OMK, sep=',', na_values='')

        # Load input text
        df_texts = pd.read_csv(PATH_TO_TEXT, sep=',')
        df_texts = df_texts.rename(columns={'requestId': 'id'})
        df_texts['text'] = df_texts['subject'] + " " + df_texts['description']

        # Fill NaN just in case
        df_time = df_time.fillna('')
        df_responsible = df_responsible.fillna('')
        df_feature_vector = df_feature_vector.fillna('')
        df_feature_received_by = df_feature_received_by.fillna('')
        df_feature_title = df_feature_title.fillna('')
        df_feature_omk = df_feature_omk.fillna('')
        df_texts = df_texts.fillna('')

        # Make sure all ids are found in all sets
        ids = list(
            set(df_feature_vector.id.to_list()) &
            set(df_feature_received_by.id.to_list()) &
            set(df_feature_title.id.to_list()) &
            set(df_feature_omk.id.to_list()) &
            set(df_texts.id.to_list()) &
            set(df_responsible.id.to_list()) &
            set(df_time.id.to_list())
        )
        ids = np.unique(np.array(ids))

        # Make sure all ids are found in all sets
        df_time = df_time[df_time.id.isin(ids)]
        df_responsible = df_responsible[df_responsible.id.isin(ids)]
        df_feature_vector = df_feature_vector[df_feature_vector.id.isin(ids)]
        df_feature_received_by = df_feature_received_by[df_feature_received_by.id.isin(ids)]
        df_feature_title = df_feature_title[df_feature_title.id.isin(ids)]
        df_feature_omk = df_feature_omk[df_feature_omk.id.isin(ids)]
        df_texts = df_texts[df_texts.id.isin(ids)]

        # Drop duplicates
        df_time = df_time.drop_duplicates(subset='id')
        df_responsible = df_responsible.drop_duplicates(subset='id')
        df_feature_vector = df_feature_vector.drop_duplicates(subset='id')
        df_feature_received_by = df_feature_received_by.drop_duplicates(subset='id')
        df_feature_title = df_feature_title.drop_duplicates(subset='id')
        df_feature_omk = df_feature_omk.drop_duplicates(subset='id')
        df_texts = df_texts.drop_duplicates(subset='id')

        # Sort by id
        df_time = df_time.sort_values(by='id')
        df_responsible = df_responsible.sort_values(by='id')
        df_feature_vector = df_feature_vector.sort_values(by='id')
        df_feature_received_by = df_feature_received_by.sort_values(by='id')
        df_feature_title = df_feature_title.sort_values(by='id')
        df_feature_omk = df_feature_omk.sort_values(by='id')
        df_texts = df_texts.sort_values(by='id')

        # Reset index for iteration purposes
        df_time = df_time.reset_index(drop=True)
        df_responsible = df_responsible.reset_index(drop=True)
        df_feature_vector = df_feature_vector.reset_index(drop=True)
        df_feature_received_by = df_feature_received_by.reset_index(drop=True)
        df_feature_title = df_feature_title.reset_index(drop=True)
        df_feature_omk = df_feature_omk.reset_index(drop=True)
        df_texts = df_texts.reset_index(drop=True)

        # ----------------------------------------------------------------------------------------------------------------------

        arr_time = np.random.rand(len(df_time), 100)
        for iy, ix in np.ndindex(arr_time.shape):
            arr_time[iy, ix] = 0  # (arr_time[iy, ix] / 2) + 0.5

        for i, el in df_time.iterrows():
            index = int(df_responsible.iloc[i].label_encoded)
            if float(el.label) == 0.0:
                arr_time[i][index] = 5.0
            if float(el.label) == 1.0:
                arr_time[i][index] = 4.0
            if float(el.label) == 2.0:
                arr_time[i][index] = 3.0
            if float(el.label) == 3.0:
                arr_time[i][index] = 2.0
            if float(el.label) == 4.0:
                arr_time[i][index] = 1.0

        # ----------------------------------------------------------------------------------------------------------------------

        arr_responsible = df_responsible[['label_encoded']].to_numpy()
        arr_texts = df_texts[['text']].to_numpy()

        # ----------------------------------------------------------------------------------------------------------------------

        arr_feature_vector = df_feature_vector.drop(['id'], axis=1)
        arr_feature_received_by = df_feature_received_by.drop(['id'], axis=1)
        arr_feature_title = df_feature_title.drop(['id'], axis=1)
        arr_feature_omk = df_feature_omk.drop(['id'], axis=1)

        arr_feature_vector = arr_feature_vector.values
        arr_feature_received_by = arr_feature_received_by.values
        arr_feature_title = arr_feature_title.values
        arr_feature_omk = arr_feature_omk.values

        if nrows is not None:
            idx = random.sample(range(0, len(arr_feature_vector)), nrows)

        if idx is not None:
            arr_feature_vector = [arr_feature_vector[i] for i in idx]
            arr_feature_received_by = [arr_feature_received_by[i] for i in idx]
            arr_feature_title = [arr_feature_title[i] for i in idx]
            arr_feature_omk = [arr_feature_omk[i] for i in idx]
            arr_texts = [arr_texts[i] for i in idx]
            arr_time = [arr_time[i] for i in idx]
            arr_responsible = [arr_responsible[i] for i in idx]

        arr_feature_vector = np.asarray(arr_feature_vector)
        arr_feature_received_by = np.asarray(arr_feature_received_by)
        arr_feature_title = np.asarray(arr_feature_title)
        arr_feature_omk = np.asarray(arr_feature_omk)
        arr_texts = np.asarray(arr_texts).reshape(-1)

        arr_time = np.asarray(arr_time)
        arr_responsible = np.asarray(arr_responsible)

        train_feature_vector, \
        validation_feature_vector, \
        train_received_by, \
        validation_received_by, \
        train_title, \
        validation_title, \
        train_omk, \
        validation_omk, \
        train_texts, \
        validation_texts, \
        train_time, \
        validation_time, \
        train_responsible, \
        validation_responsible \
            = train_test_split(
            arr_feature_vector.tolist(),
            arr_feature_received_by.tolist(),
            arr_feature_title.tolist(),
            arr_feature_omk.tolist(),
            arr_texts.tolist(),
            arr_time.tolist(),
            arr_responsible.tolist(),
            test_size=.2,
            shuffle=True
        )

        tokenized_train = defaultdict(list)
        tokenized_train['feature_vector'] = train_feature_vector
        tokenized_train['feature_received_by'] = train_received_by
        tokenized_train['feature_title'] = train_title
        tokenized_train['feature_omk'] = train_omk
        tokenized_train['text'] = train_texts

        tokenized_train_y = defaultdict(list)
        tokenized_train_y['time'] = train_time

        tokenized_validation = defaultdict(list)
        tokenized_validation['feature_vector'] = validation_feature_vector
        tokenized_validation['feature_received_by'] = validation_received_by
        tokenized_validation['feature_title'] = validation_title
        tokenized_validation['feature_omk'] = validation_omk
        tokenized_validation['text'] = validation_texts

        tokenized_validation_y = defaultdict(list)
        tokenized_validation_y['time'] = validation_time

        return tokenized_train, tokenized_train_y, tokenized_validation, tokenized_validation_y


    @staticmethod
    def shuffle_and_split(arr_x, arr_y, split, seed):

        # Shuffle the data
        seed = seed
        rng = np.random.RandomState(seed)
        rng.shuffle(arr_x)
        rng = np.random.RandomState(seed)
        rng.shuffle(arr_y)

        # Extract a training & validation split
        validation_split = split
        num_validation_samples = int(validation_split * len(arr_x))
        arr_x_train = arr_x[:-num_validation_samples]
        arr_x_test = arr_x[-num_validation_samples:]
        arr_y_train = arr_y[:-num_validation_samples]
        arr_y_test = arr_y[-num_validation_samples:]

        return arr_x_train, arr_y_train, arr_x_test, arr_y_test


    @staticmethod
    def get_td_if_vector(config, idx):

        tmp = pd.read_csv(PATH_TO_TEXT_NO_STOPWORDS)
        tmp = tmp.iloc[idx]
        tmp = tmp.fillna('')

        file_name = f'td_idf_{config.max_df}_{config.min_df}_{config.max_features}.pickle'

        if not os.path.isfile(PATH_TO_TD_IDF_FOLDER + file_name):
            print('Error: no such pickle found.')
            return []
        else:
            with open(PATH_TO_TD_IDF_FOLDER + file_name, 'rb') as pickle_file:
                tfidf = pickle.load(pickle_file)

        print('Convert with vectorizer')
        vectors = tfidf.transform(tmp['text'].to_numpy()).todense()

        return vectors


    @staticmethod
    def get_td_idf(config, idx):
        return Loader.get_td_if_vector(config, idx)


    @staticmethod
    def get_td_idf_and_time():

        tokenized_train, tokenized_train_y, tokenized_validation, tokenized_validation_y = Loader.data()

        print('Convert with vectorizer')
        with open(PATH_TO_TD_IDF_FOLDER + 'td_idf_1.0_0.0005_10000.pickle', 'rb') as pickle_file:
            tfidf = pickle.load(pickle_file)
        tokenized_train['text'] = tfidf.transform(tokenized_train['text']).todense()
        tokenized_validation['text'] = tfidf.transform(tokenized_validation['text']).todense()

        return tokenized_train, tokenized_train_y, tokenized_validation, tokenized_validation_y

    @staticmethod
    def get_td_idf_and_keep_else(config, nrows=None, idx=None):

        arr_labels = pd.read_csv(PATH_TO_LABEL_KEEP_ELSE).label

        if nrows is not None:
            idx = random.sample(range(0, len(arr_labels)), nrows)

        arr_td_idf = Loader.get_td_idf(config, idx)
        arr_td_idf = np.asarray(arr_td_idf)

        if idx is not None:
            arr_labels = [arr_labels[i] for i in idx]

        arr_x_train, arr_y_train, arr_x_test, arr_y_test = Loader.shuffle_and_split(arr_td_idf, arr_labels, .2, 1000)
        inputs = tf.keras.Input(shape=(len(arr_x_train[0]), 1), dtype="float64")

        return inputs, None,  arr_x_train, arr_y_train, arr_x_test, arr_y_test


    @staticmethod
    def get_word_embedding_and_labels(nrows=None, idx=None, labels='keep_else', word_vectors=None, dim=300, max_len=256, embeddings_regularizer='L1L2'):

        id2label = dict()
        label2id = dict()

        if labels == 'keep_else':
            df_labels = pd.read_csv(PATH_TO_LABEL_KEEP_ELSE)
        elif labels == 'top_100':
            df_labels = pd.read_csv(PATH_TO_LABEL_TOP_100, dtype={'id': int, 'label': str, 'label_encoded': int})
        else:  # TOP ALL
            df_labels = pd.read_csv(PATH_TO_LABEL_TOP_ALL, dtype=int)

        df_texts = pd.read_csv(PATH_TO_TEXT_NO_STOPWORDS)

        df = pd.merge(df_texts, df_labels, on='id', how='left')
        df = df.fillna('')

        df = df[df['text'] != '']
        df = df[df['label'] != '']

        arr_texts = df.text.to_numpy()
        arr_labels = df.label_encoded.to_numpy()

        arr_labels_encoded_unique = np.unique(arr_labels)
        for label in arr_labels_encoded_unique:
            id2label[f"{label}"] = df[df['label_encoded'] == label].iloc[0]['label']
            label2id[id2label[f"{label}"]] = label

        if nrows is not None:
            idx = random.sample(range(0, len(arr_labels)), nrows)

        if idx is not None:
            arr_labels = [arr_labels[i] for i in idx]
            arr_texts = [arr_texts[i] for i in idx]

        arr_labels = np.asarray(arr_labels)
        arr_texts = np.asarray(arr_texts)

        arr_x_train, arr_y_train, arr_x_test, arr_y_test = Loader.shuffle_and_split(arr_texts, arr_labels, .2, 1000)

        length = len(arr_x_train)
        print(f'Train on: {length}')

        if word_vectors is None:
            return None, None, arr_x_train, arr_y_train, arr_x_test, arr_y_test, id2label, label2id

        NUM_WORDS = 100000
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS)
        tokenizer.fit_on_texts(arr_texts)

        sequences_train = tokenizer.texts_to_sequences(arr_x_train)
        sequences_valid = tokenizer.texts_to_sequences(arr_x_test)

        x_train = pad_sequences(sequences_train, maxlen=max_len)
        x_test = pad_sequences(sequences_valid, maxlen=x_train.shape[1])

        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        if word_vectors is None:
            word_vectors = [KeyedVectors.load_word2vec_format('../data/output/5_data_encode/texts_word2vec/google_news_negative_word2vec_300.bin', binary=True)]

        concatenated_embedding_matrix = None
        vocabulary_size = min(len(word_index) + 1, NUM_WORDS)

        for word_vector in word_vectors:

            count = 0
            embedding_matrix = np.zeros((vocabulary_size, dim))
            for word, i in word_index.items():
                if i >= NUM_WORDS:
                    continue
                try:
                    embedding_vector = word_vector[word]
                    embedding_matrix[i] = embedding_vector
                except KeyError:
                    count = count + 1
                    embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), dim)

            if concatenated_embedding_matrix is None:
                concatenated_embedding_matrix = embedding_matrix[:]
            else:
                concatenated_embedding_matrix = np.concatenate((concatenated_embedding_matrix, embedding_matrix), axis=1)
                np.shape(concatenated_embedding_matrix)

            print(f'Missed: {count} words')

        sequence_length = len(x_train[0])
        inputs = tf.keras.layers.Input(shape=(sequence_length,))
        embedding = tf.keras.layers.Embedding(vocabulary_size, dim * len(word_vectors), weights=[concatenated_embedding_matrix], trainable=True, embeddings_regularizer=embeddings_regularizer)(inputs)

        return inputs, embedding, x_train, arr_y_train, x_test, arr_y_test