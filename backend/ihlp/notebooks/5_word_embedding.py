import os

# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")

import re
import time
import pandas as pd
import numpy as np
import tensorflow as tf

from gensim.models import Word2Vec
from tqdm import tqdm

tqdm.pandas()


class WordEmbedding:

    def train_preprocessing(self, titles_array):

        processed_array = []

        for title in tqdm(titles_array):
            processed = re.sub('[^a-zA-Z ]', '', title)
            words = processed.split()
            processed_array.append(' '.join([word for word in words if len(word) > 1]))

        return processed_array

    def train(self):

        start_time = time.time()

        df_subject = pd.read_csv('data/subject.csv')
        df_description = pd.read_csv('data/description.csv')

        df_subject = df_subject.fillna('')
        df_description = df_description.fillna('')

        df_subject['processed'] = self.train_preprocessing(df_subject.subject)
        df_description['processed'] = self.train_preprocessing(df_description.description)

        sentences = pd.concat([df_subject.processed, df_description.processed], axis=0)
        train_sentences = list(sentences.progress_apply(str.split).values)

        model = Word2Vec(sentences=train_sentences,
                         sg=1,
                         vector_size=100,
                         workers=4)

        model.wv.save_word2vec_format('data/word2vec_ihlp_100d.txt')

        print(f'Time taken : {(time.time() - start_time) / 60:.2f} mins')

    def load(self):

        df_subject = pd.read_csv('data/subject.csv')
        # df_description = pd.read_csv('data/description.csv')

        df_subject = df_subject.fillna('')
        # df_description = df_description.fillna('')

        df_subject['processed'] = self.train_preprocessing(df_subject.subject)
        # df_description['processed'] = self.train_preprocessing(df_description.description)

        vectorizer = tf.keras.layers.TextVectorization(standardize=None, max_tokens=200000, output_sequence_length=512)
        text_ds = tf.data.Dataset.from_tensor_slices(df_subject.processed.values).batch(128)
        vectorizer.adapt(text_ds)

        voc = vectorizer.get_vocabulary()
        word_index = dict(zip(voc, range(len(voc))))

        embeddings_index = {}
        word_count = 0

        skipped = 0
        we_path = 'data/word2vec_ihlp_100d.txt'
        with open(we_path) as f:
            for i, line in enumerate(f):
                if i == 0:
                    word_count, dim = line.split(maxsplit=1)
                    word_count = int(word_count)
                    dim = int(dim[:-1])
                else:
                    word, coefs = line.split(maxsplit=1)
                    coefs = np.fromstring(coefs, "f", sep=" ")
                    if len(coefs) > 0:
                        embeddings_index[word] = coefs
                    else:
                        # For some reason we got a two-word word in the Word Embedding.
                        skipped += 1


        print("Found %s word vectors." % len(embeddings_index))
        print("Skipped %s word vectors." % skipped)

        num_tokens = word_count + 2
        hits = 0
        misses = 0
        missed = []

        word_index = dict(zip(embeddings_index.keys(), range(len(embeddings_index.keys()))))

        # Prepare embedding matrix
        embedding_matrix = np.zeros((num_tokens, 100))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1
                missed.append(word)

        print("Converted %d words (%d misses)" % (hits, misses))
        print(missed[:10])

        embedding_layer = tf.keras.layers.Embedding(
            num_tokens,
            dim,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
            trainable=False,
        )

        return embedding_layer, embedding_matrix, vectorizer

    @staticmethod
    def bulk_vectorize(data, vectorizer, size=1000):
        output = None
        prev_i = 0
        for i, s in enumerate(tqdm(data)):
            if i > 0 and i % size == 0:
                # part_output = shared.vectorizer(np.array([[s] for s in data[prev_i:i]])).numpy()
                part_output = vectorizer(data[prev_i:i])
                if output is None:
                    output = part_output[:]
                else:
                    output = np.vstack((output, part_output))
                prev_i = i
        # part_output = shared.vectorizer(np.array([[s] for s in data[prev_i:]])).numpy()
        part_output = vectorizer(data[prev_i:])
        if output is None:
            output = part_output[:]
        else:
            output = np.vstack((output, part_output))
        return output


def build(input_layer, start_neurons=8, kernel_size=4, dropout=0.25):

    conv1 = tf.keras.layers.Conv1D(start_neurons * 1, kernel_size, activation="relu", padding="same")(input_layer)
    conv1 = tf.keras.layers.Conv1D(start_neurons * 1, kernel_size, activation="relu", padding="same")(conv1)
    pool1 = tf.keras.layers.MaxPooling1D(2)(conv1)
    pool1 = tf.keras.layers.Dropout(0.1)(pool1)

    conv2 = tf.keras.layers.Conv1D(start_neurons * 2, kernel_size, activation="relu", padding="same")(pool1)
    conv2 = tf.keras.layers.Conv1D(start_neurons * 2, kernel_size, activation="relu", padding="same")(conv2)
    pool2 = tf.keras.layers.MaxPooling1D(2)(conv2)
    pool2 = tf.keras.layers.Dropout(dropout)(pool2)

    conv3 = tf.keras.layers.Conv1D(start_neurons * 4, kernel_size, activation="relu", padding="same")(pool2)
    conv3 = tf.keras.layers.Conv1D(start_neurons * 4, kernel_size, activation="relu", padding="same")(conv3)
    pool3 = tf.keras.layers.MaxPooling1D(2)(conv3)
    pool3 = tf.keras.layers.Dropout(dropout)(pool3)

    conv4 = tf.keras.layers.Conv1D(start_neurons * 8, kernel_size, activation="relu", padding="same")(pool3)
    conv4 = tf.keras.layers.Conv1D(start_neurons * 8, kernel_size, activation="relu", padding="same")(conv4)
    pool4 = tf.keras.layers.MaxPooling1D(2)(conv4)
    pool4 = tf.keras.layers.Dropout(dropout)(pool4)

    # Middle
    convm = tf.keras.layers.Conv1D(start_neurons * 16, kernel_size, activation="relu", padding="same")(pool4)
    convm = tf.keras.layers.Conv1D(start_neurons * 16, kernel_size, activation="relu", padding="same")(convm)

    deconv4 = tf.keras.layers.Conv1DTranspose(start_neurons * 8, kernel_size, strides=2, padding="same")(convm)
    uconv4 = tf.keras.layers.concatenate([deconv4, conv4])
    uconv4 = tf.keras.layers.Dropout(dropout)(uconv4)
    uconv4 = tf.keras.layers.Conv1D(start_neurons * 8, kernel_size, activation="relu", padding="same")(uconv4)
    uconv4 = tf.keras.layers.Conv1D(start_neurons * 8, kernel_size, activation="relu", padding="same")(uconv4)

    deconv3 = tf.keras.layers.Conv1DTranspose(start_neurons * 4, kernel_size, strides=2, padding="same")(uconv4)
    uconv3 = tf.keras.layers.concatenate([deconv3, conv3])
    uconv3 = tf.keras.layers.Dropout(dropout)(uconv3)
    uconv3 = tf.keras.layers.Conv1D(start_neurons * 4, kernel_size, activation="relu", padding="same")(uconv3)
    uconv3 = tf.keras.layers.Conv1D(start_neurons * 4, kernel_size, activation="relu", padding="same")(uconv3)

    deconv2 = tf.keras.layers.Conv1DTranspose(start_neurons * 2, kernel_size, strides=2, padding="same")(uconv3)
    uconv2 = tf.keras.layers.concatenate([deconv2, conv2])
    uconv2 = tf.keras.layers.Dropout(dropout)(uconv2)
    uconv2 = tf.keras.layers.Conv1D(start_neurons * 2, kernel_size, activation="relu", padding="same")(uconv2)
    uconv2 = tf.keras.layers.Conv1D(start_neurons * 2, kernel_size, activation="relu", padding="same")(uconv2)

    deconv1 = tf.keras.layers.Conv1DTranspose(start_neurons * 1, kernel_size, strides=2, padding="same")(uconv2)
    uconv1 = tf.keras.layers.concatenate([deconv1, conv1])
    uconv1 = tf.keras.layers.Dropout(dropout)(uconv1)
    uconv1 = tf.keras.layers.Conv1D(start_neurons * 1, kernel_size, activation="relu", padding="same")(uconv1)
    uconv1 = tf.keras.layers.Conv1D(start_neurons * 1, kernel_size, activation="relu", padding="same")(uconv1)

    output_layer = tf.keras.layers.GlobalMaxPool1D()(uconv1)

    # output_layer = layers.Dense(128, activation='relu')(output_layer)
    # output_layer = layers.Conv1D(1, 1, padding="same", activation="sigmoid")(uconv1)

    return output_layer

embedding_layer, embedding_matrix, vectorizer = WordEmbedding().load()

input_layer = tf.keras.Input(shape=(None,), dtype="float64")
layers = embedding_layer(input_layer)
layers = build(layers, dropout=.1, kernel_size=8)
layers = tf.keras.layers.Dense(36, activation="softmax")(layers)

model = tf.keras.Model(input_layer, layers)
model.summary()

df_train = pd.read_csv('data/cached_train_subject_label_placement.csv')
df_test = pd.read_csv('data/cached_test_subject_label_placement.csv')

# df_train = df_train[:1000]
# df_test = df_test[:1000]

x_train = WordEmbedding.bulk_vectorize(df_train.text.values, vectorizer)
x_test = WordEmbedding.bulk_vectorize(df_test.text.values, vectorizer)

for i in [0.01]:
    for b in [32]:
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adamax(
                learning_rate=i,
            ),
            metrics=["acc"]
        )

        model.fit(x_train, df_train.label.values, batch_size=b, epochs=100, validation_data=(x_test, df_test.label.values))