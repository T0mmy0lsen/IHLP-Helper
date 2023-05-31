import os
<<<<<<< HEAD:backend/ihlp/notebooks/5_0_word_embedding.py
=======

# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")
>>>>>>> c33128bae6b7d9fa9030c548208e4823278f7838:backend/ihlp/notebooks/5_word_embedding.py

import re
import time
import pandas as pd
import numpy as np
import tensorflow as tf

from gensim.models import Word2Vec
from sklearn.utils import resample
from tqdm import tqdm

tqdm.pandas()


class WordEmbedding:

    def train(self, TEXT, path):

        start_time = time.time()

        df_subject = pd.read_csv(f'data/subject{TEXT}.csv')
        df_description = pd.read_csv(f'data/description{TEXT}.csv')

        df_subject = df_subject.fillna('')
        df_description = df_description.fillna('')

        df_subject['processed'] = df_subject.subject
        df_description['processed'] = df_description.description

        sentences = pd.concat([df_subject.processed, df_description.processed], axis=0)
        train_sentences = list(sentences.progress_apply(str.split).values)

        model = Word2Vec(sentences=train_sentences,
                         sg=1,
                         vector_size=100,
                         workers=4)

        model.wv.save_word2vec_format(path)

        print(f'Time taken : {(time.time() - start_time) / 60:.2f} mins')

    def load(self, TEXT, path):

<<<<<<< HEAD:backend/ihlp/notebooks/5_0_word_embedding.py
        vector_size = 100
        if path == 'data/danish_dsl_and_reddit_word2vec_word_embeddings.txt':
            vector_size = 500

        df_subject = pd.read_csv(f'data/subject{TEXT}.csv')
        df_description = pd.read_csv(f'data/description{TEXT}.csv')
=======
        df_subject = pd.read_csv('data/subject.csv')
        df_description = pd.read_csv('data/description.csv')
>>>>>>> c33128bae6b7d9fa9030c548208e4823278f7838:backend/ihlp/notebooks/5_word_embedding.py

        df_subject = df_subject.fillna('')
        df_description = df_description.fillna('')

        df_subject = pd.merge(df_subject, df_description, on='id')
<<<<<<< HEAD:backend/ihlp/notebooks/5_0_word_embedding.py
        df_subject['processed'] = df_subject.apply(lambda x: "{} {}".format(x.subject, x.description), axis=1)
=======

        df_subject['processed'] = df_subject.apply(lambda x: "{} {}".format(x.subject, x.description), axis=1)
        df_subject['processed'] = self.train_preprocessing(df_subject.processed)
>>>>>>> c33128bae6b7d9fa9030c548208e4823278f7838:backend/ihlp/notebooks/5_word_embedding.py

        vectorizer = tf.keras.layers.TextVectorization(standardize=None, max_tokens=25000, output_sequence_length=512)
        text_ds = tf.data.Dataset.from_tensor_slices(df_subject.processed.values).batch(128)
        vectorizer.adapt(text_ds)

        voc = vectorizer.get_vocabulary()
        word_index = dict(zip(voc, range(len(voc))))

        embeddings_index = {}

        skipped = 0
        with open(path) as f:
            for i, line in enumerate(f):
                if i == 0:
                    _, dim = line.split(maxsplit=1)
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

        num_tokens = len(voc) + 2
        hits = 0
        misses = 0
        missed = []

        # word_index = dict(zip(embeddings_index.keys(), range(len(embeddings_index.keys()))))

        # Prepare embedding matrix
        embedding_matrix = np.zeros((num_tokens, vector_size))
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
            input_shape=512,
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

<<<<<<< HEAD:backend/ihlp/notebooks/5_0_word_embedding.py
def curriculum_sort(df):
    df['text_length'] = df['text'].apply(lambda x: len(x.split()))
    return df.sort_values(by='text_length').drop('text_length', axis=1)
=======
# WordEmbedding().train()

embedding_layer, embedding_matrix, vectorizer = WordEmbedding().load()
>>>>>>> c33128bae6b7d9fa9030c548208e4823278f7838:backend/ihlp/notebooks/5_word_embedding.py

# Define a list of learning rates to try
learning_rates = [0.1, 0.01, 0.001, 0.0001]

TEXTS = ['_html_tags', '_raw', '_lemmatize']
LABELS = ['_placement', '_responsible', '_timeconsumption']
LABELS = ['_placement']
WORD2VECS_EXT = [True, False]
WORD2VECS_EXT = [False]

<<<<<<< HEAD:backend/ihlp/notebooks/5_0_word_embedding.py
# Define a list of learning rates to try
learning_rates = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
learning_rates = [0.005, 0.001, 0.0005, 0.0001]
=======
df_train = pd.read_csv('data/cached_train_label_placement.csv')
df_test = pd.read_csv('data/cached_test_label_placement.csv')
>>>>>>> c33128bae6b7d9fa9030c548208e4823278f7838:backend/ihlp/notebooks/5_word_embedding.py

for WORD2VEC_EXT in WORD2VECS_EXT:
    for LABEL in LABELS:
        for TEXT in TEXTS:

            df_train = pd.read_csv(f'data/cached_train{TEXT}{LABEL}.csv')
            df_test = pd.read_csv(f'data/cached_test{TEXT}{LABEL}.csv')

            # Calculate the mean number of instances per class
            mean_count = int(df_train['label'].value_counts().mean())

            df_train_balanced = pd.DataFrame()

            for label in df_train['label'].unique():
                df_label = df_train[df_train['label'] == label]

                if len(df_label) > mean_count:
                    # If this label has more instances than the mean, downsample
                    df_label = df_label.sample(mean_count)
                elif len(df_label) < mean_count:
                    # If this label has less instances than the mean, upsample
                    df_label = resample(df_label, replace=True, n_samples=mean_count, random_state=123)

                df_train_balanced = pd.concat([df_train_balanced, df_label])

            df_train = df_train_balanced

            path = f'data/word2vec{TEXT}_100d.txt'
            if WORD2VEC_EXT:
                path = 'data/danish_dsl_and_reddit_word2vec_word_embeddings.txt'

            if not os.path.isfile(path) and path != 'data/danish_dsl_and_reddit_word2vec_word_embeddings.txt':
                WordEmbedding().train(TEXT, path)
            embedding_layer, embedding_matrix, vectorizer = WordEmbedding().load(TEXT, path)

            x_train = WordEmbedding.bulk_vectorize(df_train.text.values, vectorizer)
            x_test = WordEmbedding.bulk_vectorize(df_test.text.values, vectorizer)

            # Iterate over learning rates
            for lr in learning_rates:

                # Hyperparameter tuning
                batch_sizes = [32, 64, 128]
                best_val_acc = 0
                best_batch_size = 32

                for BATCH in batch_sizes:

                    input_layer = tf.keras.Input(shape=(None,), dtype="float64")
                    layers = embedding_layer(input_layer)
                    layers = build(layers, dropout=.1, kernel_size=8)

                    if LABEL == '_timeconsumption':
                        layers = tf.keras.layers.Dense(1)(layers)
                    else:
                        layers = tf.keras.layers.Dense(len(df_train.label.unique()), activation="softmax")(layers)

                    model = tf.keras.Model(input_layer, layers)
                    # model.summary()

                    EPOCHS = 5

                    batches_per_epoch = len(x_train) // BATCH
                    total_train_steps = int(batches_per_epoch * EPOCHS)

                    print(batches_per_epoch)
                    print(total_train_steps)

                    # Use the current learning rate
                    initial_learning_rate = lr
                    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                        initial_learning_rate,
                        total_train_steps,
                        end_learning_rate=0.0005,
                        power=1.0
                    )

                    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

                    if LABEL == '_timeconsumption':
                        model.compile(
                            loss="mae",
                            optimizer=optimizer,
                            metrics=["mae"]
                        )
                    else:
                        model.compile(
                            loss="sparse_categorical_crossentropy",
                            optimizer=optimizer,
                            metrics=["acc"]
                        )

                    history = model.fit(x_train, df_train.label.values, batch_size=BATCH, epochs=EPOCHS, validation_data=(x_test, df_test.label.values))

                    print(f"Best validation accuracy: {best_val_acc}, best batch size: {best_batch_size}")
                    print(f"Setup LR: {lr}, TEXT: {TEXT}, LABEL: {LABEL}, WORD2VEC_EXT: {WORD2VEC_EXT}")