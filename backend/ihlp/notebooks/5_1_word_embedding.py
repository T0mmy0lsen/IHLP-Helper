import os

import re
import time
import pandas as pd
import numpy as np
import tensorflow as tf

from gensim.models import Word2Vec
from sklearn.utils import resample
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

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

        vector_size = 100
        if path == 'data/danish_dsl_and_reddit_word2vec_word_embeddings.txt':
            vector_size = 500

        df_subject = pd.read_csv(f'data/subject{TEXT}.csv')
        df_description = pd.read_csv(f'data/description{TEXT}.csv')

        df_subject = df_subject.fillna('')
        df_description = df_description.fillna('')

        df_subject = pd.merge(df_subject, df_description, on='id')
        df_subject['processed'] = df_subject.apply(lambda x: "{} {}".format(x.subject, x.description), axis=1)

        vectorizer = tf.keras.layers.TextVectorization(standardize=None, max_tokens=50000, output_sequence_length=512)
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
            input_length=512,
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


def build(input_layer, dropout=0.25):

    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=True))(input_layer)
    lstm = tf.keras.layers.Dropout(0.2)(lstm)
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, return_sequences=True))(lstm)
    lstm = tf.keras.layers.Dropout(0.2)(lstm)
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, return_sequences=False))(lstm)

    return lstm


def curriculum_sort(df):
    df['text_length'] = df['text'].apply(lambda x: len(x.split()))
    return df.sort_values(by='text_length').drop('text_length', axis=1)


TEXTS = ['_html_tags', '_raw', '_lemmatize']
LABELS = ['_placement', '_responsible', '_timeconsumption']
LABELS = ['_placement']
WORD2VECS_EXT = [True, False]
WORD2VECS_EXT = [False]

# Define a list of learning rates to try
learning_rates = [0.0001]

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

            df_train = df_train_balanced.sample(frac=1).reset_index(drop=True)

            print(len(df_train.label.unique()))

            path = f'data/word2vec{TEXT}_100d.txt'
            if WORD2VEC_EXT:
                path = 'data/danish_dsl_and_reddit_word2vec_word_embeddings.txt'

            if not os.path.isfile(path) and path != 'data/danish_dsl_and_reddit_word2vec_word_embeddings.txt':
                WordEmbedding().train(TEXT, path)
            embedding_layer, embedding_matrix, vectorizer = WordEmbedding().load(TEXT, path)

            x_train = WordEmbedding.bulk_vectorize(df_train.text.values, vectorizer)
            x_test = WordEmbedding.bulk_vectorize(df_test.text.values, vectorizer)

            df_train = curriculum_sort(df_train)

            # Iterate over learning rates
            for lr in learning_rates:

                # Hyperparameter tuning
                batch_sizes = [256]
                best_val_acc = 0
                best_batch_size = 32

                for BATCH in batch_sizes:

                    input_layer = tf.keras.Input(shape=(None,), dtype="float64")
                    layers = embedding_layer(input_layer)
                    layers = build(layers)

                    if LABEL == '_timeconsumption':
                        layers = tf.keras.layers.Dense(1)(layers)
                    else:
                        layers = tf.keras.layers.Dense(35, activation="softmax")(layers)

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
                        total_train_steps
                    )

                    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

                    if LABEL == '_timeconsumption':
                        model.compile(
                            loss="mae",
                            optimizer=optimizer,
                            metrics=["mae"]
                        )
                    else:
                        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                        """
                        model.compile(
                            loss="sparse_categorical_crossentropy",
                            optimizer=optimizer,
                            metrics=["acc"]
                        )
                        """

                    history = model.fit(x_train, df_train.label.values, batch_size=BATCH, epochs=EPOCHS, validation_data=(x_test, df_test.label.values))