import os
from ast import literal_eval

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

tqdm.pandas()

max_features = 5000
target = 'subject_label_placement'

df_train = pd.read_csv(f'data/cached_train_{target}.csv')
df_test = pd.read_csv(f'data/cached_test_{target}.csv')

y_train = df_train.label
y_test = df_test.label

tfidf_vectorizer = TfidfVectorizer(max_df=.9, max_features=max_features, min_df=0, ngram_range=(1, 5))
tfidf_vectorizer.fit_transform(df_train.text.values)
tf_len = len(tfidf_vectorizer.vocabulary_)
print(tf_len)


def bulk_vectorize(data, vectorizer, size=1000):
    output = None
    prev_i = 0
    for i, s in enumerate(tqdm(data)):
        if i > 0 and i % size == 0:
            part_output = np.array(vectorizer.transform(data[prev_i:i]).todense().tolist()).reshape(-1, tf_len)
            if output is None:
                output = part_output[:]
            else:
                output = np.vstack((output, part_output))
            prev_i = i
    part_output = np.array(vectorizer.transform(data[prev_i:]).todense().tolist()).reshape(-1, tf_len)
    if output is None:
        output = part_output[:]
    else:
        output = np.vstack((output, part_output))
    return output


def vectorizer(data):
    return np.array(tfidf_vectorizer.transform(data).todense().tolist()).reshape(-1, tf_len)

x_train_tfidf = bulk_vectorize(df_train.text.values, tfidf_vectorizer)
x_test_tfidf = bulk_vectorize(df_test.text.values, tfidf_vectorizer)

input_layer = tf.keras.Input(shape=(max_features,), dtype="float64")
layers = tf.keras.layers.Dense(512, activation="relu")(input_layer)
layers = tf.keras.layers.Dense(256, activation="relu")(layers)
layers = tf.keras.layers.Dense(128, activation="relu")(layers)
layers = tf.keras.layers.Dense(36, activation="softmax")(layers)

y_train = df_train.label.values
y_test = df_test.label.values

model = tf.keras.Model(input_layer, layers)
model.summary()

for i in [0.01]:
    for b in [16]:
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adamax(
                learning_rate=i,
            ),
            metrics=["acc"]
        )

        print("Time to fit.")
        model.fit(x_train_tfidf, y_train, batch_size=b, epochs=100, validation_data=(x_test_tfidf, y_test))