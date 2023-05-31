import pandas as pd
import numpy as np
import gensim.downloader as api
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

df_train = pd.read_csv(f'data/cached_train_html_tags_responsible.csv')
df_test = pd.read_csv(f'data/cached_test_html_tags_responsible.csv')

y_train = to_categorical(df_train.label.values)
y_test = to_categorical(df_test.label.values)

num_words = 10000
tokenizer = Tokenizer(num_words, lower=True)
df_total = pd.concat([df_train.text, df_test.text], axis=0)
tokenizer.fit_on_texts(df_total)

X_train = tokenizer.texts_to_sequences(df_train.text)
X_train_pad = pad_sequences(X_train, maxlen=300, padding='post')
X_test = tokenizer.texts_to_sequences(df_test.text)
X_test_pad = pad_sequences(X_test, maxlen=300, padding='post')

glove_gensim = api.load('glove-wiki-gigaword-100')

vector_size = 100
gensim_weight_matrix = np.zeros((num_words, vector_size))

for word, index in tokenizer.word_index.items():
    if index < num_words:
        if word in glove_gensim.key_to_index:
            gensim_weight_matrix[index] = glove_gensim[word]
        else:
            gensim_weight_matrix[index] = np.zeros(100)

EMBEDDING_DIM = 100
class_num = 137
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Embedding(
    input_dim=num_words,
    output_dim=EMBEDDING_DIM,
    input_length=X_train_pad.shape[1],
    weights=[gensim_weight_matrix],
    trainable=False
))

model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100,return_sequences=True)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200,return_sequences=True)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100,return_sequences=False)))

model.add(tf.keras.layers.Dense(class_num, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

from keras.callbacks import EarlyStopping, ModelCheckpoint

es = EarlyStopping(monitor='val_loss', mode='min', verbose= 1, patience=5)
mc = ModelCheckpoint('./model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

history_embedding = model.fit(X_train_pad, y_train,
    epochs=25,
    batch_size=128,
    validation_data=(X_test_pad, y_test),
    verbose=1,
    callbacks=[es, mc]
)