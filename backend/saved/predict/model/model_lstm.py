import cv2
import numpy as np
import keras
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical, pad_sequences

from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, CuDNNLSTM, Dense

import albumentations as albu

# Resizing the images as per EfficientNetB0 to size (224, 224)
from predict.model.prepare import Prepare
from predict.model.preprocess import Preprocess
from predict.model.shared import SharedDict

n_classes = 100
epochs = 15
batch_size = 8
len = 100

def get_labels(labels):
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    return le


def create_dataset():

    shared = SharedDict().revised()

    Preprocess(shared)
    Prepare(
        shared,
        category_type='responsible'
    ).fetch(
        top=100,
        categorical=False,
        lang='da',
    )

    le = get_labels(np.unique(np.concatenate([shared.y_train, shared.y_validate])))

    y_train_labels = le.transform(shared.y_train)
    y_validate_labels = le.transform(shared.y_validate)

    y_train = to_categorical(y_train_labels, n_classes)
    y_validate = to_categorical(y_validate_labels, n_classes)

    return shared.x_train, y_train, shared.x_validate, y_validate, shared

x_train, y_train, x_val, y_val, shared = create_dataset()

tokenizer = Tokenizer(num_words=20000)

tokenizer.fit_on_texts(np.concatenate((x_train, x_val)))
x_train_sequences = tokenizer.texts_to_sequences(x_train)
x_val_sequences = tokenizer.texts_to_sequences(x_val)

x_train_new = pad_sequences(x_train_sequences, maxlen=len)
x_val_new = pad_sequences(x_val_sequences, maxlen=len)

model = Sequential()
model.add(Embedding(20000, len))
model.add(CuDNNLSTM(len))
model.add(Dense(100, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer = 'RMSprop', metrics=['accuracy'])

model.fit(x_train_new, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val_new, y_val))