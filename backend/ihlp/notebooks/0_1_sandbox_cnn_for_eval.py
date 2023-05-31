import pandas as pd
import numpy as np
import tensorflow as tf
import os
import json

from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors

# Define model details
MODEL = 'cnn'
TEXT = '_html_tags'
LABEL = '_timeconsumption'
WORDVEC = 'self'

# Load model
model = load_model(f'./data/models/model_{MODEL}{TEXT}{LABEL}.h5')

# Load the test data
df_validation = pd.read_csv(f'data/cached{TEXT}_validate{LABEL}.csv')
df_train = pd.read_csv(f'data/cached_train{TEXT}{LABEL}.csv')
df_test = pd.read_csv(f'data/cached_test{TEXT}{LABEL}.csv')

num_classes = np.max(np.concatenate([df_train.label.values, df_test.label.values])) + 1

if LABEL == '_timeconsumption':
    y_validate = df_validation.label.values
else:
    y_validate = to_categorical(df_validation.label.values,num_classes=num_classes)

# Perform tokenization and padding
num_words = 20000
tokenizer = Tokenizer(num_words, lower=True)
tokenizer.fit_on_texts(df_validation.text)

X_validate = tokenizer.texts_to_sequences(df_validation.text)
X_validate_pad = pad_sequences(X_validate, maxlen=304, padding='post')

# If the task is a classification task, one-hot encode the labels
if LABEL != '_timeconsumption':
    num_classes = np.max(df_validation.label.values) + 1
    y_validate = to_categorical(df_validation.label.values, num_classes=num_classes)
else:
    y_validate = df_validation.label.values

# Evaluate the model on the validate data
results = model.evaluate(X_validate_pad, y_validate, batch_size=128)
print("Test loss, Test acc:", results)