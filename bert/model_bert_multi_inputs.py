import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from keras.utils import to_categorical
from sklearn import preprocessing

from tensorflow import keras
from keras import layers
from official.nlp import optimization

num_tags = 12
num_words = 10000
num_departments = 4

encoder_hub_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
bert_hub_url = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1'
bert_hub_url = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'

label_input = keras.layers.Input(shape=(1), dtype=tf.int32, name='x')
label_outputs = layers.Dense(100, activation='relu')(label_input)

subject_input = keras.layers.Input(shape=(), dtype=tf.string, name='subject')
subject_preprocessing_layer = hub.KerasLayer(encoder_hub_url, name='subject_preprocessing')
subject_encoder_inputs = subject_preprocessing_layer(subject_input)

description_input = keras.layers.Input(shape=(), dtype=tf.string, name='description')
description_preprocessing_layer = hub.KerasLayer(encoder_hub_url, name='description_preprocessing')
description_encoder_inputs = description_preprocessing_layer(description_input)

subject_encoder = hub.KerasLayer(bert_hub_url, trainable=True, name='subject_BERT_encoder')
subject_outputs = subject_encoder(subject_encoder_inputs)

description_encoder = hub.KerasLayer(bert_hub_url, trainable=True, name='description_BERT_encoder')
description_outputs = description_encoder(description_encoder_inputs)

description_net = description_outputs['pooled_output']
description_net = layers.Dropout(.1)(description_net)

subject_net = subject_outputs['pooled_output']
subject_net = layers.Dropout(.1)(subject_net)

x = layers.concatenate([subject_net, description_net, label_outputs])

x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(.1)(x)

label_pred = layers.Dense(100, name='label', activation='softmax')(x)

model = keras.Model(
    inputs=[subject_input, description_input, label_input],
    outputs=[label_pred],
)


""" -------------------------------------- """

data = pd.read_csv(
    "ihlp_multi_inputs/request_responsible_subject_description_label.csv",
    names=["id", "subject", "description", "label"]
)

data = data.fillna('')

top_list = data['label'].value_counts().index.tolist()
data = data[data['label'].isin(top_list[:100])]


def get_labels(labels):
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    return le


le = get_labels(np.unique(data['label'].to_numpy()))

n_classes = 100
y = le.transform(data['label'])
# y = to_categorical(y, n_classes)

subject = data['subject'].to_numpy()
description = data['description'].to_numpy()

""" -------------------------------------- """

epochs = 5

num_train_steps = 3000
num_warmup_steps = 300

init_lr = 3e-5
optimizer = optimization.create_optimizer(
    init_lr=init_lr,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    optimizer_type='adamw'
)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['acc']
)

model.summary()

model.fit(
    {'subject': subject, 'description': description, 'x': y},
    {'label': y},
    epochs=epochs,
    batch_size=32,
)