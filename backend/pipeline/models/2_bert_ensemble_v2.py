import datetime
from collections import defaultdict

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding, AutoConfig, BertForSequenceClassification, \
    TFAutoModel, BertConfig, TFBertForSequenceClassification, DistilBertTokenizerFast, TFDistilBertModel, \
    BertTokenizerFast, TFBertModel
import torch
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from official.nlp import optimization  # to create AdamW optimizer


class ModelBertEnsemble:

    def __init__(self, loss=1, batch_size=1):

        MODEL_NAME = 'bert-base-uncased'
        BATCH_SIZE = batch_size
        LEARNING_RATE = loss
        L2 = 0.01

        # df = pd.read_csv('../data/output/1_data_modeling/output_solved.csv')

        tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

        def tokenize_subject(sentences, max_length=128, padding='max_length'):
            return tokenizer(
                sentences,
                truncation=True,
                padding=padding,
                max_length=max_length,
                return_tensors="tf"
            )

        def tokenize_description(sentences, max_length=512, padding='max_length'):
            return tokenizer(
                sentences,
                truncation=True,
                padding=padding,
                max_length=max_length,
                return_tensors="tf"
            )

        # --------------------------------------------------------------------------------
        # Split data into training and validation
        # --------------------------------------------------------------------------------

        raw_train = pd.read_csv('../data/output/4_data_merged/output_merged.csv', nrows=50000)
        raw_train = raw_train.fillna('')
        raw_train['responsible_last'] = raw_train.apply(lambda x: 'keep' if x['received_by'] == '' or x['responsible_last'] == '' or x['responsible_last'] == x['received_by'] else 'else', axis=1)

        raw_train = raw_train.sample(10000)

        print(raw_train['responsible_last'].value_counts(normalize=True))

        NUM_LABELS = len(raw_train['responsible_last'].unique())

        def get_labels(labels):
            le = preprocessing.LabelEncoder()
            le.fit(labels)
            return le

        def get_received_by(received_by):
            le = preprocessing.LabelEncoder()
            le.fit(received_by)
            return le

        le_labels = get_labels(raw_train['responsible_last'].to_numpy())
        le_received_by = get_received_by(raw_train['received_by'].to_numpy())

        raw_train['label'] = raw_train.apply(lambda x: le_labels.transform([x['responsible_last']])[0], axis=1)
        raw_train['encoded_received_by'] = raw_train.apply(lambda x: le_received_by.transform([x['received_by']])[0], axis=1)

        train_hour, \
        validation_hour, \
        train_encoded_received_by, \
        validation_encoded_received_by, \
        train_subject, \
        validation_subject, \
        train_description, \
        validation_description, \
        train_label, \
        validation_label = train_test_split(
            raw_train['hour'].tolist(),
            raw_train['encoded_received_by'].tolist(),
            raw_train['subject'].tolist(),
            raw_train['description'].tolist(),
            raw_train['label'].tolist(),
            test_size=.2,
            shuffle=True
        )

        input_ids_subject, attention_mask_subject, output_subject = self.get_layer_for_subject()
        input_ids_description, attention_mask_description, output_description = self.get_layer_for_description()

        layer_hour = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='hour')
        layer_encoded_received_by = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='encoded_received_by')

        features = tf.keras.layers.concatenate([layer_hour, layer_encoded_received_by])

        features_output = tf.keras.layers.Dense(
            units=200,
            kernel_initializer='glorot_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(l2=L2),
            activation='softmax'
        )(features)

        output = tf.keras.layers.concatenate([features_output, output_subject, output_description])
        output = tf.keras.layers.Dense(
            units=NUM_LABELS,
            kernel_initializer='glorot_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(l2=L2),
            activation='softmax',
            name='softmax'
        )(output)

        model = tf.keras.Model(
            inputs=[layer_hour, layer_encoded_received_by, input_ids_subject, attention_mask_subject, input_ids_description, attention_mask_description],
            outputs=[output],
        )

        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            metrics=['accuracy']
        )
        
        model.summary()

        tokenized_train = defaultdict(list)
        tokenized_train['hour'] = tf.convert_to_tensor(train_hour)
        tokenized_train['encoded_received_by'] = tf.convert_to_tensor(train_encoded_received_by)
        tokenized_train['description.input_ids'] = dict(tokenize_description(train_description))['input_ids']
        tokenized_train['description.attention_mask'] = dict(tokenize_description(train_description))['attention_mask']
        tokenized_train['subject.input_ids'] = dict(tokenize_subject(train_subject))['input_ids']
        tokenized_train['subject.attention_mask'] = dict(tokenize_subject(train_subject))['attention_mask']

        tokenized_validation = defaultdict(list)
        tokenized_validation['hour'] = tf.convert_to_tensor(validation_hour)
        tokenized_validation['encoded_received_by'] = tf.convert_to_tensor(validation_encoded_received_by)
        tokenized_validation['description.input_ids'] = dict(tokenize_description(validation_description))['input_ids']
        tokenized_validation['description.attention_mask'] = dict(tokenize_description(validation_description))['attention_mask']
        tokenized_validation['subject.input_ids'] = dict(tokenize_subject(validation_subject))['input_ids']
        tokenized_validation['subject.attention_mask'] = dict(tokenize_subject(validation_subject))['attention_mask']

        X = tf.data.Dataset.from_tensor_slices((
            dict(tokenized_train),
            train_label
        )).batch(BATCH_SIZE).prefetch(1)

        V = tf.data.Dataset.from_tensor_slices((
            dict(tokenized_validation),
            validation_label
        )).batch(BATCH_SIZE).prefetch(1)

        f = open("logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), "a")

        class MetricCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                _loss = logs['loss']
                _accuracy = logs['accuracy']
                f.write(f'learning_rate={loss}, batch_size={batch_size}, loss={_loss}, accuracy={_accuracy}')

        model.fit(
            x=X,
            y=None,
            epochs=5,
            batch_size=BATCH_SIZE,
            validation_data=V,
            callbacks=[MetricCallback()]
        )

        f.close()


    def get_layer_for_description(self, MAX_SEQUENCE_LENGTH=512, MODEL_NAME='bert-base-uncased', NUM_BASE_MODEL_OUTPUT=768):

        base = TFBertModel.from_pretrained(MODEL_NAME)

        # Inputs for token indices and attention masks
        input_ids = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='description.input_ids')
        attention_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='description.attention_mask')

        for layer in base.layers:
            layer.trainable = False

        output = base([input_ids, attention_mask]).last_hidden_state[:, 0, :]
        output = tf.keras.layers.Dropout(rate=0.1)(output)

        return input_ids, attention_mask, output

    def get_layer_for_subject(self, MAX_SEQUENCE_LENGTH=128, MODEL_NAME='bert-base-uncased', NUM_BASE_MODEL_OUTPUT=768):

        base = TFBertModel.from_pretrained(MODEL_NAME)

        # Inputs for token indices and attention masks
        input_ids = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='subject.input_ids')
        attention_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='subject.attention_mask')

        for layer in base.layers:
            layer.trainable = False

        output = base([input_ids, attention_mask]).last_hidden_state[:, 0, :]
        output = tf.keras.layers.Dropout(rate=0.1)(output)

        return input_ids, attention_mask, output


# for loss in [.1, .05, .01, .005, .001, .0005, .0001, .00005, .00001]:
for loss in [.0005, .00025, .0001, .000075, .00005, .000025, .00001]:
    for batch_size in [16]:  # OOM on size 128
        ModelBertEnsemble(loss=loss, batch_size=batch_size)