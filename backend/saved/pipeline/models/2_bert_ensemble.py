import datetime
from collections import defaultdict

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding, AutoConfig, BertForSequenceClassification, \
    TFAutoModel, BertConfig, TFBertForSequenceClassification, DistilBertTokenizerFast, TFDistilBertModel
import torch
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from official.nlp import optimization  # to create AdamW optimizer


class ModelBertEnsemble:

    def __init__(self):



        MODEL_NAME = 'distilbert-base-uncased'
        FREEZE_BASE = True
        BATCH_SIZE = 16
        LEARNING_RATE = 1e-2 if FREEZE_BASE else 3e-5
        L2 = 0.01

        df = pd.read_csv('../data/output/1_data_modeling/output_solved.csv')

        NUM_LABELS = len(df['responsible_last'].unique())

        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

        def tokenize_subject(sentences, max_length=128, padding='max_length'):
            return tokenizer(
                sentences,
                truncation=True,
                padding=padding,
                max_length=max_length,
                return_tensors="tf"
            )

        def tokenize_description(sentences, max_length=256, padding='max_length'):
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

        raw_train = pd.read_csv('../data/output/1_data_modeling/output_solved.csv')
        raw_train = raw_train.dropna(subset='responsible_last')
        raw_train = raw_train.fillna('')

        def get_labels(labels):
            le = preprocessing.LabelEncoder()
            le.fit(labels)
            return le

        le = get_labels(raw_train['responsible_last'].to_numpy())
        raw_train['label'] = raw_train.apply(lambda x: le.transform([x['responsible_last']])[0], axis=1)

        train_subject, \
        validation_subject, \
        train_description, \
        validation_description, \
        train_label, \
        validation_label = train_test_split(
            raw_train['subject'].tolist(),
            raw_train['description'].tolist(),
            raw_train['label'].tolist(),
            test_size=.2,
            shuffle=True
        )

        input_ids_subject, attention_mask_subject, output_subject = self.get_layer_for_subject()
        input_ids_description, attention_mask_description, output_description = self.get_layer_for_description()

        output = tf.keras.layers.concatenate([output_subject, output_description])
        output = tf.keras.layers.Dense(
            units=NUM_LABELS,
            kernel_initializer='glorot_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(l2=L2),
            activation='softmax',
            name="softmax"
        )(output)

        model = tf.keras.Model(
            inputs=[input_ids_subject, attention_mask_subject, input_ids_description, attention_mask_description],
            outputs=[output],
        )

        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            metrics=['accuracy']
        )
        
        model.summary()

        """
        tokenized_train = defaultdict(list)
        for i, row in tqdm(enumerate(train_label)):
            t = tokenizer(
                train_description[i],
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors="tf"
            )
            tokenized_train['description.input_ids'].append(t['input_ids'])
            tokenized_train['description.attention_mask'].append(t['attention_mask'])
            t = tokenizer(
                train_subject[i],
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors="tf"
            )
            tokenized_train['subject.input_ids'].append(t['input_ids'])
            tokenized_train['subject.attention_mask'].append(t['attention_mask'])

        tokenized_validation = defaultdict(list)
        for i, row in tqdm(enumerate(validation_label)):
            t = tokenizer(
                validation_description[i],
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors="tf"
            )
            tokenized_validation['description.input_ids'].append(t['input_ids'])
            tokenized_validation['description.attention_mask'].append(t['attention_mask'])
            t = tokenizer(
                validation_subject[i],
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors="tf"
            )
            tokenized_validation['subject.input_ids'].append(t['input_ids'])
            tokenized_validation['subject.attention_mask'].append(t['attention_mask'])
        """

        tokenized_train = defaultdict(list)
        tokenized_train['description.input_ids'] = dict(tokenize_description(train_description))['input_ids']
        tokenized_train['description.attention_mask'] = dict(tokenize_description(train_description))['attention_mask']
        tokenized_train['subject.input_ids'] = dict(tokenize_subject(train_subject))['input_ids']
        tokenized_train['subject.attention_mask'] = dict(tokenize_subject(train_subject))['attention_mask']

        tokenized_validation = defaultdict(list)
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

        model.fit(
            x=X,
            y=None,
            epochs=10,
            batch_size=BATCH_SIZE,
            validation_data=V,
        )

    def get_layer_for_description(self, MAX_SEQUENCE_LENGTH=256, MODEL_NAME='distilbert-base-uncased', NUM_BASE_MODEL_OUTPUT=768):

        base = TFDistilBertModel.from_pretrained(MODEL_NAME)

        # Inputs for token indices and attention masks
        input_ids = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='description.input_ids')
        attention_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='description.attention_mask')

        for layer in base.layers:
            layer.trainable = False

        output = base([input_ids, attention_mask]).last_hidden_state[:, 0, :]
        output = tf.keras.layers.Dropout(rate=0.15)(output)
        output = tf.keras.layers.Dense(
            units=NUM_BASE_MODEL_OUTPUT,
            kernel_initializer='glorot_uniform',
            activation=None,
        )(output)

        return input_ids, attention_mask, output

    def get_layer_for_subject(self, MAX_SEQUENCE_LENGTH=128, MODEL_NAME='distilbert-base-uncased', NUM_BASE_MODEL_OUTPUT=768):

        base = TFDistilBertModel.from_pretrained(MODEL_NAME)

        # Inputs for token indices and attention masks
        input_ids = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='subject.input_ids')
        attention_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='subject.attention_mask')

        for layer in base.layers:
            layer.trainable = False

        output = base([input_ids, attention_mask]).last_hidden_state[:, 0, :]
        output = tf.keras.layers.Dropout(rate=0.15)(output)
        output = tf.keras.layers.Dense(
            units=NUM_BASE_MODEL_OUTPUT,
            kernel_initializer='glorot_uniform',
            activation=None,
        )(output)

        return input_ids, attention_mask, output


ModelBertEnsemble()