import datetime

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding, AutoConfig, BertForSequenceClassification, \
    TFAutoModel, BertConfig, TFBertForSequenceClassification, DistilBertTokenizerFast, TFDistilBertModel
import torch
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
from official.nlp import optimization  # to create AdamW optimizer


class ModelBert:

    def __init__(self):

        TIMESTAMP = datetime.datetime.now().strftime("%Y%b%d%H%M").upper()
        DATA_COLUMN = 'text'
        LABEL_COLUMN = 'label'
        MAX_SEQUENCE_LENGTH = 512
        MODEL_NAME = 'distilbert-base-uncased'
        NUM_BASE_MODEL_OUTPUT = 768
        FREEZE_BASE = True
        USE_CUSTOM_HEAD = False
        BATCH_SIZE = 16
        LEARNING_RATE = 5e-5 if FREEZE_BASE else 3e-5
        L2 = 0.01

        if not os.path.isfile('data/1_bert_description/output.csv'):

            df = pd.read_csv('../data/output/4_data_merged/output_merged.csv', usecols=['requestId', 'description', 'responsible_last'])
            df = df.rename(columns={'description': 'text', 'responsible_last': 'label'})
            df = df.dropna(subset='label')
            df = df.dropna(subset='text')

            def get_labels(labels):
                le = preprocessing.LabelEncoder()
                le.fit(labels)
                return le

            le = get_labels(df['label'].to_numpy())
            df['label'] = df.apply(lambda x: le.transform([x['label']])[0], axis=1)

            sample = df.sample(frac=.2, random_state=200)
            sample_inverse = df[df['requestId'].isin(sample['requestId'])]

            df = df.drop(['requestId'], axis=1)
            sample = sample.drop(['requestId'], axis=1)
            sample_inverse = sample_inverse.drop(['requestId'], axis=1)

            df.to_csv('data/1_bert_description/output.csv', index=False)
            sample.to_csv('data/1_bert_description/output_test.csv', index=False)
            sample_inverse.to_csv('data/1_bert_description/output_train.csv', index=False)

        df = pd.read_csv('data/1_bert_description/output.csv')

        NUM_LABELS = len(df['label'].unique())

        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

        def tokenize(sentences, max_length=MAX_SEQUENCE_LENGTH, padding='max_length'):
            return tokenizer(
                sentences,
                truncation=True,
                padding=padding,
                max_length=max_length,
                return_tensors="tf"
            )

        base = TFDistilBertModel.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS
        )

        # Inputs for token indices and attention masks
        input_ids = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_ids')
        attention_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='attention_mask')

        # Freeze the base model weights.
        if FREEZE_BASE:
            for layer in base.layers:
                layer.trainable = False

        # [CLS] embedding is last_hidden_state[:, 0, :]
        output = base([input_ids, attention_mask]).last_hidden_state[:, 0, :]

        if USE_CUSTOM_HEAD:

            output = tf.keras.layers.Dropout(rate=0.15, name="01_dropout")(output)

            # -------------------------------------------------------------------------------
            # Classification layer 01
            # --------------------------------------------------------------------------------

            output = tf.keras.layers.Dense(
                units=NUM_BASE_MODEL_OUTPUT,
                kernel_initializer='glorot_uniform',
                activation=None,
                name="01_dense_relu_no_regularizer",
            )(output)
            output = tf.keras.layers.BatchNormalization(name="01_bn")(output)
            output = tf.keras.layers.Activation("relu", name="01_relu")(output)

            # --------------------------------------------------------------------------------
            # Classification layer 02
            # --------------------------------------------------------------------------------

            output = tf.keras.layers.Dense(
                units=NUM_BASE_MODEL_OUTPUT,
                kernel_initializer='glorot_uniform',
                activation=None,
                name="02_dense_relu_no_regularizer",
            )(output)
            output = tf.keras.layers.BatchNormalization(name="02_bn")(output)
            output = tf.keras.layers.Activation("relu",name="02_relu")(output)

        output = tf.keras.layers.Dense(
            units=NUM_LABELS,
            kernel_initializer='glorot_uniform',
            kernel_regularizer=tf.keras.regularizers.l2(l2=L2),
            activation='softmax',
            name="softmax"
        )(output)

        # --------------------------------------------------------------------------------
        # Split data into training and validation
        # --------------------------------------------------------------------------------

        raw_train = pd.read_csv('data/1_bert_description/output.csv', nrows=10000)

        train_data, validation_data, train_label, validation_label = train_test_split(
            raw_train[DATA_COLUMN].tolist(),
            raw_train[LABEL_COLUMN].tolist(),
            test_size=.2,
            shuffle=True
        )

        x = dict(tokenize(train_data))

        X = tf.data.Dataset.from_tensor_slices((
            dict(tokenize(train_data)),
            train_label
        )).batch(BATCH_SIZE).prefetch(1)

        V = tf.data.Dataset.from_tensor_slices((
            dict(tokenize(validation_data)),
            validation_label
        )).batch(BATCH_SIZE).prefetch(1)

        epochs = 10

        name = f"{TIMESTAMP}_{MODEL_NAME.upper()}"
        model = tf.keras.models.Model(inputs=[input_ids, attention_mask], outputs=output, name=name)
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            metrics=['accuracy']
        )
        
        model.summary()
        model.fit(
            x=X,
            y=None,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            validation_data=V,
        )

ModelBert()