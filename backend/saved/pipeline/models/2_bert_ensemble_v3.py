import datetime
import tensorflow as tf
import pandas as pd

from collections import defaultdict
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, TFBertModel
from sklearn import preprocessing
from sklearn.preprocessing import normalize


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

        raw_train = pd.read_csv('../data/output/4_data_merged/output_merged.csv')
        raw_train = raw_train.fillna('')
        raw_train['responsible_last'] = raw_train.apply(
            lambda x: 'keep' if x['received_by'] == '' or x['responsible_last'] == '' or x['responsible_last'] == x[
                'received_by'] else 'else', axis=1)

        raw_train = raw_train.sample(5000)

        print(raw_train['responsible_last'].value_counts(normalize=True))

        NUM_LABELS = len(raw_train['responsible_last'].unique())

        def get_le_labels(labels):
            le = preprocessing.LabelEncoder()
            le.fit(labels)
            return le

        def get_le_received_by(received_by):
            le = preprocessing.LabelEncoder()
            le.fit(received_by)
            return le

        def get_le_user(user):
            le = preprocessing.LabelEncoder()
            le.fit(user)
            return le

        le_user = get_le_user(raw_train['user'].to_numpy())
        le_labels = get_le_labels(raw_train['responsible_last'].to_numpy())
        le_received_by = get_le_received_by(raw_train['received_by'].to_numpy())

        raw_train['user'] = raw_train.apply(lambda x: le_user.transform([x['user']])[0], axis=1)
        raw_train['label'] = raw_train.apply(lambda x: le_labels.transform([x['responsible_last']])[0], axis=1)
        raw_train['encoded_received_by'] = raw_train.apply(lambda x: le_received_by.transform([x['received_by']])[0], axis=1)

        train_month, \
        validation_month, \
        train_day_of_week, \
        validation_day_of_week, \
        train_hour, \
        validation_hour, \
        train_user, \
        validation_user, \
        train_encoded_received_by, \
        validation_encoded_received_by, \
        train_subject, \
        validation_subject, \
        train_description, \
        validation_description, \
        train_label, \
        validation_label = train_test_split(
            normalize([raw_train['month'].tolist()], norm="max")[0],
            normalize([raw_train['day_of_week'].tolist()], norm="max")[0],
            normalize([raw_train['hour'].tolist()], norm="max")[0],
            raw_train['user'].tolist(),
            raw_train['encoded_received_by'].tolist(),
            raw_train['subject'].tolist(),
            raw_train['description'].tolist(),
            raw_train['label'].tolist(),
            test_size=.2,
            shuffle=True
        )

        input_ids_subject, attention_mask_subject, output_subject = self.get_layer_for_subject()
        input_ids_description, attention_mask_description, output_description = self.get_layer_for_description()

        layer_month = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='month')
        layer_day_of_week = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='day_of_week')
        layer_hour = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='hour')
        layer_user = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='user')
        layer_encoded_received_by = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name='encoded_received_by')

        features = tf.keras.layers.concatenate([layer_month, layer_day_of_week, layer_hour, layer_user, layer_encoded_received_by])

        features = tf.keras.layers.Dense(units=128, activation='softmax')(features)
        features = tf.keras.layers.Dropout(.2)(features)
        features = tf.keras.layers.Dense(units=64, activation='softmax')(features)
        features = tf.keras.layers.Dropout(.2)(features)
        features = tf.keras.layers.Dense(units=32, activation='softmax')(features)
        features = tf.keras.layers.Dropout(.2)(features)

        output = tf.keras.layers.concatenate([features, output_subject, output_description])
        output = tf.keras.layers.Dense(units=NUM_LABELS, activation='softmax')(output)

        model = tf.keras.Model(
            inputs=[layer_month, layer_day_of_week, layer_hour, layer_user, layer_encoded_received_by, input_ids_subject, attention_mask_subject, input_ids_description, attention_mask_description],
            outputs=[output],
        )

        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            metrics=['accuracy']
        )

        model.summary()

        tokenized_train = defaultdict(list)
        tokenized_train['month'] = tf.convert_to_tensor(train_month)
        tokenized_train['day_of_week'] = tf.convert_to_tensor(train_day_of_week)
        tokenized_train['hour'] = tf.convert_to_tensor(train_hour)
        tokenized_train['user'] = tf.convert_to_tensor(train_user)
        tokenized_train['encoded_received_by'] = tf.convert_to_tensor(train_encoded_received_by)
        tokenized_train['description.input_ids'] = dict(tokenize_description(train_description))['input_ids']
        tokenized_train['description.attention_mask'] = dict(tokenize_description(train_description))['attention_mask']
        tokenized_train['subject.input_ids'] = dict(tokenize_subject(train_subject))['input_ids']
        tokenized_train['subject.attention_mask'] = dict(tokenize_subject(train_subject))['attention_mask']

        tokenized_validation = defaultdict(list)
        tokenized_validation['month'] = tf.convert_to_tensor(validation_month)
        tokenized_validation['day_of_week'] = tf.convert_to_tensor(validation_day_of_week)
        tokenized_validation['hour'] = tf.convert_to_tensor(validation_hour)
        tokenized_validation['user'] = tf.convert_to_tensor(validation_user)
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

        file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        class MetricCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                _loss = logs['loss']
                _accuracy = logs['accuracy']
                f = open("logs/scalars/" + file_name, "a")
                f.write(f'learning_rate={loss}, batch_size={batch_size}, loss={_loss}, accuracy={_accuracy}')
                f.close()

        model.fit(
            x=X,
            y=None,
            epochs=5,
            batch_size=BATCH_SIZE,
            validation_data=V,
            callbacks=[MetricCallback()]
        )

    def get_layer_for_description(self, MAX_SEQUENCE_LENGTH=512, MODEL_NAME='bert-base-uncased'):

        base = TFBertModel.from_pretrained(MODEL_NAME)

        # Inputs for token indices and attention masks
        input_ids = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='description.input_ids')
        attention_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='description.attention_mask')

        for layer in base.layers:
            layer.trainable = False

        output = base([input_ids, attention_mask]).last_hidden_state[:, 0, :]
        output = tf.keras.layers.Dropout(rate=0.1)(output)

        return input_ids, attention_mask, output

    def get_layer_for_subject(self, MAX_SEQUENCE_LENGTH=128, MODEL_NAME='bert-base-uncased'):

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
for loss in [.0005]:
    for batch_size in [16]:  # OOM on size 128
        ModelBertEnsemble(loss=loss, batch_size=batch_size)
