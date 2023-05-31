import re
from datetime import timedelta, datetime

import bs4 as bs4
import numpy as np
from django.conf import settings

from ihlp.models import Predict
from ihlp.models_ihlp import Request
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

import pandas as pd
import tensorflow as tf


class CustomModel(tf.keras.Model):
    def __init__(self, base_model):
        super(CustomModel, self).__init__()
        self.base_model = base_model
        self.output_layer = tf.keras.layers.Dense(units=1, activation='linear')

    def call(self, inputs, **args):
        x = self.base_model(inputs)
        x = self.output_layer(x[0])
        return x


def calculatePrediction(
        amount=0,
        delete=False,
):
    queryset_requests = Request.objects.using('ihlp').order_by('-id')[:amount]

    df = pd.DataFrame.from_records(queryset_requests.values('id', 'subject', 'description'))

    if delete:
        Predict.objects.filter(request_id__in=list(df.id.values)).delete()

    df_predictions = pd.DataFrame.from_records(
        Predict.objects.filter(request_id__in=list(df.id.values)).values('request_id')
    )

    if len(df_predictions) > 0:
        df = df[~df.id.isin(df_predictions.request_id.values)]
        df = df.reset_index()

    if len(df) == 0:
        return False

    def text_combine_and_clean(x):
        x = x['subject'] + ". " + x['description']
        x = bs4.BeautifulSoup(x, "lxml").text
        x = x.replace('/\s\s+/g', ' ')
        x = x.replace('/\n\n+/g', ' ')
        x = x.replace('/\t\t+/g', ' ')
        x = x.replace(u'\u00A0', ' ')
        x = x.lower()
        return x

    df['text'] = df.apply(lambda x: text_combine_and_clean(x)[:512], axis=1)

    # tokenizer = AutoTokenizer.from_pretrained(f'{settings.BASE_DIR}/ihlp/notebooks/data/models/XLM-RoBERTa-Tokenizer')
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def tokenize_texts(sentences, max_length=512, padding='max_length'):
        return tokenizer(
            sentences,
            truncation=False,
            padding=padding,
            max_length=max_length,
            return_tensors="tf"
        )

    tokenized_text = dict(tokenize_texts(list(df.text.values)))

    model_responsible = TFAutoModelForSequenceClassification.from_pretrained(f'{settings.BASE_DIR}/ihlp/notebooks/data/models/IHLP-XLM-RoBERTa-Responsible')
    model_responsible.compile(metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)])
    predict_responsible = model_responsible.predict(tokenized_text, batch_size=1, verbose=False)

    df_label_responsible = pd.read_csv(f'{settings.BASE_DIR}/ihlp/notebooks/data/label_responsible.csv')
    df_tmp_responsible = df_label_responsible.drop_duplicates(subset=['label_responsible', 'label_encoded'])
    df_tmp_responsible = df_tmp_responsible.sort_values(by='label_encoded')
    responsible_index = df_tmp_responsible.label_responsible.values

    model_placement = TFAutoModelForSequenceClassification.from_pretrained(f'{settings.BASE_DIR}/ihlp/notebooks/data/models/IHLP-XLM-RoBERTa-Placement')
    model_placement.compile(metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)])
    predict_placement = model_placement.predict(tokenized_text, batch_size=1, verbose=False)

    df_label_placement = pd.read_csv(f'{settings.BASE_DIR}/ihlp/notebooks/data/label_placement.csv')
    df_tmp_placement = df_label_placement.drop_duplicates(subset=['label_placement', 'label_encoded'])
    df_tmp_placement = df_tmp_placement.sort_values(by='label_encoded')
    placement_index = df_tmp_placement.label_placement.values

    model_path = f'{settings.BASE_DIR}/ihlp/notebooks/data/models/IHLP-XLM-RoBERTa-Time-Consumption'
    model = TFAutoModelForSequenceClassification.from_pretrained(model_path)

    predict_time_consumption = model.predict(tokenized_text, batch_size=1, verbose=False)

    def get_responsible_as_list(predictions):
        lst = []
        for i, val in enumerate(predictions):
            lst.append({
                'name': responsible_index[i],
                'prediction_log': predictions[i],
            })
        lst.sort(key=lambda x: x['prediction_log'], reverse=True)
        return lst

    def get_placement_as_list(predictions):
        lst = []
        for i, val in enumerate(predictions):
            lst.append({
                'name': placement_index[i],
                'prediction_log': predictions[i],
            })
        lst.sort(key=lambda x: x['prediction_log'], reverse=True)
        return lst

    def inverse_min_max_scaling(scaled_data, original_min=1, original_max=120, new_min=-1, new_max=1):
        return ((scaled_data - new_min) * (original_max - original_min)) / (new_max - new_min) + original_min

    print('Predict, saving:', len(df))
    for i, el in df.iterrows():
        Predict(
            request_id=el.id,
            data={
                'timeconsumption': predict_time_consumption[0][i],  # inverse_min_max_scaling(predict_time_consumption[0][i]),  # 120, # original_predictions,
                'responsible': get_responsible_as_list(predict_responsible[0][i]),
                'placement': get_placement_as_list(predict_placement[0][i]),
            }
        ).save()

    return True