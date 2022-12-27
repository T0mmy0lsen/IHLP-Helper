from datetime import timedelta, datetime

import numpy as np
from django.db.models import Q

from ihlp.models import Predict
from ihlp.models_ihlp import Request
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

import pandas as pd
import tensorflow as tf

def calculatePrediction(time=datetime.strptime("2022-02-01 00:00:00", "%Y-%m-%d %H:%M:%S"), limit=1):

    # TODO: Get a list of requests without predictions. Should be limited.

    # We limit the resultset to be within the last n days from 'time'.
    latest = time - timedelta(days=limit)

    # The result should show all that does not have a solution and have been recevied after 'latest' and before 'time'.
    # Note that 'time' is used to simulate a different current time.
    queryset_requests = Request.objects.using('ihlp').filter(
        Q(receiveddate__lte=time) & Q(receiveddate__gte=latest)
    )

    # We can't write in the Request table, so we need to keep track of which has been predicted seperately.
    # So we get all Request, and filter out those we allready predicted.
    df = pd.DataFrame.from_records(queryset_requests.values('id', 'subject', 'description'))
    if len(df) == 0:
        return False

    df_predictions = pd.DataFrame.from_records(Predict.objects.filter(request_id__in=list(df.id.values)).values('request_id'))

    if len(df_predictions) > 0:
        df = df[~df.id.isin(df_predictions.request_id.values)]
        df = df.reset_index()

    if len(df) == 0:
        return False

    df['text'] = df.apply(lambda x: f'{x.subject} {x.description}'[:512], axis=1)

    # TODO: Use model to do predictions

    PATH_RELATIVE = './ihlp/notebooks'

    model = TFAutoModelForSequenceClassification.from_pretrained(f'{PATH_RELATIVE}/models/IHLP-XLM-RoBERTa-Time-Encoded')
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

    model.compile(metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10)])
    predict = model.predict(tokenized_text, batch_size=1, verbose=False)

    df_label_users_top_100 = pd.read_csv(f'{PATH_RELATIVE}/data/label_users_top_100.csv')
    tmp = df_label_users_top_100.drop_duplicates(subset=['label_closed', 'label_users_top_100'])
    tmp = tmp.sort_values(by='label_users_top_100')
    user_index = tmp.label_closed.values

    def get_as_list(i):
        obj = {}
        for val in user_index:
            obj[val] = {}
        predictions = predict[0][i]
        for i, val in enumerate(predictions):
            if i % 5 == 0:
                user = user_index[int(i / 5)]
                user_predictions = predictions[i:i + 5]
                obj[user]['predictions'] = user_predictions
                obj[user]['predictions_sum'] = sum(user_predictions)
                obj[user]['predictions_index'] = np.argsort(-np.array(user_predictions))
        return obj

    # TODO: Save predictions
    for i, el in df.iterrows():
        Predict(
            request_id=el.id,
            data={'prediction': predict[0][i], 'list': get_as_list(i)}
        ).save()

    return True