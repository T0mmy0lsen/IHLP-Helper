import pandas as pd
import numpy as np

from django.db.models import Q
from ihlp.models import Workload, Predict
from ihlp.models_ihlp import Request, Item, RelationHistory, ObjectHistory
from datetime import datetime, timedelta

def calculateWorkload(time=datetime.strptime("2022-02-01 00:00:00", "%Y-%m-%d %H:%M:%S"), limit=1):

    # We limit the resultset to be within the last n days from 'time'.
    latest = time - timedelta(days=limit)

    # The result should show all that does not have a solution and have been recevied after 'latest' and before 'time'.
    # Note that 'time' is used to simulate a different current time.
    queryset_requests = Request.objects.using('ihlp').filter(
        Q(receiveddate__lte=time) & Q(receiveddate__gte=latest)
    )

    df_requests = pd.DataFrame.from_records(queryset_requests.values('id'))
    if len(df_requests) == 0:
        return False

    df_predictions = pd.DataFrame.from_records(
        Workload.objects.filter(request_id__in=list(df_requests.id.values)).values('request_id'))

    if len(df_predictions) > 0:
        df_requests = df_requests[~df_requests.id.isin(df_predictions.request_id.values)]
        df_requests = df_requests.reset_index()

    if len(df_requests) == 0:
        return False

    queryset_relations = RelationHistory.objects.using('ihlp').filter(leftid__in=df_requests.id.values)
    df_relations = pd.DataFrame.from_records(queryset_relations.values('leftid', 'rightid', 'tblid'))

    queryset_objects = ObjectHistory.objects.using('ihlp').filter(tblid__in=df_relations.tblid.values)
    queryset_items = Item.objects.using('ihlp').filter(id__in=df_relations.rightid.values)

    df_objects = pd.DataFrame.from_records(queryset_objects.values())
    df_objects = df_objects[df_objects['name'].isin([
        'RequestServiceResponsible',
        'RequestServiceReceivedBy',
        'RequestIncidentResponsible',
        'RequestIncidentReceivedBy',
    ])]

    df_items = pd.DataFrame.from_records(queryset_items.values())
    df_items = df_items.rename(columns={'id': 'itemid'})
    df_items = df_items[df_items['username'] != '']

    df = pd.merge(df_requests, df_relations, left_on='id', right_on='leftid', how='left')
    df = pd.merge(df, df_objects, on='tblid', how='inner')
    df = pd.merge(df, df_items, left_on='rightid', right_on='itemid', how='inner')

    df = df.sort_values(by=['tblid'])
    df = df.rename(columns={'id_x': 'id'})
    df = df.drop_duplicates(keep='last', subset=['id'])
    df = df[['id', 'username']]
    df.username = df.apply(lambda x: x.username.lower(), axis=1)

    PATH_RELATIVE = './ihlp/notebooks/data'

    df_label_users_top_100 = pd.read_csv(f'{PATH_RELATIVE}/label_users_top_100.csv')
    tmp = df_label_users_top_100.drop_duplicates(subset=['label_closed', 'label_users_top_100'])
    tmp = tmp.sort_values(by='label_users_top_100')
    user_index = tmp.label_closed.values

    def get_top_index(x):
        return np.argsort(-np.array(x))[0]

    for i, el in df.iterrows():
        index = np.where(user_index == el.username)[0]
        if len(index) > 0:
            index = index[0]
            p = Predict.objects.filter(request_id=el.id).first()
            predictions = p.data['prediction']
            predictions = predictions[index * 5:(index * 5) + 5]
            predictions = get_top_index(predictions)
        else:
            predictions = 5
        w = Workload(request_id=el.id, username=el.username, data={
            'workload': predictions,
            'username': el.username
        })
        w.save()

    return df