import pandas as pd
import numpy as np

from django.db.models import Q
from ihlp.models import Workload, Predict
from ihlp.models_ihlp import Request, Item, RelationHistory, ObjectHistory
from datetime import datetime, timedelta


def calculateWorkloadTotal(
    time=datetime.strptime("2022-02-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
    limit=14,
    df=None
):

    latest = time - timedelta(days=limit)

    # The result should show all that does not have a solution and have been received after 'latest' and before 'time'.
    # Note that 'time' is used to simulate a different current time.
    queryset_requests = Request.objects.using('ihlp').filter(
        (Q(closingcode=None) | Q(solutiondate__gte=time)) & Q(receiveddate__lte=time) & Q(receiveddate__gte=latest)
    )

    # We can't write in the Request table, so we need to keep track of which has been predicted separately.
    # So we get all Request, and filter out those we already predicted.
    df = pd.DataFrame.from_records(queryset_requests.values('id'))

    if len(df) == 0:
        return False

    df_workloads = pd.DataFrame.from_records(
        Workload.objects.filter(request_id__in=list(df.id.values)).values())

    if len(df_workloads) == 0:
        return False

    PATH_RELATIVE = './ihlp/notebooks/data'

    df_label_users_top_100 = pd.read_csv(f'{PATH_RELATIVE}/label_users_top_100.csv')
    tmp = df_label_users_top_100.drop_duplicates(subset=['label_closed', 'label_users_top_100'])
    tmp = tmp.sort_values(by='label_users_top_100')
    user_index = tmp.label_closed.values

    workloads = {}
    for user in user_index:
        workloads[user] = sum(e['workload'] for e in df_workloads[df_workloads.username == user].data)
    return workloads


def calculateWorkload(
        time=datetime.strptime("2022-02-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
        limit=1,
        df=None
):
    if df is None:
        # We limit the result-set to be within the last n days from 'time'.
        latest = time - timedelta(days=limit)

        queryset_requests = Request.objects.using('ihlp').filter(
            Q(receiveddate__lte=time) & Q(receiveddate__gte=latest)
        )

        # We limit the set to only Request.id
        df = pd.DataFrame.from_records(queryset_requests.values('id'))
        if len(df) == 0:
            return False
    else:
        df = df[['id']]

    # We get all prior predictions from the Request.id set. No need to predict twice (unless we change model).
    df_workloads = pd.DataFrame.from_records(
        Workload.objects.filter(request_id__in=list(df.id.values)).values('request_id'))

    if len(df_workloads) > 0:
        df = df[~df.id.isin(df_workloads.request_id.values)]
        df = df.reset_index()

    if len(df) == 0:
        return False

    # Here we join the needed data to retrieve current Responsible and/or ReceivedBy
    queryset_relations = RelationHistory.objects.using('ihlp').filter(leftid__in=df.id.values)
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

    df = pd.merge(df, df_relations, left_on='id', right_on='leftid', how='left')
    df = pd.merge(df, df_objects, on='tblid', how='inner')
    df = pd.merge(df, df_items, left_on='rightid', right_on='itemid', how='inner')

    if len(df) == 0:
        print('The dataframe is empty. Something probably went wrong.')
        return False

    df = df.sort_values(by=['tblid'])
    df = df.rename(columns={'id_x': 'id'})
    df = df.drop_duplicates(keep='last', subset=['id'])
    df = df[['id', 'username']]
    df.username = df.apply(lambda x: x.username.lower(), axis=1)

    # This bit creates a list of users which index (almost) corresponds to the output of the model.
    # The model with 500 labels; the output time is 'output % 5' and the output username is 'int(output / 5)'.

    PATH_RELATIVE = './ihlp/notebooks/data'

    df_label_users_top_100 = pd.read_csv(f'{PATH_RELATIVE}/label_users_top_100.csv')
    tmp = df_label_users_top_100.drop_duplicates(subset=['label_closed', 'label_users_top_100'])
    tmp = tmp.sort_values(by='label_users_top_100')
    user_index = tmp.label_closed.values

    # Get an array of indexes that is sorted by the value of the index, e.g. [21, 31, 10] -> [1, 0, 2]
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