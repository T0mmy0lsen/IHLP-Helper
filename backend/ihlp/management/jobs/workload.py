import pandas as pd
import numpy as np

from django.db.models import Q
from ihlp.models import Workload, Predict
from ihlp.models_ihlp import Request, Item, Relation, Object
from datetime import datetime, timedelta


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
    queryset_relations = Relation.objects.using('ihlp').filter(leftid__in=df.id.values)
    df_relations = pd.DataFrame.from_records(queryset_relations.values('leftid', 'rightid', 'id'))
    df_relations = df_relations.rename(columns={'id': 'relationid'})

    queryset_objects = Object.objects.using('ihlp').filter(id__in=df_relations.relationid.tolist() + df_relations.rightid.tolist())
    queryset_items = Item.objects.using('ihlp').filter(id__in=df_relations.rightid.values)

    df_objects = pd.DataFrame.from_records(queryset_objects.values())
    df_objects = df_objects.rename(columns={'id': 'objectid'})

    df_items = pd.DataFrame.from_records(queryset_items.values())
    df_items = df_items.rename(columns={'id': 'itemid'})
    df_items = df_items[df_items['username'] != '']

    df = pd.merge(df, df_relations, left_on='id', right_on='leftid', how='left')
    df = pd.merge(df, df_objects, left_on='relationid', right_on='objectid', how='inner')
    df = pd.merge(df, df_objects, left_on='rightid', right_on='objectid', how='inner')
    df = pd.merge(df, df_items, left_on='rightid', right_on='itemid', how='left')

    df['placement'] = df.apply(lambda x: x['name_y'] if x['name_x'][-9:] == 'Placement' else 'unknown', axis=1)
    df['responsible'] = df.apply(lambda x: x['username'] if x['name_x'][-11:] == 'Responsible' else 'unknown', axis=1)

    if len(df) == 0:
        print('The dataframe is empty. Something probably went wrong.')
        return False

    df = df[['id', 'placement', 'responsible']]

    cc = df.groupby(['id']).cumcount() + 1
    df = df.set_index(['id', cc]).unstack().sort_index(1, level=1)
    df.columns = ['_'.join(map(str, i)) for i in df.columns]
    df = df.reset_index()
    df = df.fillna('unknown')

    def get_placement(x):
        if x['placement_1'] != 'unknown':
            return x['placement_1'].lower()
        if x['placement_2'] != 'unknown':
            return x['placement_2'].lower()
        if x['placement_3'] != 'unknown':
            return x['placement_3'].lower()
        if x['placement_4'] != 'unknown':
            return x['placement_4'].lower()
        return 'unknown'

    def get_responsible(x):
        if x['responsible_1'] != 'unknown':
            return x['responsible_1'].lower()
        if x['responsible_2'] != 'unknown':
            return x['responsible_2'].lower()
        if x['responsible_3'] != 'unknown':
            return x['responsible_3'].lower()
        if x['responsible_4'] != 'unknown':
            return x['responsible_4'].lower()
        return 'unknown'

    df['placement'] = df.apply(lambda x: get_placement(x), axis=1)
    df['responsible'] = df.apply(lambda x: get_responsible(x), axis=1)
    df = df[['id', 'placement', 'responsible']]

    for i, el in df.iterrows():
        p = Predict.objects.filter(request_id=el.id).first()
        w = Workload(request_id=el.id, data={
            'true_placement': el.placement,
            'true_responsible': el.responsible,
            'predict_placement': p.data['placement'][0:3],
            'predict_responsible': p.data['responsible'][0:3],
        })
        w.save()

    return df