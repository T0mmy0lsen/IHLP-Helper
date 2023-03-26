import pandas as pd
import numpy as np

from django.db.models import Q
from ihlp.models import Predict, Responsible, Workload
from ihlp.models_ihlp import Request, Item, Relation, Object
from datetime import datetime, timedelta


def calculateResponsibles(
        amount=0,
        delete=False,
):

    queryset_requests = Request.objects.using('ihlp').order_by('-id')[:amount]

    df = pd.DataFrame.from_records(queryset_requests.values('id'))

    if len(df) == 0:
        return False

    if delete:
        Responsible.objects.filter(request_id__in=list(df.id.values)).delete()

    # We get all prior predictions from the Request.id set. No need to predict twice (unless we change model).
    df_responsible = pd.DataFrame.from_records(
        Responsible.objects.filter(request_id__in=list(df.id.values)).values('request_id'))

    if len(df_responsible) > 0:
        df = df[~df.id.isin(df_responsible.request_id.values)]
        df = df.reset_index()

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

    def get_role(x, check):
        user = 'unknown'
        for el in check:
            user_next = x[el].lower()
            if user_next != 'unknown':
                if user == 'unknown':
                    user = user_next
                elif user == 'it help line1':
                    user = user_next
        return user

    check_placement = [x for x in df.columns if x[:9] == 'placement']
    check_responsible = [x for x in df.columns if x[:11] == 'responsible']

    check_placement.sort()
    check_responsible.sort()

    df['placement'] = df.apply(lambda x: get_role(x, check_placement), axis=1)
    df['responsible'] = df.apply(lambda x: get_role(x, check_responsible), axis=1)
    df = df[['id', 'placement', 'responsible']]


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

    def get_role(x, check, role):
        user = 'unknown'
        for el in check:
            user_next = x[el].lower()
            if user_next != 'unknown':
                if user == 'unknown':
                    user = user_next
                elif user == 'it help line1':
                    user = user_next
        return user

    check_placement = [x for x in df.columns if x[:9] == 'placement']
    check_responsible = [x for x in df.columns if x[:11] == 'responsible']

    check_placement.sort()
    check_responsible.sort()

    df['placement'] = df.apply(lambda x: get_role(x, check_placement, 'Placement'), axis=1)
    df['responsible'] = df.apply(lambda x: get_role(x, check_responsible, 'Responsible'), axis=1)
    df = df[['id', 'placement', 'responsible']]

    for i, el in df.iterrows():

        p = Predict.objects.filter(request_id=el.id).first()
        p_placement = []
        p_responsible = []

        if p:
            p_placement = p.data['placement'][0:3]
            p_responsible = p.data['responsible'][0:3]

        w = Responsible(request_id=el.id, data={
            'true_placement': el.placement,
            'true_responsible': el.responsible,
            'predict_placement': p_placement,
            'predict_responsible': p_responsible,
        })
        w.save()

    return df