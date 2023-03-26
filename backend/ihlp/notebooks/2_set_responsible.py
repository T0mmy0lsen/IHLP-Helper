import os

import pandas as pd

from sklearn import preprocessing
from tqdm import tqdm
from datetime import datetime

tqdm.pandas()

HAS_CACHE_DATA = os.path.isfile('data/label_users.csv')
PATH_REQUESTS = 'data/subject.csv'
DEBUG = False
DEBUG_REQUEST = '85598869'

if not HAS_CACHE_DATA:

    df_requests = pd.read_csv(
        'database/Request.csv',
        encoding='UTF-8',
        delimiter=';',
        quotechar='"',
        parse_dates=True,
        usecols=['id', 'timeConsumption'],
        dtype=str
    )

    df_requests = df_requests.fillna('')

    df = pd.read_csv(PATH_REQUESTS, usecols=['id'], dtype=str)
    df_requests = df_requests[df_requests.id.isin(df.id.values)]

    if DEBUG:
        df_requests = df_requests[df_requests['id'] == DEBUG_REQUEST]

    # df_requests = df_requests[50000:50100]

    PATH_RELATIONS = 'database/RelationHistory.csv'

    df_relations = pd.read_csv(PATH_RELATIONS, encoding='UTF-8', delimiter=';', quotechar='"', dtype=str, usecols=['id', 'tblid', 'leftID', 'rightID'])

    if DEBUG:
        df_relations = df_relations[df_relations['leftID'] == DEBUG_REQUEST]

    df_relations = df_relations.fillna('')
    df_relations = df_relations.sort_values(by=['id'])
    df_relations = df_relations.rename(columns={'tblid': 'relationId'})

    df_r = pd.merge(df_requests, df_relations, left_on='id', right_on='leftID', how='left')
    df_r = df_r.drop_duplicates(subset='relationId', keep='last')

    PATH_OBJECTS = 'database/Object.csv'

    df_objects = pd.read_csv(PATH_OBJECTS, encoding='UTF-8', delimiter=';', quotechar='"', dtype=str, usecols=['id', 'name'])
    df_objects = df_objects.fillna('')

    df_objects = df_objects.sort_values(by=['id'])
    df_objects = df_objects.rename(columns={'id': 'objectId'})

    df_o = pd.merge(df_r, df_objects, left_on='relationId', right_on='objectId', how='inner')
    df_o = pd.merge(df_o, df_objects, left_on='rightID', right_on='objectId', how='inner')
    # df_o = df_o[(df_o.name_x.str.contains('Placement')) | (df_o.name_x.str.contains('Responsible'))]

    PATH_ITEMS = 'database/Item.csv'

    df_items = pd.read_csv(PATH_ITEMS, encoding='UTF-8', delimiter=';', quotechar='"', dtype=str, usecols=['id', 'username'])

    df_items = df_items.fillna('')
    df_items = df_items[df_items['username'] != '']
    df_items = df_items.rename(columns={'id': 'itemId'})

    df_i = pd.merge(df_o, df_items, left_on='rightID', right_on='itemId', how='left')

    df_i['username'] = df_i.apply(lambda x: x['name_y'] if x['name_x'][-9:] == 'Placement' else x['username'], axis=1)
    df_i = df_i.sort_values(by=['id_y'])

    cc = df_i.groupby(['leftID']).cumcount() + 1
    df = df_i.set_index(['leftID', cc]).unstack().sort_index(1, level=1)
    df.columns = ['_'.join(map(str, i)) for i in df.columns]
    df.reset_index()
    df = df.fillna('unknown')

    def get_role(x, check, role):
        user = 'unknown'
        for el in check:
            if x[el][-len(role):] == role:
                user_next = x[f'username_{el[7:]}'].lower()
                if user_next != 'unknown':
                    if user == 'unknown':
                        user = user_next
                    elif user == 'it help line1':
                        user = user_next
        return user

    def get_closed(x):
        if x['label_placement'] != 'unknown':
            return x['label_placement']
        if x['label_responsible'] != 'unknown':
            return x['label_responsible']
        return x['label_received']

    check = [x for x in df.columns if x[:6] == 'name_x']
    check.sort()

    df['label_timeconsumption'] = df.apply(lambda x: x['timeConsumption_1'], axis=1)
    df['label_responsible'] = df.apply(lambda x: get_role(x, check, 'Responsible'), axis=1)
    df['label_placement'] = df.apply(lambda x: get_role(x, check, 'Placement'), axis=1)

    df = df.rename(columns={'id_x_1': 'id'})
    df.to_csv('data/label_users_full.csv', index=False)
    df[['id', 'label_timeconsumption', 'label_responsible', 'label_placement']].to_csv('data/label_users.csv', index=False)


def get_labels(labels):
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    return le


df = pd.read_csv('data/label_users.csv', dtype=str)
df['id'] = df['id'].astype(int)
df = df.sort_values(by='id')
df_latest = df[-25000:]

df_for_timeconsumption = df[df.label_timeconsumption != '']
df_for_timeconsumption['label_encoded'] = df_for_timeconsumption.label_timeconsumption

df_for_placement = df_latest[df_latest.label_placement != 'unknown']
top_list_label_placement = df_for_placement.label_placement.value_counts().index.tolist()[:250]

df_for_responsible = df_latest[df_latest.label_responsible != 'unknown']
top_list_label_responsible = df_for_responsible.label_responsible.value_counts().index.tolist()[:250]

df_for_placement = df[df.label_placement.isin(top_list_label_placement)]
label_encoder_for_placement = get_labels(top_list_label_placement)
df_for_placement['label_encoded'] = label_encoder_for_placement.transform(df_for_placement.label_placement)

df_for_responsible = df[df.label_responsible.isin(top_list_label_responsible)]
label_encoder_for_responsible = get_labels(top_list_label_responsible)
df_for_responsible['label_encoded'] = label_encoder_for_responsible.transform(df_for_responsible.label_responsible)

df_for_placement[['id', 'label_placement', 'label_encoded']].to_csv('data/label_placement.csv', index=False)
df_for_responsible[['id', 'label_responsible', 'label_encoded']].to_csv('data/label_responsible.csv', index=False)
df_for_timeconsumption[['id', 'label_timeconsumption', 'label_encoded']].to_csv('data/label_timeconsumption.csv', index=False)

print(df_for_placement.label_placement.describe())
print(df_for_responsible.label_responsible.describe())
print(df_for_timeconsumption.label_timeconsumption.describe())
