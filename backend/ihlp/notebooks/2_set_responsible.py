import os

import pandas as pd

from sklearn import preprocessing
from tqdm import tqdm
from datetime import datetime

tqdm.pandas()

HAS_CACHE_DATA = os.path.isfile('data/label_users.csv')
PATH_REQUESTS = 'data/text.csv'
DEBUG = False

if not HAS_CACHE_DATA:
    df_requests = pd.read_csv(PATH_REQUESTS, usecols=['id'], dtype=str)
    df_requests = df_requests.fillna('')

    if DEBUG:
        df_requests = df_requests[df_requests['id'] == '109842828']

    PATH_RELATIONS = 'database/Relation.csv'

    df_relations = pd.read_csv(PATH_RELATIONS, encoding='UTF-8', delimiter=';', quotechar='"', dtype=str, usecols=['id', 'leftID', 'rightID'])

    if DEBUG:
        df_relations = df_relations[df_relations['leftID'] == '109842828']

    df_relations = df_relations.fillna('')
    df_relations = df_relations.sort_values(by=['id'])
    df_relations = df_relations.rename(columns={'id': 'relationId'})

    df_r = pd.merge(df_requests, df_relations, left_on='id', right_on='leftID', how='left')

    def name_changed(x):
        if x['name'] == 'RequestIncidentPlacement':
            return 'RequestServicePlacement'
        if x['name'] == 'RequestIncidentResponsible':
            return 'RequestServiceResponsible'
        if x['name'] == 'RequestIncidentReceivedBy':
            return 'RequestServiceReceivedBy'
        if x['name'] == 'RequestIncidentUser':
            return 'RequestServiceUser'
        return x['name']

    PATH_OBJECTS = 'database/Object.csv'

    df_objects = pd.read_csv(PATH_OBJECTS, encoding='UTF-8', delimiter=';', quotechar='"', dtype=str, usecols=['id', 'name'])
    df_objects = df_objects.fillna('')

    df_objects['name_changed'] = df_objects.apply(lambda x: name_changed(x), axis=1)

    df_objects = df_objects.sort_values(by=['id'])
    df_objects = df_objects.rename(columns={'id': 'objectId'})

    df_o = pd.merge(df_r, df_objects, left_on='relationId', right_on='objectId', how='inner')
    df_o = pd.merge(df_o, df_objects, left_on='rightID', right_on='objectId', how='inner')

    PATH_ITEMS = 'database/Item.csv'

    df_items = pd.read_csv(PATH_ITEMS, encoding='UTF-8', delimiter=';', quotechar='"', dtype=str, usecols=['id', 'username'])

    df_items = df_items.fillna('')
    df_items = df_items[df_items['username'] != '']
    df_items = df_items.rename(columns={'id': 'itemId'})

    df_i = pd.merge(df_o, df_items, left_on='rightID', right_on='itemId', how='left')
    df_i = df_i.sort_values(by='name_changed_x')

    df_i['username'] = df_i.apply(lambda x: x['name_y'] if x['name_x'][-9:] == 'Placement' else x['username'], axis=1)

    cc = df_i.groupby(['leftID']).cumcount() + 1
    df = df_i.set_index(['leftID', cc]).unstack().sort_index(1, level=1)
    df.columns = ['_'.join(map(str,i)) for i in df.columns]
    df.reset_index()
    df = df.fillna('unknown')

    def get_role(x, role):
        if x['name_changed_x_1'][-len(role):] == role:
            return x['username_1'].lower()
        if x['name_changed_x_2'][-len(role):] == role:
            return x['username_2'].lower()
        if x['name_changed_x_3'][-len(role):] == role:
            return x['username_3'].lower()
        if x['name_changed_x_4'][-len(role):] == role:
            return x['username_4'].lower()
        return 'unknown'

    def get_closed(x):
        if x['label_placement'] != 'unknown':
            return x['label_placement']
        if x['label_responsible'] != 'unknown':
            return x['label_responsible']
        return x['label_received']

    df['label_user'] = df.apply(lambda x: get_role(x, 'User'), axis=1)
    df['label_received'] = df.apply(lambda x: get_role(x, 'ReceivedBy'), axis=1)
    df['label_responsible'] = df.apply(lambda x: get_role(x, 'Responsible'), axis=1)
    df['label_placement'] = df.apply(lambda x: get_role(x, 'Placement'), axis=1)
    df['label_closed'] = df.apply(lambda x: get_closed(x), axis=1)

    df = df[['id_1', 'label_user', 'label_received', 'label_responsible', 'label_placement', 'label_closed']]
    df = df.rename(columns={'id_1': 'id'})
    df.to_csv('data/label_users.csv', index=False)


def get_labels(labels):
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    return le

df_requests = pd.read_csv('database/Request.csv',  encoding='UTF-8',  delimiter=';', quotechar='"', parse_dates=True)
df_requests.receivedDate = pd.to_datetime(df_requests.receivedDate)
df_requests = df_requests[df_requests.receivedDate > datetime.strptime("2022-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")]
ids = df_requests.id.values

df = pd.read_csv('data/label_users.csv')

df_tmp = df[df.id.isin(ids)]

df_for_placement = df_tmp[df_tmp.label_placement != 'unknown']
top_list_label_placement = df_for_placement.label_placement.value_counts().index.tolist()[:100]

df_for_responsible = df_tmp[df_tmp.label_responsible != 'unknown']
top_list_label_responsible = df_for_responsible.label_responsible.value_counts().index.tolist()[:100]

df_for_placement = df[df.label_placement.isin(top_list_label_placement)]
label_encoder_for_placement = get_labels(top_list_label_placement)
df_for_placement['label_encoded'] = label_encoder_for_placement.transform(df_for_placement.label_placement)

df_for_responsible = df[df.label_responsible.isin(top_list_label_responsible)]
label_encoder_for_responsible = get_labels(top_list_label_responsible)
df_for_responsible['label_encoded'] = label_encoder_for_responsible.transform(df_for_responsible.label_responsible)

df_for_placement[['id', 'label_placement', 'label_encoded']].to_csv('data/label_placement.csv', index=False)
df_for_responsible[['id', 'label_responsible', 'label_encoded']].to_csv('data/label_responsible.csv', index=False)
