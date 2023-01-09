#%%

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

PATH_REQUESTS = 'data/text.csv'
DEBUG = False

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

from sklearn import preprocessing

def get_labels(labels):
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    return le

df = pd.read_csv('data/label_users.csv')
df = df[df['label_closed'] != 'unknown']

top_list = df['label_closed'].value_counts().index.tolist()
tmp = df[df['label_closed'].isin(top_list[:100])]
label_encoder = get_labels(tmp['label_closed'].to_numpy())
tmp['label_encoded'] = label_encoder.transform(tmp['label_closed'])
tmp[['id', 'label_closed', 'label_encoded']].to_csv('data/label_users_top_100.csv', index=False)
