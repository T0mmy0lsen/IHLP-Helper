from tqdm import tqdm

import numpy as np
import pandas as pd
import socket


def create_relations():

    PATH_ITEMS = 'database/Item.csv'
    PATH_OBJECTS = 'database/Object.csv'
    PATH_RELATIONS = 'database/RelationHistory.csv'

    df_relations = pd.read_csv(PATH_RELATIONS, encoding='UTF-8', delimiter=';', quotechar='"', dtype=str, usecols=['id', 'tblid', 'tblTimeStamp', 'leftID', 'rightID'])
    # df_relations = df_relations[df_relations.leftID == '133780042']

    df_relations = df_relations.fillna('')
    df_relations = df_relations.sort_values(by=['id'])
    df_relations = df_relations.rename(columns={'tblid': 'relationId', 'tblTimeStamp': 'relationTblTimeStamp'})
    df_relations['relationTblTimeStamp'] = pd.to_datetime(df_relations['relationTblTimeStamp'])

    df_objects = pd.read_csv(PATH_OBJECTS, encoding='UTF-8', delimiter=';', quotechar='"', dtype=str, usecols=['id', 'name', 'createdBy'])
    df_objects = df_objects.fillna('')
    df_objects = df_objects.sort_values(by=['id'])
    df_objects = df_objects.rename(columns={'id': 'objectId'})

    df_items = pd.read_csv(PATH_ITEMS, encoding='UTF-8', delimiter=';', quotechar='"', dtype=str, usecols=['id', 'username', 'description'])
    df_items = df_items.fillna('')
    df_items = df_items.sort_values(by=['id'])
    df_items = df_items.rename(columns={'id': 'itemId'})

    print('32')
    df_o = pd.merge(df_relations, df_objects, left_on='relationId', right_on='objectId', how='left')
    df_o = pd.merge(df_o, df_objects, left_on='rightID', right_on='objectId', how='left')

    print('36')
    df_i = pd.merge(df_o, df_items, left_on='rightID', right_on='itemId', how='left')
    df_i = pd.merge(df_i, df_items, left_on='createdBy_y', right_on='itemId', how='left')

    df_i = df_i.fillna('')
    df_i = df_i[(df_i.name_x.str.contains('Request') & ~df_i.name_x.str.contains('Item'))]

    print('43')
    df_i['matching_id'] = 0
    df_i['type'] = ''

    columns = ['leftID', 'relationTblTimeStamp', 'name_y', 'name_x', 'username_y', 'username_x', 'createdBy_y', 'createdBy_x']

    print('49')
    df_i = df_i[columns]
    df_i.to_csv('bunch_of_relations.csv')


def create_requests():

    print('56')
    df = pd.read_csv('database/RequestHistory.csv', sep=';', low_memory=False)
    df.tblid = df.tblid.astype(str)

    df['event'] = 'update'
    df['leftID'] = df['tblid']
    df['relationTblTimeStamp'] = df['tblTimeStamp']

    print('64')
    df = df[['leftID', 'relationTblTimeStamp', 'timeConsumption', 'event']]
    df.to_csv('bunch_of_requests.csv')


def create_merge():

    print('71')
    df_requests = pd.read_csv('bunch_of_requests.csv')
    df_relations = pd.read_csv('bunch_of_relations.csv')

    for column in list(set(df_requests.columns).union(df_relations.columns)):
        if column not in df_requests.columns:
            df_requests[column] = None
        if column not in df_relations.columns:
            df_relations[column] = None

    print('81')
    df = pd.concat([df_requests, df_relations], ignore_index=True)
    df = df.sort_values('relationTblTimeStamp', ignore_index=True)

    print('85')
    df = df[['leftID', 'relationTblTimeStamp', 'timeConsumption', 'event', 'username_y', 'username_x', 'name_x', 'name_y','createdBy_y', 'createdBy_x']]
    df.to_csv('bunch_of_merged.csv', index=False)





# create_relations()
# create_requests()
# create_merge()
pass
