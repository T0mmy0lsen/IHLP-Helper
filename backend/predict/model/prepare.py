import datetime
import os
import shutil

import numpy as np
from langdetect import detect

import config as cf
import pandas as pd


class Prepare:

    df_relation_history = None
    df_object_history = None
    df_request = None
    df_item = None

    def __init__(self, shared, label_index='responsible', type='responsible'):

        self.shared = shared
        self.label_index = label_index
        self.run(type)

        path = f'{cf.BASE_PATH}/data/output/prepare/{type}/{self.shared.hashed}'
        self.df = pd.read_csv(f'{path}/{self.shared.dfs_names_train}_{self.label_index}.csv')
        self.df = self.df.fillna('')
        self.df = self.df[self.df['label'] != '']

    def fetch(self,
              categorical: bool = True,
              categorical_index: bool = True,
              amount: int = 1000,
              index_label: str = 'label',
              index_text: str = 'text',
              split=.25,
              seed=123,
              filter=None,
              lang='da'):

        print("[Prepare] Start")

        if filter is not None:
            self.df = self.df[self.df[index_label].isin(filter)]

        if lang is not None:

            def detect_lang(x):
                try:
                    return detect(x[index_text])
                except:
                    return lang

            self.df['lang'] = self.df.progress_apply(lambda x: detect_lang(x), axis=1)
            self.df = self.df[self.df['lang'] == lang]

        categories = []
        if categorical:
            categories = self.df.head(amount)[index_label].to_numpy()
            categories = np.sort(np.unique(categories))

        arr_x = self.df.head(amount)[index_text].to_numpy()
        arr_y = []

        if categorical_index:
            for idx, el in self.df.head(amount).iterrows():
                i = np.where(categories == el[index_label])
                arr_y.append(i)
        else:
            arr_y = self.df.head(amount)[index_label].to_numpy()

        arr_y = np.array(arr_y).flatten()

        x_train, y_train, x_validate, y_validate, categories = self.shuffle_and_split(arr_x, arr_y, split, seed, categories)

        self.shared.x_train = x_train
        self.shared.y_train = y_train
        self.shared.x_validate = x_validate
        self.shared.y_validate = y_validate
        self.shared.categories = categories

        print(f"\t Length of x_train = {len(x_train)}")
        print(f"\t Length of x_validate = {len(x_validate)}")
        print(f"\t Length of categories = {len(categories)}")
        print("[Prepare] End")
        print("")

    def construct_labels_time(self, label_path):

        def set_time(x):

            result_communication = 0
            result_solution = 0

            if x['rightType'] == 'CommunicationSimple' and str(x['receivedDate'])[0] != '0':
                received = datetime.datetime.strptime(str(x['receivedDate']), "%Y-%m-%d %H:%M:%S")
                solution = datetime.datetime.strptime(str(x['tblTimeStamp']), "%Y-%m-%d %H:%M:%S")
                result_communication = int(solution.timestamp()) - int(received.timestamp())
                if result_communication <= 0:
                    result_communication = 0

            if str(x['receivedDate'])[0] != '0' and str(x['solutionDate'])[0] != '0':
                received = datetime.datetime.strptime(str(x['receivedDate']), "%Y-%m-%d %H:%M:%S")
                solution = datetime.datetime.strptime(str(x['solutionDate']), "%Y-%m-%d %H:%M:%S")
                result_solution = int(solution.timestamp()) - int(received.timestamp())
                if result_solution <= 0:
                    result_solution = 0

            if result_communication == 0 and result_solution == 0:
                return 0
            return np.min([result_solution, result_communication])


        def set_time_bins(x, max_val, bins):
            i = ((x.name + 1) * bins)
            return int(i / (max_val + 1))


        df_re = self.df_request
        print(f"\t Expected length: {len(df_re)}")

        df_rh = self.df_relation_history
        df_rh = df_rh.sort_values(by='tblTimeStamp')
        df_rh = df_rh[df_rh['leftType'].isin(['RequestService', 'RequestIncident'])]
        df_rh = df_rh.drop_duplicates(subset=['rightId'], keep='last')

        df = pd.merge(df_rh, df_re, left_on='leftId', right_on='requestId')

        df_with_communication = df[df['rightType'] == 'CommunicationSimple']
        df_with_communication = df_with_communication.drop_duplicates(subset=['leftId'], keep='last')

        df_without_communication = df[~df['leftId'].isin(np.unique(df_with_communication['leftId'].to_numpy()))]
        df_without_communication = df_without_communication.drop_duplicates(subset=['leftId'], keep='last')

        df = pd.concat([df_with_communication, df_without_communication])

        df['time_seconds'] = df.apply(lambda x: set_time(x), axis=1)
        df = df[df['time_seconds'] > 0]
        df = df.sort_values(by='time_seconds')
        df = df.reset_index()

        max_val = len(df)
        df['time'] = df.apply(lambda x: set_time_bins(x, max_val, 5), axis=1)

        print(f"\t Actual length: {len(df)}")

        df.to_csv(label_path, index=False, columns=['requestId', 'time'])
        return 0

    def construct_labels_responsible(self, label_path):

        df_rh = self.df_relation_history
        df_oh = self.df_object_history
        df_it = self.df_item

        df_rh = df_rh.sort_values(by='tblTimeStamp')
        df_oh = df_oh[df_oh['name'].isin(
            ['RequestServiceResponsible', 'RequestIncidentResponsible', 'RequestServiceReceivedBy', 'RequestIncidentReceivedBy']
        )]

        df_rh_tmp = df_rh.drop_duplicates(subset=['leftId'], keep='last')
        df_rh_tmp = df_rh_tmp[df_rh_tmp['leftType'].isin(['RequestService', 'RequestIncident'])]
        length_expected = len(df_rh_tmp)
        print("[Prepare] Expected length:", length_expected)

        # We expect 1/4 not having an Object with Responsible and/or ReveivedBy
        df = pd.merge(df_rh, df_oh, left_on='rhTblId', right_on='ohTblId')
        df = pd.merge(df, df_it, left_on='rightId', right_on='itemId', how='left')
        df = df[df['username'] != '']
        df = df.drop_duplicates(subset=['leftId'], keep='last')
        length_actual = len(df_rh_tmp)
        print("[Prepare] Actual length:", length_actual)

        df = df.rename(columns={'leftId': 'requestId', 'username': 'responsible'})
        df.to_csv(label_path, index=False, columns=['requestId', 'responsible'])

    def construct_text(self, df):

        def make_text(x):
            text = []
            for index in self.shared.dfs_index_train:
                text.append(x[index])
            return " ".join(text)
        df['text'] = df.apply(lambda x: make_text(x), axis=1)
        return df

    def load(self):

        self.df_relation_history = pd.read_csv(f'{cf.BASE_PATH}/data/input/relation_history.csv')
        self.df_object_history = pd.read_csv(f'{cf.BASE_PATH}/data/input/object_history.csv')
        self.df_request = pd.read_csv(f'{cf.BASE_PATH}/data/input/request.csv', nrows=self.shared.nrows)
        self.df_item = pd.read_csv(f'{cf.BASE_PATH}/data/input/item.csv', low_memory=False)

        self.df_relation_history = self.df_relation_history.rename(columns={'id': 'rhId', 'tblid': 'rhTblId'})
        self.df_object_history = self.df_object_history.rename(columns={'id': 'ohId', 'tblid': 'ohTblId'})
        self.df_request = self.df_request.rename(columns={'id': 'requestId'})
        self.df_item = self.df_item.rename(columns={'id': 'itemId'})

        self.df_relation_history = self.df_relation_history.fillna('')
        self.df_object_history = self.df_object_history.fillna('')
        self.df_request = self.df_request.fillna('')
        self.df_item = self.df_item.fillna('')


    def run(self, type):

        label_path = f'{cf.BASE_PATH}/data/output/prepare/labels_{type}.csv'
        print(f"[Prepare] Creating labels for: {type}")
        if not os.path.exists(label_path):
            self.load()
            if type == 'responsible':
                self.construct_labels_responsible(label_path)
            elif type == 'time':
                self.construct_labels_time(label_path)
            else:
                print('Type is unknown. Use either \'responsible\' or \'time\'')
                return 0

        generate_path = f'{cf.BASE_PATH}/data/output/prepare/{type}/{self.shared.hashed}'
        if not os.path.isdir(generate_path):

            os.makedirs(generate_path)

            df_labels = pd.read_csv(label_path)

            df = pd.read_csv(f'{cf.BASE_PATH}/data/output/preprocessed/{self.shared.hashed}/{self.shared.dfs_names_train}.csv')
            df = df.fillna('')
            df = pd.merge(df, df_labels, left_on='id', right_on='requestId')
            df = df.rename(columns={f'{self.label_index}': 'label'})
            df = self.construct_text(df)            
            df.to_csv(f'{generate_path}/{self.shared.dfs_names_train}_{self.label_index}.csv', columns=['id', 'text', 'label'], index=False)


    @staticmethod
    def shuffle_and_split(arr_x, arr_y, split, seed, categories):

        # Shuffle the data
        seed = seed
        rng = np.random.RandomState(seed)
        rng.shuffle(arr_x)
        rng = np.random.RandomState(seed)
        rng.shuffle(arr_y)

        # Extract a training & validation split
        validation_split = split
        num_validation_samples = int(validation_split * len(arr_x))
        arr_x_train = arr_x[:-num_validation_samples]
        arr_x_validate = arr_x[-num_validation_samples:]
        arr_y_train = arr_y[:-num_validation_samples]
        arr_y_validate = arr_y[-num_validation_samples:]

        return arr_x_train, arr_y_train, arr_x_validate, arr_y_validate, categories
