import os
import shutil

import numpy as np

import config as cf
import pandas as pd


class Prepare:

    df_relation_history = None
    df_request = None
    df_item = None

    def __init__(self, shared=None, label_index='assignee'):

        self.shared = shared
        self.label_index = label_index
        self.run()

        path = f'{cf.BASE_PATH}/model/output/prepare/{self.shared.hashed}'
        self.df = pd.read_csv(f'{path}/{self.shared.dfs_names_train}_{self.label_index}.csv')
        self.df = self.df.fillna('')
        self.df = self.df[self.df['label'] != '']

    def fetch(self,
              categorical: bool = True,
              amount: int = 1000,
              index_label: str = 'label',
              index_text: str = 'text',
              split=.25,
              seed=123):

        categories = []
        if categorical:
            categories = self.df.head(amount)[index_label].to_numpy()
            categories = np.sort(np.unique(categories))

        arr_x = self.df.head(amount)[index_text].to_numpy()
        arr_y = []

        if categorical:
            for idx, el in self.df.head(amount).iterrows():
                i = np.where(categories == el[index_label])
                arr_y.append(i)
        else:
            arr_y = self.df.head(amount)[index_label].to_numpy()

        arr_y = np.array(arr_y).flatten()

        return self.shuffle_and_split(arr_x, arr_y, split, seed, categories)

    def construct_labels(self, label_path):
        self.load()
        df = pd.merge(self.df_request, self.df_relation_history, left_on='requestId', right_on='leftId')
        df = df[df['rightType'] == 'ItemRole']
        df = pd.merge(df, self.df_item, left_on='rightId', right_on='itemId')
        df = df.sort_values(by='tblTimeStamp')
        df = df[df['username'] != '']
        df = df.drop_duplicates(subset=['requestId'], keep='last')
        df = df.rename(columns={'username': 'assignee'})
        df.to_csv(label_path, index=False, columns=['requestId', 'assignee'])

    def construct_text(self, df):
        def make_text(x):
            text = []
            for index in self.shared.dfs_index_train:
                text.append(x[index])
            return " ".join(text)
        df['text'] = df.progress_apply(lambda x: make_text(x), axis=1)
        return df

    def load(self):

        self.df_relation_history = pd.read_csv(f'{cf.BASE_PATH}/data/relation_history.csv', na_values='')
        self.df_request = pd.read_csv(f'{cf.BASE_PATH}/data/request.csv', na_values='', nrows=self.shared.nrows)
        self.df_item = pd.read_csv(f'{cf.BASE_PATH}/data/item.csv', na_values='', low_memory=False)

        self.df_relation_history = self.df_relation_history.rename(columns={'id': 'relationHistoryId'})
        self.df_request = self.df_request.rename(columns={'id': 'requestId'})
        self.df_item = self.df_item.rename(columns={'id': 'itemId'})

        self.df_relation_history = self.df_relation_history.fillna('')
        self.df_request = self.df_request.fillna('')
        self.df_item = self.df_item.fillna('')


    def run(self):
        df = pd.read_csv(
            f'{cf.BASE_PATH}/model/output/preprocessed/{self.shared.hashed}/{self.shared.dfs_names_train}.csv')

        label_path = f'{cf.BASE_PATH}/model/output/prepare/labels.csv'
        if not os.path.exists(label_path):
            self.construct_labels(label_path)

        df_labels = pd.read_csv(label_path)

        df = df.fillna('')
        df = pd.merge(df, df_labels, left_on='id', right_on='requestId')
        df = df.rename(columns={f'{self.label_index}': 'label'})
        df = self.construct_text(df)

        generate_path = f'{cf.BASE_PATH}/model/output/prepare/{self.shared.hashed}'
        if os.path.isdir(generate_path):
            shutil.rmtree(generate_path)
        os.makedirs(generate_path)

        # TO-DO: Should probably not make this file each time.
        df.to_csv(f'{generate_path}/{self.shared.dfs_names_train}_{self.label_index}.csv', columns=['id', 'text', 'label'], index=False)

    def shuffle_and_split(self, arr_x, arr_y, split, seed, categories):

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
