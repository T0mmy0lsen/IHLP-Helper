

import pandas as pd
import numpy as np
import os
import re
import pickle

from sklearn import preprocessing
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

PATH_FROM_MERGED = 'data/output/4_data_merged/output_merged.csv'

PATH_TO_FEATURE_VECTOR_RECEIVED_BY = 'models/notebooks/data/features/output_received_by.csv'
PATH_TO_FEATURE_VECTOR_TITLE = 'models/notebooks/data/features/output_title.csv'
PATH_TO_FEATURE_VECTOR_OMK = 'models/notebooks/data/features/output_omk.csv'
PATH_TO_FEATURE_VECTOR = 'models/notebooks/data/features/output.csv'

PATH_TO_TEXT_ORIGINAL = 'data/output/5_data_encode/texts_original/output.csv'
PATH_TO_ONLY_TEXT = 'data/output/5_data_encode/texts_only_text/output.csv'
PATH_TO_TEXT_NO_STOPWORDS = 'data/output/5_data_encode/texts_no_stopwords/output.csv'

PATH_TO_FEATURE_VECTOR_FOLDER = 'data/output/5_data_encode/feature_vector/'
PATH_TO_TEXT_ORIGINAL_FOLDER = 'data/output/5_data_encode/texts_original/'
PATH_TO_ONLY_TEXT_FOLDER = 'data/output/5_data_encode/texts_only_text/'
PATH_TO_TEXT_NO_STOPWORDS_FOLDER = 'data/output/5_data_encode/texts_no_stopwords/'

PATH_TO_LABEL_TIME = 'data/output/5_data_encode/label_time/output.csv'
PATH_TO_LABEL_TOP_100 = 'data/output/5_data_encode/label_top_100/output.csv'
PATH_TO_LABEL_TOP_ALL = 'data/output/5_data_encode/label_top_all/output.csv'
PATH_TO_LABEL_KEEP_ELSE = 'data/output/5_data_encode/label_keep_else/output.csv'

PATH_TO_LABEL_TOP_100_FOLDER = 'data/output/5_data_encode/label_top_100/'
PATH_TO_LABEL_TOP_ALL_FOLDER = 'data/output/5_data_encode/label_top_all/'
PATH_TO_LABEL_KEEP_ELSE_FOLDER = 'data/output/5_data_encode/label_keep_else/'

PATH_TO_TD_IDF_FOLDER = 'data/output/5_data_encode/texts_td_idf/'
PATH_OUTPUT_CHECKPOINT_TIME = 'data/output/1_data_modeling/output_time.csv'


class Config:
    def __init__(self, max_df, min_df, max_features):
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features


class DataEncode:

    def __init__(self):
        self.df = pd.read_csv(PATH_FROM_MERGED, sep=",", quotechar="\"", dtype={
            'year': int,
            'day_of_week': int,
            'month': int,
            'hour': int,
        })
        self.df = self.df.fillna('')
        self.run()

    @staticmethod
    def get_labels_one_hot(labels):
        le = preprocessing.OneHotEncoder()
        le.fit(labels)
        return le

    @staticmethod
    def get_labels(labels):
        le = preprocessing.LabelEncoder()
        le.fit(labels)
        return le

    def run(self):

        if not os.path.isfile(PATH_TO_FEATURE_VECTOR):
            self.checkpoint_feature_vector()
        else:
            print(f'Skip checkpoint_feature_vector(). To rerun delete {PATH_TO_FEATURE_VECTOR}')

        if not os.path.isfile(PATH_TO_TEXT_ORIGINAL):
            self.checkpoint_text_original()
        else:
            print(f'Skip checkpoint_text_original(). To rerun delete {PATH_TO_TEXT_ORIGINAL}')

        if not os.path.isfile(PATH_TO_ONLY_TEXT):
            self.checkpoint_only_text()
        else:
            print(f'Skip checkpoint_only_text(). To rerun delete {PATH_TO_ONLY_TEXT}')

        if not os.path.isfile(PATH_TO_TEXT_NO_STOPWORDS):
            self.checkpoint_no_stopwords()
        else:
            print(f'Skip checkpoint_no_stopwords(). To rerun delete {PATH_TO_TEXT_NO_STOPWORDS}')

        if not os.path.isfile(PATH_TO_LABEL_KEEP_ELSE):
            self.checkpoint_labels_keep_else()
        else:
            print(f'Skip checkpoint_labels_keep_else(). To rerun delete {PATH_TO_LABEL_KEEP_ELSE}')

        if not os.path.isfile(PATH_TO_LABEL_TOP_100):
            self.checkpoint_labels_top_100()
        else:
            print(f'Skip checkpoint_labels_top_100(). To rerun delete {PATH_TO_LABEL_TOP_100}')

        if not os.path.isfile(PATH_TO_LABEL_TOP_ALL):
            self.checkpoint_labels_top_all()
        else:
            print(f'Skip checkpoint_labels_top_all(). To rerun delete {PATH_TO_LABEL_TOP_ALL}')

        if not os.path.isfile(PATH_TO_LABEL_TIME):
            self.checkpoint_labels_time()
        else:
            print(f'Skip checkpoint_labels_top_all(). To rerun delete {PATH_TO_LABEL_TIME}')

        self.checkpoint_text_td_idf()
        
    
    def checkpoint_feature_vector(self):

        df = self.df[['requestId', 'year', 'day_of_week' ,'month', 'hour', 'user', 'received_by']]
        df = df.rename(columns={'requestId': 'id'})
        print(len(df))

        ad = pd.read_excel('data/input/5_data_encode/ADAnsatte.xlsx')
        ad = ad.drop_duplicates(subset='sAMAccountName', keep='last')

        df = pd.merge(df, ad, left_on='user', right_on='sAMAccountName', how='left')
        print(len(df))

        df = df.fillna('')

        df['AD.Expires'] = df.apply(lambda x: '2030-01-01' if x['AD.Expires'] == '4712-12-31' else x['AD.Expires'], axis=1)
        df['AD.Expires'] = df.apply(lambda x: '2000-01-01' if x['AD.Expires'] == '' else x['AD.Expires'], axis=1)

        df['AD.Created'] = df.apply(lambda x: '2030-01-01' if x['AD.Created'] == '4712-12-31' else x['AD.Created'], axis=1)
        df['AD.Created'] = df.apply(lambda x: '2000-01-01' if x['AD.Created'] == '' else x['AD.Created'], axis=1)

        df['expires'] = pd.to_datetime(df['AD.Expires']).astype('int64')
        df['created'] = pd.to_datetime(df['AD.Created']).astype('int64')

        _max = df.expires.max()
        _min = df.expires.min()
        df['expires_normalized'] = (df.expires - _min) / (_max - _min)

        _max = df.created.max()
        _min = df.created.min()
        df['created_normalized'] = (df.created - _min) / (_max - _min)

        _max = df.year.max()
        _min = df.year.min()
        df['year_normalized'] = (df.day_of_week - _min) / (_max - _min)

        _max = df.day_of_week.max()
        _min = df.day_of_week.min()
        df['day_of_week_normalized'] = (df.day_of_week - _min) / (_max - _min)

        _max = df.month.max()
        _min = df.month.min()
        df['month_normalized'] = (df.month - _min) / (_max - _min)

        _max = df.hour.max()
        _min = df.hour.min()
        df['hour_normalized'] = (df.hour - _min) / (_max - _min)

        def encoder(col):
            enc = preprocessing.OneHotEncoder()
            le = preprocessing.LabelEncoder()
            df_encoded = df[[col]]
            df_encoded = df_encoded.apply(le.fit_transform)
            enc.fit(df_encoded)
            return enc.transform(df_encoded).toarray()

        enc_omk = encoder('omk')
        enc_title = encoder('title')
        enc_received_by = encoder('received_by')

        # enc_merged = np.concatenate((enc_omk, enc_title, enc_received_by), axis=1)

        # enc_df_merged = pd.DataFrame(enc_merged)
        enc_df_received_by = pd.DataFrame(enc_received_by)
        enc_df_title = pd.DataFrame(enc_title)
        enc_df_omk = pd.DataFrame(enc_omk)

        enc_df_received_by = pd.merge(
            df[['id']],
            enc_df_received_by,
            left_index=True,
            right_index=True
        )

        enc_df_title = pd.merge(
            df[['id']],
            enc_df_title,
            left_index=True,
            right_index=True
        )

        enc_df_omk = pd.merge(
            df[['id']],
            enc_df_omk,
            left_index=True,
            right_index=True
        )

        enc_df_received_by.to_csv(PATH_TO_FEATURE_VECTOR_RECEIVED_BY, index=False)
        enc_df_title.to_csv(PATH_TO_FEATURE_VECTOR_TITLE, index=False)
        enc_df_omk.to_csv(PATH_TO_FEATURE_VECTOR_OMK, index=False)

        df = df[['id', 'expires_normalized', 'created_normalized', 'day_of_week_normalized', 'month_normalized', 'hour_normalized']]
        df.to_csv(PATH_TO_FEATURE_VECTOR, index=False)


    def checkpoint_text_td_idf(self, config=None):

        if config is None:
            configs = [
                Config(1.0, 0.0001, 10000),
                Config(1.0, 0.00025, 10000),
                Config(1.0, 0.0005, 10000),
                Config(1.0, 0.00075, 10000),
                Config(1.0, 0.001, 10000),
                Config(1.0, 0.0025, 10000),
                Config(1.0, 0.005, 10000),
                Config(1.0, 0.010, 10000),
                Config(1.0, 0.050, 10000),
            ]
        else:
            configs = [config]

        def create_vectorizer(train, max_df, min_df, max_features):
            tfidf_vectorizer = TfidfVectorizer(
                max_df=max_df,
                min_df=min_df,
                max_features=max_features,
                stop_words=None,
                use_idf=True,
                ngram_range=(1, 5)
            )
            tfidf_vectorizer.fit_transform(train)
            return tfidf_vectorizer

        for config in configs:

            file_name = f'td_idf_{config.max_df}_{config.min_df}_{config.max_features}.pickle'

            if not os.path.isfile(PATH_TO_TD_IDF_FOLDER + file_name):

                tmp = pd.read_csv(PATH_TO_TEXT_NO_STOPWORDS)
                tmp = tmp.fillna('')

                tfidf = create_vectorizer(tmp['text'].to_numpy(), config.max_df, config.min_df, config.max_features)

                with open(PATH_TO_TD_IDF_FOLDER + file_name, 'wb') as fin:
                    pickle.dump(tfidf, fin)


    def checkpoint_labels_keep_else(self):

        self.df['keep_else'] = self.df.apply(lambda x: 'keep' if x['received_by'] == '' or x['responsible_last'] == '' or x['responsible_last'] == x['received_by'] else 'else', axis=1)
        tmp = self.df[['requestId', 'keep_else']]
        tmp = tmp.rename(columns={'requestId': 'id', 'keep_else': 'label'})
        label_encoder = DataEncode.get_labels(tmp['label'].to_numpy())
        tmp['label_encoded'] = label_encoder.transform(tmp['label'])
        tmp.to_csv(PATH_TO_LABEL_KEEP_ELSE, index=False)


    def checkpoint_labels_top_100(self):

        def get_responsible(x):
            if x['responsible_last'] != '':
                return x['responsible_last']
            if x['received_by'] != '':
                return x['received_by']
            return 'unknown'

        self.df['top_100'] = self.df.apply(lambda x: get_responsible(x), axis=1)

        top_list = self.df['top_100'].value_counts().index.tolist()
        tmp = self.df[self.df['top_100'].isin(top_list[:100])]

        tmp = tmp[['requestId', 'top_100']]
        tmp = tmp.rename(columns={'requestId': 'id', 'top_100': 'label'})
        label_encoder = DataEncode.get_labels(tmp['label'].to_numpy())
        tmp['label_encoded'] = label_encoder.transform(tmp['label'])
        tmp.to_csv(PATH_TO_LABEL_TOP_100, index=False)


    def checkpoint_labels_time(self):

        tmp = pd.read_csv(PATH_OUTPUT_CHECKPOINT_TIME)
        tmp = tmp[['requestId', 'time_bins']]
        tmp = tmp.fillna('')
        tmp = tmp[tmp['time_bins'] != '']
        tmp = tmp.rename(columns={'requestId': 'id', 'time_bins': 'label'})

        label_encoder = DataEncode.get_labels(tmp['label'].to_numpy())
        tmp['label_encoded'] = label_encoder.transform(tmp['label'])
        tmp.to_csv(PATH_TO_LABEL_TIME, index=False)


    def checkpoint_labels_top_all(self):

        def get_responsible(x):
            if x['responsible_last'] != '':
                return x['responsible_last']
            if x['received_by'] != '':
                return x['received_by']
            return 'unknown'

        self.df['top_all'] = self.df.apply(lambda x: get_responsible(x), axis=1)
        tmp = self.df[['requestId', 'top_all']]
        tmp = tmp.rename(columns={'requestId': 'id', 'top_all': 'label'})
        label_encoder = DataEncode.get_labels(tmp['label'].to_numpy())
        tmp['label_encoded'] = label_encoder.transform(tmp['label'])
        tmp.to_csv(PATH_TO_LABEL_TOP_ALL, index=False)


    def checkpoint_only_text(self):
        self.df['only_text'] = self.df.apply(lambda x: re.sub(' +', ' ', re.sub(r'[^a-z]', ' ', x['subject'].lower() + " " + x['description'].lower())), axis=1)
        tmp = self.df[['requestId', 'only_text']]
        tmp = tmp.rename(columns={'requestId': 'id', 'only_text': 'text'})
        tmp.to_csv(PATH_TO_ONLY_TEXT, index=False)


    def checkpoint_no_stopwords(self):

        with open('data/output/5_data_encode/texts_no_stopwords/stopwords.txt') as file:
            lines = file.readlines()
            stopwords = [line.rstrip() for line in lines]

        tmp = pd.read_csv(PATH_TO_ONLY_TEXT)
        tmp['text'] = tmp.apply(lambda x: " ".join([e for e in x['text'].split(' ') if e not in stopwords]), axis=1)
        tmp.to_csv(PATH_TO_TEXT_NO_STOPWORDS, index=False)


    def checkpoint_text_original(self):
        self.df['original_text'] = self.df.apply(lambda x: x['subject'].lower() + " " + x['description'].lower(), axis=1)
        tmp = self.df[['requestId', 'original_text']]
        tmp = tmp.rename(columns={'requestId': 'id', 'original_text': 'text'})
        tmp.to_csv(PATH_TO_TEXT_ORIGINAL, index=False)


DataEncode()