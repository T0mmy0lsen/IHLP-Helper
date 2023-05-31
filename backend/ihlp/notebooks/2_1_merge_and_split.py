from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

PATH_RELATIVE = 'data/'

TEXTS = ['_html_tags_validate', '_html_tags', '_raw', '_raw_validate', '_lemmatize']
LABELS = ['_placement', '_responsible', '_timeconsumption']

for TEXT in TEXTS:
    for LABEL in LABELS:

        df_labels = pd.read_csv(
            filepath_or_buffer=PATH_RELATIVE + f'label{LABEL}.csv',
            dtype={'id': int, 'label_encoded': int},
            sep=','
        )

        df_subject = pd.read_csv(
            filepath_or_buffer=PATH_RELATIVE + f'subject{TEXT}.csv',
            sep=','
        )

        df_description = pd.read_csv(
            filepath_or_buffer=PATH_RELATIVE + f'description{TEXT}.csv',
            sep=','
        )

        df_texts = pd.merge(df_subject, df_description, on='id', how='inner')
        df_texts = df_texts.fillna('')
        df_texts['text'] = df_texts.apply(lambda x:  x['subject'] + ". " + x['description'], axis=1)

        df = pd.merge(df_texts, df_labels, on='id', how='inner')
        df = df[['id', 'text', 'label_encoded']]
        df = df.rename(columns={'label_encoded': 'label'})
        df = df.fillna('')

        # if LABEL == '_timeconsumption':
        #    df.label = np.log10(df.label.astype('float'))

        df = df[df['text'] != '']

        if 'validate' not in TEXT:
            train, test = train_test_split(df, test_size=0.2, random_state=1)
            train.to_csv(PATH_RELATIVE + f'cached_train{TEXT}{LABEL}.csv', index=False)
            test.to_csv(PATH_RELATIVE + f'cached_test{TEXT}{LABEL}.csv', index=False)
        else:
            df.to_csv(PATH_RELATIVE + f'cached{TEXT}{LABEL}.csv', index=False)