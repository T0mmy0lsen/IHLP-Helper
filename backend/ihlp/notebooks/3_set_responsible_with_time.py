
import numpy as np
import pandas as pd

from tqdm import tqdm

tqdm.pandas()

PATH_REQUESTS = 'database/Request.csv'

df = pd.read_csv(PATH_REQUESTS,
                 encoding='UTF-8',
                 delimiter=';',
                 quotechar='"',
                 dtype=str,
                 usecols=['id', 'timeConsumption'])

df = df.fillna('')

df = df[df['timeConsumption'] != '']
df['timeConsumption_reduced'] = df.apply(
    lambda x: float(x['timeConsumption']) if float(x['timeConsumption']) < 50.0 else 50.0, axis=1)

df['label_time'] = df['timeConsumption_reduced']
df['label_bins'] = pd.cut(df['label_time'], bins=[0.0, 2.0, 5.0, 10.0, 25.0, 50.0], labels=[0, 1, 2, 3, 4])
df[['id', 'label_bins', 'label_time']].to_csv('data/label_time.csv', index=False)



df_user = pd.read_csv('data/label_users_top_100.csv')
df_time = pd.read_csv('data/label_time.csv')

print(len(df_user))

df = pd.merge(df_user, df_time, on='id', how='left')
df = df.fillna('')
df = df[df['label_bins'] != '']

df[['id', 'label_bins']].to_csv('data/label_bins.csv', index=False)

df['tmp'] = (df.label_encoded * 5) + df.label_bins
df['label_time_encoded'] = df.apply(lambda x: int(x['tmp']), axis=1)

df[['id', 'label_closed', 'label_time_encoded']].to_csv('data/label_time_encoded.csv', index=False)