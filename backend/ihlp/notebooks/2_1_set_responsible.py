
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('bunch_of_tasks_but_better.csv')
df = df[df.event != 'created']
df['reaction_timestamp'] = pd.to_datetime(df['reaction_timestamp'])

df_p = df[(df.current_placement != 'unset')]
df_p = df_p.drop_duplicates(subset=['id'])

df_r = df[(df.current_responsible != 'unset')]
df_r = df_r.drop_duplicates(subset=['id'])

date = pd.to_datetime('2022-01-01')

tmp_p = df_p[(df_p['reaction_timestamp'] >= date)]
tmp_r = df_r[(df_r['reaction_timestamp'] >= date)]

df_p = df_p[df_p.current_placement.isin(tmp_p.current_placement.values)]
df_r = df_r[df_r.current_responsible.isin(tmp_r.current_responsible.values)]

encoder_p = LabelEncoder()
encoder_r = LabelEncoder()

df_p['label_encoded'] = encoder_p.fit_transform(df_p['current_placement'])
df_r['label_encoded'] = encoder_r.fit_transform(df_r['current_responsible'])

df_p = df_p.rename(columns={'current_placement': 'label_placement'})
df_r = df_r.rename(columns={'current_responsible': 'label_responsible'})

df_p[['id', 'label_encoded', 'label_placement']].to_csv('encoded_current_placement.csv', index=False)
df_r[['id', 'label_encoded', 'label_responsible']].to_csv('encoded_current_responsible.csv', index=False)