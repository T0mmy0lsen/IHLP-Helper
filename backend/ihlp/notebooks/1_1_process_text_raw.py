import bs4
import pandas as pd

from tqdm import tqdm
from datetime import datetime

tqdm.pandas()

PATH_REQUESTS = 'database/Request.csv'

def text_clean(x):
    x = x.replace(u'\u00A0', ' ')
    return x

# Load the data.
df = pd.read_csv(PATH_REQUESTS,  encoding='UTF-8',  delimiter=';', quotechar='"', dtype=str, parse_dates=True)
print('Length:', len(df))

# Filter out for evaluation.
df.receivedDate = pd.to_datetime(df.receivedDate)

# Clean the data and save it.
df = df.fillna('')
df.subject = df.progress_apply(lambda x: text_clean(x.subject), axis=1)
df.description = df.progress_apply(lambda x: text_clean(x.description), axis=1)

df_train = df[df['receivedDate'] < datetime.strptime("2023-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")]
df_valid = df[df['receivedDate'] >= datetime.strptime("2023-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")]

df_train[['id', 'subject']].to_csv('data/subject_raw.csv', index=False)
df_train[['id', 'description']].to_csv('data/description_raw.csv', index=False)
print('Length of train:', len(df_train))

df_valid[['id', 'subject']].to_csv('data/subject_raw_validate.csv', index=False)
df_valid[['id', 'description']].to_csv('data/description_raw_validate.csv', index=False)
print('Length of validate:', len(df_valid))