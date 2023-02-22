import bs4
import pandas as pd

from tqdm import tqdm
from datetime import datetime

tqdm.pandas()

PATH_REQUESTS = 'database/Request.csv'

def text_clean(x):
    x = bs4.BeautifulSoup(x, "lxml").text
    x = x.replace('/\s\s+/g', ' ')
    x = x.replace('/\n\n+/g', '\n')
    x = x.replace('/\t\t+/g', '\t')
    x = x.replace(u'\u00A0', ' ')
    x = x.lower()
    return x

# Load the data.
df = pd.read_csv(PATH_REQUESTS,  encoding='UTF-8',  delimiter=';', quotechar='"', dtype=str, parse_dates=True)
print('Length:', len(df))

# Filter out for evaluation.
df.receivedDate = pd.to_datetime(df.receivedDate)
df = df[df['receivedDate'] < datetime.strptime("2022-11-01 00:00:00", "%Y-%m-%d %H:%M:%S")]
df = df.fillna('')
print('Length:', len(df))

# Clean the data and save it.
df.subject = df.progress_apply(lambda x: text_clean(x.subject), axis=1)
df.description = df.progress_apply(lambda x: text_clean(x.description), axis=1)

df[['id', 'subject']].to_csv('data/subject.csv', index=False)
df[['id', 'description']].to_csv('data/description.csv', index=False)