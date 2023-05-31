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
<<<<<<< HEAD:backend/ihlp/notebooks/1_0_process_text_html_tags.py
=======
print('Length:', len(df))

# Filter out for evaluation.
df.receivedDate = pd.to_datetime(df.receivedDate)
df = df[df['receivedDate'] < datetime.strptime("2023-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")]
>>>>>>> c33128bae6b7d9fa9030c548208e4823278f7838:backend/ihlp/notebooks/1_process_text.py
df = df.fillna('')
print('Length:', len(df))

# Clean the data.
df.subject = df.progress_apply(lambda x: text_clean(x.subject), axis=1)
df.description = df.progress_apply(lambda x: text_clean(x.description), axis=1)

# Filter out for evaluation.
df.receivedDate = pd.to_datetime(df.receivedDate)
df_train = df[df['receivedDate'] < datetime.strptime("2023-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")]
df_valid = df[df['receivedDate'] >= datetime.strptime("2023-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")]

df_train[['id', 'subject']].to_csv('data/subject_html_tags.csv', index=False)
df_train[['id', 'description']].to_csv('data/description_html_tags.csv', index=False)
print('Length of train:', len(df_train))

df_valid[['id', 'subject']].to_csv('data/subject_html_tags_validate.csv', index=False)
df_valid[['id', 'description']].to_csv('data/description_html_tags_validate.csv', index=False)
print('Length of validate:', len(df_valid))