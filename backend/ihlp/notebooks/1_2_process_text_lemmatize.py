import os
import pandas as pd
import spacy as spacy

from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

tqdm.pandas()

PATH_REQUESTS = 'database/Request.csv'
PATH_RELATIVE = 'data/'
HAS_CACHE_DATA = os.path.isfile('data/subject_with_lang.csv')

if not HAS_CACHE_DATA:

    def text_detect(x):
        try:
            return detect(x)
        except:
            return 'unknown'

    df_s = pd.read_csv('data/subject_html_tags.csv')
    df_d = pd.read_csv('data/description_html_tags.csv')

    df_s.subject = df_s.progress_apply(lambda x: text_detect(x.subject), axis=1)
    df_d.description = df_d.progress_apply(lambda x: text_detect(x.description), axis=1)

    df_s[['id', 'subject']].to_csv('data/subject_with_lang.csv', index=False)
    df_d[['id', 'description']].to_csv('data/description_with_lang.csv', index=False)

# Load and merge data
df_s_lang = pd.read_csv(f'{PATH_RELATIVE}subject_with_lang.csv')
df_d_lang = pd.read_csv(f'{PATH_RELATIVE}description_with_lang.csv')

df_s = pd.read_csv(f'{PATH_RELATIVE}subject_html_tags.csv')
df_d = pd.read_csv(f'{PATH_RELATIVE}description_html_tags.csv')

df_s = pd.merge(df_s, df_s_lang, on='id')
df_d = pd.merge(df_d, df_d_lang, on='id')

df_s.lang = df_s.subject_y
df_d.lang = df_d.description_y

df_s = df_s.rename(columns={'subject_x': 'subject'})
df_d = df_d.rename(columns={'description_x': 'description'})

df_d = df_d[df_d.lang == 'da']
df_s = df_s[df_s.id.isin(df_d.id.values)]

nlp = spacy.load('da_core_news_sm')


def lemmatize_text(text):
    if isinstance(text, str):
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc if not token.is_stop])
    return text

# Lemmatize subject and description for Danish text
df_s['subject'] = df_s.progress_apply(lambda x: lemmatize_text(x.subject), axis=1)
df_d['description'] = df_d.progress_apply(lambda x: lemmatize_text(x.description), axis=1)

# Save the lemmatized data
df_s[['id', 'subject']].to_csv(f'{PATH_RELATIVE}subject_lemmatize.csv', index=False)
df_d[['id', 'description']].to_csv(f'{PATH_RELATIVE}description_lemmatize.csv', index=False)
