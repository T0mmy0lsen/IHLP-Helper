import pandas as pd
from tqdm import tqdm

import config as cf


def preprocessing():
    import lemmy as lemmy
    from danlp.models import load_spacy_model
    from nltk import SnowballStemmer

    import config
    from model.preprocess import Preprocess
    from model.shared import SharedDict

    stemmer = SnowballStemmer('danish')
    lemmatizer = lemmy.load('da')
    nlp = load_spacy_model()

    with open(f'{config.BASE_PATH}/data/text.txt', encoding='utf-8') as f:
        lines = f.readlines()
        text = " ".join(lines)

    shared = SharedDict().default()

    text = Preprocess.get_remove_html_tags(text)
    text = Preprocess.get_lemmatize(text, nlp, lemmatizer)
    text = Preprocess.get_remove_stopwords(text, shared.stopwords)
    text = Preprocess.get_replace_match(text, shared.replace_match_regex)
    text = Preprocess.get_remove_special_chars(text)
    text = Preprocess.get_remove_extra_spaces(text)

    print(text)


def firstResponsible(df_rh=None, df_oh=None, df_it=None, id='3710927'):

    tmp_df_rh = df_rh.rename(columns={'id': 'requestHistoryId', 'tblid': 'rhTblId'})
    tmp_df_oh = df_oh.rename(columns={'id': 'objectHistoryId', 'tblid': 'ohTblId'})
    # tmp_df_it = df_it.rename(columns={'id': 'itemId'})

    tmp_df_rh = tmp_df_rh[tmp_df_rh['leftId'] == id]

    df_0 = tmp_df_rh.drop_duplicates(subset=['rightId'], keep='last')
    df_1 = pd.merge(df_0, tmp_df_oh, left_on='rhTblId', right_on='ohTblId')
    df_2 = df_1.drop_duplicates(subset=['requestHistoryId'], keep='last')
    # df_3 = pd.merge(df_2, df_it, left_on='rightId', right_on='itemId', how='left')
    # df_4 = df_3.drop_duplicates(subset=['requestHistoryId'], keep='last')

    tmp = df_2[df_2['name'].isin([
        'RequestServiceResponsible',
        'RequestIncidentResponsible',
        'RequestServiceReceivedBy',
        'RequestIncidentReceivedBy',
    ])]

    if len(tmp) == 0:
        print(df_2)


def construct_labels(df_rh, df_oh, df_it):

    label_path = f'{cf.BASE_PATH}/model/output/prepare/labels.csv'
    df_rh = df_rh.sort_values(by='tblTimeStamp')
    df_oh = df_oh[df_oh['name'].isin(
        ['RequestServiceResponsible', 'RequestIncidentResponsible', 'RequestServiceReceivedBy',
         'RequestIncidentReceivedBy']
    )]

    df_rh_tmp = df_rh.drop_duplicates(subset=['leftId'], keep='last')
    df_rh_tmp = df_rh_tmp[df_rh_tmp['leftType'].isin(['RequestService', 'RequestIncident'])]
    length_expected = len(df_rh_tmp)
    print("[Prepare] Expected length:", length_expected)

    # We expect 1/4 not having an Object with Responsible and/or ReveivedBy
    df = pd.merge(df_rh, df_oh, left_on='rhTblId', right_on='ohTblId')
    df = pd.merge(df, df_it, left_on='rightId', right_on='itemId', how='left')
    df = df.fillna('')
    df = df[df['username'] != '']
    df = df.drop_duplicates(subset=['leftId'], keep='last')
    length_actual = len(df)
    print("[Prepare] Actual length:", length_actual)

    df = df.rename(columns={'leftId': 'requestId', 'username': 'assignee'})
    df.to_csv(label_path, index=False, columns=['requestId', 'assignee'])

    print('Done')


def iterFirstResponsible(df_rh, df_oh):
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        id = row['id']
        firstResponsible(df_rh=df_rh, df_oh=df_oh, id=f"{id}")


def findMissing(df_rh, df_oh):

    df_rh = df_rh.sort_values(by='tblTimeStamp')

    df_oh_tmp = df_oh[df_oh['name'].isin([
        'RequestServiceResponsible',
        'RequestIncidentResponsible',
        'RequestServiceReceivedBy',
        'RequestIncidentReceivedBy',
    ])]

    df = df_rh[df_rh['leftType'].isin(['RequestService', 'RequestIncident'])]
    df = df.drop_duplicates(subset=['leftId'], keep='last')
    print("All unique leftId in relationHistory:", len(df))

    df = df_rh[df_rh['leftType'].isin(['RequestService', 'RequestIncident'])]
    df = pd.merge(df, df_oh_tmp, left_on='rhTblId', right_on='ohTblId', how='right')
    df_sub = df.drop_duplicates(subset=['leftId'], keep='last')
    print("Right join requestHistory on subset-objectHistory:", len(df_sub))

    df = df.drop_duplicates(subset=['leftId'], keep='last')
    print("All unique leftId in join between requestHistory and objectHistory:", len(df))

    df = df_rh[df_rh['leftType'].isin(['RequestService', 'RequestIncident'])]
    df = pd.merge(df, df_oh, left_on='rhTblId', right_on='ohTblId', how='left')
    print("Left join requestHistory on objectHistory:", len(df))

    df = df[~df['leftId'].isin(df_sub['leftId']).to_numpy()]
    print("Remove all leftId that was in the subset-join:", len(df))

    df = df[~df['leftId'].isin(df_sub['rightId']).to_numpy()]
    print("Remove all rightId dublicates:", len(df))

    df_tmp = df.drop_duplicates(subset=['leftId'], keep='last')
    print("All unique leftId after removal:", len(df_tmp))

    df_tmp.to_csv(f'{cf.BASE_PATH}/data/sandbox/missing.csv')


df = pd.read_csv(f'{cf.BASE_PATH}/data/request.csv')
df_rh = pd.read_csv(f'{cf.BASE_PATH}/data/relation_history.csv', dtype=str)
df_oh = pd.read_csv(f'{cf.BASE_PATH}/data/object_history.csv', dtype=str, low_memory=False)
df_it = pd.read_csv(f'{cf.BASE_PATH}/data/item.csv', dtype=str, low_memory=False)

df_rh = df_rh.fillna('')
df_oh = df_oh.fillna('')
df_it = df_it.fillna('')

df_rh = df_rh.rename(columns={'id': 'requestHistoryId', 'tblid': 'rhTblId'})
df_oh = df_oh.rename(columns={'id': 'objectHistoryId', 'tblid': 'ohTblId'})
df_it = df_it.rename(columns={'id': 'itemId'})

df_rh = df_rh.sort_values(by='tblTimeStamp')

# findMissing(df_rh, df_oh)
# firstResponsible(df_rh, df_oh, id='33376492')
construct_labels(df_rh, df_oh, df_it)

