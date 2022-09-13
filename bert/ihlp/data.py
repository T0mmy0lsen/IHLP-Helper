import os
import random
import shutil
import pandas as pd
from translate import Translate

from tqdm import tqdm

folders = os.listdir('translated')

df_en = pd.read_csv('request_responsible_en.csv')
df_en = df_en[df_en['label'].isin(folders)]


def clean():

    try:
        shutil.rmtree('test/')
        shutil.rmtree('train/')
    except:
        pass

    os.mkdir('test/')
    os.mkdir('train/')


def make_file(folder, d, translate):
    path = f'{folder}/{d.label}/{d.id}.txt'
    if not os.path.exists(path):
        with open(f'{folder}/{d.label}/{d.id}.txt', 'w') as f:
            text = d['text']
            text = text.lower()
            f.write(text)


def f(label):
    df_tmp = df_en[df_en['label'] == label]
    for i, d in tqdm(df_tmp.iterrows()):
        rand = random.randint(0, 6)
        if rand == 0:
            make_file('test', d)
        else:
            make_file('train', d)


def run():
    try:
        for label in folders:
            os.mkdir('test/' + label)
            os.mkdir('train/' + label)
    except:
        pass

    for label in folders:
        f(label)
    for label in folders:
        for file in os.listdir(f'translated/{label}'):
            if '.txt' in file:
                rand = random.randint(0, 6)
                if rand == 0:
                    shutil.copy(f'translated/{label}/{file}', f'test/{label}/{file}')
                else:
                    shutil.copy(f'translated/{label}/{file}', f'train/{label}/{file}')

# clean()
run()

"""
for i, d in tqdm(df_en.iterrows()):
    r = random.randint(0, 6)
    if r == 0:
        with open(f'ihlp/test/{d.label}/{d.id}.txt', 'w') as f:
            text = d['text'].lower()[:text_length]
            f.write(text)
    else:
        with open(f'ihlp/train/{d.label}/{d.id}.txt', 'w') as f:
            text = d['text'].lower()[:text_length]
            f.write(text)
"""


