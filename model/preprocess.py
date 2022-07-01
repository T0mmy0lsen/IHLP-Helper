import hashlib
import json
import os

import warnings
from time import sleep

import pandas as pd
import config as cf
import re

from tqdm import tqdm
from bs4 import BeautifulSoup

folder = f'{cf.BASE_PATH}/model/output/preprocessed'
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
tqdm.pandas()


class Preprocess:

    def __init__(self, config):
        self.config = config
        self.hashed, exists = self.get_hash()

        if not exists:
            self.df_communication = pd.read_csv(cf.PATH_INPUT_COMMUNICATION)
            self.df_request = pd.read_csv(cf.PATH_INPUT_REQUEST)
            self.df_communication = self.df_communication.fillna('')
            self.df_request = self.df_request.fillna('')
            self.run()
        else:
            print(f'There is already a preprocessing-folder for that configuration')
            print(f'Remove the folder {folder}/{self.hashed} to rebuild it')

    def run(self):
        if 'beautify' in self.config and self.config['beautify']:
            self.df_communication = \
                self.beautify(name='Communication', df=self.df_communication, indexes=['subject', 'message'])
            self.df_request = \
                self.beautify(name='Request', df=self.df_request, indexes=['subject', 'description'])

        os.makedirs(f'{folder}/{self.hashed}')
        self.df_communication.to_csv(f'{folder}/{self.hashed}/communication.csv')
        self.df_request.to_csv(f'{folder}/{self.hashed}/request.csv')

    def get_hash(self):
        obj_str = f'{json.dumps(self.config)}'
        hashed = hashlib.md5(obj_str.encode()).hexdigest()
        hashed = f'{hashed}'.upper()[0:8]
        exists = os.path.isdir(f'{folder}/{hashed}')
        return hashed, exists

    def beautify(self, name, df, indexes=None):
        if indexes is None:
            indexes = []

        def get_beautiful_text(line):
            text = BeautifulSoup(line, "lxml").text
            text = re.sub('[\n.]', ' ', text)
            return text

        for idx, e in enumerate(indexes):
            print(f'Beautify data from {name}, step {idx + 1}/{len(indexes)}')
            sleep(.1)
            df[e] = df.progress_apply(lambda x: get_beautiful_text(x[e]), axis=1)
        return df