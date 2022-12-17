


import pandas as pd
import re
import os

from bs4 import BeautifulSoup

PATH_FROM_MODELING = 'data/output/1_data_modeling/output_roles.csv'

PATH_OUTPUT_CHECKPOINT_REGEX_LIGHT = 'data/output/2_data_cleaning/output_light.csv'
PATH_OUTPUT_CHECKPOINT_REGEX_HEAVY = 'data/output/2_data_cleaning/output_heavy.csv'
PATH_OUTPUT_CHECKPOINT_ONLY_WORDS = 'data/output/2_data_cleaning/output_only_words.csv'


class DataCleaning:

    def light(self, x):
        # x = x.encode('ascii', 'ignore').decode()
        x = BeautifulSoup(x, "lxml").text
        x = x.lower()
        x = x.replace(u'\u00A0', ' ')
        return x

    def heavy(self, x):
        # x = x.encode('ascii', 'ignore').decode()
        x = BeautifulSoup(x, "lxml").text
        x = x.lower()
        x = x.replace(u'\u00A0', ' ')
        x = re.sub(r'https*\S+', ' ', x)
        x = re.sub(r'http*\S+', ' ', x)
        x = re.sub(r'www*\S+', ' ', x)
        x = re.sub(r'\'\w+', '', x)
        x = re.sub(r'\w*\d+\w*', '', x)
        x = re.sub(r'\s{2,}', ' ', x)
        x = re.sub(r'\s[^\w\s]\s', '', x)
        return x


    def __init__(self, debug=True):
        self.df = pd.read_csv(PATH_FROM_MODELING, sep=",", quotechar="\"", dtype=str)
        print("[Cleaning] Loading data completed.")

        if debug:
            self.df = self.df.head(100)

        self.run()


    def checkpoint_regex_light(self):
        tmp = self.df.fillna('')
        tmp['subject'] = tmp.apply(lambda x: self.light(x['subject']), axis=1)
        tmp['description'] = tmp.apply(lambda x: self.light(x['description']), axis=1)
        tmp.to_csv(PATH_OUTPUT_CHECKPOINT_REGEX_LIGHT, index=False)


    def checkpoint_regex_heavy(self):
        tmp = self.df.fillna('')
        tmp['subject'] = tmp.apply(lambda x: self.heavy(x['subject']), axis=1)
        tmp['description'] = tmp.apply(lambda x: self.heavy(x['description']), axis=1)
        tmp.to_csv(PATH_OUTPUT_CHECKPOINT_REGEX_HEAVY, index=False)


    def run(self):

        if not os.path.isfile(PATH_OUTPUT_CHECKPOINT_REGEX_LIGHT):
            self.checkpoint_regex_light()
        else:
            print(f'Skip checkpoint_regex_light(). To rerun delete {PATH_OUTPUT_CHECKPOINT_REGEX_LIGHT}')

        if not os.path.isfile(PATH_OUTPUT_CHECKPOINT_REGEX_HEAVY):
            self.checkpoint_regex_heavy()
        else:
            print(f'Skip checkpoint_regex_heavy(). To rerun delete {PATH_OUTPUT_CHECKPOINT_REGEX_HEAVY}')



DataCleaning(debug=False)