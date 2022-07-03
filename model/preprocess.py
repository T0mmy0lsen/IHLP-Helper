import csv
import hashlib
import json
import os
import shutil
import warnings

import pandas as pd
import config as cf
import re

from tqdm import tqdm
from bs4 import BeautifulSoup

folder = f'{cf.BASE_PATH}/model/output/preprocessed'
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
tqdm.pandas()


class Preprocess:

    def __init__(self, config, env='dev'):

        self.config = config
        # Hash the config and check if there is a folder with the same config
        self.hashed, exists = self.get_hash()

        if env == 'dev' and exists:
            shutil.rmtree(f'{folder}/{self.hashed}')
            exists = False

        if not exists:
            # Load in the original files
            self.df_communication = pd.read_csv(cf.PATH_INPUT_COMMUNICATION, nrows=1000)
            self.df_request_original = pd.read_csv(cf.PATH_INPUT_REQUEST, nrows=1000)
            self.df_request = pd.read_csv(cf.PATH_INPUT_REQUEST, nrows=1000)
            # Avoid problems with NaN-values
            self.df_communication = self.df_communication.fillna('')
            self.df_request = self.df_request.fillna('')
            # First array is the data, second array shows which indexes should be preprocessed.
            self.run(
                [self.df_communication, self.df_request],
                [['subject', 'message'], ['subject', 'description', 'solution']]
            )
        else:
            print(f'There is already a preprocessing-folder for that configuration')
            print(f'Remove the folder {folder}/{self.hashed} to rebuild it')

    def run(self, dfs, idxs):

        for i, df in enumerate(dfs):
            for idx in idxs[i]:
                self.remove_html_tags(df, idx)              # Remove HTML-tags, weird characters and lowercases text
                self.replace_match(df, idx)                 # Configurable
                self.replace_match_regex(df, idx)           # Configurable
                self.remove_everything_after(df, idx)       # Configurable
                if self.config.remove_special_chars:
                    self.remove_special_chars(df, idx)      # Remove special chars
                self.remove_extra_spaces(df, idx)           # Remove extra spacing

        # Make a folder for the preprocessed files
        os.makedirs(f'{folder}/{self.hashed}')

        # Output the files
        self.df_communication.to_csv(f'{folder}/{self.hashed}/communication.csv', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        self.df_request.to_csv(f'{folder}/{self.hashed}/request.csv', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

        # Output the description-field from request for inspection
        self.df_request.to_csv(f'{folder}/{self.hashed}/request_description.csv', quotechar='"', quoting=csv.QUOTE_NONNUMERIC, columns=['description'])

        # Saved the config as JSON
        f = open(f'{folder}/{self.hashed}/config.json', 'a')
        f.write(self.config.get_json())
        f.close()

    def get_hash(self):
        obj_str = f'{self.config.get_json()}'
        hashed = hashlib.md5(obj_str.encode()).hexdigest()
        hashed = f'{hashed}'.upper()[0:8]
        exists = os.path.isdir(f'{folder}/{hashed}')
        return hashed, exists

    @staticmethod
    def remove_html_tags(df, index):

        def get_remove_html_tags(line):
            text = BeautifulSoup(line, "lxml").text
            text = text.replace(u'\u00A0', ' ')
            text = text.lower()
            return text

        df[index] = df.progress_apply(lambda x: get_remove_html_tags(x[index]), axis=1)
        return df

    def remove_everything_after(self, df, index):

        # Removes everything after a specific match.
        # This is mainly for removing mail-signatures.
        def get_remove_everything_after(text):
            for e in self.config.remove_after:
                text = re.sub(f'({e}).*', ' ', text)
            return text

        df[index] = df.progress_apply(lambda x: get_remove_everything_after(x[index]), axis=1)

    def replace_match(self, df, index):

        def get_replace_match(line):
            text = line
            values = list(self.config.replace_match.values())
            for i, e in enumerate(list(self.config.replace_match.keys())):
                value = values[i]
                text = text.replace(f'{e}', f'{value}')
            return text

        df[index] = df.progress_apply(lambda x: get_replace_match(x[index]), axis=1)
        return df

    def replace_match_regex(self, df, index):

        def get_replace_match(line):
            text = line
            values = list(self.config.replace_match_regex.values())
            for i, e in enumerate(list(self.config.replace_match_regex.keys())):
                regexes = values[i]
                for regex in regexes:
                    text = re.sub(regex, f' {e} ', text)
            return text

        df[index] = df.progress_apply(lambda x: get_replace_match(x[index]), axis=1)
        return df

    @staticmethod
    def remove_special_chars(df, index):

        # This removes special chars. Note that this is run second-to-last,
        # s.t. any meaning from special characters can be extracted before this is run.
        def get_remove_special_chars(text):
            text = re.sub('[^\w\s.<>]', ' ', text)
            text = re.sub('_', ' ', text)
            return text

        df[index] = df.progress_apply(lambda x: get_remove_special_chars(x[index]), axis=1)

    @staticmethod
    def remove_extra_spaces(df, index):

        def get_remove_extra_spaces(line):
            text = re.sub('\s\s+', ' ', line)
            text = text.strip()
            return text

        df[index] = df.progress_apply(lambda x: get_remove_extra_spaces(x[index]), axis=1)