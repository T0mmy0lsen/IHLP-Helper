import csv
import os
import shutil
import warnings

import pandas as pd
import config as cf
import re

from tqdm import tqdm
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
tqdm.pandas()


class Preprocess:

    def __init__(self, shared, env='staging'):

        self.shared = shared

        if env == 'dev' and self.shared.exists:
            shutil.rmtree(f'{self.shared.folder}/{self.shared.hashed}')
            self.shared.set_exists(False)

        if not self.shared.exists:
            self.dfs = [
                pd.read_csv(cf.PATH_INPUT_COMMUNICATION, nrows=self.shared.nrows),
                pd.read_csv(cf.PATH_INPUT_REQUEST, nrows=self.shared.nrows)
            ]
            for idx, df in enumerate(self.dfs):
                self.dfs[idx] = df.fillna('')
            self.run()
        else:
            print(f'There is already a preprocessing-folder for that configuration')
            print(f'Remove the folder {self.shared.folder}/{self.shared.hashed} to rebuild it')


    def run(self):

        for i, df in enumerate(self.dfs):
            for idx in self.shared.dfs_index[i]:
                self.remove_html_tags(df, idx)              # Remove HTML-tags, weird characters and lowercases text
                self.replace_match_regex(df, idx)           # Configurable

                if self.shared.remove_special_chars:
                    self.remove_special_chars(df, idx)      # Remove special chars
                self.remove_extra_spaces(df, idx)           # Remove extra spacing

        # Make a folder for the preprocessed files
        os.makedirs(f'{self.shared.folder}/{self.shared.hashed}')

        # Output the files
        for idx, df_name in enumerate(self.shared.dfs_names):
            self.dfs[idx].to_csv(f'{self.shared.folder}/{self.shared.hashed}/{df_name}.csv', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            for index in self.shared.dfs_index[idx]:
                self.dfs[idx].to_csv(f'{self.shared.folder}/{self.shared.hashed}/{df_name}_{index}.csv', quotechar='"', quoting=csv.QUOTE_NONNUMERIC, columns=[index], index=False)

        # Saved the config as JSON
        f = open(f'{self.shared.folder}/{self.shared.hashed}/config.json', 'a')
        f.write(self.shared.get_json())
        f.close()


    @staticmethod
    def remove_html_tags(df, index):

        def get_remove_html_tags(line):
            text = BeautifulSoup(line, "lxml").text
            text = text.replace(u'\u00A0', ' ')
            text = text.lower()
            return text

        df[index] = df.progress_apply(lambda x: get_remove_html_tags(x[index]), axis=1)
        return df

    def replace_match_regex(self, df, index):

        def get_replace_match(line):
            text = line
            values = list(self.shared.replace_match_regex.values())
            for i, e in enumerate(list(self.shared.replace_match_regex.keys())):
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

        def get_remove_extra_spaces(text):
            text = re.sub('\s\s+', ' ', text)
            text = text.strip()
            return text

        df[index] = df.progress_apply(lambda x: get_remove_extra_spaces(x[index]), axis=1)