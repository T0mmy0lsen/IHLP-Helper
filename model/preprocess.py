import csv
import os
import shutil
import warnings
from time import sleep

import lemmy as lemmy
import pandas as pd
from nltk import SnowballStemmer

from danlp.models import load_spacy_model

import config as cf
import re

from tqdm import tqdm
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
tqdm.pandas()


class Preprocess:

    def __init__(self, shared, env='staging'):

        print("[Preprocess] Start")
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
            print(f'\tThere is already a preprocessing-folder for that configuration')
            print(f'\tRemove the folder {self.shared.folder}/{self.shared.hashed} to rebuild it')

        print("[Preprocess] End")
        print("")


    def run(self):

        for i, df in enumerate(self.dfs):
            for idx in self.shared.dfs_index[i]:

                self.remove_html_tags(df, idx)

                if self.shared.lemmatize:
                    self.lemmatize(df, idx)

                if self.shared.stemming:
                    self.stemming(df, idx)

                if self.shared.stopwords is not None:
                    self.remove_stopwords(df, idx, self.shared)

                if self.shared.replace_match_regex is not None:
                    self.replace_match_regex(df, idx, self.shared)

                if self.shared.remove_special_chars:
                    self.remove_special_chars(df, idx)

                self.remove_extra_spaces(df, idx)

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
    def get_lemmatize(line, nlp, lemmatizer):
        doc = nlp(line)
        texts = []
        skip = False
        for idx, tok in enumerate(doc):
            text = tok.lower_
            # We keep the tokens between <>
            if tok.lower_ == '<':
                skip = True
            elif tok.lower_ == '>':
                skip = False
                text = "".join([e.text for e in doc[idx - 2: idx + 1]])
            # Some tokens we keep
            elif tok.lower_ in ['jer', 'mange']:
                text = tok.lower_
            # Some tokens we alter
            elif tok.lower_[-3:] == 'rne':
                text = tok.lower_[:-1]
            # Some tokens we remove
            elif tok.lower_ in [',']:
                continue
            else:
                text = lemmatizer.lemmatize(tok.tag_, tok.lower_)[0]
            if not skip:
                texts.append(text)
        return " ".join(texts)

    @staticmethod
    def lemmatize(df, index):
        sleep(.2)
        print('\tLemmatize')
        lemmatizer = lemmy.load('da')
        nlp = load_spacy_model()
        df[index] = df.progress_apply(lambda x: Preprocess.get_lemmatize(x[index], nlp, lemmatizer), axis=1)

    @staticmethod
    def get_stemmer(line, stemmer):
        text = " ".join([stemmer.stem(e) for e in line.split(" ")])
        return text

    @staticmethod
    def stemming(df, index):
        sleep(.2)
        print('\tStemming')
        stemmer = SnowballStemmer('danish')
        df[index] = df.progress_apply(lambda x: Preprocess.get_stemmer(x[index], stemmer), axis=1)

    @staticmethod
    def get_remove_html_tags(line):
        text = BeautifulSoup(line, "lxml").text
        text = text.replace(u'\u00A0', ' ')
        return text

    @staticmethod
    def remove_html_tags(df, index):
        sleep(.2)
        print('\tRemoving HTML tags')
        df[index] = df.progress_apply(lambda x: Preprocess.get_remove_html_tags(x[index]), axis=1)
        return df

    @staticmethod
    def get_replace_match(line, regex_dict):
        text = line
        values = list(regex_dict.values())
        for i, e in enumerate(list(regex_dict.keys())):
            regexes = values[i]
            for regex in regexes:
                text = re.sub(regex, f' {e} ', text)
        return text

    @staticmethod
    def replace_match_regex(df, index, shared):
        sleep(.2)
        print('\tReplace by regex')
        df[index] = df.progress_apply(lambda x: Preprocess.get_replace_match(x[index], shared.replace_match_regex), axis=1)

    @staticmethod
    def get_remove_stopwords(line, stopwords):
        text = " ".join([e for e in line.split(" ") if e not in stopwords])
        return text

    @staticmethod
    def remove_stopwords(df, index, shared):
        sleep(.2)
        print('\tRemoving stopwords')
        df[index] = df.progress_apply(lambda x: Preprocess.get_remove_stopwords(x[index], shared.stopwords), axis=1)

    @staticmethod
    def get_remove_special_chars(text):
        text = re.sub('[^\w\s.<>]', ' ', text)
        text = re.sub('_', ' ', text)
        return text

    @staticmethod
    def remove_special_chars(df, index):
        sleep(.2)
        print('\tRemoving special charaters')
        df[index] = df.progress_apply(lambda x: Preprocess.get_remove_special_chars(x[index]), axis=1)

    @staticmethod
    def get_remove_extra_spaces(text):
        text = re.sub('\s\s+', ' ', text)
        text = text.strip()
        return text

    @staticmethod
    def remove_extra_spaces(df, index):
        sleep(.2)
        print('\tRemoving spacing')
        df[index] = df.progress_apply(lambda x: Preprocess.get_remove_extra_spaces(x[index]), axis=1)