import csv
import os
import shutil
import warnings
import swifter

import lemmy as lemmy
import pandas as pd
import numpy as np
from nltk import SnowballStemmer

from danlp.models import load_spacy_model

import re

from numpy import loadtxt
from tqdm import tqdm
from bs4 import BeautifulSoup

from predict import config

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
tqdm.pandas()


class Preprocess:

    dfs = None
    shared = None
    for_predict = False

    def __init__(self, shared, env='staging', for_predict=False):

        self.shared = shared
        self.for_predict = for_predict
        self.nlp = load_spacy_model()
        self.stemmer = SnowballStemmer('danish')
        self.lemmatizer = lemmy.load('da')
        self.word_list = loadtxt(f'{config.BASE_PATH}/data/input/danish-words.txt', dtype=str)

        if not self.for_predict:

            print("[Preprocess] Start")

            # If in dev-mode we delete existing data.
            if env == 'dev' and self.shared.exists:
                shutil.rmtree(f'{self.shared.folder}/{self.shared.hashed}')
                self.shared.set_exists(False)

            if not self.shared.exists:
                self.dfs = [
                    # pd.read_csv(config.PATH_INPUT_COMMUNICATION, nrows=self.shared.nrows),
                    pd.read_csv(config.PATH_INPUT_REQUEST, nrows=self.shared.nrows)
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

                if self.shared.remove_unknown_words:
                    self.remove_unknown_words(df, idx)

                self.remove_extra_spaces(df, idx)

        if not self.for_predict:
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
    def get_lemmatize(x, nlp, lemmatizer):
        doc = nlp(x)
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

    def lemmatize(self, df, index):
        df[index] = df[index].swifter.apply(Preprocess.get_lemmatize, nlp=self.nlp, lemmatizer=self.lemmatizer)
        """
                if self.for_predict:
            df[index] = df.apply(lambda x: Preprocess.get_lemmatize(x[index], self.nlp, self.lemmatizer), axis=1)
        else:
            df[index] = df.progress_apply(lambda x: Preprocess.get_lemmatize(x[index], self.nlp, self.lemmatizer), axis=1)
        """


    @staticmethod
    def get_stemmer(line, stemmer):
        text = " ".join([stemmer.stem(e) for e in line.split(" ")])
        return text

    def stemming(self, df, index):
        if self.for_predict:
            df[index] = df.apply(lambda x: Preprocess.get_stemmer(x[index], self.stemmer), axis=1)
        else:
            df[index] = df.progress_apply(lambda x: Preprocess.get_stemmer(x[index], self.stemmer), axis=1)

    @staticmethod
    def get_remove_html_tags(x):
        text = BeautifulSoup(x, "lxml").text
        text = text.replace(u'\u00A0', ' ')
        return text

    def remove_html_tags(self, df, index):
        df[index] = df[index].swifter.apply(Preprocess.get_remove_html_tags)
        """
        if self.for_predict:
            df[index] = df.apply(lambda x: Preprocess.get_remove_html_tags(x[index]), axis=1)
        else:
            df[index] = df.progress_apply(lambda x: Preprocess.get_remove_html_tags(x[index]), axis=1)

        """

    @staticmethod
    def get_replace_match(line, regex_dict):
        text = line
        values = list(regex_dict.values())
        for i, e in enumerate(list(regex_dict.keys())):
            regexes = values[i]
            for regex in regexes:
                text = re.sub(regex, f' {e} ', text)
        return text

    def replace_match_regex(self, df, index, shared):
        if self.for_predict:
            df[index] = df.apply(lambda x: Preprocess.get_replace_match(x[index], shared.replace_match_regex), axis=1)
        else:
            df[index] = df.progress_apply(lambda x: Preprocess.get_replace_match(x[index], shared.replace_match_regex), axis=1)

    @staticmethod
    def get_remove_stopwords(line, stopwords):
        text = " ".join([e for e in line.split(" ") if e not in stopwords])
        return text

    def remove_stopwords(self, df, index, shared):
        df[index] = df.swifter.apply(lambda x: Preprocess.get_remove_stopwords(x[index], shared.stopwords), axis=1)
        """
        if self.for_predict:
            df[index] = df.apply(lambda x: Preprocess.get_remove_stopwords(x[index], shared.stopwords), axis=1)
        else:
            df[index] = df.progress_apply(lambda x: Preprocess.get_remove_stopwords(x[index], shared.stopwords), axis=1)
        """

    @staticmethod
    def get_remove_special_chars(text):
        text = re.sub('[^\w\s.<>]', ' ', text)
        text = re.sub('_', ' ', text)
        return text

    def remove_special_chars(self, df, index):
        if self.for_predict:
            df[index] = df.apply(lambda x: Preprocess.get_remove_special_chars(x[index]), axis=1)
        else:
            df[index] = df.progress_apply(lambda x: Preprocess.get_remove_special_chars(x[index]), axis=1)

    @staticmethod
    def get_remove_extra_spaces(text):
        text = re.sub('\s\s+', ' ', text)
        text = text.strip()
        return text

    def remove_extra_spaces(self, df, index):
        if self.for_predict:
            df[index] = df.apply(lambda x: Preprocess.get_remove_extra_spaces(x[index]), axis=1)
        else:
            df[index] = df.progress_apply(lambda x: Preprocess.get_remove_extra_spaces(x[index]), axis=1)

    @staticmethod
    def get_remove_unknown_words(x, bad_words):
        text = " ".join([e for e in x.split(" ") if e.lower() not in bad_words])
        return text

    def remove_unknown_words(self, df, index):
        words_dictionary = [e.lower() for e in self.word_list]
        lines = df[index].to_numpy()
        words = np.concatenate([[g.lower() for g in e.split(" ")] for e in lines])
        words = np.unique(words)
        bad_words = [e for e in words if e not in words_dictionary]
        df[index] = df[index].swifter.apply(Preprocess.get_remove_unknown_words, bad_words=bad_words)
        # df[index] = df.progress_apply(lambda x: Preprocess.get_remove_unknown_words(x[index], self.word_list), axis=1)