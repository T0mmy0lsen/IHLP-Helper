import hashlib
import json
import os
import re

from predict import config


class SharedDict:

    def default(self):

        return Shared(
            model=1,
            word_embedding_dim=128,
            word_embedding_epochs=50,
            nrows=None,
            stemming=False,  # Makes the tekst hard to read. I don't think it's doing what is intended.
            lemmatize=True,
            stopwords_path=f"{config.BASE_PATH}/data/input/stopwords.txt",
            replace_match_regex={
                '': [
                    re.compile(r'gmt\+\d{2}:00', flags=re.MULTILINE),
                    re.compile(r'med venlig hilsen(.|\n)*', flags=re.MULTILINE),
                ],
                '.': [
                    re.compile(r'[?!]', flags=re.MULTILINE),
                ],
                ' ': [
                    re.compile(r'\t|\n|>|<|_|\.\s+', flags=re.MULTILINE)
                ],
                'mail': [
                    re.compile(r'email|e-mail', flags=re.MULTILINE)
                ],
                '<identifier>': [
                    re.compile(r'[a-z]+-\d+'),
                    re.compile(r'\d+-[a-z]+'),
                ],
                '<link>': [
                    re.compile(r'http?[^\s]+', flags=re.MULTILINE)
                ],
                '<email>': [
                    re.compile(r'\S*@\S*\s?', flags=re.MULTILINE)
                ],
                '<datetime>': [
                    re.compile(r'(\d{2}|\d{4})(-|\.|\/)(\d{2})(-|\.|\/)(\d{2}|\d{4}) \d{2}:\d{2}', flags=re.MULTILINE)
                ],
                '<date>': [
                    re.compile(r'([^\w])(\d|\d{2})(-|\.|\/)(\d|\d{2})(-|\.|\/)(\d{4}|\d{2})([^\w])',
                               flags=re.MULTILINE),
                    re.compile(r'([^\w])([0-3][0-9])(-|\.|/)([0-1][0-9])([^\w])', flags=re.MULTILINE),
                ],
                '<time>': [
                    re.compile(r'(kl|kl\.) \d{2}\.\d{2}\s', flags=re.MULTILINE),
                    re.compile(r'\s\d{2}:\d{2}\s', flags=re.MULTILINE),
                ],
                '<phone>': [
                    re.compile(r'\s(\+\d{2})? ?\d{2} ?\d{2} ?\d{2} ?\d{2}\s', flags=re.MULTILINE)
                ],
                '<measure>': [
                    re.compile(r'([^\w])\d+(\.|-|\d)+\d+([^\w])', flags=re.MULTILINE)
                ],
                '<number>': [
                    re.compile(r'([^\w])\d+([^\w])', flags=re.MULTILINE)
                ]
            }
        )


class Shared:

    vectorizer = None
    layer = None
    exists = None
    hashed = None

    model = None
    nrows = None
    stemming = None
    lemmatize = None
    stopwords = None
    replace_match_regex = None
    word_embedding_dim = None
    word_embedding_epochs = None
    remove_special_chars = None

    x_train = None
    y_train = None
    x_validate = None
    y_validate = None
    categories = None

    folder = f'{config.BASE_PATH}/data/output/preprocessed'

    dfs_names = ['communication', 'request']
    dfs_index = [
        ['subject', 'message'],
        ['subject', 'solution', 'description']
    ]
    dfs_names_train = 'request'
    dfs_index_train = ['subject', 'description']

    def __init__(self,
                 model,
                 nrows,
                 stemming,
                 lemmatize,
                 replace_match_regex,
                 word_embedding_epochs,
                 word_embedding_dim,
                 stopwords_path=None,
                 remove_special_chars=True):
        self.model = model
        self.nrows = nrows
        self.stemming = stemming
        self.lemmatize = lemmatize
        self.replace_match_regex = replace_match_regex
        self.remove_special_chars = remove_special_chars
        self.word_embedding_dim = word_embedding_dim
        self.word_embedding_epochs = word_embedding_epochs

        if stopwords_path is not None:
            with open(f"{config.BASE_PATH}/data/input/stopwords.txt") as f:
                self.stopwords = [line.replace('\n', '') for line in f]

        # Hash the config and check if there is a folder with the same config
        self.set_hash()

    def set_vectorizer(self, vectorizer):
        self.vectorizer = vectorizer

    def set_word_embedding_layer(self, layer):
        self.layer = layer

    def set_exists(self, exists):
        self.exists = exists

    def set_hash(self):
        obj_str = f'{self.get_json()}'
        hashed = hashlib.md5(obj_str.encode()).hexdigest()
        self.hashed = f'{hashed}'.upper()[0:8]
        self.exists = os.path.isdir(f'{self.folder}/{self.hashed}')

    def get_json(self):

        def extract_regex(x):
            tmp = []
            for arrays in x.values():
                for e in arrays:
                    tmp.append(e.pattern)
            return tmp

        return json.dumps({
            'nrows': self.nrows,
            'stopwords': self.stopwords,
            'remove_special_chars': self.remove_special_chars,
            'replace_match_regex': extract_regex(self.replace_match_regex),
            'word_embedding_dim': self.word_embedding_dim,
            'word_embedding_epochs': self.word_embedding_epochs
        })