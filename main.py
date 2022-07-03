# This is a fun Python script.
import json
import re

from model.preprocess import Preprocess


class Config:

    replace_match: {}
    replace_match_regex: {}
    remove_after: []
    remove_special_chars: None

    def __init__(self,
                 replace_match,
                 replace_match_regex,
                 remove_after,
                 remove_special_chars=True):
        self.replace_match = replace_match
        self.replace_match_regex = replace_match_regex
        self.remove_after = remove_after
        self.remove_special_chars = remove_special_chars


    def get_json(self):

        def extract_regex(x):
            tmp = []
            for arrays in x.values():
                for e in arrays:
                    tmp.append(e.pattern)
            return tmp

        return json.dumps({
            'remove_after': self.remove_after,
            'remove_special_chars': self.remove_special_chars,
            'replace_match': self.replace_match,
            'replace_match_regex': extract_regex(self.replace_match_regex)
        })


def preprocess():

    Preprocess(Config(
        replace_match={
            '\n': ' ',
            '\t': ' ',
            'email': 'mail',
            'e-mail': 'mail',
        },
        replace_match_regex={
            '': [
                re.compile(r'gmt\+\d{2}:00')
            ],
            '<link>': [
                re.compile(r'http?:\/\/.*[\r\n]*', flags=re.MULTILINE)
            ],
            '<email>': [
                re.compile(r'\S*@\S*\s?', flags=re.MULTILINE)
            ],
            '<datetime>': [
                re.compile(r'(\d{2}|\d{4})(-|\.|\/)(\d{2})(-|\.|\/)(\d{2}|\d{4}) \d{2}:\d{2}', flags=re.MULTILINE)
            ],
            '<date>': [
                re.compile(r'(\d|\d{2})(-|\.|\/)(\d|\d{2})(\.|-)(\d{4}|\d{2})', flags=re.MULTILINE)
            ],
            '<time>': [
                re.compile(r'(kl|kl\.) \d{2}\.\d{2}', flags=re.MULTILINE),
                re.compile(r'\d{2}:\d{2}', flags=re.MULTILINE),
            ],
            '<phone>': [
                re.compile(r'(\+\d{2})? ?\d{2} ?\d{2} ?\d{2} ?\d{2}', flags=re.MULTILINE)
            ],
            '<number>': [
                re.compile(r'\s\d+\s', flags=re.MULTILINE)
            ]
        },
        remove_after=[
            'med venlig hilsen'
        ],
        remove_special_chars=True
    ))


def run():
    preprocess()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()