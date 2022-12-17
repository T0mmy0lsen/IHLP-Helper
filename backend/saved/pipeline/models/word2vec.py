from xml.etree.ElementTree import iterparse

import numpy as np
import pandas as pd
import os
import re

from bs4 import BeautifulSoup
from gensim.models import Word2Vec

PATH_TO_TEXT_NO_STOPWORDS = '../data/output/5_data_encode/texts_no_stopwords/output.csv'

PATH_TO_IHLP_WORD2VEC = 'word2vec/ihlp_word2vec.model'
PATH_TO_IHLP_WORD2VEC_STACKOVERFLOW = 'word2vec/ihlp_word2vec_stackoverflow.model'

PATH_TO_STACKOVERFLOW_WORD2VEC = 'word2vec/stackoverflow_word2vec.model'

VECTOR_SIZE = 300


class MakeWord2Vec:

    def __init__(self):

        if not os.path.isfile(PATH_TO_IHLP_WORD2VEC):
            self.train_ihlp()

        if not os.path.isfile(PATH_TO_IHLP_WORD2VEC_STACKOVERFLOW):
            # self.train_ihlp_stackoverflow()
            pass

        if not os.path.isfile(PATH_TO_STACKOVERFLOW_WORD2VEC):
            self.train_stackoverflow()

        pass

    def train_ihlp(self):
        arr_texts = pd.read_csv(PATH_TO_TEXT_NO_STOPWORDS).text
        arr_texts = [e.split(' ') for e in arr_texts if isinstance(e, str)]
        model = Word2Vec(sentences=arr_texts, vector_size=VECTOR_SIZE, window=5, min_count=1, workers=4, negative=5, epochs=20)
        model.save(PATH_TO_IHLP_WORD2VEC)


    def train_ihlp_stackoverflow(self):

        file_path = r"word2vec/stackoverflow/Posts.xml"
        text_list = []
        count = 0

        model = Word2Vec.load(PATH_TO_IHLP_WORD2VEC)

        for _, elem in iterparse(file_path, events=("end",)):
            if elem.tag == "row":
                text = ""
                if 'Title' in elem.attrib:
                    text = text + elem.attrib['Title'] + " "
                if 'Body' in elem.attrib:
                    text = text + elem.attrib['Body'] + " "
                if text != "":
                    x = BeautifulSoup(text, "lxml").text
                    x = re.sub(' +', ' ', re.sub(r'[^a-z]', ' ', x.lower()))
                    text_list.append(x.split(" "))
                elem.clear()
            if len(text_list) > 100000:
                count = count + 1
                print(f"Step: {count}")
                model.train(text_list, total_examples=model.corpus_count, epochs=2)
                model.save(PATH_TO_IHLP_WORD2VEC_STACKOVERFLOW)
                text_list = []


    def train_stackoverflow(self):

        file_path = r"word2vec/stackoverflow/Posts.xml"
        vocab = []
        text_list = []
        count = 0

        for _, elem in iterparse(file_path, events=("end",)):
            if elem.tag == "row":
                text = ""
                if 'Title' in elem.attrib:
                    text = text + elem.attrib['Title'] + " "
                if 'Body' in elem.attrib:
                    text = text + elem.attrib['Body'] + " "
                if text != "":
                    x = BeautifulSoup(text, "lxml").text
                    x = re.sub(' +', ' ', re.sub(r'[^a-z]', ' ', x.lower()))
                    text_list = text_list + x.split(" ")
                elem.clear()
            if len(text_list) > 100000:
                count = count + 1
                vocab = list(np.unique(vocab + list(np.unique(text_list))))
                print(f"Step: {count}, vocabulary: {len(vocab)}")
                text_list = []

        model = Word2Vec(sentences=vocab, vector_size=VECTOR_SIZE, window=5, min_count=1, workers=4, negative=5, epochs=1)
        model.save(PATH_TO_STACKOVERFLOW_WORD2VEC)

        for _, elem in iterparse(file_path, events=("end",)):
            if elem.tag == "row":
                text = ""
                if 'Title' in elem.attrib:
                    text = text + elem.attrib['Title'] + " "
                if 'Body' in elem.attrib:
                    text = text + elem.attrib['Body'] + " "
                if text != "":
                    x = BeautifulSoup(text, "lxml").text
                    x = re.sub(' +', ' ', re.sub(r'[^a-z]', ' ', x.lower()))
                    text_list.append(x.split(" "))
                elem.clear()
            if len(text_list) > 100000:
                count = count + 1
                print(f"Step: {count}")
                model.train(text_list, total_examples=model.corpus_count, epochs=5)
                model.save(PATH_TO_STACKOVERFLOW_WORD2VEC)
                text_list = []





MakeWord2Vec()