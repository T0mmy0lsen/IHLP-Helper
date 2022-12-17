from functools import partial

from scipy.special import softmax
from gensim.models import KeyedVectors, Word2Vec
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pipeline.models.loader import Loader

import numpy as np
import pandas as pd
import tensorflow as tf

PATH_TO_TEXT_NO_STOPWORDS = '../data/output/5_data_encode/texts_no_stopwords/output.csv'


class Validator:


    def __init__(self):

        # self.similar_words()
        # self.check_labels()
        # self.missing_words()
        self.load_and_test()
        pass

    def predict_proba(self, text_list, model, tokenizer):
        encodings = tokenizer(text_list,
                              max_length=512,
                              truncation=True,
                              padding=True)
        dataset = tf.data.Dataset.from_tensor_slices((dict(encodings)))
        preds = model.predict(dataset.batch(1)).logits
        res = tf.nn.softmax(preds, axis=1).numpy()

        return res

    def load_and_test(self):

        model = AutoModelForSequenceClassification.from_pretrained("saved/model_bert_by_example", num_labels=100, from_tf=True)
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        text = "i need to reset my password"
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        labels = pd.read_csv('../data/output/5_data_encode/label_top_100/output.csv')

        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        for i in range(scores.shape[0]):
            l = labels[labels['label_encoded'] == ranking[i]].iloc[0].label
            s = scores[ranking[i]]
            print(f"{i + 1}) {l} {np.round(float(s), 4)}")

        pass

    def missing_words(self):
        pass

    def check_labels(self):
        inputs, embedding, x_train, y_train, x_test, y_test = Loader.get_word_embedding_and_labels(nrows=10000)

    def similar_words(self):

        model = KeyedVectors.load_word2vec_format('../data/output/5_data_encode/texts_word2vec/google_news_negative_word2vec_300.bin', binary=True)
        print(model.most_similar('computer', topn=10))

        model = Word2Vec.load(
            'word2vec/ihlp_word2vec.model').wv
        print(model.most_similar('computer', topn=10))

        model = Word2Vec.load(
            'word2vec/ihlp_word2vec_stackoverflow.model').wv
        print(model.most_similar('computer', topn=10))


Validator()
