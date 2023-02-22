import os
import time
import numpy as np
import pandas as pd

from joblib import dump
from ast import literal_eval
from math import sqrt
from keras.losses import mean_squared_error, mean_absolute_error
from sklearn import svm
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, top_k_accuracy_score, log_loss, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

HAS_CACHE_DATA = os.path.isfile('data/cached_train_tfidf.csv')


class Config:
    def __init__(self, max_df, min_df, max_features):
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features


class ModelBaseline:

    def __init__(self):
        self.run()

    def run(self):

        target = 'subject_label_placement'

        def evaluate(X, y, X_v, y_v, name, clf=None, time_it=False, do_top_k=False, save_it=False):

            clf = clf.fit(X.tolist(), y.tolist())
            dump(clf, f'data/models/sklearn/{target}_{name}.joblib')

            # We may find classes in the validation set that we did not find in the training set.
            indexes = [i for i, e in enumerate(y_v) if e not in clf.classes_]
            y_v = y_v.drop(index=indexes, axis=1)
            X_v = X_v.drop(index=indexes, axis=1)

            if do_top_k:
                probs = clf.predict_proba(X_v.tolist())
                print(f"[{name}] Score:", top_k_accuracy_score(y_v.tolist(), probs, k=1, labels=clf.classes_))
                print(f"[{name}] Score:", top_k_accuracy_score(y_v.tolist(), probs, k=3, labels=clf.classes_))
                # print(f"[{name}] Score:", top_k_accuracy_score(y_v.tolist(), probs, k=5, labels=clf.classes_))
            else:

                probs = clf.predict(X_v)
                total = 0

                for i, el in enumerate(probs):
                    input_list = list(el)
                    max_value = max(input_list)
                    index = input_list.index(max_value)
                    left = probs[i][index]
                    right = y_v[i][index]
                    total += sqrt(pow(left - right, 2))

                print(f"[{name}] Error:", total / len(y_v))
                print(f"[{name}] Loss:", classification_report(y_v, probs))

        df_train = pd.read_csv(f'data/cached_train_{target}.csv')
        df_test = pd.read_csv(f'data/cached_test_{target}.csv')

        # df_train = df_train[:1000]

        x_train = df_train.text
        y_train = df_train.label

        x_test = df_test.text
        y_test = df_test.label

        if not HAS_CACHE_DATA:

            tfidf_vectorizer = TfidfVectorizer(max_df=.9, max_features=1000, min_df=0, ngram_range=(1, 5))
            tfidf_vectorizer.fit_transform(x_train)
            tf_len = len(tfidf_vectorizer.vocabulary_)
            print(tf_len)

            def vectorizer(data):
                return pd.Series(tfidf_vectorizer.transform(data).todense().tolist())

            x_train_tfidf = vectorizer(x_train)
            x_test_tfidf = vectorizer(x_test)

            x_train_tfidf.to_csv('data/cached_train_tfidf.csv')
            x_test_tfidf.to_csv('data/cached_test_tfidf.csv')

        x_train_tfidf = pd.read_csv('data/cached_train_tfidf.csv')
        x_train_tfidf['0'] = x_train_tfidf.apply(lambda x: literal_eval(x['0']), axis=1)
        x_train_tfidf = x_train_tfidf['0']

        x_test_tfidf = pd.read_csv('data/cached_test_tfidf.csv')
        x_test_tfidf['0'] = x_test_tfidf.apply(lambda x: literal_eval(x['0']), axis=1)
        x_test_tfidf = x_test_tfidf['0']

        evaluate(x_train_tfidf, y_train, x_test_tfidf, y_test, 'dum', do_top_k=True, clf=DummyClassifier(strategy='prior'))
        evaluate(x_train_tfidf, y_train, x_test_tfidf, y_test, 'log', do_top_k=True, clf=LogisticRegression())
        evaluate(x_train_tfidf, y_train, x_test_tfidf, y_test, 'rfc', do_top_k=True, clf=RandomForestClassifier())
        evaluate(x_train_tfidf, y_train, x_test_tfidf, y_test, 'knn', do_top_k=True, clf=KNeighborsClassifier())
        evaluate(x_train_tfidf, y_train, x_test_tfidf, y_test, 'svm', clf=svm.SVC(probability=True))


ModelBaseline()