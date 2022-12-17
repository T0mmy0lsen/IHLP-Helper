

import time
from math import sqrt

import numpy as np
from keras.losses import mean_squared_error, mean_absolute_error
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, top_k_accuracy_score, log_loss, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from pipeline.models.loader import Loader

DATA_PATH_X_TRAIN = 'data/0_baselines/data_x_train.csv'
DATA_PATH_Y_TRAIN = 'data/0_baselines/data_y_train.csv'
DATA_PATH_X_TEST = 'data/0_baselines/data_x_test.csv'
DATA_PATH_Y_TEST = 'data/0_baselines/data_y_test.csv'


class Config:
    def __init__(self, max_df, min_df, max_features):
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features


class ModelBaseline:

    def __init__(self):
        self.run()

    def run(self):

        def evaluate(X, y, X_v, y_v, name, clf=None, time_it=False, do_top_k=False, save_it=False):

            clf = clf.fit(X, y)

            # We may find classes in the validation set that we did not find in the training set.
            # indexes = [i for i, e in enumerate(y_v) if e not in clf.classes_]
            # x_test = np.delete(X_v, indexes, axis=0)
            # y_test = np.delete(y_v, indexes, axis=0)

            if do_top_k:
                probs = clf.predict_proba(X_v)
                print(f"[{name}] Score:", top_k_accuracy_score(y_v, probs, k=1, labels=clf.classes_))
                print(f"[{name}] Score:", top_k_accuracy_score(y_v, probs, k=3, labels=clf.classes_))
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

        x_train, y_train, x_test, y_test = Loader.get_td_idf_and_time()

        evaluate(x_train['text'], y_train['time'], x_test['text'], y_test['time'], 'dum',
                 clf=DummyRegressor(
                     strategy='constant',
                     constant=[3.0] * 100
                 ))
        # evaluate(x_train['text'], y_train['time'], x_test['text'], y_test['time'], 'log', clf=LogisticRegression(max_iter=150, C=1.0))
        # evaluate(x_train, y_train, x_test, y_test, 'rfc', clf=RandomForestClassifier(n_estimators=100))
        # evaluate(x_train, y_train, x_test, y_test, 'knn', clf=KNeighborsClassifier(n_neighbors=2))
        # evaluate(x_train, y_train, x_test, y_test, 'ada', clf=AdaBoostClassifier())
        # evaluate(x_train, y_train, x_test, y_test, 'mnb', clf=MultinomialNB())
        # evaluate(x_train, y_train, x_test, y_test, 'svm', clf=svm.SVC(probability=True, max_iter=-1))


ModelBaseline()