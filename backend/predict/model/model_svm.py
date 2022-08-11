import os
import pickle

import numpy as np
from sklearn import preprocessing, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, top_k_accuracy_score

import config
from model.prepare import Prepare
from model.preprocess import Preprocess
from model.shared import SharedDict


class ModelSVM:

    def __init__(self, shared=None, type='responsible'):
        if shared is None:
            self.shared = SharedDict().default()
            self.prepare()
        else:
            self.shared = shared
        self.run(type)

    def prepare(self):
        Preprocess(self.shared)
        # Prepare(self.shared).fetch(amount=86000, categorical_index=False)
        # Prepare(self.shared).fetch(amount=86000, categorical_index=False, filter=['thoje', 'tpieler', 'alib', 'ep'])
        Prepare(self.shared, type=type, label_index='time').fetch(amount=86000, categorical_index=False, lang=None)


    def run(self, type):

        path_pickle = f"{config.BASE_PATH}/data/output/pickle/{type}/{self.shared.hashed}"
        path_pickle_clf = f"{path_pickle}/tfidf.pickle"
        path_pickle_tfi = f"{path_pickle}/clf.pickle"
        path_pickle_le = f"{path_pickle}/le.pickle"

        if not os.path.isdir(path_pickle):
            os.mkdir(path_pickle)
        def get_labels(labels):
            le = preprocessing.LabelEncoder()
            le.fit(labels)
            return le

        def create_dataset(x, y, vectorizer, le):
            vectors = vectorizer.transform(x).todense()
            labels = le.transform(y)
            return vectors, labels

        def create_vectorizer():

            tfidf_vectorizer = TfidfVectorizer(
                max_df=.9,
                max_features=200000,
                min_df=.05,
                stop_words=None,
                use_idf=True,
                ngram_range=(1, 3)
            )

            # training_data = [e.split(" ") for e in self.shared.x_train]

            tfidf_vectorizer.fit_transform(self.shared.x_train)
            pickle.dump(tfidf_vectorizer, open(path_pickle_tfi, "wb"))
            return tfidf_vectorizer

        def train_svm_classifier(x_train, y_train):
            clf = svm.SVC(C=1, kernel='linear', decision_function_shape='ovo', probability=True)
            clf = clf.fit(x_train, y_train)
            pickle.dump(clf, open(path_pickle_clf, "wb"))
            return clf

        def evaluate(clf, x_validate, y_validate):
            x_validate = np.array(x_validate[:])
            y_validate = np.array(y_validate[:])
            # We may find classes in the validation set that we did not find in the training set.
            indexes = [i for i, e in enumerate(y_validate) if e not in clf.classes_]
            x_validate = np.delete(x_validate, indexes, axis=0)
            y_validate = np.delete(y_validate, indexes, axis=0)
            probs = clf.predict_proba(x_validate)
            print("[SVM] Actual Score:", top_k_accuracy_score(y_validate, probs, k=3, labels=clf.classes_))

        vectorizer = create_vectorizer()

        le = get_labels(self.shared.categories)
        pickle.dump(le, open(path_pickle_le, "wb"))

        x_train, y_train = create_dataset(self.shared.x_train, self.shared.y_train, vectorizer, le)
        x_validate, y_validate = create_dataset(self.shared.x_validate, self.shared.y_validate, vectorizer, le)

        clf = train_svm_classifier(x_train, y_train)
        evaluate(clf, x_validate, y_validate)