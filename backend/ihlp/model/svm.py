import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import top_k_accuracy_score

from model.preprocess import Preprocess
from model.shared import SharedDict


class SVM:

    clf = None
    vec = None
    le = None

    type = None
    shared = None
    preprocess = None

    def __init__(self):
        pass

    def load(self, type='responsible', base_path='C:\\Users\\tool\\git\\ihlp-helper\\'):

        self.type = type
        self.shared = SharedDict().default()
        self.shared.dfs_index = [['text']]
        self.preprocess = Preprocess(self.shared, for_predict=True)

        path_pickle = f"{base_path}/model/output/pickle/{self.type}/EEF986E6"
        path_pickle_clf = f"{path_pickle}/tfidf.pickle"
        path_pickle_tfi = f"{path_pickle}/clf.pickle"
        path_pickle_le = f"{path_pickle}/le.pickle"

        with open(path_pickle_clf, 'rb') as pickle_file:
            self.clf = pickle.load(pickle_file)

        with open(path_pickle_tfi, 'rb') as pickle_file:
            self.vec = pickle.load(pickle_file)

        with open(path_pickle_le, 'rb') as pickle_file:
            self.le = pickle.load(pickle_file)


    def evaluate(self):

        Prepare(self.shared, type=self.type, label_index=self.type).fetch(amount=86000, categorical_index=False, lang=None)

        x_validate = self.vec.transform(self.shared.x_validate).todense()
        y_validate = self.le.transform(self.shared.y_validate)

        # We may find classes in the validation set that we did not find in the training set.
        indexes = [i for i, e in enumerate(y_validate) if e not in self.clf.classes_]
        x_validate = np.delete(x_validate, indexes, axis=0)
        y_validate = np.delete(y_validate, indexes, axis=0)

        probs = self.clf.predict_proba(x_validate)
        print("Score:", top_k_accuracy_score(y_validate, probs, k=1, labels=self.clf.classes_))


    def predict(self, text):

        start = time.time()

        self.preprocess.dfs = [pd.DataFrame([text], columns=['text'])]
        self.preprocess.run()

        vector = self.vec.transform(self.preprocess.dfs[0]['text'].to_numpy()).todense()
        probs = self.clf.predict_proba(vector)
        top_3 = np.argpartition(probs[0], -3)[-3:]
        labels = [str(e).lower() for e in self.le.classes_[top_3]]

        end = time.time()
        print(end - start)

        return labels
