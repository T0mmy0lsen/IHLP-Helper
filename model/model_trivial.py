from collections import Counter

import numpy as np
from nltk.probability import FreqDist
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing


class ModelTrivial:

    def __init__(self, shared):
        self.shared = shared
        self.run()

    def run(self):

        def sort_index(lst, rev=True):
            index = range(len(lst))
            s = sorted(index, reverse=rev, key=lambda i: lst[i])
            return s

        def predict_trivial(x_train, y_train, x_validate, y_validate, le, constant=False):
            if constant:
                dummy_clf = DummyClassifier(constant=y_validate)
            else:
                dummy_clf = DummyClassifier(strategy='prior')
            dummy_clf.fit(x_train, y_train)
            y_pred = dummy_clf.predict(x_validate)
            y_pred_proba = dummy_clf.predict_proba(x_validate)
            y_pred_proba_top = []
            for r in y_pred_proba:
                y_pred_proba_top.append(sort_index(r)[:3]) # Top K = 3
            # print(classification_report(y_validate, y_pred))
            return sum([1 for i, e in enumerate(y_pred_proba_top) if y_validate[i] in e]) / len(y_validate)

        def tokenize_and_stem(sentence):
            return sentence.split(" ")

        def create_vectorizer():
            tfidf_vectorizer = TfidfVectorizer(
                max_df=.9,
                max_features=200000,
                min_df=.05,
                stop_words=None,
                use_idf=True,
                tokenizer=tokenize_and_stem,
                ngram_range=(1, 3)
            )
            tfidf_vectorizer.fit_transform(self.shared.x_train)
            return tfidf_vectorizer

        def get_labels(labels):
            le = preprocessing.LabelEncoder()
            le.fit(labels)
            return le

        def create_dataset(x, y, vectorizer, le):
            vectors = vectorizer.transform(x).todense()
            labels = le.transform(y)
            return vectors, labels


        # data_dict = get_data()
        # get_stats(data_dict)

        c = Counter(self.shared.y_train)
        most_common = [e[0] for e in c.most_common(3)]
        most_common_count = len([e for e in self.shared.y_train if e in most_common])
        score_expected_train = most_common_count / len(self.shared.y_train)

        c = Counter(self.shared.y_validate)
        most_common = [e[0] for e in c.most_common(3)]
        most_common_count = len([e for e in self.shared.y_validate if e in most_common])
        score_expected_validate = most_common_count / len(self.shared.y_validate)

        print("[Trivial] Create Vectorizer")
        tfidf_vec = create_vectorizer()
        le = get_labels(self.shared.categories)

        print("[Trivial] Create Dataset")
        x_train, y_train = create_dataset(self.shared.x_train, self.shared.y_train, tfidf_vec, le)
        x_validate, y_validate = create_dataset(self.shared.x_validate, self.shared.y_validate, tfidf_vec, le)

        print("[Trivial] Predict by Highest Occurence")
        score_actual = predict_trivial(x_train, y_train, x_validate, y_validate, le)

        print("[Trivial] Expected Score Train:", score_expected_train)
        print("[Trivial] Expected Score Validate:", score_expected_validate)
        print("[Trivial] Actual Score:", score_actual)