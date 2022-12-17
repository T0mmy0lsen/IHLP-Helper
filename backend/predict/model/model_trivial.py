

import time
import numpy as np
import pickle
import predict.config as config
from sklearn.dummy import DummyClassifier
from sklearn.metrics import top_k_accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier


class ModelTrivial:

    def __init__(self, shared, category_type):
        self.category_type = category_type
        self.shared = shared
        self.run()

    def run(self):

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
                ngram_range=(1, 5)
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

        def evaluate(X, y, X_v, y_v, name, clf=None):

            # Train the model
            tic = time.perf_counter()
            clf = clf.fit(X, y)
            toc = time.perf_counter()
            print(f"Done in {toc - tic:0.4f} seconds")

            # Save the model
            path_pickle = f"{config.BASE_PATH}/data/output/pickle/{self.category_type}/{self.shared.hashed}"
            path_pickle_clf = f"{path_pickle}/{name}_tfidf.pickle"
            pickle.dump(clf, open(path_pickle_clf, "wb"))

            # We may find classes in the validation set that we did not find in the training set.
            indexes = [i for i, e in enumerate(y_v) if e not in clf.classes_]
            x_validate = np.delete(X_v, indexes, axis=0)
            y_validate = np.delete(y_v, indexes, axis=0)
            probs = clf.predict_proba(x_validate)
            print(f"[{name}] Score:", top_k_accuracy_score(y_validate, probs, k=3, labels=clf.classes_))
            print(f"[{name}] Score:", top_k_accuracy_score(y_validate, probs, k=1, labels=clf.classes_))

        print("[Trivial] Create Vectorizer")
        tfidf_vec = create_vectorizer()
        le = get_labels(self.shared.categories)

        print("[Trivial] Create Dataset")
        x_train, y_train = create_dataset(self.shared.x_train, self.shared.y_train, tfidf_vec, le)
        x_validate, y_validate = create_dataset(self.shared.x_validate, self.shared.y_validate, tfidf_vec, le)

        evaluate(x_train, y_train, x_validate, y_validate, 'dum', clf=DummyClassifier(strategy='prior'))
        evaluate(x_train, y_train, x_validate, y_validate, 'log', clf=LogisticRegression())
        evaluate(x_train, y_train, x_validate, y_validate, 'xgb', clf=XGBClassifier())
        evaluate(x_train, y_train, x_validate, y_validate, 'ada', clf=AdaBoostClassifier())
        evaluate(x_train, y_train, x_validate, y_validate, 'mnb', clf=MultinomialNB())
        evaluate(x_train, y_train, x_validate, y_validate, 'svm', clf=svm.SVC(probability=True))