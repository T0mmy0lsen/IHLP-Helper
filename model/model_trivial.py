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

        def predict_trivial(x_train, y_train, x_validate, y_validate, le):

            dummy_clf = DummyClassifier(strategy='uniform')
            dummy_clf.fit(x_train, y_train)
            y_pred = dummy_clf.predict(x_validate)
            # print(dummy_clf.score(x_validate, y_validate))
            print("")
            print(classification_report(y_validate, y_pred))

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

        print("[Trivial] Create Vectorizer")
        tfidf_vec = create_vectorizer()
        le = get_labels(self.shared.categories)

        print("[Trivial] Create Dataset")
        x_train, y_train = create_dataset(self.shared.x_train, self.shared.y_train, tfidf_vec, le)
        x_validate, y_validate = create_dataset(self.shared.x_validate, self.shared.y_validate, tfidf_vec, le)

        print("[Trivial] Predict")
        predict_trivial(x_train, y_train, x_validate, y_validate, le)