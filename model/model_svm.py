from sklearn import preprocessing, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

from model.prepare import Prepare
from model.preprocess import Preprocess
from model.shared import SharedDict


class ModelSVM:

    def __init__(self, shared=None):
        if shared is None:
            self.shared = SharedDict().default()
            self.prepare()
        else:
            self.shared = shared
        self.run()

    def prepare(self):
        Preprocess(self.shared)
        # Prepare(self.shared).fetch(amount=86000, categorical_index=False)
        Prepare(self.shared).fetch(amount=86000, categorical_index=False, filter=['thoje', 'tpieler', 'alib', 'ep'])

    def run(self):

        def get_labels(labels):
            le = preprocessing.LabelEncoder()
            le.fit(labels)
            return le

        def create_dataset(x, y, vectorizer, le):
            vectors = vectorizer.transform(x).todense()
            labels = le.transform(y)
            return vectors, labels

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

        def train_svm_classifier(x_train, y_train):
            clf = svm.SVC(C=1, kernel='linear', decision_function_shape='ovo')
            clf = clf.fit(x_train, y_train)
            return clf

        def evaluate(clf, x_validate, y_validate):
            y_pred = clf.predict(x_validate)
            print(classification_report(y_validate, y_pred))

        vectorizer = create_vectorizer()
        le = get_labels(self.shared.categories)

        x_train, y_train = create_dataset(self.shared.x_train, self.shared.y_train, vectorizer, le)
        x_validate, y_validate = create_dataset(self.shared.x_validate, self.shared.y_validate, vectorizer, le)

        clf = train_svm_classifier(x_train, y_train)
        evaluate(clf, x_validate, y_validate)