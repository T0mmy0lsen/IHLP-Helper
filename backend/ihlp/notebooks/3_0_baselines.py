
import pandas as pd

from joblib import dump
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import top_k_accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from tqdm import tqdm

tqdm.pandas()


class Config:
    def __init__(self, max_df, min_df, max_features):
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features


class ModelBaseline:

    def __init__(self):
        self.run()

    def run(self):

        def evaluate(X, y, MODEL, TEXT, LABEL, clf=None):

            clf = clf.fit(X.tolist(), y.tolist())
            dump(clf, f'data/models/sklearn/{MODEL}{TEXT}{LABEL}.joblib')

            probs = clf.predict_proba(X.tolist())
            y_true_filtered, probs_filtered = [], []

            for yt, prob in zip(y.tolist(), probs):
                if yt in clf.classes_:
                    y_true_filtered.append(yt)
                    probs_filtered.append(prob)

            print(f"[{MODEL}] Score:", top_k_accuracy_score(y_true_filtered, probs_filtered, k=1, labels=clf.classes_))


        TEXTS = ['_html_tags', '_raw', '_lemmatize']
        TEXTS = ['_html_tags']
        LABELS = ['_placement']

        for LABEL in LABELS:
            for TEXT in TEXTS:

                df_train = pd.read_csv(f'data/cached_train{TEXT}{LABEL}.csv')

                tfidf_vectorizer = TfidfVectorizer(max_df=.9, max_features=1000, min_df=0, ngram_range=(1, 5))
                tfidf_vectorizer.fit_transform(df_train.text)

                dump(tfidf_vectorizer, f'data/models/tfidf/tfidf_vectorizer{TEXT}{LABEL}.joblib')

                def vectorizer(data):
                    return pd.Series(tfidf_vectorizer.transform(data).todense().tolist())

                x_train_tfidf = vectorizer(df_train.text)

                if LABEL != '_timeconsumption':
                    evaluate(x_train_tfidf, df_train.label, 'DUM', TEXT, LABEL, clf=DummyClassifier(strategy='prior'))
                    evaluate(x_train_tfidf, df_train.label, 'LOG', TEXT, LABEL, clf=LogisticRegression(max_iter=500))
                    evaluate(x_train_tfidf, df_train.label, 'RFC', TEXT, LABEL, clf=RandomForestClassifier())
                    evaluate(x_train_tfidf, df_train.label, 'KNN', TEXT, LABEL, clf=KNeighborsClassifier())
                    # evaluate(x_train_tfidf, y_train, x_test_tfidf, y_test, 'SVM', TEXT, LABEL, clf=svm.SVC(max_iter=10, probability=True))
                else:
                    evaluate(x_train_tfidf, df_train.label, 'DUM', TEXT, LABEL, clf=DummyRegressor(strategy='mean'))
                    evaluate(x_train_tfidf, df_train.label, 'LIN', TEXT, LABEL, clf=LinearRegression())
                    evaluate(x_train_tfidf, df_train.label, 'RFR', TEXT, LABEL, clf=RandomForestRegressor(n_estimators=64))
                    evaluate(x_train_tfidf, df_train.label, 'KNN', TEXT, LABEL, clf=KNeighborsRegressor())

ModelBaseline()