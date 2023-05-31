
import pandas as pd

from joblib import load
from math import sqrt
from sklearn.metrics import top_k_accuracy_score, classification_report
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

        def evaluate(X, y, MODEL, TEXT, LABEL, do_top_k=True):

            clf = load(f'data/models/sklearn/{MODEL}{TEXT}{LABEL}.joblib')

            probs = clf.predict_proba(X.tolist())
            y_true_filtered, probs_filtered = [], []

            for yt, prob in zip(y.tolist(), probs):
                if yt in clf.classes_:
                    y_true_filtered.append(yt)
                    probs_filtered.append(prob)

            if do_top_k:
                print(f"[{MODEL}] Score:", top_k_accuracy_score(y_true_filtered, probs_filtered, k=1, labels=clf.classes_))
                print(f"[{MODEL}] Score:", top_k_accuracy_score(y_true_filtered, probs_filtered, k=3, labels=clf.classes_))
            else:
                total = 0
                for i, el in enumerate(probs):
                    input_list = list(el)
                    max_value = max(input_list)
                    index = input_list.index(max_value)
                    left = probs_filtered[i][index]
                    right = y_true_filtered[i][index]
                    total += sqrt(pow(left - right, 2))

                print(f"[{MODEL}] Error:", total / len(y_true_filtered))
                print(f"[{MODEL}] Loss:", classification_report(y_true_filtered, probs_filtered))

        TEXTS = ['_html_tags', '_raw', '_lemmatize']
        LABELS = ['_responsible', '_placement']

        for LABEL in LABELS:
            for TEXT in TEXTS:

                df_validate = pd.read_csv(f'data/cached_test{TEXT}{LABEL}.csv')
                tfidf_vectorizer = load(f'data/models/tfidf/tfidf_vectorizer{TEXT}{LABEL}.joblib')

                def vectorizer(data):
                    return pd.Series(tfidf_vectorizer.transform(data).todense().tolist())

                x_tfidf = vectorizer(df_validate.text)

                print('Loaded.')

                evaluate(x_tfidf, df_validate.label, 'DUM', TEXT, LABEL)
                evaluate(x_tfidf, df_validate.label, 'LOG', TEXT, LABEL)
                evaluate(x_tfidf, df_validate.label, 'RFC', TEXT, LABEL)
                evaluate(x_tfidf, df_validate.label, 'KNN', TEXT, LABEL)


ModelBaseline()