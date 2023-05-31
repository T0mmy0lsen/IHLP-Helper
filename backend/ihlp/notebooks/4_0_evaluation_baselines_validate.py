
import pandas as pd
import numpy as np

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

            # Get the indices of the classes that exist in y
            valid_class_indices = [i for i, class_ in enumerate(clf.classes_) if class_ in set(y)]

            # Predict probabilities
            probs = clf.predict_proba(X.tolist())

            # Only keep probabilities for valid classes
            valid_classes = clf.classes_[valid_class_indices]

            # Filter y and probs to only contain labels and corresponding probabilities present in valid_classes
            y_filtered, probs_filtered = [], []
            for label, prob in zip(y, probs):
                if label in valid_classes:
                    y_filtered.append(label)
                    probs_filtered.append(prob[valid_class_indices])

            if do_top_k:
                print(f"[{MODEL}] Score:", top_k_accuracy_score(y_filtered, probs_filtered, k=1, labels=valid_classes))
                print(f"[{MODEL}] Score:", top_k_accuracy_score(y_filtered, probs_filtered, k=3, labels=valid_classes))
            else:
                total = 0
                for i, el in enumerate(probs_filtered):
                    input_list = list(el)
                    max_value = max(input_list)
                    index = input_list.index(max_value)
                    left = el[index]
                    right = y_filtered[i]
                    total += sqrt(pow(left - right, 2))

                print(f"[{MODEL}] Error:", total / len(y_filtered))
                print(f"[{MODEL}] Loss:", classification_report(y_filtered, [valid_classes[np.argmax(prob)] for prob in probs_filtered]))

        TEXTS = ['_html_tags', '_raw']
        LABELS = ['_responsible', '_placement']

        for LABEL in LABELS:
            for TEXT in TEXTS:

                df_validate = pd.read_csv(f'data/cached{TEXT}_validate{LABEL}.csv')
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