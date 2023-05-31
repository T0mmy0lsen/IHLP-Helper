
import pandas as pd

from joblib import load
from keras.losses import mean_absolute_error
from sklearn.metrics import r2_score
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

        def evaluate(X, y, MODEL, TEXT):

            clf = load(f'data/models/sklearn/{MODEL}{TEXT}_timeconsumption.joblib')
            y_preds = clf.predict(X.tolist())

            print(f"[{MODEL}] Mean Absolute Error:", mean_absolute_error(y, y_preds))
            print(f"[{MODEL}] R^2 Score:", r2_score(y, y_preds))

        TEXTS = ['_html_tags', '_raw', '_lemmatize']

        for TEXT in TEXTS:

            df_validate = pd.read_csv(f'data/cached_test{TEXT}_timeconsumption.csv')
            tfidf_vectorizer = load(f'data/models/tfidf/tfidf_vectorizer{TEXT}_timeconsumption.joblib')

            def vectorizer(data):
                return pd.Series(tfidf_vectorizer.transform(data).todense().tolist())

            x_tfidf = vectorizer(df_validate.text)

            print('Loaded.')

            evaluate(x_tfidf, df_validate.label, 'DUM', TEXT)
            evaluate(x_tfidf, df_validate.label, 'LIN', TEXT)
            evaluate(x_tfidf, df_validate.label, 'RFR', TEXT)
            evaluate(x_tfidf, df_validate.label, 'KNN', TEXT)


ModelBaseline()