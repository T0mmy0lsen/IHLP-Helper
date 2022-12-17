import pandas as pd

from predict.model.prepare import Prepare
from predict.model.preprocess import Preprocess
from predict.model.shared import SharedDict


def calc_diff_in_responsible():

    base_path = 'C:/Git/ihlp-helper/backend/predict/data/output/prepare'
    index_label = 'responsible'

    df = pd.read_csv(f'{base_path}/labels_responsible.csv')
    df_first = pd.read_csv(f'{base_path}/labels_responsible_first.csv')
    top_list = df[index_label].value_counts().index.tolist()

    for i in list(range(100)):
        count = 0
        df_tmp = df[df[index_label].isin(top_list[:i])]
        for j, e in df_tmp.iterrows():
            tmp = df_first[df_first['requestId'] == e['requestId']]
            if tmp.iloc[0]['responsible_first'] == e['responsible']:
                count += 1

        print("{}: {}".format(i, (count / len(df))))

calc_diff_in_responsible()


def create_data():
    shared = SharedDict().revised_no_stem()
    Preprocess(shared)
    Prepare(
        shared,
        category_type='responsible'
    ).fetch(
        top=100,
    )
