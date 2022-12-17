from os import walk
import pandas as pd
import numpy as np
import tensorflow as tf

def check_if_can_load():

    # We have the raw files from the database.
    # These are extracted in Microsoft SQL Manager Studio 18, by calling a SELECT * FROM TABLE.
    # Then, on the result-set right-click and press 'save result as'

    import pandas as pd

    df = pd.read_csv('input/0_sandbox/Items.csv', encoding='UTF-8', delimiter=';', quotechar='"', dtype=str)
    df.to_csv('input/1_data_modeling/items.csv', index=False)
    print("Items:", len(df))

    df = pd.read_csv('input/0_sandbox/ObjectHistory.csv', encoding='UTF-8', delimiter=';', quotechar='"', dtype=str)
    df.to_csv('input/1_data_modeling/object_history.csv', index=False)
    print("Objects:", len(df))

    df = pd.read_csv('input/0_sandbox/RelationHistory.csv', encoding='UTF-8', delimiter=';', quotechar='"', dtype=str)
    df = df.rename(columns={'leftID': 'leftId', 'rightID': 'rightId'})
    df.to_csv('input/1_data_modeling/relation_history.csv', index=False)
    print("Relations:", len(df))

    df = pd.read_csv('input/0_sandbox/Requests.csv', encoding='UTF-8', delimiter=';', quotechar='"', dtype=str)
    df.to_csv('input/1_data_modeling/request.csv', index=False)
    print("Requests:", len(df))


def check_roles_count():

    import pandas as pd

    df = pd.read_csv('output/1_data_modeling/output_roles.csv')
    df = df.fillna('empty')

    print(df['user'].value_counts(normalize=True)[:10])
    print(df['responsible'].value_counts(normalize=True)[:10])
    print(df['received_by'].value_counts(normalize=True)[:10])


def check_if_responsible_and_received_by_is_same():

    import pandas as pd

    df = pd.read_csv('data/output/1_data_modeling/output_roles.csv')
    df = df.fillna('empty')

    df['allIsEmpty'] = df.apply(lambda x: (
        # All is empty
        (x['responsible_first'] == 'empty' and x['received_by'] == 'empty')
    ), axis=1)

    print(df['allIsEmpty'].value_counts(normalize=True)[:10])

    df['theDispatcherSolvedIt'] = df.apply(lambda x: (
        # The dispatcher received the request and was the first and final responsible
        (x['responsible_first'] != 'empty' and x['responsible_first'] == x['received_by'] and x['responsible_last'] == x['received_by']) or
        (x['responsible_first'] == 'empty' and x['received_by'] != 'empty')
    ), axis=1)

    print(df['theDispatcherSolvedIt'].value_counts(normalize=True)[:10])

    df['theDispatcherSolvedItButRerouted'] = df.apply(lambda x: (
        # The dispatcher received the request and was the final responsible, but not the first
        (x['responsible_first'] != 'empty' and x['responsible_first'] != x['received_by'] and x['responsible_last'] == x['received_by'])
    ), axis=1)

    print(df['theDispatcherSolvedItButRerouted'].value_counts(normalize=True)[:10])

    df['theFirstResponsibleSolvedIt'] = df.apply(lambda x: (
        # The first responsible solved it, which is not the dispatcher
        (x['responsible_first'] != 'empty' and x['responsible_first'] == x['responsible_last']) and not x['theDispatcherSolvedIt'] and not x['theDispatcherSolvedItButRerouted']
    ), axis=1)

    print(df['theFirstResponsibleSolvedIt'].value_counts(normalize=True)[:10])

    df['theFirstResponsibleDidNotSolveIt'] = df.apply(lambda x: (
        # The first responsible did not solved it
        (x['responsible_first'] != 'empty' and x['responsible_first'] != x['responsible_last']) and not x['theDispatcherSolvedIt'] and not x['theDispatcherSolvedItButRerouted']
    ), axis=1)

    print(df['theFirstResponsibleDidNotSolveIt'].value_counts(normalize=True)[:10])

    # 63.853% is solved by the receiver
    # 06.056% is not processed
    # 30.091% is given to other than the receiver


def inspect_subject_text():

    import pandas as pd

    df = pd.DataFrame(columns=['requestId', 'subject_translated', 'subject'])
    for (_, _, filenames) in walk('output/3_data_translate/da_en_subject'):
        df = df.concat([df, pd.read_csv('output/3_data_translate/da_en_subject/{}')])
    pass


def inspect_request_test_set():

    import pandas as pd

    PATH_REQUEST = 'input/1_data_modeling/request.csv'
    PATH_RELATION_HISTORY = 'input/1_data_modeling/relation_history.csv'

    df_request = pd.read_csv(PATH_REQUEST, sep=",", quotechar="\"", dtype=str)
    df_relation_history = pd.read_csv(PATH_RELATION_HISTORY, sep=",", quotechar="\"", dtype=str)

    df_request = df_request.sample(1000)
    df_request.to_csv('output/0_sandbox/test_requests.csv')

    ids = [str(e) for e in df_request['id'].to_numpy().tolist()]

    df_relation_history = df_relation_history[df_relation_history['leftId'].isin(ids)]
    df_relation_history.to_csv('output/0_sandbox/test_relation_history.csv')


def inspect_request():

    import pandas as pd
    import numpy as np

    PATH_REQUEST = 'output/0_sandbox/test_requests.csv'
    PATH_RELATION_HISTORY = 'output/0_sandbox/test_relation_history.csv'

    PATH_ITEM = 'input/1_data_modeling/items.csv'
    PATH_OBJECT_HISTORY = 'input/1_data_modeling/object_history.csv'

    df_item = pd.read_csv(PATH_ITEM, sep=",", quotechar="\"", dtype=str)
    df_request = pd.read_csv(PATH_REQUEST, sep=",", quotechar="\"", dtype=str)
    df_object = pd.read_csv(PATH_OBJECT_HISTORY, sep=",", quotechar="\"", dtype=str)
    df_relation = pd.read_csv(PATH_RELATION_HISTORY, sep=",", quotechar="\"", dtype=str)

    count = 0

    for i, row in df_request.iterrows():
        df_relation_tmp = df_relation[df_relation['leftId'] == row['id']]
        df = pd.merge(df_relation_tmp, df_item, left_on='rightId', right_on='id')
        df = pd.merge(df, df_object, left_on='tblid', right_on='tblid', how='left')
        df = df[df['name'].isin(['RequestServiceResponsible', 'RequestIncidentResponsible'])]
        if len(np.unique(df['username'].to_numpy())) > 1:
            count = count + 1
        print((count + 1) / (i + 1))
        

def inspect_value_counts():

    df = pd.read_csv('data/output/4_data_merged/output_merged.csv', usecols=['description', 'responsible_last'])
    print(df['responsible_last'].value_counts(normalize=True))
    print(df['responsible_last'].value_counts())


def check_label_count():

    import pandas as pd

    df = pd.read_csv('models/notebooks/data/output.csv')
    df = df.fillna('empty')

    print(df['label'].value_counts(normalize=True)[:10])


# check_if_responsible_and_received_by_is_same()
# inspect_request_test_set()
# inspect_request()  # 0.0625
# inspect_value_counts()
# check_label_count()

y_pred = tf.constant([
    [0.0, 1.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 3.0, 2.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 2.0, 2.0, 0.0],
])

y_true = tf.constant([
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 3.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2.0, 0.0],
])

y_true_ = tf.convert_to_tensor([e[tf.argmax(y_true, 1, name=None)[i]] for i, e in enumerate(tf.data.Dataset.from_tensors(y_true))])
y_pred_ = tf.convert_to_tensor([e[tf.argmax(y_true, 1, name=None)[i]] for i, e in enumerate(tf.data.Dataset.from_tensors(y_pred))])
rmse = tf.math.sqrt(tf.reduce_mean(tf.square(y_pred_ - y_true_)))
print(rmse / tf.reduce_mean(tf.square(y_true_)))

map = tf.argmax(y_true, 1, name=None)

tf.argmax(y_true, 1, name=None)
i = 0
def outer_comp(x, i):
    i += 1
    def inner_comp(y):
        return y
    return tf.map_fn(inner_comp, elems=x, dtype=tf.float32)
patches = tf.map_fn(outer_comp, elems=y_true, dtype=tf.float32)
print(patches)