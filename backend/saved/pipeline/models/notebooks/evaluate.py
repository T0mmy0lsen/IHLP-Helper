from math import sqrt

import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tqdm import tqdm

tqdm.pandas()

def eval(baseline=False):

    total_e = 0

    df_i = pd.read_csv('predict/colab_21_11_1608/2311_ids.csv')
    df_t = pd.read_csv('predict/colab_21_11_1608/2311_true.csv')
    df_p = pd.read_csv('predict/colab_21_11_1608/2311_prediction.csv')

    df_i.drop(columns=df_i.columns[0], axis=1, inplace=True)
    df_t.drop(columns=df_t.columns[0], axis=1, inplace=True)
    df_p.drop(columns=df_p.columns[0], axis=1, inplace=True)

    labels = []
    # labels_i = []

    if baseline:
        for _ in df_t.iterrows():
            total_e += random.choice([1.0, 2.0, 3.0, 4.0, 5.0])
    else:
        scale = 0.075
        for i, el in tqdm(df_t.iterrows()):
            input_list = list(el.to_numpy())
            max_value = max(input_list)
            index = input_list.index(max_value)
            label_t = df_t.iloc[i][index] * scale
            label_p = df_p.iloc[i][index]
            # label_i = df_p.iloc[i][0]
            labels.append([label_t, 'true'])
            labels.append([label_p, 'predict'])
            # labels.append([label_t - label_p, 'diff'])
            # labels_i.append([label_i, label_p, label_t])
            # total_e += sqrt(pow(label_t - label_p, 2))

    df = pd.DataFrame(labels, columns=['Time', 'Group'])

    sns.histplot(data=df, x='Time', hue='Group', bins=25, stat='density', common_norm=False)
    plt.title("Density Histogram")
    plt.show()

    # df_i = pd.DataFrame(labels_i, columns=['id', 'predict', 'true'])
    # df_i.to_csv('predict/colab_21_11_1608/1951_merged.csv')

    print(total_e / len(df_t))


def eval_loss(baseline=False):

    t = pd.read_csv('predict/colab_21_11_1608/1609_true.csv')
    p = pd.read_csv('predict/colab_21_11_1608/1609_prediction.csv')

    t.drop(columns=t.columns[0], axis=1, inplace=True)
    p.drop(columns=p.columns[0], axis=1, inplace=True)

    t = t.values[:16]
    p = p.values[:16]

    if baseline:
        for i, row in enumerate(p):
            max_value = max(list(row))
            index = list(row).index(max_value)
            p[i][index] = random.choice([1.0, 2.0, 3.0, 4.0, 5.0])

    t = tf.convert_to_tensor(t)
    p = tf.convert_to_tensor(p)

    # loss_0 = tf.reduce_mean(tf.math.square(tf.math.square((p * t) - t)), axis=-1)  # -0.618
    # loss_1 = tf.reduce_mean(tf.math.square(tf.math.square(p * t) - t), axis=-1)
    # loss_2 = tf.reduce_mean(tf.math.square(p * t) - t)
    # loss_3 = tf.reduce_mean(tf.math.square(tf.math.pow((p * t) - t, 2)), axis=-1)
    # loss_4 = tf.reduce_mean(tf.math.square(tf.math.pow(p * t, 2) - t) * 100)

    e_1 = tf.math.pow(p * t, 2).numpy()
    e_2 = (t * t).numpy()

    loss_6 = tf.reduce_mean(tf.math.square(tf.math.pow((p - t) * t, 2)))
    loss_7 = tf.reduce_mean(tf.math.square(tf.math.pow((p - t) * tf.math.log(tf.math.log(t + 1) + 1), 2)), axis=-1)

    print(loss_6)
    print(loss_7)

    # loss = tf.reduce_mean(tf.abs(tf.sqrt(p * t) - t))
    # print(0.001 * loss)

    # mse = tf.reduce_mean(tf.square(p - t))
    # rmse = tf.math.sqrt(mse)
    # loss = rmse / tf.reduce_mean(tf.square(t)) - 1
    # print(0.001 * loss)

    # mae = tf.keras.losses.MeanAbsoluteError()
    # mae(y_true, y_pred).numpy()

    # loss = tf.reduce_mean(tf.math.square(tf.math.square(y_pred * y_true) - y_true), axis=-1).numpy()
    # mae_loss = tf.reduce_mean(y_pred - y_true, axis=-1).numpy()

    # print(mae(y_true, y_pred).numpy())
    # print(sum(mae_loss))
    # print(sum(loss))


def load_model_and_predict():
    """
    # model.load_weights('weights/colab_06_11_1023/model_weight.h5')

    model.fit(
        x=X,
        y=None,
        epochs=epochs,
        validation_data=V
    )

    validation = np.concatenate([y['time'] for x, y in V])
    prediction = model.predict(V)

    pd.DataFrame(validation).to_csv('predict/colab_06_11_1023/true.csv', index_label=False)
    pd.DataFrame(prediction).to_csv('predict/colab_06_11_1023/prediction.csv', index_label=False)
    """


def eval_labels():

    import matplotlib.pyplot as plt

    arr_time = pd.read_csv('cache/arr_time.csv')
    arr_responsible = pd.read_csv('cache/arr_responsible.csv')

    counts = []

    for i in list(range(100)):
        counts.append({'index': i, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0})

    for i, e in arr_responsible.iterrows():
        responsible = e[0]
        time = arr_time.iloc[i][responsible]
        counts[responsible][int(time)] = counts[responsible][int(time)] + 1

    df = pd.DataFrame(counts)
    df[:].plot(x='index', kind='bar', stacked=True, align='center')

    plt.show()


    pass


def eval_schedule(scale=1):
    
    df_ids = pd.read_csv('predict/colab_21_11_1608/2311_ids.csv')

    df_predict = pd.read_csv('predict/colab_21_11_1608/2311_prediction.csv')
    df_predict = pd.merge(df_predict, df_ids, left_index=True, right_index=True)

    df_true = pd.read_csv('predict/colab_21_11_1608/2311_true.csv')
    df_true = pd.merge(df_true, df_ids, left_index=True, right_index=True)
    
    df_time = pd.read_csv('data/time/output_time.csv')
    df_time = df_time.sort_values(by='received_time')

    bucket_t = [0] * 100
    bucket_p = [0] * 100

    for i, el in tqdm(df_time[:10000].iterrows()):
        tmp_p = df_predict[df_predict['0_y'] == el['requestId']]
        tmp_t = df_true[df_true['0_y'] == el['requestId']]
        if len(tmp_t) > 0 and len(tmp_p) > 0:
            arr_t = tmp_t.iloc[0][1:-1].to_numpy()
            arr_p = tmp_p.iloc[0][1:-1].to_numpy()

            max_index_t = list(arr_t).index(max(list(arr_t)))
            max_index_p = list(arr_p).index(max(list(arr_p)))

            bucket_t[max_index_t] += 1
            bucket_p[max_index_p] += 1
            pass

    print(max(bucket_t))
    print(max(bucket_p))


eval(baseline=False)
# eval_schedule(0.075)
# eval_loss(baseline=False)
# eval_labels()

