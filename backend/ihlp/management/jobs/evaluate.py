
from datetime import timedelta, datetime
from django.db.models import Q
from ihlp.models import Predict, Workload
from ihlp.models_ihlp import Request

import pandas as pd
import matplotlib.pyplot as plt


class Workloads:

    def set(self, key, value):
        self.__setattr__(key, value)

    def get(self, key):
        return self.__getattribute__(key)
    

def initialDataset(
    time=datetime.strptime("2022-02-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
    limit=1,
    df=None
):
    
    if df is None:
        # TODO: Get a list of requests without predictions. Should be limited.

        # We limit the result set to be within the last n days from 'time'.
        latest = time - timedelta(days=limit)

        # The result should show all that does not have a solution and have been received after 'latest' and before 'time'.
        # Note that 'time' is used to simulate a different current time.
        queryset_requests = Request.objects.using('ihlp').filter(
            Q(receiveddate__lte=time) & Q(receiveddate__gte=latest)
        )

        # We can't write in the Request table, so we need to keep track of which has been predicted separately.
        # So we get all Request, and filter out those we already predicted.
        df = pd.DataFrame.from_records(queryset_requests.values('id', 'receiveddate', 'solutiondate', 'timeconsumption'))
    else:
        df = df[['id', 'receiveddate', 'solutiondate', 'timeconsumption']]

    if len(df) == 0:
        return False


    df_workloads = pd.DataFrame.from_records(
        Workload.objects.filter(request_id__in=list(df.id.values)).values('request_id', 'data'))
    df_predictions = pd.DataFrame.from_records(
        Predict.objects.filter(request_id__in=list(df.id.values)).values('request_id', 'data'))

    df = df.rename(columns={'id': 'request_id'})
    df_workloads = df_workloads.rename(columns={'data': 'workload_object'})
    df_predictions = df_predictions.rename(columns={'data': 'prediction_object'})

    df = pd.merge(df, df_predictions, on='request_id')
    df = pd.merge(df, df_workloads, on='request_id')

    # Prepare the workload data.
    df['true_time'] = df.apply(lambda x: float(x.timeconsumption) if float(x.timeconsumption) < 50.0 else 50.0, axis=1)
    df['true_time'] = pd.cut(df['true_time'], bins=[0.0, 2.0, 5.0, 10.0, 25.0, 50.0], labels=[1, 2, 3, 4, 5])
    df['true_time'] = df.apply(lambda x: int(x.workload_true), axis=1)

    df['predict_time_for_true_responsible'] = df.workload_true
    df['predict_time_for_predict_responsible'] = df.workload_true

    df['true_responsible'] = df.data['responsible']
    df['true_placement'] = df.data['placement']

    df['predict_responsible'] = df.data['responsible']
    df['predict_placement'] = df.data['placement']

    if len(df) == 0:
        print('The dataframe is empty. Something probably went wrong.')
        return False

    df = df.sort_values(by='receiveddate')

    # We expect the dataset to hold enough days into the past to calculate an initial workload.

    initial_date = df.iloc[0].receiveddate + timedelta(days=14)
    initial_df = df[df.receiveddate > initial_date]
    initial_df = initial_df[~initial_df.receiveddate.isnull()]
    initial_df = initial_df.reset_index(drop=True)
    initial_df['workload_predict_for_username_true'] = initial_df.apply(lambda x: int(x.workload_object['workload']) + 1, axis=1)

    # For each time-step calculate the workload, including 14 days back in time.
    # Recall that we say that the workload is constant from receiving a Request until solving it.

    users = [key for (key, value) in df.loc[0, 'prediction_object']['list'].items()]
    
    return df, initial_df, users


def evaluate(
    time=datetime.strptime("2022-02-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
    limit=1,
    df=None
):
    df_evaluation = pd.read_csv('output/2022-02-01.csv')

    user = 'it help indkøb'

    plt.plot(df_evaluation[f't_{user}'].values, label=f't_{user}')
    plt.plot(df_evaluation[f'p_{user}'].values, label=f'p_{user}')

    user = 'sd nord skranke'

    plt.plot(df_evaluation[f't_{user}'].values, label=f't_{user}')
    plt.plot(df_evaluation[f'p_{user}'].values, label=f'p_{user}')

    plt.legend()
    plt.show()



def evaluateWorkloadWithUserPredictionAndSchedule(
    time=datetime.strptime("2022-02-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
    limit=1,
    df=None
):
    df, initial_df, users = initialDataset(time, limit, df)

    columns = []
    for user in users:
        columns.append(f'p_{user}')
        columns.append(f't_{user}')

    df_evaluation = pd.DataFrame(columns=columns)

    # For initial_df I should find username and workload_true.
    # The initial_df is what should be predicted and df is the source of the prediction.

    count_user = 0
    count_time = 0

    for i, el in initial_df.iterrows():

        # Firstly we set some decision boundaries.
        # These boundaries are determined by all request that does not have a solution

        # Give me all elements that comes before and including el.received.
        # Remove all elements that has a solution before and including el.received.

        tmp = df[df.receiveddate < el.receiveddate]
        tmp = tmp[tmp.solutiondate > el.receiveddate]

        # From that we find the current workload.
        for user in users:

            tmp_for_user_true = tmp[tmp.username_true == user]
            workload_sum_true = tmp_for_user_true.workload_true.sum()
            df_evaluation.loc[i, f't_{user}'] = workload_sum_true

            tmp_for_user_predict = tmp[tmp.username_predict == user]
            workload_sum_predict = tmp_for_user_predict.workload_predict_for_username_predict.sum()
            df_evaluation.loc[i, f'p_{user}'] = workload_sum_predict

        # This is exactly like evaluateWorkload, so we might refactor this later.
        # Next, we must take a decision about whom should solve the request.

        predictions = [dict(value, **{
            'user': key,
            'predictions_high': value['predictions'][value['predictions_index'][0]]
        }) for (key, value) in el.prediction_object['list'].items()]

        predictions = sorted(predictions, key=lambda x: x['predictions_high'], reverse=True)



        # Who in the top 3 has the lowest workload?
        top_index = 0
        top_1 = predictions[0]['user']
        top_2 = predictions[1]['user']
        top_3 = predictions[2]['user']

        if df_evaluation.loc[i, f'p_{top_2}'] < df_evaluation.loc[i, f'p_{top_1}']:
            top_index = 1
        if top_index == 1 and df_evaluation.loc[i, f'p_{top_3}'] < df_evaluation.loc[i, f'p_{top_2}']:
            top_index = 2

        username_was = df[df.request_id == el.request_id].iloc[0].username_true
        username_predict = predictions[top_index]['user']

        workload_was = df[df.request_id == el.request_id].iloc[0].workload_true
        workload_predict = int(predictions[top_index]['predictions_index'][0]) + 1

        if username_was == username_predict and username_was == 'tpieler':
            count_user += 1
        if workload_was == workload_predict and username_was == 'tpieler':
            count_time += 1

        df.loc[df.request_id == el.request_id, 'username_predict'] = username_predict
        df.loc[df.request_id == el.request_id, 'workload_predict_for_username_predict'] = workload_predict

    df_evaluation.to_csv('output/2022-02-01.csv')


def evaluateWorkloadWithUserPrediction(
    time=datetime.strptime("2022-02-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
    limit=1,
    df=None
):
    df, initial_df, users = initialDataset(time, limit, df)

    columns = []
    for user in users:
        columns.append(f'p_{user}')
        columns.append(f't_{user}')

    df_evaluation = pd.DataFrame(columns=columns)

    # For initial_df I should find username and workload_true.
    # The initial_df is what should be predicted and df is the source of the prediction.

    count_user = 0
    count_time = 0

    for i, el in initial_df.iterrows():

        # Firstly we set som decision boundaries.
        # These boundaries are determined by all request that does not have a solution

        # Give me all elements that comes before and including el.received.
        # Remove all elements that has a solution before and including el.received.

        tmp = df[df.receiveddate < el.receiveddate]
        tmp = tmp[tmp.solutiondate > el.receiveddate]

        # From that we find the current workload.
        for user in users:

            tmp_for_user_true = tmp[tmp.username_true == user]
            workload_sum_true = tmp_for_user_true.workload_true.sum()
            df_evaluation.loc[i, f't_{user}'] = workload_sum_true

            tmp_for_user_predict = tmp[tmp.username_predict == user]
            workload_sum_predict = tmp_for_user_predict.workload_predict_for_username_predict.sum()
            df_evaluation.loc[i, f'p_{user}'] = workload_sum_predict

        # This is exactly like evaluateWorkload, so we might refactor this later.
        # Next, we must take a decision about whom should solve the request.

        predictions = [dict(value, **{
            'user': key,
            'predictions_high': value['predictions'][value['predictions_index'][0]]
        }) for (key, value) in el.prediction_object['list'].items()]

        predictions = sorted(predictions, key=lambda x: x['predictions_high'], reverse=True)

        username_was = df[df.request_id == el.request_id].iloc[0].username_true
        username_predict = predictions[0]['user']

        workload_was = df[df.request_id == el.request_id].iloc[0].workload_true
        workload_predict = int(predictions[0]['predictions_index'][0]) + 1

        if username_was == username_predict and username_was == 'tpieler':
            count_user += 1
        if workload_was == workload_predict and username_was == 'tpieler':
            count_time += 1

        df.loc[df.request_id == el.request_id, 'username_predict'] = username_predict
        df.loc[df.request_id == el.request_id, 'workload_predict_for_username_predict'] = workload_predict

        # print(df.head())

    print(count_user / len(initial_df[initial_df.username_true == 'tpieler']))
    print(count_time / len(initial_df[initial_df.username_true == 'tpieler']))

    for user in users[:1]:
        plt.plot(df_evaluation[f't_{user}'].values, label=f't_{user}')
        plt.plot(df_evaluation[f'p_{user}'].values, label=f'p_{user}')

    plt.legend()
    plt.show()

    print(df.head())
    df_evaluation.to_csv('output/2022-02-01.csv')
    # We let the model choose both responsible and workload.



def evaluateWorkload(
        time=datetime.strptime("2022-02-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
        limit=1,
        df=None
):
    df, initial_df, users = initialDataset(time, limit, df)

    columns = []
    for user in users:
        columns.append(f'a_{user}')
        columns.append(f't_{user}')
        columns.append(f'p_{user}')

    df_evaluation = pd.DataFrame(columns=columns)

    for i, el in initial_df.iterrows():

        # Give me all elements that comes before and including el.received.
        # Remove all elements that has a solution before and including el.received.
        tmp = df[df.receiveddate <= el.receiveddate]
        tmp = tmp[tmp.solutiondate > el.receiveddate]

        for user in users:
            tmp_for_user = tmp[tmp.username == user]
            df_evaluation.loc[i, f'a_{user}'] = len(tmp_for_user)
            df_evaluation.loc[i, f't_{user}'] = tmp_for_user.workload_true.sum()
            df_evaluation.loc[i, f'p_{user}'] = tmp_for_user.workload_predict_for_username_true.sum()

    # Add the average workload
    for i, el in df_evaluation.iterrows():
        for user in users:
            df_evaluation.loc[i, f'a_{user}'] = df_evaluation.loc[i, f'a_{user}'] * (df[df.username == user].workload_true.sum() / len(df[df.username == user]))

    for user in users[:1]:
        plt.plot(df_evaluation[f't_{user}'].values, label=f't_{user}')
        plt.plot(df_evaluation[f'p_{user}'].values, label=f'p_{user}')
        plt.plot(df_evaluation[f'a_{user}'].values, label=f'a_{user}')

    plt.legend()
    plt.show()

    return True