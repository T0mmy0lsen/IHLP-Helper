
import os
import pandas as pd
import numpy as np

from datetime import datetime
from tqdm import tqdm

tqdm.pandas()


PATH_ITEM = 'data/input/1_data_modeling/items.csv'
PATH_REQUEST = 'data/input/1_data_modeling/request.csv'
PATH_OBJECT_HISTORY = 'data/input/1_data_modeling/object_history.csv'
PATH_RELATION_HISTORY = 'data/input/1_data_modeling/relation_history.csv'

PATH_OUTPUT_CHECKPOINT_REQUEST = 'data/output/1_data_modeling/output_request.csv'
PATH_OUTPUT_CHECKPOINT_SOLVED = 'data/output/1_data_modeling/output_solved.csv'
PATH_OUTPUT_CHECKPOINT_ROLES = 'data/output/1_data_modeling/output_roles.csv'
PATH_OUTPUT_CHECKPOINT_TIME = 'data/output/1_data_modeling/output_time.csv'

#   run()
#       checkpoint_request()
#           get_request_start_time()
#           get_request_no_newlines_and_lower()
#       checkpoint_roles()
#           get_label_roles(index='user', ...)
#           get_label_roles(index='responsible', ...)
#           get_label_roles(index='received_by', ...)


class DataModeling:

    df = None

    def __init__(self, debug=True):

        self.df_item = pd.read_csv(PATH_ITEM, sep=",", quotechar="\"", dtype=str)
        self.df_request = pd.read_csv(PATH_REQUEST, sep=",", quotechar="\"", dtype=str)
        self.df_object_history = pd.read_csv(PATH_OBJECT_HISTORY, sep=",", quotechar="\"", dtype=str)
        self.df_relation_history = pd.read_csv(PATH_RELATION_HISTORY, sep=",", quotechar="\"", dtype=str)

        print("[Modeling] Loading data completed.")

        self.df_relation_history = self.df_relation_history.rename(columns={'id': 'rhId', 'tblid': 'rhTblId'})
        self.df_object_history = self.df_object_history.rename(columns={'id': 'ohId', 'tblid': 'ohTblId'})
        self.df_request = self.df_request.rename(columns={'id': 'requestId'})
        self.df_item = self.df_item.rename(columns={'id': 'itemId'})

        self.df_request = self.df_request.fillna('')

        if debug:
            self.df_request = self.df_request.head(100)

        self.df = self.df_request
        self.run()
        
    def get_request_start_time(self):

        def year(x):
            if x['receivedDate'] != '' and x['receivedDate'][0] != '0':
                return int(x['receivedDate'][0:4])
            return 0

        def day_of_week(x):
            if x['receivedDate'] != '' and x['receivedDate'][0] != '0':
                return datetime.strptime(x['receivedDate'][:-4], '%Y-%m-%d %H:%M:%S').weekday()
            return 0

        def month(x):
            if x['receivedDate'] != '' and x['receivedDate'][0] != '0':
                return int(x['receivedDate'][5:7])
            return 0

        def hour(x):
            if x['receivedDate'] != '' and x['receivedDate'][0] != '0':
                return int(x['receivedDate'][11:13])
            return 0

        self.df['day_of_week'] = self.df.progress_apply(lambda x: day_of_week(x), axis=1)
        self.df['month'] = self.df.progress_apply(lambda x: month(x), axis=1)
        self.df['hour'] = self.df.progress_apply(lambda x: hour(x), axis=1)
        self.df['year'] = self.df.progress_apply(lambda x: year(x), axis=1)


    def get_request_no_newlines_and_lower(self):

        self.df['subject'] = self.df.progress_apply(lambda x: x['subject'].lower(), axis=1)
        self.df['subject'] = self.df.progress_apply(lambda x: x['subject'].replace('\n', ' '), axis=1)

        self.df['description'] = self.df.progress_apply(lambda x: x['description'].lower(), axis=1)
        self.df['description'] = self.df.progress_apply(lambda x: x['description'].replace('\n', ' '), axis=1)


    def get_label_roles(self, index, filter, keep='first'):

        print(f'get_label_roles for index \'{index}\'')

        df_oh = self.df_object_history[self.df_object_history['name'].isin(filter)]
        df_oh = df_oh[['ohTblId', 'name']]
        print(len(df_oh))

        df_it = self.df_item
        df_it = df_it[['itemId', 'username']]
        df_it = df_it[df_it['username'] != '']
        print(len(df_it))

        df_rh = self.df_relation_history[self.df_relation_history['leftType'].isin(['RequestService', 'RequestIncident'])]
        df_rh = df_rh[['rhTblId', 'leftId', 'rightId', 'tblTimeStamp']]
        df_rh = df_rh.sort_values(by='tblTimeStamp')
        print(len(df_rh))

        df = pd.merge(df_rh, df_oh, left_on='rhTblId', right_on='ohTblId')
        df = pd.merge(df, df_it, left_on='rightId', right_on='itemId', how='left')
        df = df.drop_duplicates(subset=['leftId'], keep=keep)
        df = df[['leftId', 'username']]

        self.df = pd.merge(self.df, df, left_on='requestId', right_on='leftId', how='left')
        self.df = self.df.drop(['leftId'], axis=1)
        self.df = self.df.rename(columns={'username': index})

    def get_last_received_time(self, x):

        if x['receivedDate'] == '':
            return 0

        if str(x['receivedDate'])[0] == '0':
            return 0

        received = datetime.strptime(str(x['receivedDate'][:-4]), "%Y-%m-%d %H:%M:%S")
        return received.timestamp()


    def get_last_communication_time(self, x):

        if x['receivedDate'] == '':
            return 0

        if str(x['receivedDate'])[0] == '0':
            return 0

        result_communication = 0

        if x['rightType'] == 'CommunicationSimple':
            received = datetime.strptime(str(x['receivedDate'][:-4]), "%Y-%m-%d %H:%M:%S")
            solution = datetime.strptime(str(x['tblTimeStamp'][:-4]), "%Y-%m-%d %H:%M:%S")
            result_communication = int(solution.timestamp()) - int(received.timestamp())
            if result_communication <= 0:
                result_communication = 0

        return result_communication

    def get_last_communication_time(self, x):

        if x['receivedDate'] == '':
            return 0

        if str(x['receivedDate'])[0] == '0':
            return 0

        result_communication = 0

        if x['rightType'] == 'CommunicationSimple':
            received = datetime.strptime(str(x['receivedDate'][:-4]), "%Y-%m-%d %H:%M:%S")
            solution = datetime.strptime(str(x['tblTimeStamp'][:-4]), "%Y-%m-%d %H:%M:%S")
            result_communication = int(solution.timestamp()) - int(received.timestamp())
            if result_communication <= 0:
                result_communication = 0

        return result_communication

    def get_solution_time(self, x):

        if x['receivedDate'] == '' or x['solutionDate'] == '':
            return 0

        if str(x['receivedDate'])[0] == '0' or str(x['solutionDate'])[0] == '0':
            return 0

        received = datetime.strptime(str(x['receivedDate'][:-4]), "%Y-%m-%d %H:%M:%S")
        solution = datetime.strptime(str(x['solutionDate'][:-4]), "%Y-%m-%d %H:%M:%S")
        result_solution = int(solution.timestamp()) - int(received.timestamp())
        if result_solution <= 0:
            result_solution = 0

        return result_solution

    def get_derived_completed_time(self, x):
        if x['solution_time'] == 0:
            return x['last_communication_time']
        if x['last_communication_time'] == 0:
            return x['solution_time']
        return np.min([x['solution_time'], x['last_communication_time']])


    def get_time_bins(self, x, max_val, bins):
        i = ((x.name + 1) * bins)
        return int(i / (max_val + 1))


    def checkpoint_time(self):

        print('checkpoint_time (4 steps)')

        df_rh = self.df_relation_history
        df_rh = df_rh.sort_values(by='tblTimeStamp')
        df_rh = df_rh[df_rh['leftType'].isin(['RequestService', 'RequestIncident'])]
        df_rh = df_rh.drop_duplicates(subset=['rightId'], keep='last')

        df = pd.merge(self.df_request, df_rh, left_on='requestId', right_on='leftId', how='right')
        df = df.fillna('')

        df_with_communication = df[df['rightType'] == 'CommunicationSimple']
        df_with_communication = df_with_communication.drop_duplicates(subset=['leftId'], keep='last')

        df_without_communication = df[~df['leftId'].isin(np.unique(df_with_communication['leftId'].to_numpy()))]
        df_without_communication = df_without_communication.drop_duplicates(subset=['leftId'], keep='last')

        df = pd.concat([df_with_communication, df_without_communication])

        df['solution_time'] = df.progress_apply(lambda x: self.get_solution_time(x), axis=1)
        df['last_communication_time'] = df.progress_apply(lambda x: self.get_last_communication_time(x), axis=1)
        df['derived_completed_time'] = df.progress_apply(lambda x: self.get_derived_completed_time(x), axis=1)
        df['received_time'] = df.progress_apply(lambda x: self.get_last_received_time(x), axis=1)

        df = df[df['derived_completed_time'] > 0]
        df = df.sort_values(by='derived_completed_time')
        df = df.reset_index()

        df['time_bins'] = df.progress_apply(lambda x: self.get_time_bins(x, len(df), 5), axis=1)

        df = df[['requestId', 'received_time', 'solution_time', 'last_communication_time', 'derived_completed_time', 'time_bins']]

        self.df = pd.merge(self.df, df, on='requestId', how='left')
        self.df.to_csv(PATH_OUTPUT_CHECKPOINT_TIME, index=False)

        print('Length for \'time\' checkpoint:', len(self.df))


    def checkpoint_request(self):

        print('checkpoint_request (7 steps)')

        self.get_request_start_time()
        self.get_request_no_newlines_and_lower()

        self.df = self.df[['requestId', 'subject', 'description', 'year', 'day_of_week', 'month', 'hour']]
        self.df.to_csv(PATH_OUTPUT_CHECKPOINT_REQUEST, index=False)

        print('Length after \'request\' checkpoint:', len(self.df))


    def checkpoint_roles(self):

        print('checkpoint_request (3 steps)')

        filter_user = [
            'RequestServiceUser',
            'RequestIncidentUser',
        ]

        filter_received_by = [
            'RequestServiceReceivedBy',
            'RequestIncidentReceivedBy'
        ]

        filter_responsible = [
            'RequestServiceResponsible',
            'RequestIncidentResponsible',
        ]

        self.get_label_roles(index='user', filter=filter_user)
        self.get_label_roles(index='responsible_first', filter=filter_responsible, keep='first')
        self.get_label_roles(index='responsible_last', filter=filter_responsible, keep='last')
        self.get_label_roles(index='received_by', filter=filter_received_by)

        self.df.to_csv(PATH_OUTPUT_CHECKPOINT_ROLES, index=False)
        print('Length after \'roles\' checkpoint:', len(self.df))


    def checkpoint_solved(self):

        df = self.df
        df = df.fillna('empty')

        df['allIsEmpty'] = df.apply(lambda x: (
            # All is empty
            (x['responsible_first'] == 'empty' and x['received_by'] == 'empty')
        ), axis=1)

        # print(df['allIsEmpty'].value_counts(normalize=True)[:10])

        df['theDispatcherSolvedIt'] = df.apply(lambda x: (
            # The dispatcher received the request and was the first and final responsible
                (x['responsible_first'] != 'empty' and x['responsible_first'] == x['received_by'] and x[
                    'responsible_last'] == x['received_by']) or
                (x['responsible_first'] == 'empty' and x['received_by'] != 'empty')
        ), axis=1)

        # print(df['theDispatcherSolvedIt'].value_counts(normalize=True)[:10])

        df['theDispatcherSolvedItButRerouted'] = df.apply(lambda x: (
            # The dispatcher received the request and was the final responsible, but not the first
            (x['responsible_first'] != 'empty' and x['responsible_first'] != x['received_by'] and x[
                'responsible_last'] == x['received_by'])
        ), axis=1)

        # print(df['theDispatcherSolvedItButRerouted'].value_counts(normalize=True)[:10])

        df['theFirstResponsibleSolvedIt'] = df.apply(lambda x: (
            # The first responsible solved it, which is not the dispatcher
                (x['responsible_first'] != 'empty' and x['responsible_first'] == x['responsible_last']) and not x[
            'theDispatcherSolvedIt'] and not x['theDispatcherSolvedItButRerouted']
        ), axis=1)

        # print(df['theFirstResponsibleSolvedIt'].value_counts(normalize=True)[:10])

        df['theFirstResponsibleDidNotSolveIt'] = df.apply(lambda x: (
            # The first responsible did not solved it
                (x['responsible_first'] != 'empty' and x['responsible_first'] != x['responsible_last']) and not x[
            'theDispatcherSolvedIt'] and not x['theDispatcherSolvedItButRerouted']
        ), axis=1)

        # print(df['theFirstResponsibleDidNotSolveIt'].value_counts(normalize=True)[:10])

        self.df.to_csv(PATH_OUTPUT_CHECKPOINT_SOLVED, index=False)
        print('Length after \'solved\' checkpoint:', len(self.df))

        """
        False    0.939438
        True     0.060562
        Name: allIsEmpty, dtype: float64
        
        True     0.624931
        False    0.375069
        Name: theDispatcherSolvedIt, dtype: float64
        
        False    0.993435
        True     0.006565
        Name: theDispatcherSolvedItButRerouted, dtype: float64
        
        False    0.751308
        True     0.248692
        Name: theFirstResponsibleSolvedIt, dtype: float64
        
        False    0.940749
        True     0.059251
        Name: theFirstResponsibleDidNotSolveIt, dtype: float64
        """




    def run(self):

        if not os.path.isfile(PATH_OUTPUT_CHECKPOINT_REQUEST):
            self.checkpoint_request()
        else:
            print(f'Skip checkpoint_request(). To rerun delete {PATH_OUTPUT_CHECKPOINT_REQUEST}')
            self.df = pd.read_csv(PATH_OUTPUT_CHECKPOINT_REQUEST, dtype=str)
            print('Length:', len(self.df))

        if not os.path.isfile(PATH_OUTPUT_CHECKPOINT_TIME):
            self.checkpoint_time()
        else:
            print(f'Skip checkpoint_roles(). To rerun delete {PATH_OUTPUT_CHECKPOINT_TIME}')
            self.df = pd.read_csv(PATH_OUTPUT_CHECKPOINT_TIME, dtype=str)
            print('Length:', len(self.df))

        if not os.path.isfile(PATH_OUTPUT_CHECKPOINT_ROLES):
            self.checkpoint_roles()
        else:
            print(f'Skip checkpoint_roles(). To rerun delete {PATH_OUTPUT_CHECKPOINT_ROLES}')
            self.df = pd.read_csv(PATH_OUTPUT_CHECKPOINT_ROLES, dtype=str)
            print('Length:', len(self.df))

        if not os.path.isfile(PATH_OUTPUT_CHECKPOINT_SOLVED):
            self.checkpoint_solved()
        else:
            print(f'Skip checkpoint_roles(). To rerun delete {PATH_OUTPUT_CHECKPOINT_SOLVED}')
            self.df = pd.read_csv(PATH_OUTPUT_CHECKPOINT_SOLVED, dtype=str)
            print('Length:', len(self.df))


DataModeling(debug=False)
