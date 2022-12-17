


import pandas as pd
import os


PATH_FROM_TRANSLATE_DESCRIPTION = 'data/output/3_data_translate/da_en_description'
PATH_FROM_TRANSLATE_SUBJECT = 'data/output/3_data_translate/da_en_subject'

# This is the last file for step 1.
# Length: 302832
PATH_FROM_MODELING = 'data/output/1_data_modeling/output_roles.csv'

PATH_FROM_MERGED_DESCRIPTION = 'data/output/4_data_merged/output_description_merged.csv'
PATH_FROM_MERGED_SUBJECT = 'data/output/4_data_merged/output_subject_merged.csv'
PATH_FROM_MERGED = 'data/output/4_data_merged/output_merged.csv'


class DataMerged:

    def __init__(self):

        self.df_description = pd.read_csv(PATH_FROM_TRANSLATE_DESCRIPTION + '/1000_output_en.csv', sep=",", quotechar="\"", dtype=str)
        self.df_subject = pd.read_csv(PATH_FROM_TRANSLATE_SUBJECT + '/1000_output_en.csv', sep=",", quotechar="\"", dtype=str)
        self.df_description_merge = pd.DataFrame([], columns=self.df_description.columns)
        self.df_subject_merge = pd.DataFrame([], columns=self.df_subject.columns)
        self.df = pd.read_csv(PATH_FROM_MODELING, sep=",", quotechar="\"", dtype=str, usecols=[
            'requestId', 'year', 'day_of_week', 'month', 'hour', 'solution_time', 'last_communication_time', 'derived_completed_time', 'time_bins', 'user', 'responsible_first', 'responsible_last', 'received_by'
        ])

        self.run()


    def run(self):

        if not os.path.isfile(PATH_FROM_MERGED_SUBJECT):
            self.checkpoint_merged_subject()
        else:
            self.df_subject = pd.read_csv(PATH_FROM_MERGED_SUBJECT, sep=",", quotechar="\"", dtype=str)
            print(f'Skip checkpoint_merged(). To rerun delete {PATH_FROM_MERGED_SUBJECT}')

        if not os.path.isfile(PATH_FROM_MERGED_DESCRIPTION):
            self.checkpoint_merged_description()
        else:
            self.df_description = pd.read_csv(PATH_FROM_MERGED_DESCRIPTION, sep=",", quotechar="\"", dtype=str)
            print(f'Skip checkpoint_merged(). To rerun delete {PATH_FROM_MERGED_DESCRIPTION}')

        if not os.path.isfile(PATH_FROM_MERGED):
            self.checkpoint_merged()
        else:
            self.df = pd.read_csv(PATH_FROM_MERGED, sep=",", quotechar="\"", dtype=str)
            print(f'Skip checkpoint_merged(). To rerun delete {PATH_FROM_MERGED}')


    def checkpoint_merged(self):
        self.df = self.df.drop_duplicates(subset=['requestId'])
        print('Merging: {}'.format(len(self.df)))
        self.df = pd.merge(self.df, self.df_subject, on='requestId', how='left')
        print('Merging: {}'.format(len(self.df)))
        self.df = pd.merge(self.df, self.df_description, on='requestId', how='left')
        print('Merging: {}'.format(len(self.df)))
        self.df.to_csv(PATH_FROM_MERGED, index=False)


    def checkpoint_merged_subject(self):
        files = os.listdir(PATH_FROM_TRANSLATE_SUBJECT)
        for file in files:
            df = pd.read_csv(f"{PATH_FROM_TRANSLATE_SUBJECT}/" + file)
            self.df_subject_merge = pd.concat([self.df_subject_merge, df])
        self.df_subject_merge.to_csv(PATH_FROM_MERGED_SUBJECT, index=False)


    def checkpoint_merged_description(self):
        files = os.listdir(PATH_FROM_TRANSLATE_DESCRIPTION)
        for file in files:
            df = pd.read_csv(f"{PATH_FROM_TRANSLATE_DESCRIPTION}/" + file)
            self.df_description_merge = pd.concat([self.df_description_merge, df])
        self.df_description_merge.to_csv(PATH_FROM_MERGED_DESCRIPTION, index=False)


DataMerged()