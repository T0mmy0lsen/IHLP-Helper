from tqdm import tqdm

import numpy as np
import pandas as pd
import socket


def create_tasks():

    df = pd.read_csv('bunch_of_merged.csv', low_memory=False)
    df = df.fillna('')

    df['relationTblTimeStamp'] = pd.to_datetime(df['relationTblTimeStamp'])
    df = df.sort_values('relationTblTimeStamp')

    # Group the DataFrame by leftID
    groups = df.groupby('leftID')

    # Create an empty DataFrame to store the tasks
    tasks = []

    # Iterate through the groups
    for _, group in tqdm(groups):

        current_user = 'unset'
        current_placement = 'unset'
        current_responsible = 'unset'

        time_consumption = 0

        tasks.append({
            'id': group.iloc[0].leftID,
            'event': 'created',
            'duration': 1,
            'reaction_time': 0,
            'reaction_timestamp': group.iloc[0]['relationTblTimeStamp'],
            'current_user': '',
            'current_placement': 'created',
            'current_responsible': 'created',
            'created_by_x': '',
            'created_by_y': '',
        })

        for i, task in group.iterrows():

            event = task.event

            if 'User' in task.name_x:
                current_user = task.name_y.lower()
            if 'Placement' in task.name_x:
                current_placement = task.name_y.lower()
            if 'Responsible' in task.name_x:
                current_responsible = task.name_y.lower()
            if 'Communication' in task.name_x:
                event = 'communication'

            if task['timeConsumption'] == '':
                duration = 0
            else:
                duration = task['timeConsumption'] - time_consumption
                time_consumption = task['timeConsumption']

            reaction_time = (task['relationTblTimeStamp'] - tasks[-1]['reaction_timestamp']).total_seconds() / 60

            if reaction_time < 1.0 and i != 0 and tasks[-1]['event'] != 'created' and tasks[-1]['event'] != 'communication':
                tasks[-1] = {
                    'id': task.leftID,
                    'event': event,
                    'duration': tasks[-1]['duration'] + duration,
                    'reaction_time': tasks[-1]['reaction_time'] + reaction_time,
                    'reaction_timestamp': task['relationTblTimeStamp'],
                    'current_user': current_user,
                    'current_placement': current_placement,
                    'current_responsible': current_responsible,
                    'created_by_x': task['username_x'],
                    'created_by_y': task['username_y'],
                }
            else:
                tasks.append({
                    'id': task.leftID,
                    'event': event,
                    'duration': duration,
                    'reaction_time': reaction_time,
                    'reaction_timestamp': task['relationTblTimeStamp'],
                    'current_user': current_user,
                    'current_placement': current_placement,
                    'current_responsible': current_responsible,
                    'created_by_x': task['username_x'],
                    'created_by_y': task['username_y'],
                })

        # df_tasks = pd.DataFrame(tasks)
        # pass

    df_tasks = pd.DataFrame(tasks)
    df_tasks = df_tasks[(df_tasks.duration > 0) | (df_tasks.reaction_time > 1)]
    df_tasks.to_csv('bunch_of_tasks.csv', index=False)


def create_tasks_but_better():

    df = pd.read_csv('bunch_of_tasks.csv')
    # Group the DataFrame by leftID
    groups = df.groupby('id')

    # Create an empty DataFrame to store the tasks
    tasks = []

    # Iterate through the groups
    for _, group in tqdm(groups):

        for i, task in group.iterrows():

            task_type = 'follow-up'

            if task['event'] == 'created':
                task_type = 'created'
            else:

                if task['event'] == 'communication':
                    if task.current_user == task['created_by_y']:
                        task_type = 'communication-from-user'
                    elif task.current_responsible == task['created_by_y']:
                        task_type = 'communication-from-responsible'
                    elif task['created_by_y'] == 'svcihlp':
                        task_type = 'communication-from-mail'
                    else:
                        task_type = 'communication-unknown'


            tasks.append({
                'id': task.id,
                'event': task.event,
                'duration': task.duration,
                'reaction_time': task.reaction_time,
                'reaction_timestamp': task.reaction_timestamp,
                'current_user': task.current_user,
                'current_placement': task.current_placement,
                'current_responsible': task.current_responsible,
                'task_type': task_type,
                'created_by_x': task.created_by_x,
                'created_by_y': task.created_by_y,
            })

        # df_tasks = pd.DataFrame(tasks)
        # pass

    df_tasks = pd.DataFrame(tasks)
    df_tasks.to_csv('bunch_of_tasks_but_better.csv', index=False)

    pass


create_tasks()
create_tasks_but_better()
