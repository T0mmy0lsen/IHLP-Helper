import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import numpy as np

from matplotlib import MatplotlibDeprecationWarning

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

cols = [
    'cross_validate', 'algorithm', 'horizontal_scaling', 'vertical_scaling', 'queue_for_awaiting_release',
    'queue_size', 'idle_time', 'completion_time', 'reaction_time', 'deadline_time', 'machines_hours_active',
    'jobs_extra_active', 'model'
]

df = pd.read_csv('results_validate_0.csv')
df.columns = cols

# Define the columns to filter by
columns = ['horizontal_scaling', 'vertical_scaling', 'machines_hours_active', 'jobs_extra_active', 'model']

# Get unique combinations of the specified columns
unique_combinations = df[columns].drop_duplicates()

# Define the columns to average
average_columns = ['queue_size', 'completion_time', 'reaction_time', 'deadline_time']


def dataframe_to_image(df, title):
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.axis('tight')
    ax.axis('off')

    # Determine the top two scores for each model and set the colors accordingly
    cell_colors = []
    for _, row_data in df.iterrows():
        row_colors = ['white']  # Add a neutral color for the 'algorithm' column
        for model in df.columns[1:]:  # Skip the 'algorithm' column
            model_scores = df[model].values
            top_scores = np.sort(model_scores)[-2:]  # Get the two highest scores
            if row_data[model] == top_scores[-1]:  # Top score
                row_colors.append('limegreen')
            elif len(top_scores) > 1 and row_data[model] == top_scores[-2]:  # Second top score
                row_colors.append('palegreen')
            else:
                row_colors.append('white')
        cell_colors.append(row_colors)

    ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', fontsize=14,
             cellColours=cell_colors)
    ax.set_title(title, fontsize=14)
    plt.savefig(f"{title}.png", dpi=300, bbox_inches='tight')
    plt.show()



def score(row, columns, data):
    points = 0
    for col in columns:
        if data[col].std() == 0:  # Skip scoring if all values are the same
            continue

        rank = data[col].rank(method='min')
        if rank[row.name] == 1:
            if col in ['queue_size', 'completion_time']:
                points += 3
            else:
                points += 2
        elif rank[row.name] == 2:
            if col in ['queue_size', 'completion_time']:
                points += 2
            else:
                points += 1
    return points


all_mean_data = []

for index, row in unique_combinations.iterrows():
    title = f"Combination {row['horizontal_scaling']} {row['vertical_scaling']} {row['machines_hours_active']} {row['jobs_extra_active']}"
    filtered_data = df[(df[columns] == row).all(axis=1)]
    mean_data = filtered_data.groupby(['algorithm', 'model'])[average_columns].mean().reset_index()
    mean_data = mean_data.round(1)
    mean_data.columns = ['algorithm', 'model'] + [f"{col}" for col in mean_data.columns if col not in ['algorithm', 'model']]
    mean_data['score'] = mean_data.apply(lambda x: score(x, average_columns, mean_data), axis=1)
    all_mean_data.append(mean_data)
    # dataframe_to_image(mean_data, title)

total_scores = pd.concat(all_mean_data, axis=0).groupby(['algorithm', 'model'])['score'].sum().reset_index()

# Reshape the DataFrame
pivot_total_scores = total_scores.pivot_table(index='algorithm', columns='model', values='score', fill_value=0)

# Convert index back to column
pivot_total_scores.reset_index(inplace=True)

dataframe_to_image(pivot_total_scores, "Total Scores [algorithm, model]")

