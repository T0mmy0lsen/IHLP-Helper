import pandas as pd
import matplotlib.pyplot as plt
import warnings

from matplotlib import MatplotlibDeprecationWarning

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

df_0 = pd.read_csv('results_0.csv')
df_1 = pd.read_csv('results_1.csv')
df_2 = pd.read_csv('results_2.csv')
df_3 = pd.read_csv('results_3.csv')
df_4 = pd.read_csv('results_4.csv')

df = pd.concat([df_0, df_1, df_2, df_3, df_4])

# Define the columns to filter by
columns = ['horizontal_scaling', 'vertical_scaling', 'machines_hours_active', 'jobs_extra_active']

# Get unique combinations of the specified columns
unique_combinations = df[columns].drop_duplicates()

# Define the columns to average
average_columns = ['queue_size', 'idle_time', 'completion_time', 'reaction_time', 'deadline_time']


def dataframe_to_image(df, title):
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.axis('tight')
    ax.axis('off')

    # Determine the top two scores and set the colors accordingly
    top_scores = df['score'].nlargest(2).values
    cell_colors = []
    for _, row_data in df.iterrows():
        if row_data['score'] == top_scores[0]:
            cell_colors.append(['limegreen'] * len(df.columns))
        elif len(top_scores) > 1 and row_data['score'] == top_scores[1]:
            cell_colors.append(['palegreen'] * len(df.columns))
        else:
            cell_colors.append(['white'] * len(df.columns))

    ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', fontsize=14,
             cellColours=cell_colors)
    ax.set_title(title, fontsize=14)
    plt.savefig(f"{title}.png", dpi=300, bbox_inches='tight')
    plt.show()

def score(row, columns, data):
    points = 0
    for col in columns:
        if row[col] == data[col].min():
            if col in ['queue_size', 'completion_time', 'deadline_time']:
                points += 2
            else:
                points += 1
    return points

for index, row in unique_combinations.iterrows():
    title = f"Combination {row['horizontal_scaling']} {row['vertical_scaling']} {row['machines_hours_active']} {row['jobs_extra_active']}"
    filtered_data = df[(df[columns] == row).all(axis=1)]
    mean_data = filtered_data.groupby('algorithm')[average_columns].mean().reset_index()
    mean_data = mean_data.round(1)
    mean_data.columns = ['algorithm'] + [f"{col}" for col in mean_data.columns if col != 'algorithm']
    mean_data['score'] = mean_data.apply(lambda x: score(x, average_columns, mean_data), axis=1)
    dataframe_to_image(mean_data, title)