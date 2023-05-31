import pandas as pd
import matplotlib

"""
role = 'Placement'

df = pd.read_csv('data/label_users_full.csv', dtype=str)
check = [x for x in df.columns if x[:6] == 'name_x']
check.sort()

def makeVector(x):
    vector = ''
    for el in check:
        if x[el][-len(role):] == role:
            user = x[f'username_{el[7:]}'].lower()
            if user != 'unknown':
                vector += f"{user};"
    return vector

df['vector'] = df.apply(lambda x: makeVector(x), axis=1)
df[['id', 'vector']].to_csv('data/vector.csv')
print(df.vector.describe())
"""

"""
original_min, original_max = df['label'].min(), df['label'].max()

def inverse_min_max_scaling(scaled_data, original_min, original_max, new_min=-1, new_max=1):
    return ((scaled_data - new_min) * (original_max - original_min)) / (new_max - new_min) + original_min
    
# After making predictions with your model
scaled_predictions = model.predict(...)  # Replace this with the actual prediction code

# Apply inverse min-max scaling to the predictions
original_predictions = inverse_min_max_scaling(scaled_predictions, original_min, original_max)
"""

"""
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt

df_t = pd.read_csv('data/label_timeconsumption.csv', dtype=int)
df_t = df_t.fillna(0)

df_p = pd.read_csv('data/label_placement.csv')
df_t = df_t[df_t.id.isin(df_p.id.values)]

print(len(df_t[df_t['label_encoded'] > 60]))
df_t = df_t[df_t['label_encoded'] <= 60]

print(len(df_t))

ax = df_t.hist(column='label_encoded', bins=60)
plt.savefig('histogram.png')
"""

# df = pd.read_csv('data/label_timeconsumption.csv', dtype=int)
# print(df.hist(column='label_encoded', bins=11))
