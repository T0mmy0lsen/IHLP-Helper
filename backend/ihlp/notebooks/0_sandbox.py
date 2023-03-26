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

df = pd.read_csv('data/label_timeconsumption.csv', dtype=int)
print(df.hist(column='label_encoded', bins=11))

