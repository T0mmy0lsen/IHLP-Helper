
import pandas as pd
import sys

sys.setrecursionlimit(10000)

df = pd.read_csv('results.csv')
df['use_active_machine'] = False
df['root'] = False

for index, row in df.iterrows():
    dv = index % 4
    if dv == 0:
        df.at[index, 'use_active_machine'] = True
        df.at[index, 'root'] = True
    elif dv == 1:
        df.at[index, 'use_active_machine'] = True
    elif dv == 2:
        df.at[index, 'root'] = True

df = df.sort_values(by=['horizontal_scaling', 'vertical_scaling', 'use_active_machine', 'root'])

print(df)