import pandas as pd

df = pd.read_csv('data/label_responsible.csv', dtype=str)
print(df.label_responsible.describe())