import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../../data/Heart.csv')
featureName = df.columns.tolist()

for i in featureName:
    print(i)
    print(df.groupby('target')[i].describe())
