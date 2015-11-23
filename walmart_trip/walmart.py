import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')

for day in ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']:
    df[day] = df.Weekday==day

df.drop('Weekday',1,inplace=True)

