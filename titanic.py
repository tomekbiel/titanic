import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset('titanic')

print(df.head())

print(df.info())

print(df.describe())

print(df.columns)

print(df['deck'])

print(df[[ 'sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alive']])
df = df.drop(columns=['survived','pclass','embarked','who','adult_male','alone'])

df.rename(columns={'class':'pclass',
                   'alive':'survived'}, inplace=True)

change = {'no':0,
          'yes':1}

df['survived_num'] = df['survived'].replace(change)

df['deck'].unique()

df['embark_town'].nunique()

df['embark_town'].value_counts()

df['age_bucket'] = df['age']//10*10
df.groupby('sex')
#Out[27]: <pandas.core.groupby.generic.DataFrameGroupBy object at 0x00000256010BD700>

df.groupby('sex').age.mean()

df.groupby('sex').age.agg([np.min, np.mean, np.max])

df.groupby('age_bucket').survived_num.mean()

df.groupby(['sex','age_bucket']).survived_num.agg([np.mean, np.size])

df.sort_values(by='age')





