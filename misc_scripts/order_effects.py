# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 00:10:51 2016

@author: ian
"""

import matplotlib.pyplot as plt

b=[]
count = 0
w = data.iloc[0].worker_id
for i,row in a.iterrows():
    if row.worker_id != w:
        w = row.worker_id
        count = 0
    b.append(count)
    count+=1
data.loc[:,'count'] = b


b = data.groupby('experiment_exp_id')['count'].mean().sort_values()
pd.Series(['survey' in i for i in b.index]).cumsum().plot()
plt.xlabel('Place in seq')
plt.ylabel('number of surveys')

t=data.groupby('experiment_exp_id').total_time.mean()/60
plt.plot([t.loc[i] for i in b.index])

