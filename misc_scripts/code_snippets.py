#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 23:03:06 2016

@author: ian
"""
import numpy as np
import pandas as pd
import sys
sys.path.append('../../utils')
from utils import get_behav_data
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import cross_val_score

data = get_behav_data(file = 'taskdata_imputed.csv')
n_components = range(1,12)
best_score = -np.Inf
best_c = 0
for c in n_components:
    fa=FactorAnalysis(c)
    score = np.mean(cross_val_score(fa,data))
    if score>best_score:
        best_score = score
        best_c = c
print(best_c)
fa=FactorAnalysis(best_c)
result = fa.fit_transform(data.corr())

# print top factors
n = 6 # number of variables to display
for i,column in enumerate(result.T):
    sort_index = np.argsort(abs(column))[::-1]
    top_vars = data.columns[sort_index][0:n]
    loadings = column[sort_index][0:n]
    print('\nFACTOR %s' % i)
    print(pd.DataFrame({'var':top_vars,'loading':loadings}, columns = ['var','loading']))
