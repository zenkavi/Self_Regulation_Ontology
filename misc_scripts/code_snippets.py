#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 23:03:06 2016

@author: ian
"""
import numpy as np
import pandas as pd
import sys
sys.path.append('../utils')
from utils import get_behav_data
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import KFold
from r_to_py_utils import GPArotation, missForest

# find best number of components
data = get_behav_data(file = 'taskdata_clean.csv', full_dataset = True)
n_components = range(1,12)
best_score = -np.Inf
best_c = 0
for c in n_components:
    fa=FactorAnalysis(c)
    kf = KFold(n_splits = 4)
    scores = []
    for train_index, test_index in kf.split(data.values):
        data_train, data_test = data.iloc[train_index], data.iloc[test_index]
        imputed_train,e = missForest(data_train)
        imputed_test,e = missForest(data_test)
        fa.fit(imputed_train)
        scores.append(fa.score(imputed_test))
    score = np.mean(scores)
    if score>best_score:
        best_score = score
        best_c = c
print(best_c)


imputed_data = get_behav_data(dataset = 'Complete_12-15-2016', file = 'taskdata_imputed.csv')
fa=FactorAnalysis(best_c)
result = pd.DataFrame(fa.fit_transform(imputed_data.corr().values), imputed_data.columns)
result = GPArotation(result, method='oblimin')
# print top factors
n = 6 # number of variables to display
for i,column in result.iteritems():
    sort_index = np.argsort(abs(column))[::-1]
    top_vars = data.columns[sort_index][0:n]
    loadings = list(column[sort_index][0:n])
    print('\nFACTOR %s' % i)
    print(pd.DataFrame({'var':top_vars,'loading':loadings}, columns = ['var','loading']))
