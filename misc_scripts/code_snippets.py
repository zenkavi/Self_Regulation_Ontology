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
from fancyimpute import SoftImpute

# using for speed, for the time being
def SoftImpute_df(data):
    imputed_mat = SoftImpute(verbose=False).complete(data)
    return pd.DataFrame(data=imputed_mat, columns=data.columns, index=data.index)
    
# find best number of components
data = get_behav_data(file = 'taskdata_clean.csv', full_dataset = True)
n_components = range(1,12)
best_score = -np.Inf
best_c = 0
for c in n_components:
    print('N Components: %s' % c)
    fa=FactorAnalysis(c)
    kf = KFold(n_splits = 4)
    scores = []
    for train_index, test_index in kf.split(data.values):
        data_train, data_test = data.iloc[train_index], data.iloc[test_index]
        # replace with missForest later
        imputed_train = SoftImpute_df(data_train)
        imputed_test = SoftImpute_df(data_test)
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

# *****************************************************************************
# print top factors
# *****************************************************************************

n = 6 # number of variables to display
for i,column in result.iteritems():
    sort_index = np.argsort(abs(column))[::-1]
    top_vars = data.columns[sort_index][0:n]
    loadings = list(column[sort_index][0:n])
    print('\nFACTOR %s' % i)
    print(pd.DataFrame({'var':top_vars,'loading':loadings}, columns = ['var','loading']))

# *****************************************************************************
# visualize the similarity of the measurements in FA space
# *****************************************************************************

from data_preparation_utils import convert_var_names
from graph_utils import distcorr_mat
from sklearn import manifold
from sklearn.metrics import euclidean_distances
import seaborn as sns

seed = np.random.RandomState(seed=3)
mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=1)

tsne = manifold.TSNE(n_components=2, random_state=seed, metric="precomputed")

# compute distances between variables
# using distance correlation
# result_distances = 1-distcorr_mat(result.T.values)
# euclidean
result_distances = euclidean_distances(result)
# transform
mds_transform = mds.fit_transform(result_distances)
tsne_transform = tsne.fit_transform(result_distances)

# plot
tasks = [i.split('.')[0] for i in result.index]
colors = sns.color_palette("husl", len(np.unique(tasks)))

fig, ax = sns.plt.subplots(figsize = (20,20))
ax.scatter(mds_transform[:,0], mds_transform[:,1])

variables = convert_var_names(list(result.index))
for i, txt in enumerate(variables):
    ax.annotate(txt, (mds_transform[i,0],mds_transform[i,1]), size = 15)
