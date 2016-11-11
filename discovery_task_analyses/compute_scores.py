"""
from http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html
"""

import os,glob,sys
import numpy,pandas
import json

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

def compute_pca_cval(df,flag,n_components=None):
    X=df.values
    if n_components is None:
        n_components=range(1,int(X.shape[1]/2))
    pca = PCA(svd_solver='full')

    pca_scores = []
    for n in n_components:
        pca.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X)))
        #print(n,pca_scores[-1])
    pca.n_components = n_components[numpy.argsort(pca_scores)[-1]]
    data=pandas.DataFrame(pca.fit_transform(X),index=df.index,columns=['%s-PC%d'%(flag,i) for i in range(1,pca.n_components+1)])
    return data,pca.explained_variance_ratio_
