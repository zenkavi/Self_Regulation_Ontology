#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 13:49:06 2017

@author: ian
"""
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import squareform
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import scale
from utils import abs_pdist, load_results

datafile = 'Complete_10-08-2017'
subset = 'survey'
# load results
results = load_results(datafile)
r_sub = results[subset]
data = r_sub.data
c = r_sub.EFA.get_metric_cs()['c_metric-BIC']
EFA_data = r_sub.EFA.get_loading(c)

EFA_data = results['survey'].EFA.get_loading(8)

# dimensionality reduction to reduce noise
scaled_data = scale(data)
pca = PCA()
data_reduced = pca.fit_transform(scaled_data.T)
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'o-')
plt.plot(pca.explained_variance_ratio_, 'o-')

# construct distance matrix
dist = squareform(abs_pdist(EFA_data))

# perform HDSBScan clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True,
                            metric='precomputed')
clusterer.fit(dist)
# get labels
labels = clusterer.labels_
probs = clusterer.probabilities_

# visualize clustering in lower dimension
def plot_labels(data, labels):
    mds = MDS(2)
    mds_coords = mds.fit_transform(data)
    # get colors
    colors = sns.palettes.hls_palette(len(np.unique(labels)))
    colors[0] = [0,0,0]
    c = [colors[i+1] for i in labels]
    plt.scatter(mds_coords[:,0], mds_coords[:,1], c=c)





label_names = [[(data.columns[i], probs[i]) 
                for i,l in enumerate(labels) if l == ii] for ii in np.unique(labels)]

clusterer.condensed_tree_.plot()

clusterer.condensed_tree_.plot(select_clusters=True,
                               selection_palette=sns.color_palette('deep', 8))


clusterer.single_linkage_tree_.plot()


# convert to pandas and scipy
from scipy.cluster.hierarchy import dendrogram
Z=clusterer.single_linkage_tree_.to_pandas()
dendrogram(Z.iloc[:,1:])
