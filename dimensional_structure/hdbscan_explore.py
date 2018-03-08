#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 13:49:06 2017

@author: ian
"""
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.preprocessing import scale
from dimensional_structure.utils import abs_pdist, load_results, hierarchical_cluster
from selfregulation.utils.plot_utils import dendroheatmap

def plot_labels(data, labels, probs=None, labels_on=True, random_state=None,
                ax=None):
    basesize=80
    sizes=basesize*2
    c='b'
    N_labels = len(np.unique([i for i in labels if i>-1]))
    
    mds = MDS(2, random_state=random_state)
    mds_coords = mds.fit_transform(data)
    # get colors
    if labels_on:
        colors = sns.palettes.hls_palette(N_labels+1)
        c = [colors[i] if i>-1 else [.5,.5,.5] for i in labels ]
        # set sizes if probs are added
        if probs is not None:
            sizes = [basesize*(1+p*4) for p in probs]
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(15,15))
    ax.scatter(*mds_coords.T, c=c, s=sizes)
    if labels_on:
        ax.set_title('Num Labels: %s' % N_labels, fontsize=20)
        
# set up hyper variables
datafile = 'Complete_10-08-2017'
subset = 'task'

# load results
results = load_results(datafile)
r_sub = results[subset]
data = r_sub.data
c = r_sub.EFA.get_metric_cs()['c_metric-BIC']
EFA_data = r_sub.EFA.get_loading(c)

EFA_data = r_sub.EFA.get_loading(8)

# dimensionality reduction to reduce noise
scaled_data = scale(data).T
pca = PCA()
data_reduced = pca.fit_transform(scaled_data)
# plot PCA
plt.figure(figsize=(12,8))
plt.subplot(121)
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'o-')
plt.title('PCA: Cumulative Variance Explained')
plt.subplot(122)
plt.plot(pca.explained_variance_ratio_, 'o-')
plt.title('PCA: Variance per component')
# redo PCA with 10 dimensions
pca = PCA(15)
data_reduced = pca.fit_transform(scaled_data)

# ****************************************************************************
# construct distance matrix
input_data = data_reduced
dist = pd.DataFrame(squareform(abs_pdist(input_data)), 
                    index=data.columns,
                    columns=data.columns)

# ***********************HDBSCAN*********************************

# perform HDSBScan clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=4, min_samples=4, 
                            gen_min_span_tree=True,
                            metric='precomputed',
                            cluster_selection_method='leaf')
clusterer.fit(dist)
# get labels
hdbscan_labels = clusterer.labels_
hdbscan_probs = clusterer.probabilities_
hdbscan_label_names = [[(data.columns[i], hdbscan_probs[i]) 
                        for i,l in enumerate(hdbscan_labels) if l == ii] 
                        for ii in np.unique(hdbscan_labels)]

# visualize clustering in lower dimension
f, axes = plt.subplots(1,2,figsize=(25,15))
plot_labels(dist, hdbscan_labels, hdbscan_probs, labels_on=False,
            random_state=5, ax=axes[0])
plot_labels(dist, hdbscan_labels, hdbscan_probs, 
            random_state=5, ax=axes[1])



plt.figure(figsize=(18,12))
clusterer.condensed_tree_.plot()

plt.figure(figsize=(18,12))
clusterer.condensed_tree_.plot(select_clusters=True,
                               selection_palette=sns.color_palette('deep', 8))

# convert to pandas and scipy
Z=clusterer.single_linkage_tree_.to_pandas().iloc[:,1:]
plt.figure(figsize=(30,10))
d = dendrogram(Z, labels=data.columns)
plt.tick_params(labelsize=15)

dendroheatmap(Z, dist)

# ************************Using agglomerative Clustering***********************
clustering = hierarchical_cluster(dist, compute_dist=False)
agglomerative_labels = clustering['clustering']['labels']
agglomerative_label_names = [[data.columns[i] 
                            for i,l in enumerate(agglomerative_labels) if l == ii] 
                            for ii in np.unique(agglomerative_labels)]
dendroheatmap(clustering['linkage'], clustering['distance_df'])

f, axes = plt.subplots(1,2,figsize=(25,15))
plot_labels(dist, agglomerative_labels, random_state=5, ax=axes[0])
plot_labels(dist, hdbscan_labels, hdbscan_probs, 
            random_state=5, ax=axes[1])
axes[0].set_xlabel('Agglomerative Clustering', fontsize=20)
axes[1].set_xlabel('HDBSCAN Clustering', fontsize=20)

# ******************Clustering Similarity****************************
adjusted_mutual_info_score(hdbscan_labels, agglomerative_labels)


adjusted_mutual_info_score(hdbscan_labels[hdbscan_labels!=-1], 
                           agglomerative_labels[hdbscan_labels!=-1])











