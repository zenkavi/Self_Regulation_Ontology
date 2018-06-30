#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 19:48:18 2018

@author: ian
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
import seaborn as sns

from dimensional_structure.graph_utils import Graph_Analysis
from selfregulation.utils.utils import get_behav_data
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.r_to_py_utils import qgraph_cor
data = get_behav_data(file='meaningful_variables_imputed.csv')

all_results = load_results('Complete_03-29-2018')
def get_EFA_HCA(results, EFA):
    if EFA == False:
        return results.HCA.results['data']
    else:
        c = results.EFA.results['num_factors']
        return results.HCA.results['EFA%s_oblimin' % c]

EFA=True
survey_HCA = get_EFA_HCA(all_results['survey'], EFA)
survey_order = survey_HCA['reorder_vec']
task_HCA = get_EFA_HCA(all_results['task'], EFA)
task_order = task_HCA['reorder_vec']


all_data = pd.concat([all_results['task'].data.iloc[:, task_order], 
                      all_results['survey'].data.iloc[:, survey_order]], 
                    axis=1)

    
        
out, tuning = qgraph_cor(all_data, glasso=True, gamma=.5)

from sklearn.covariance import GraphLassoCV
from sklearn.preprocessing import scale
data = scale(all_data)
clf = GraphLassoCV()
clf.fit(data)


def add_attributes(g):
    g.vs['measurement'] = ['task']*len(task_order) + ['survey']*len(survey_order)
    task_clusters = task_HCA['labels'][task_order]
    survey_clusters = survey_HCA['labels'][survey_order] + max(task_clusters)
    g.vs['cluster'] = np.append(task_clusters, survey_clusters)
    
# unweighted
g = Graph_Analysis()
g.setup(abs(out), weighted=False)
add_attributes(g.G)
g.save_graph('/home/ian/tmp/graph.graphml', 'graphml')

# weighted
g = Graph_Analysis()
g.setup(abs(out), weighted=True)
add_attributes(g.G)
g.save_graph('/home/ian/tmp/weighted_graph.graphml', 'graphml')

"""
gephi settings using weighted graph

Yifan Hu Layout
-Optimal Distance 200
-Relative Strength .1
-Default everything else 
-Edge Weights filter at .01

Force Atlas Layout
...just play with it until it looks right. I started with
Yifan Hu to get it in a relatively nice layout, then used force atlas
"""

task_within = squareform(g.graph_to_dataframe().iloc[:len(task_order), :len(task_order)])
survey_within = squareform(g.graph_to_dataframe().iloc[len(task_order):, len(task_order):])
across = g.graph_to_dataframe().iloc[:len(task_order), len(task_order):].values.flatten()

#task_within = squareform(all_data.corr().iloc[:len(task_order), :len(task_order)].replace(1,0))
#survey_within = squareform(all_data.corr().iloc[len(task_order):, len(task_order):].replace(1,0))
#across = all_data.corr().iloc[:len(task_order), len(task_order):].values.flatten()

titles = ['Within Tasks', 'Within Surveys', 'Across']
colors = [sns.color_palette('Blues_d',3)[0],
          sns.color_palette('Reds_d',3)[0],
          [0,0,0]]

with sns.axes_style('white'):
    f, axes = plt.subplots(3,1, figsize=(6,6))
for i, corr in enumerate([task_within, survey_within, across]):
    sns.stripplot(corr, jitter=.2, alpha=.3, orient='h', ax=axes[i],
                  color=colors[i])
    
max_x = max([ax.get_xlim()[1] for ax in axes])*1.1
for i, ax in enumerate(axes):
    ax.set_xlim([0, max_x])
    ax.text(max_x*.1, -.35, titles[i], color=colors[i], ha='left',
            fontsize=16)
    if i!=(len(axes)-1):
        ax.tick_params(labelsize=0, pad=0)
    else:
        ax.tick_params(labelsize=12)
axes[-1].set_xlabel('Edge Weight', fontsize=16)
plt.subplots_adjust(hspace=0)



