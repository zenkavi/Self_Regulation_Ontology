#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 10:40:19 2018

@author: ian
"""
import matplotlib.pyplot as plt
import numpy as np
from os import path
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from selfregulation.utils.data_preparation_utils import (remove_outliers, 
                                                         transform_remove_skew)
from selfregulation.utils.plot_utils import format_num, save_figure
from selfregulation.utils.r_to_py_utils import get_attr, missForest
from selfregulation.utils.utils import get_behav_data
from selfregulation.utils.result_utils import load_results

all_results = load_results('Complete_03-29-2018')
def calc_EFA_retest(results, verbose=True):
    name = results.ID.split('_')[0].title()
    data = results.data
    positive_skewed = [i.replace('.logTr', '') for i in data.columns if ".logTr" in i]
    negative_skewed = [i.replace('.ReflogTr', '') for i in data.columns if ".ReflogTr" in i]
    DVs = [i.replace('.logTr','').replace('.ReflogTr','') for i in data.columns]
    
    # load and clean retest data exactly like original data
    retest_data_raw = get_behav_data(dataset=results.dataset.replace('Complete','Retest'),
                                     file='meaningful_variables.csv')
    shared_ids = set(retest_data_raw.index) & set(data.index)
    retest_data_raw = retest_data_raw.loc[shared_ids, :]
    retest_data = retest_data_raw.loc[:, DVs]
    retest_data = remove_outliers(retest_data)
    retest_data = transform_remove_skew(retest_data,
                                        positive_skewed=positive_skewed,
                                        negative_skewed=negative_skewed)
    retest_data_imputed, error = missForest(retest_data)
    
    # scale and perform the factor score transformation
    scaled_retest = scale(retest_data_imputed)
    EFA = results.EFA
    c = EFA.results['num_factors']
    scores = EFA.get_scores(c=c).loc[retest_data_imputed.index, :]
    weights = get_attr(EFA.results['factor_tree_Rout_oblimin'][c], 'weights')
    retest_scores = pd.DataFrame(scaled_retest.dot(weights),
                                 index=retest_data_imputed.index,
                                 columns=[i+' Retest' for i in scores.columns])
    combined = pd.concat([scores, retest_scores], axis=1)
    cross_diag = [combined.corr().iloc[i,i+len(scores.columns)] 
                    for i in range(len(scores.columns))]
    
    if verbose:
        print('%s, Avg Correlation: %s\n' % (name, format_num(np.mean(cross_diag))))
        for factor, num in zip(scores.columns, cross_diag):
            print('%s: %s' % (factor, format_num(num)))
    return combined, cross_diag


def plot_EFA_retest(results, size=4.6, dpi=300, ext='png', plot_dir=None):
    combined, cross_diag = calc_EFA_retest(results, verbose=False)
    corr = combined.corr()
    max_val = abs(corr).max().max()
    
    fig = plt.figure(figsize=(size,size)); 
    ax = fig.add_axes([.1, .1, .8, .8])
    cbar_ax = fig.add_axes([.92, .15, .04, .7])
    sns.heatmap(corr, square=True, ax=ax, cbar_ax=cbar_ax,
                cbar_kws={'orientation': 'vertical',
                          'ticks': [-max_val, 0, max_val]}); 
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.tick_params(labelsize=size)
    
    # format cbar axis
    cbar_ax.set_yticklabels([format_num(-max_val), 0, format_num(max_val)])
    cbar_ax.tick_params(labelsize=size, length=0, pad=size/2)
    cbar_ax.set_ylabel('Factor Loading', rotation=-90, 
                   fontsize=size, labelpad=size/2)
    
    # set divider lines
    n = corr.shape[1]
    ax.axvline(n//2, 0, n, color='k', linewidth=size/3)
    ax.axhline(n//2, 0, n, color='k', linewidth=size/3)
    
    if plot_dir is not None:
            save_figure(fig, path.join(plot_dir, 'EFA_test_retest_heatmap.%s' % ext),
                        {'bbox_inches': 'tight', 'dpi': dpi})
            plt.close()
            


def plot_EFA_change(results, size=4.6, dpi=300, ext='png', plot_dir=None):
    combined, cross_diag = calc_EFA_retest(results, verbose=False)
    n = combined.shape[1]//2
    orig = combined.iloc[:,:n]
    retest = combined.iloc[:,n:]
    pca = PCA(2)    
    orig_pca= pca.fit_transform(orig)    
    retest_pca= pca.transform(retest)
    
    # get color range
    with sns.axes_style('white'):
        mins = np.min(orig_pca)
        ranges = np.max(orig_pca)-mins
        fig = plt.figure(figsize=(size,size))
        markersize = size**2*.05
        for i in range(len(orig_pca)):
            label = [None, None]
            if i==0:
                label=['Original Scores', 'Retest Scores']
            color = list((orig_pca[i,:]-mins)/ranges)
            color = [color[0]] + [0] + [color[1]]
            plt.plot(*zip(orig_pca[i,:], retest_pca[i,:]), marker='o',
                     markersize=markersize, color=color,
                     markeredgewidth=size/5, markerfacecolor='w',
                     linewidth=size/5, label=label[0])
            plt.plot(retest_pca[i,0], retest_pca[i,1], marker='o', 
                     markersize=markersize, color=color, label=label[1])
        plt.tick_params(labelsize=size)
        plt.xlabel('PC 1', fontsize=size*2)
        plt.ylabel('PC 2', fontsize=size*2)
        plt.ylim(plt.xlim())
        plt.legend(fontsize=size)
        
    if plot_dir is not None:
            save_figure(fig, path.join(plot_dir, 'EFA_test_retest_sticks.%s' % ext),
                        {'bbox_inches': 'tight', 'dpi': dpi})
            plt.close()
    
