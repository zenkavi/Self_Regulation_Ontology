#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 10:40:19 2018

@author: ian
"""
import math
import matplotlib.pyplot as plt
import numpy as np
from os import path
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from selfregulation.utils.data_preparation_utils import (remove_outliers, 
                                                         transform_remove_skew)
from selfregulation.utils.plot_utils import format_num, place_letter, save_figure
from selfregulation.utils.r_to_py_utils import get_attr, missForest
from selfregulation.utils.utils import get_behav_data

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
    # reorder scores
    reorder_vec = EFA.get_factor_reorder(c)
    scores = scores.iloc[:, reorder_vec]
    retest_scores = retest_scores.iloc[:, reorder_vec]
    combined = pd.concat([scores, retest_scores], axis=1)
    cross_diag = [combined.corr().iloc[i,i+len(scores.columns)] 
                    for i in range(len(scores.columns))]
    
    if verbose:
        print('%s, Avg Correlation: %s\n' % (name, format_num(np.mean(cross_diag))))
        for factor, num in zip(scores.columns, cross_diag):
            print('%s: %s' % (factor, format_num(num)))
    return combined, cross_diag

def calc_EFA_retest_held_out(results, verbose=True):
    name = results.ID.split('_')[0].title()
    orig_data = results.data
    positive_skewed = [i.replace('.logTr', '') for i in orig_data.columns if ".logTr" in i]
    negative_skewed = [i.replace('.ReflogTr', '') for i in orig_data.columns if ".ReflogTr" in i]
    DVs = [i.replace('.logTr','').replace('.ReflogTr','') for i in orig_data.columns]
    orig_scores = results.EFA.get_scores()
    
    # load and clean retest data exactly like original data
    data_raw = get_behav_data(dataset=results.dataset,
                              file='meaningful_variables.csv')
    retest_data_raw = get_behav_data(dataset=results.dataset.replace('Complete','Retest'),
                                     file='meaningful_variables.csv')
    shared_ids = set(retest_data_raw.index) & set(data_raw.index)
    data_raw = data_raw.loc[shared_ids, :]
    retest_data_raw = retest_data_raw.loc[shared_ids, :]
    raw_data = {'T1': data_raw, 'T2': retest_data_raw}
    imputed_data = {}
    for name, data in raw_data.items():  
        tmp_data = data.loc[:, DVs]
        tmp_data = remove_outliers(tmp_data)
        tmp_data = transform_remove_skew(tmp_data,
                                         positive_skewed=positive_skewed,
                                         negative_skewed=negative_skewed)
        tmp_data_imputed, error = missForest(tmp_data)  
        scaled_tmp_data = scale(tmp_data_imputed)
        imputed_data[name] = scaled_tmp_data
    
    from selfregulation.utils.r_to_py_utils import get_attr, psychFA
    # get subjects not in the retest set
    ind_data = orig_data.loc[set(orig_data.index)-shared_ids]
    fa, output = psychFA(ind_data, results.EFA.results['num_factors'], 
                         method='ml', rotate='oblimin')
    weights = get_attr(fa, 'weights')
    scores = {}
    for name, data in imputed_data.items():
        suffix=''
        if name=='T2': suffix='T2'
        tmp_scores = pd.DataFrame(data.dot(weights),
                                  index=shared_ids,
                                  columns=[i+' '+suffix for i in orig_scores.columns])
        scores[name] = tmp_scores
    combined = pd.concat([scores['T1'], scores['T2']], axis=1)
    cross_diag = [combined.corr().iloc[i,i+len(orig_scores.columns)] 
                    for i in range(len(orig_scores.columns))]
    return combined, cross_diag, (fa, output)
    
def plot_EFA_retest(combined, size=4.6, dpi=300, 
                    ext='png', plot_dir=None):
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
    ax.tick_params(labelsize=size/len(corr)*40)
    
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
            


def plot_EFA_change(combined, ax=None, color_on=False,
                    size=4.6, dpi=300, ext='png', plot_dir=None):
    n = combined.shape[1]//2
    orig = combined.iloc[:,:n]
    retest = combined.iloc[:,n:]
    pca = PCA(2)    
    orig_pca= pca.fit_transform(orig)    
    retest_pca= pca.transform(retest)
    
    color=[.2,.2,.2, .9]
    # get color range
    mins = np.min(orig_pca)
    ranges = np.max(orig_pca)-mins
    if ax is None:
        with sns.axes_style('white'):
            fig, ax = plt.subplots(figsize=(size,size))
    markersize = size
    markeredge = size/5
    linewidth = size/3
    for i in range(len(orig_pca)):
        label = [None, None]
        if i==0:
            label=['T1 Scores', 'T2 Scores']
        if color_on == True:
            color = list((orig_pca[i,:]-mins)/ranges)
            color = [color[0]] + [0] + [color[1]]
        elif color_on != False:
            color = color_on
        ax.plot(*zip(orig_pca[i,:], retest_pca[i,:]), marker='o',
                 markersize=markersize, color=color,
                 markeredgewidth=markeredge, markerfacecolor='w',
                 linewidth=linewidth, label=label[0])
        ax.plot(retest_pca[i,0], retest_pca[i,1], marker='o', 
                 markersize=markersize, color=color, 
                 linewidth=linewidth, label=label[1])
    ax.tick_params(labelsize=0, pad=size/2)
    ax.set_xlabel('PC 1', fontsize=size*2.5)
    ax.set_ylabel('PC 2', fontsize=size*2.5)
    ax.set_ylim(ax.get_xlim())
    ax.legend(fontsize=size*1.5)
        
    if plot_dir is not None:
            save_figure(fig, path.join(plot_dir, 'EFA_test_retest_sticks.%s' % ext),
                        {'bbox_inches': 'tight', 'dpi': dpi})
            plt.close()
    

def plot_cross_EFA_change(all_results, size=4.6, dpi=300, 
                          ext='png', plot_dir=None):
    keys = list(all_results.keys())
    num_cols = 2
    num_rows = math.ceil(len(keys)/num_cols)
    with sns.axes_style('white'):
        fig, axes = plt.subplots(num_rows, num_cols, 
                                 figsize=(size, size/2*num_rows))
    axes = fig.get_axes()
    for i, (name,results) in enumerate(all_results.items()):
        ax = axes[i]
        plot_EFA_change(results, ax=ax, size=size/2)
        ax.set_title(name.title(), fontsize=size)
        if ax != axes[0]:
            ax.get_legend().set_visible(False)
            
    if plot_dir is not None:
        save_figure(fig, path.join(plot_dir, 'EFA_test_retest_sticks.%s' % ext),
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
    
def plot_cross_EFA_retest(all_results, size=4.6, dpi=300, EFA_retest_fun=None,
                          annot_heatmap=False, ext='png', plot_dir=None):
    if EFA_retest_fun is None:
        EFA_retest_fun = calc_EFA_retest
    colors = {'survey': sns.color_palette('Reds_d',3)[0], 
              'task': sns.color_palette('Blues_d',3)[0]}
    letters = [chr(i).upper() for i in range(ord('a'),ord('z')+1)]
    keys = list(all_results.keys())
    num_cols = 2
    num_rows = math.ceil(len(keys)*2/num_cols)
    with sns.axes_style('white'):
        fig, axes = plt.subplots(num_rows, num_cols, 
                                 figsize=(size, size/2*num_rows))
    axes = fig.get_axes()
    cbar_ax = fig.add_axes([.2, .03, .2, .02])
    # get fontsize for factor labels
    for i, (name,results) in enumerate(all_results.items()):
        combined, *the_rest = EFA_retest_fun(results)
        color = list(colors.get(name, [.2,.2,.2])) + [.8]
        ax2 = axes[i*2]; ax = axes[i*2+num_rows//2]
        plot_EFA_change(combined=combined,  color_on=color, ax=ax, size=size/2)
        ax.set_xlabel('PC 1', fontsize=size*1.8)
        ax.set_ylabel('PC 2', fontsize=size*1.8)
        # plot corr between test and retest
        num_labels = combined.shape[1]//2
        corr = combined.corr().iloc[:num_labels, num_labels:]
        annot_fontsize = size/num_labels*7
        annot=False
        if annot_heatmap:
            annot=True
        if i == len(all_results)-1:
            sns.heatmap(corr, square=True, ax=ax2, cbar_ax=cbar_ax, 
                        vmin=-1, vmax=1,
                        cbar_kws={'orientation': 'horizontal',
                                  'ticks': [-1, 0, 1]},
                        annot=annot,
                        annot_kws={'fontsize': annot_fontsize}); 
            
            cbar_ax.set_xlabel('Pearson Correlation', fontsize=size*1.5)
            cbar_ax.tick_params(labelsize=size, pad=size/2)
        else:
            sns.heatmap(corr, square=True, ax=ax2, vmin=-1, vmax=1,
                        cbar=False, annot=annot,
                        annot_kws={'fontsize': annot_fontsize})
        ax2.set_xticklabels('')
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
        ax2.tick_params(labelsize=min(size/num_labels/num_rows*20, size*1.6), 
                        pad=size/2)
        ax2.set_xlabel('Retest (T2)', fontsize=size*1.8)
        ax2.set_ylabel('Test (T1)', fontsize=size*1.8)
        # add text for measurement category
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.text(x=xlim[1]+(xlim[1]-xlim[0])*0.05, 
                y=ylim[0]+(ylim[1]-ylim[0])/2, 
                s=name.title(),
                rotation=-90,
                size=size/num_rows*5,
                fontweight='bold')
        place_letter(ax2, letters.pop(0), fontsize=size*9/4.6)
        place_letter(ax, letters.pop(0), fontsize=size*9/4.6)
        [i.set_linewidth(size*.1) for i in ax.spines.values()]
        [i.set_linewidth(size*.1) for i in ax2.spines.values()]
        
    if plot_dir is not None:
        save_figure(fig, path.join(plot_dir, 'EFA_test_retest.%s' % ext),
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    