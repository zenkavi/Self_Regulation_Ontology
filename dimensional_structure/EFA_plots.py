# imports
import argparse
from math import ceil
from dimensional_structure.utils import (
        create_factor_tree, find_optimal_components, get_factor_groups,
        get_hierarchical_groups, get_scores_from_subset,
        get_loadings, plot_factor_tree, get_top_factors, 
        quantify_lower_nesting, save_figure,
        visualize_factors, visualize_task_factors
        )
import matplotlib.pyplot as plt
import numpy as np
from os import path
import pandas as pd
import seaborn as sns
sns.set_context('notebook', font_scale=1.4)



def plot_BIC_SABIC(EFA, plot_dir=None):
    # Plot BIC and SABIC curves
    with sns.axes_style('white'):
        x = list(EFA.results['cscores_metric-BIC'].keys())
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        # BIC
        BIC_scores = list(EFA.results['cscores_metric-BIC'].values())
        BIC_c = EFA.results['c_metric-BIC']
        ax1.plot(x, BIC_scores, c='c', lw=3, label='BIC')
        ax1.set_ylabel('BIC', fontsize=20)
        ax1.plot(BIC_c, BIC_scores[BIC_c],'k.', markersize=30)
        # SABIC
        SABIC_scores = list(EFA.results['cscores_metric-SABIC'].values())
        SABIC_c = EFA.results['c_metric-SABIC']
        ax2.plot(x, SABIC_scores, c='m', lw=3, label='SABIC')
        ax2.set_ylabel('SABIC', fontsize=20)
        ax2.plot(SABIC_c, SABIC_scores[SABIC_c],'k.', markersize=30)
        # set up legend
        ax1.plot(np.nan, c='m', lw=3, label='SABIC')
        ax1.legend(loc='upper center')
        if plot_dir is not None:
            save_figure(fig, path.join(plot_dir, 'BIC_SABIC_curves.png'),
                        {'bbox_inches': 'tight'})

def plot_nesting(EFA, thresh=.5, plot_dir=None):
    explained_scores, sum_explained = EFA.get_nesting_matrix(thresh)

    # plot lower nesting
    fig, ax = plt.subplots(1, 1, figsize=(30,30))
    cbar_ax = fig.add_axes([.905, .3, .05, .3])
    sns.heatmap(sum_explained, annot=explained_scores,
                fmt='.2f', mask=(explained_scores==-1), square=True,
                ax = ax, vmin=.2, cbar_ax=cbar_ax,
                xticklabels = range(1,sum_explained.shape[1]+1),
                yticklabels = range(1,sum_explained.shape[0]+1))
    ax.set_xlabel('Higher Factors (Explainer)', fontsize=25)
    ax.set_ylabel('Lower Factors (Explainee)', fontsize=25)
    ax.set_title('Nesting of Lower Level Factors based on R2', fontsize=30)
    if plot_dir is not None:
        filename = 'lower_nesting_heatmap.png'
        save_figure(fig, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight'})
    
def plot_bar_factors(EFA, c, plot_dir=None):
    loadings = EFA.results['factor_tree'][c]
    sorted_vars = get_top_factors(loadings) # sort by loading
            
    grouping = get_factor_groups(loadings)
    flattened_factor_order = []
    for sublist in [i[1] for i in grouping]:
        flattened_factor_order += sublist
        
    n_factors = len(sorted_vars)
    f = plt.figure(figsize=(30, n_factors*3))
    axes = []
    for i in range(n_factors):
        axes.append(plt.subplot2grid((n_factors, 4), (i,0), colspan=3))
        axes.append(plt.subplot2grid((n_factors, 4), (i,3), colspan=1))
    with sns.plotting_context(font_scale=1.3) and sns.axes_style('white'):
        # plot optimal factor breakdown in bar format to better see labels
        for i, (k,v) in list(enumerate(sorted_vars.items())):
            ax1 = axes[2*i]
            ax2 = axes[2*i+1]
            # plot distribution of factors
            colors = [['r','b'][int(i)] for i in (np.sign(v)+1)/2]
            abs(v).plot(kind='bar', ax=ax2, color=colors)
            # plot actual values
            ordered_v = v[flattened_factor_order]
            ordered_colors = [['r','b'][int(i)] for i in (np.sign(ordered_v)+1)/2]
            abs(ordered_v).plot(kind='bar', ax=ax1, color=ordered_colors)
            # draw lines separating groups
            for x_val in np.cumsum([len(i[1]) for i in grouping]):
                ax1.vlines(x_val, 0, 1.1, lw=2, color='grey')
            # set axes properties
            ax1.set_ylim(0,1.1); ax2.set_ylim(0,1.1)
            ax1.set_yticklabels(''); ax2.set_yticklabels('')
            ax2.set_xticklabels('')
            labels = ax1.get_xticklabels()
            locs = ax1.xaxis.get_ticklocs()
            ax1.set_ylabel('Factor %s' % (i+1))
            if i == 0:
                ax_copy = ax1.twiny()
                ax_copy.set_xticks(locs[::2])
                ax_copy.set_xticklabels(labels[::2], rotation=90)
                ax2.set_title('Factor Loading Distribution')
            if i == len(sorted_vars)-1:
                # and other half on bottom
                ax1.set_xticks(locs[1::2])
                ax1.set_xticklabels(labels[1::2], rotation=90)
            else:
                ax1.set_xticklabels('')
    if plot_dir:
        filename = 'factor_bars_EFA%s.png' % c
        save_figure(f, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight'})

def plot_polar_factors(EFA, c, plot_dir=None):
    loadings = EFA.results['factor_tree'][c]
    groups = get_factor_groups(loadings)    
    # plot polar plot factor visualization for metric loadings
    filename =  'factor_polar_EFA%s.png' % c
    fig = visualize_factors(loadings, n_rows=4, groups=groups)
    if plot_dir is not None:
        save_figure(fig, path.join(plot_dir, filename),
                    {'bbox_inches': 'tight'})

    # plot factor tree around optimal metric
    filename2 = None
    if plot_dir is not None:
        filename2 = 'factor_tree_EFA%s.png' % c
        filename2 = path.join(plot_dir, filename2)
    
    plot_factor_tree({i: EFA.results['factor_tree'][i] for i in [c-1,c,c+1]},
                      groups=groups, filename = filename2)
    
def plot_task_factors(EFA, c, task_sublists=None, plot_dir=None):
    """
    Args:
        EFA: EFA_Analysis object
        c: number of components for EFA
        task_sublists: a dictionary whose values are sets of tasks, and 
                        whose keywords are labels for those lists
    """
    # plot task factor loading
    entropies = EFA.results['entropies']
    loadings = EFA.results['factor_tree'][c]
    tasks = np.unique([i.split('.')[0] for i in loadings.index])
    ncols = 6
    
    if task_sublists is None:
        task_sublists = {'surveys': [t for t in tasks if 'survey' in t],
                        'tasks': [t for t in tasks if 'survey' not in t]}

    for sublist_name, task_sublist in task_sublists.items():
        nrows = ceil(len(task_sublist)/ncols)
        adjusted_cols = min(ncols, len(task_sublist))
        # plot loading distributions. Each measure is scaled so absolute
        # comparisons are impossible. Only the distributions can be compared
        f, axes = plt.subplots(nrows, adjusted_cols, 
                               figsize=(adjusted_cols*10,nrows*(8+nrows)),
                               subplot_kw={'projection': 'polar'})
        axes = f.get_axes()
        for i, task in enumerate(task_sublist):
            task_loadings = loadings.filter(regex=task, axis=0)
            # add entropy to index
            task_entropies = entropies[c][task_loadings.index]
            task_loadings.index = [i+'(%.2f)' % task_entropies.loc[i] for i in task_loadings.index]
            # plot
            visualize_task_factors(task_loadings, axes[i])
            axes[i].set_title(' '.join(task.split('_')), 
                              y=1.14, fontsize=25)
            
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        plt.subplots_adjust(hspace=.5, wspace=.5)
        filename = 'factor_DVdistributions_EFA%s_subset-%s.png' % (c, sublist_name)
        if plot_dir is not None:
            save_figure(f, path.join(plot_dir, filename),
                        {'bbox_inches': 'tight'})
            
def plot_entropies(EFA, plot_dir=None): 
    # plot entropies
    entropies = EFA.results['entropies'].copy()
    null_entropies = EFA.results['null_entropies'].copy()
    entropies.loc[:, 'group'] = 'real'
    null_entropies.loc[:, 'group'] = 'null'
    plot_entropies = pd.concat([entropies, null_entropies], 0)
    plot_entropies = plot_entropies.melt(id_vars= 'group',
                                         var_name = 'EFA',
                                         value_name = 'entropy')
    with sns.plotting_context('notebook', font_scale=1.8):
        f = plt.figure(figsize=(20,8))
        sns.boxplot(x='EFA', y='entropy', data=plot_entropies, hue='group')
        plt.xlabel('# Factors')
        plt.ylabel('Entropy')
        plt.title('Distribution of Measure Specificity across Factor Solutions')
        if plot_dir is not None:
            f.savefig(path.join(plot_dir, 'entropies_across_factors.png'), 
                      bbox_inches='tight')
            
def plot_EFA(EFA, c, plot_dir=None):
    plot_BIC_SABIC(EFA, plot_dir)
    plot_nesting(EFA, plot_dir=plot_dir)
    plot_bar_factors(EFA, c, plot_dir)
    plot_polar_factors(EFA, c, plot_dir)
    plot_task_factors(EFA< c, plot_dir=plot_dir)
    plot_entropies(EFA, plot_dir)