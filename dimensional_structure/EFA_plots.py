# imports
import matplotlib
matplotlib.use('Agg')
from math import ceil
from utils import (get_factor_groups, plot_factor_tree, get_top_factors, 
         save_figure, visualize_factors, visualize_task_factors
        )
import matplotlib.pyplot as plt
import numpy as np
from os import path
import pandas as pd
import seaborn as sns
sns.set_context('notebook', font_scale=1.4)
sns.set_palette("Set1", 8, .75)



def plot_BIC_SABIC(results, dpi=300, ext='png', plot_dir=None):
    """ Plots BIC and SABIC curves
    
    Args:
        results: a dimensional structure results object
        dpi: the final dpi for the image
        ext: the extension for the saved figure
        plot_dir: the directory to save the figure. If none, do not save
    """
    EFA = results.EFA
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
            save_figure(fig, path.join(plot_dir, 'BIC_SABIC_curves.%s' % ext),
                        {'bbox_inches': 'tight', 'dpi': dpi})

def plot_nesting(results, thresh=.5, dpi=300, figsize=12, ext='png', plot_dir=None):
    """ Plots nesting of factor solutions
    
    Args:
        results: a dimensional structure results object
        thresh: the threshold to pass to EFA.get_nesting_matrix
        dpi: the final dpi for the image
        figsize: scalar - the width and height of the (square) image
        ext: the extension for the saved figure
        plot_dir: the directory to save the figure. If none, do not save
    """
    EFA = results.EFA
    explained_scores, sum_explained = EFA.get_nesting_matrix(thresh)

    # plot lower nesting
    fig, ax = plt.subplots(1, 1, figsize=(figsize, figsize))
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
        filename = 'lower_nesting_heatmap.%s' % ext
        save_figure(fig, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})
    
def plot_bar_factors(results, c, figsize=12, dpi=300, ext='png', plot_dir=None):
    """ Plots factor analytic results as bars
    
    Args:
        results: a dimensional structure results object
        c: the number of components to use
        dpi: the final dpi for the image
        figsize: scalar - the width of the plot. The height is determined
            by the number of factors
        ext: the extension for the saved figure
        plot_dir: the directory to save the figure. If none, do not save
    """
    EFA = results.EFA
    loadings = EFA.results['factor_tree'][c]
    sorted_vars = get_top_factors(loadings) # sort by loading
            
    grouping = get_factor_groups(loadings)
    flattened_factor_order = []
    for sublist in [i[1] for i in grouping]:
        flattened_factor_order += sublist
        
    n_factors = len(sorted_vars)
    f = plt.figure(figsize=(figsize, n_factors*(figsize/10)))
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
                ax1.vlines(x_val-.5, 0, 1.1, lw=2, color='grey', linestyle='dashed')
            # set axes properties
            ax1.set_ylim(0,1.1); ax2.set_ylim(0,1.1)
            ax1.set_yticklabels(''); ax2.set_yticklabels('')
            ax2.set_xticklabels('')
            labels = ax1.get_xticklabels()
            locs = ax1.xaxis.get_ticklocs()
            # add factor label to plot
            DV_fontsize = figsize/(len(labels)//2)*45
            ax1.set_ylabel(k, rotation=0, ha='right', fontsize=DV_fontsize*1.5)
            # add labels of measures to top and bottom
            if i == 0:
                ax_copy = ax1.twiny()
                ax_copy.set_xticks(locs[::2])
                ax_copy.set_xticklabels(labels[::2], 
                                        rotation=90,
                                        fontsize=DV_fontsize)
                ax2.set_title('Factor Loading Distribution', 
                              fontsize=DV_fontsize*1.5)
            if i == len(sorted_vars)-1:
                # and other half on bottom
                ax1.set_xticks(locs[1::2])
                ax1.set_xticklabels(labels[1::2], 
                                    rotation=90, 
                                    fontsize=DV_fontsize)
            else:
                ax1.set_xticklabels('')
    if plot_dir:
        filename = 'factor_bars_EFA%s.%s' % (c, ext)
        save_figure(f, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})

def plot_polar_factors(results, c, color_by_group=True, 
                       dpi=300, ext='png', plot_dir=None):
    """ Plots factor analytic results as polar plots
    
    Args:
        results: a dimensional structure results object
        c: the number of components to use
        color_by_group: whether to color the polar plot by factor groups. Groups
            are defined by the factor each measurement loads most highly on
        dpi: the final dpi for the image
        ext: the extension for the saved figure
        plot_dir: the directory to save the figure. If none, do not save
    """
    EFA = results.EFA
    loadings = EFA.results['factor_tree'][c]
    groups = get_factor_groups(loadings)    
    # plot polar plot factor visualization for metric loadings
    filename =  'factor_polar_EFA%s.%s' % (c, ext)
    if color_by_group==True:
        colors=None
    else:
        colors=['b']*len(loadings.columns)
    fig = visualize_factors(loadings, n_rows=2, groups=groups, colors=colors)
    if plot_dir is not None:
        save_figure(fig, path.join(plot_dir, filename),
                    {'bbox_inches': 'tight', 'dpi': dpi})

    
def plot_task_factors(results, c, task_sublists=None, figsize=10,
                      dpi=300, ext='png', plot_dir=None):
    """ Plots task factors as polar plots
    
    Args:
        results: a dimensional structure results object
        c: the number of components to use
        task_sublists: a dictionary whose values are sets of tasks, and 
                        whose keywords are labels for those lists
        dpi: the final dpi for the image
        figsize: scalar - a width multiplier for the plot
        ext: the extension for the saved figure
        plot_dir: the directory to save the figure. If none, do not save
    """
    EFA = results.EFA
    # plot task factor loading
    entropies = EFA.results['entropies']
    loadings = EFA.results['factor_tree'][c]
    max_loading = abs(loadings).max().max()
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
                               figsize=(adjusted_cols*figsize,nrows*(figsize*.8+nrows)),
                               subplot_kw={'projection': 'polar'})
        axes = f.get_axes()
        for i, task in enumerate(task_sublist):
            task_loadings = loadings.filter(regex='^%s' % task, axis=0)
            # add entropy to index
            task_entropies = entropies[c][task_loadings.index]
            task_loadings.index = [i+'(%.2f)' % task_entropies.loc[i] for i in task_loadings.index]
            # plot
            if i%(ncols*2)==0 or i%(ncols*2)==5:
                visualize_task_factors(task_loadings, axes[i], ymax=max_loading,
                                       xticklabels=True)
            else:
                visualize_task_factors(task_loadings, axes[i], ymax=max_loading,
                                       xticklabels=False)
            axes[i].set_title(' '.join(task.split('_')), 
                              y=1.14, fontsize=25)
            
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        plt.subplots_adjust(hspace=.5, wspace=.5)
        filename = 'factor_DVdistributions_EFA%s_subset-%s.%s' % (c, sublist_name, ext)
        if plot_dir is not None:
            save_figure(f, path.join(plot_dir, filename),
                        {'bbox_inches': 'tight', 'dpi': dpi})
            
def plot_entropies(results, dpi=300, figsize=(20,8), ext='png', plot_dir=None): 
    """ Plots factor analytic results as bars
    
    Args:
        results: a dimensional structure results object
        c: the number of components to use
        task_sublists: a dictionary whose values are sets of tasks, and 
                        whose keywords are labels for those lists
        dpi: the final dpi for the image
        figsize: scalar - the width of the plot. The height is determined
            by the number of factors
        ext: the extension for the saved figure
        plot_dir: the directory to save the figure. If none, do not save
    """
    EFA = results.EFA
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
        f = plt.figure(figsize=figsize)
        sns.boxplot(x='EFA', y='entropy', data=plot_entropies, hue='group')
        plt.xlabel('# Factors')
        plt.ylabel('Entropy')
        plt.title('Distribution of Measure Specificity across Factor Solutions')
        if plot_dir is not None:
            f.savefig(path.join(plot_dir, 'entropies_across_factors.%s' % ext), 
                      bbox_inches='tight', dpi=dpi)
            
def plot_EFA(results, plot_dir=None, verbose=False, dpi=300, ext='png',
             plot_task_kws={}):

    c = results.EFA.num_factors
    #if verbose: print("Plotting BIC/SABIC")
    #plot_BIC_SABIC(EFA, plot_dir)
    if verbose: print("Plotting entropies")
    plot_entropies(results, plot_dir=plot_dir, dpi=dpi,  ext=ext)
    if verbose: print("Plotting factor bars")
    plot_bar_factors(results, c, plot_dir=plot_dir, dpi=dpi,  ext=ext)
    if verbose: print("Plotting factor polar")
    plot_polar_factors(results, c=c, plot_dir=plot_dir, dpi=dpi,  ext=ext)
    if verbose: print("Plotting task factors")
    plot_task_factors(results, c, plot_dir=plot_dir, dpi=dpi,  ext=ext, **plot_task_kws)
