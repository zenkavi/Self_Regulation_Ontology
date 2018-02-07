# imports
import matplotlib
matplotlib.use('Agg')
from math import ceil
from utils import (get_factor_groups, plot_factor_tree, get_top_factors, 
         save_figure, visualize_factors, visualize_task_factors
        )
import matplotlib.pyplot as plt
import numpy as np
from os import makedirs, path
import pandas as pd
import seaborn as sns
from selfregulation.utils.r_to_py_utils import get_attr
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
 
def plot_factor_correlation(results, c, figsize=12, dpi=300, ext='png', plot_dir=None):
    EFA = results.EFA
    loading = EFA.get_loading(c)
    # get factor correlation matrix
    reorder_vec = EFA._get_factor_reorder(c)
    phi = get_attr(EFA.results['factor_tree_Rout'][c],'Phi')
    phi = pd.DataFrame(phi, columns=loading.columns, index=loading.columns)
    phi = phi.iloc[reorder_vec, reorder_vec]
    f = plt.figure(figsize=(figsize*5/4, figsize))
    ax1 = f.add_axes([0,0,.75,.75])
    ax1_cbar = f.add_axes([.7, .05, .03, .65])
    sns.heatmap(phi, ax=ax1, square=True, vmax=.5, vmin=-.5,
                cbar_ax = ax1_cbar,
                cmap=sns.diverging_palette(220,15,n=100,as_cmap=True))
    ax1.set_title('%s 1st-Level Factor Correlations' % results.ID.split('_')[0],
              weight='bold')
    
    # get higher order correlations
    if 'factor2_tree' in EFA.results.keys() and c in EFA.results['factor2_tree'].keys():
        higher_loading = EFA.results['factor2_tree'][c].iloc[reorder_vec]
        ax2 = f.add_axes([.85,0,.04*higher_loading.shape[1],.75])
        sns.heatmap(higher_loading, ax=ax2, cbar=True,
                    yticklabels=False,
                    cmap=sns.diverging_palette(220,15,n=100,as_cmap=True))
        ax2.set_title('2nd-Order Factor Loadings', weight='bold')
        ax2.yaxis.set_label_position('right')
    if plot_dir:
        filename = 'factor_correlations_EFA%s.%s' % (c, ext)
        save_figure(f, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})
        
    
def plot_bar_factors(results, c, figsize=20, thresh=75,
                     dpi=300, ext='png', plot_dir=None):
    """ Plots factor analytic results as bars
    
    Args:
        results: a dimensional structure results object
        c: the number of components to use
        dpi: the final dpi for the image
        figsize: scalar - the width of the plot. The height is determined
            by the number of factors
        thresh: proportion of factor loadings to keep
        ext: the extension for the saved figure
        plot_dir: the directory to save the figure. If none, do not save
    """
    EFA = results.EFA
    loadings = EFA.reorder_factors(EFA.get_loading(c))            
    grouping = get_factor_groups(loadings)
    flattened_factor_order = []
    for sublist in [i[1] for i in grouping]:
        flattened_factor_order += sublist
    loadings = loadings.loc[flattened_factor_order]
    # bootstrap CI
    bootstrap_CI = EFA.get_boot_stats(c)
    if bootstrap_CI is not None:
        bootstrap_CI = bootstrap_CI['sds'] * 1.96
        bootstrap_CI = bootstrap_CI.loc[flattened_factor_order]
    # get threshold for loadings
    if thresh>0:
        thresh_val = np.percentile(abs(loadings).values, thresh)
        print('Thresholding all loadings less than %s' % np.round(thresh_val, 3))
        loadings = loadings.mask(abs(loadings) <= thresh_val, 0)
        # remove variables that don't cross the threshold for any factor
        kept_vars = list(loadings.index[loadings.mean(1)!=0])
        print('%s Variables out of %s are kept after threshold' % (len(kept_vars), loadings.shape[0]))
        loadings = loadings.loc[kept_vars]
        if bootstrap_CI is not None:
            bootstrap_CI = bootstrap_CI.mask(abs(loadings) <= thresh_val, 0)
            bootstrap_CI = bootstrap_CI.loc[kept_vars]
        # remove masked variabled from grouping
        threshed_groups = []
        for factor, group in grouping:
            group = [x for x in group if x in kept_vars]
            threshed_groups.append([factor,group])
        grouping = threshed_groups
    n_factors = len(loadings.columns)
    f, axes = plt.subplots(1, n_factors, figsize=(n_factors*(figsize/12), figsize))
    with sns.plotting_context(font_scale=1.3):
        # plot optimal factor breakdown in bar format to better see labels
        for i, k in enumerate(loadings.columns):
            loading = loadings[k]
            ax1 = axes[i]
            # plot actual values
            colors = sns.diverging_palette(220,15,n=2)
            ordered_colors = [colors[int(i)] for i in (np.sign(loading)+1)/2]
            if bootstrap_CI is None:
                abs(loading).plot(kind='barh', ax=ax1, color=ordered_colors,
                                    width=.7)
            else:
                abs(loading).plot(kind='barh', ax=ax1, color=ordered_colors,
                                    width=.7, xerr=bootstrap_CI[k])
            # draw lines separating groups
            factor_breaks = np.cumsum([len(i[1]) for i in grouping])[:-1]
            for y_val in factor_breaks:
                ax1.hlines(y_val-.5, 0, 1.1, lw=2, color='grey', linestyle='dashed')
            # set axes properties
            ax1.set_xlim(0,1.1); 
            ax1.set_yticklabels(''); 
            ax1.set_xticklabels('')
            labels = ax1.get_yticklabels()
            locs = ax1.yaxis.get_ticklocs()
            # add factor label to plot
            DV_fontsize = figsize/(len(labels)//2)*45
            if i%2 == 0:
                ax1.set_title(k, ha='center', fontsize=figsize*.75,
                              weight='bold')
            else:
                ax1.set_xlabel(k, ha='center', fontsize=figsize*.75,
                               weight='bold')
            # add labels of measures to top and bottom
            tick_colors = ['#000000','#444098']
            ax1.set_facecolor('#DBDCE7')
            for location in locs[2::3]:
                ax1.axhline(y=location, xmin=0, xmax=1, color='w', zorder=-1)
            if i == n_factors-1:
                for i, label in enumerate(labels):
                    label.set_text('%s  %s' % (i+1, label.get_text()))
                ax_copy = ax1.twinx()
                ax_copy.set_ybound(ax1.get_ybound())
                ax_copy.set_yticks(locs[::2])
                labels = ax_copy.set_yticklabels(labels[::2], 
                                                 fontsize=DV_fontsize)
                ax_copy.yaxis.set_tick_params(size=5, width=2, color='#666666')
                # change colors of ticks based on factor group
                color_i = 1
                last_group = None
                for j, label in enumerate(labels):
                    group = np.digitize(locs[::2][j], factor_breaks)
                    if last_group is None or group != last_group:
                        color_i = 1-color_i
                        last_group = group
                    color = tick_colors[color_i]
                    label.set_color(color) 
            if i == 0:
                for i, label in enumerate(labels):
                    label.set_text('%s  %s' % (label.get_text(), i+1))
                # and other half on bottom
                ax1.set_yticks(locs[1::2])
                labels=ax1.set_yticklabels(labels[1::2], 
                                           fontsize=DV_fontsize)
                ax1.yaxis.set_tick_params(size=5, width=2, color='#666666')
                # change colors of ticks based on factor group
                color_i = 1
                last_group = None
                for j, label in enumerate(labels):
                    group = np.digitize(locs[1::2][j], factor_breaks)
                    if last_group is None or group != last_group:
                        color_i = 1-color_i
                        last_group = group
                    color = tick_colors[color_i]
                    label.set_color(color) 
            else:
                ax1.set_yticklabels('')
                ax1.yaxis.set_tick_params(size=0)
                
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
    loadings = EFA.get_loading(c)
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
    loadings = EFA.get_loading(c)
    max_loading = abs(loadings).max().max()
    tasks = np.unique([i.split('.')[0] for i in loadings.index])
    
    if task_sublists is None:
        task_sublists = {'surveys': [t for t in tasks if 'survey' in t],
                        'tasks': [t for t in tasks if 'survey' not in t]}

    for sublist_name, task_sublist in task_sublists.items():
        for i, task in enumerate(task_sublist):
            # plot loading distributions. Each measure is scaled so absolute
            # comparisons are impossible. Only the distributions can be compared
            f, ax = plt.subplots(1,1, subplot_kw={'projection': 'polar'})
            task_loadings = loadings.filter(regex='^%s' % task, axis=0)
            # add entropy to index
            task_entropies = entropies[c][task_loadings.index]
            task_loadings.index = [i+'(%.2f)' % task_entropies.loc[i] for i in task_loadings.index]
            # plot
            visualize_task_factors(task_loadings, ax, ymax=max_loading,
                                   xticklabels=True)
            ax.set_title(' '.join(task.split('_')), 
                              y=1.14, fontsize=25)
            
            if plot_dir is not None:
                function_directory = 'factor_DVdistributions_EFA%s_subset-%s' % (c, sublist_name)
                makedirs(path.join(plot_dir, function_directory), exist_ok=True)
                filename = '%s.%s' % (task, ext)
                save_figure(f, path.join(plot_dir, function_directory, filename),
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
    if verbose: print("Plotting factor correlations")
    plot_factor_correlation(results, c, plot_dir=plot_dir, dpi=dpi,  ext=ext)