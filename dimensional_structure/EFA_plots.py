# imports
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from os import makedirs, path
import pandas as pd
import seaborn as sns

from dimensional_structure.plot_utils import visualize_factors, visualize_task_factors
from dimensional_structure.utils import get_factor_groups
from selfregulation.utils.plot_utils import beautify_legend, format_num, format_variable_names, save_figure
from selfregulation.utils.r_to_py_utils import get_attr
from selfregulation.utils.utils import get_retest_data

sns.set_context('notebook', font_scale=1.4)
sns.set_palette("Set1", 8, .75)



def plot_BIC_SABIC(results, size=2.3, dpi=300, ext='png', plot_dir=None):
    """ Plots BIC and SABIC curves
    
    Args:
        results: a dimensional structure results object
        dpi: the final dpi for the image
        ext: the extension for the saved figure
        plot_dir: the directory to save the figure. If none, do not save
    """
    EFA = results.EFA
    # Plot BIC and SABIC curves
    colors = ['c', 'm']
    with sns.axes_style('white'):
        x = list(EFA.results['cscores_metric-BIC'].keys())
        fig, ax1 = plt.subplots(1,1, figsize=(size, size*.75))
        # BIC
        BIC_scores = [EFA.results['cscores_metric-BIC'][i] for i in x]
        BIC_c = EFA.results['c_metric-BIC']
        ax1.plot(x, BIC_scores,  'o-', c=colors[0], lw=3, label='BIC',
                 markersize=size*2)
        ax1.set_xlabel('# Factors', fontsize=size*3)
        ax1.set_ylabel('BIC', fontsize=size*3)
        ax1.plot(BIC_c, BIC_scores[BIC_c-1], '.', color='white',
                 markeredgecolor=colors[0], markeredgewidth=size/2, 
                 markersize=size*4)
        ax1.tick_params(labelsize=size*2)
        if 'cscores_metric-SABIC' in EFA.results.keys():
            # SABIC
            ax2 = ax1.twinx()
            SABIC_scores = list(EFA.results['cscores_metric-SABIC'].values())
            SABIC_c = EFA.results['c_metric-SABIC']
            ax2.plot(x, SABIC_scores, c=colors[1], lw=3, label='SABIC',
                     markersize=size*2)
            ax2.set_ylabel('SABIC', fontsize=size*4)
            ax2.plot(SABIC_c, SABIC_scores[SABIC_c],'k.',
                 markeredgecolor=colors[0], markeredgewidth=size/2, 
                 markersize=size*4)
            # set up legend
            ax1.plot(np.nan, c='m', lw=3, label='SABIC')
            leg = ax1.legend(loc='right center')
            beautify_legend(leg, colors=colors)
        if plot_dir is not None:
            save_figure(fig, path.join(plot_dir, 'BIC_SABIC_curves.%s' % ext),
                        {'bbox_inches': 'tight', 'dpi': dpi})
            plt.close()

def plot_communality(results, c, rotate='oblimin', retest_threshold=.2,
                     size=4.6, dpi=300, ext='png', plot_dir=None):
    EFA = results.EFA
    loading = EFA.get_loading(c, rotate=rotate)
    # get communality from psych out
    fa = EFA.results['factor_tree_Rout_%s' % rotate][c]
    communality = get_attr(fa, 'communalities')
    communality = pd.Series(communality, index=loading.index)
    # alternative calculation
    #communality = (loading**2).sum(1).sort_values()
    communality.index = [i.replace('.logTr','').replace('.ReflogTr','') for i in communality.index]
    # load retest data
    retest_data = get_retest_data(dataset=results.dataset.replace('Complete','Retest'))
    if retest_data is None:
        print('No retest data found for datafile: %s' % results.dataset)
        return
    
    # reorder data in line with communality
    retest_data = retest_data.loc[communality.index]
    # reformat variable names
    communality.index = format_variable_names(communality.index)
    retest_data.index = format_variable_names(retest_data.index)
    if len(retest_data) > 0:
        # noise ceiling
        noise_ceiling = retest_data.pearson
        # remove very low reliabilities
        if retest_threshold:
            noise_ceiling[noise_ceiling<retest_threshold]= np.nan
        # adjust
        adjusted_communality = communality/noise_ceiling
        # correlation
        correlation = pd.concat([communality, noise_ceiling], axis=1).corr().iloc[0,1]
        kept_vars = np.logical_not(noise_ceiling.isnull())
        noise_ceiling = noise_ceiling[kept_vars]
        communality = communality[kept_vars]
        adjusted_communality = adjusted_communality[kept_vars]
        
    # plot communality bars woo!
    if len(retest_data)>0:
        f, axes = plt.subplots(1, 3, figsize=(3*(size/10), size))
    
        plot_bar_factor(communality, axes[0], width=size/10, height=size,
                        label_loc='leftall',  title='Communality')
        plot_bar_factor(noise_ceiling, axes[1], width=size/10, height=size,
                        label_loc=None,  title='Test-Retest')
        plot_bar_factor(adjusted_communality, axes[2], width=size/10, height=size,
                        label_loc=None,  title='Adjusted Communality')
    else:
        f = plot_bar_factor(communality, label_loc='both', 
                            width=size/3, height=size*2, title='Communality')
    if plot_dir:
        filename = 'communality_bars-EFA%s.%s' % (c, ext)
        save_figure(f, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
    
    # plot communality histogram
    if len(retest_data) > 0:
        with sns.axes_style('white'):
            colors = sns.color_palette(n_colors=2, desat=.75)
            f, ax = plt.subplots(1,1,figsize=(size,size))
            sns.kdeplot(communality, linewidth=size/4, 
                        shade=True, label='Communality', color=colors[0])
            sns.kdeplot(adjusted_communality, linewidth=size/4, 
                        shade=True, label='Adjusted Communality', color=colors[1])
            ylim = ax.get_ylim()
            ax.vlines(np.mean(communality), ylim[0], ylim[1],
                      color=colors[0], linewidth=size/4, linestyle='--')
            ax.vlines(np.mean(adjusted_communality), ylim[0], ylim[1],
                      color=colors[1], linewidth=size/4, linestyle='--')
            leg=ax.legend(fontsize=size*2, loc='upper right')
            beautify_legend(leg, colors)
            plt.xlabel('Communality', fontsize=size*2)
            plt.ylabel('Normalized Density', fontsize=size*2)
            ax.set_yticks([])
            ax.tick_params(labelsize=size)
            ax.set_ylim(0, ax.get_ylim()[1])
            ax.set_xlim(0, ax.get_xlim()[1])
            ax.spines['right'].set_visible(False)
            #ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            # add correlation
            correlation = format_num(np.mean(correlation))
            ax.text(1.1, 1.25, 'Correlation Between Communality \nand Test-Retest: %s' % correlation,
                    size=size*2)

        if plot_dir:
            filename = 'communality_dist-EFA%s.%s' % (c, ext)
            save_figure(f, path.join(plot_dir, filename), 
                        {'bbox_inches': 'tight', 'dpi': dpi})
            plt.close()
        
    
        
    
def plot_nesting(results, thresh=.5, rotate='oblimin', title=True,
                 dpi=300, figsize=12, ext='png', plot_dir=None):
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
    explained_scores, sum_explained = EFA.get_nesting_matrix(thresh, 
                                                             rotate=rotate)

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
        plt.close()
        
def plot_factor_correlation(results, c, rotate='oblimin', title=True,
                            DA=False, size=4.6, dpi=300, ext='png', plot_dir=None):
    if DA:
        EFA = results.DA
    else:
        EFA = results.EFA
    loading = EFA.get_loading(c, rotate=rotate)
    # get factor correlation matrix
    reorder_vec = EFA.get_factor_reorder(c)
    phi = get_attr(EFA.results['factor_tree_Rout_%s' % rotate][c],'Phi')
    phi = pd.DataFrame(phi, columns=loading.columns, index=loading.columns)
    phi = phi.iloc[reorder_vec, reorder_vec]
    with sns.plotting_context('notebook', font_scale=2) and sns.axes_style('white'):
        f = plt.figure(figsize=(size*5/4, size))
        ax1 = f.add_axes([0,0,.9,.9])
        cbar_ax = f.add_axes([.91, .05, .03, .8])
        sns.heatmap(phi, ax=ax1, square=True, vmax=1, vmin=-1,
                    cbar_ax=cbar_ax, 
                    cmap=sns.diverging_palette(220,15,n=100,as_cmap=True))
        yticklabels = ax1.get_yticklabels()
        ax1.set_yticklabels(yticklabels, rotation=0, ha="right")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
        if title == True:
            ax1.set_title('%s Factor Correlations' % results.ID.split('_')[0].title(),
                      weight='bold', y=1.05, fontsize=size*3)
        ax1.tick_params(labelsize=size*3)
        # format cbar
        cbar_ax.tick_params(axis='y', length=0)
        cbar_ax.tick_params(labelsize=size*2)
        cbar_ax.set_ylabel('Pearson Correlation', rotation=-90, labelpad=size*4, fontsize=size*3)
    
    if plot_dir:
        filename = 'factor_correlations_EFA%s.%s' % (c, ext)
        save_figure(f, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
        

def plot_bar_factor(loading, ax=None, bootstrap_err=None, grouping=None,
                    width=4, height=8, label_loc='left', title=None,
                    color_grouping=False, separate_ticklabels=True):
    """ Plots one factor loading as a vertical bar plot
    
    Args:
        loading: factor loadings as a dataframe or series
        ax: optional, plot axis
        bootstrap_err: a dataframe/series with the same index as loading. Used
            to plot confidence intervals on bars
        grouping: optional, output of "get_factor_groups", used to plot separating
            horizontal lines
        label_loc: 'left', 'right', or None. Plots half the variables names, either
            on the left or the right
    """
    
    # longest label for drawing lines
    DV_fontsize = height/(loading.shape[0]//2)*20
    longest_label = max([len(i) for i in loading.index])
    line_length = DV_fontsize*longest_label*.013/width
    # set up plot variables
    if ax is None:
        f, ax = plt.subplots(1,1, figsize=(width, height))
    with sns.plotting_context(font_scale=1.3):
        # plot optimal factor breakdown in bar format to better see labels
        # plot actual values
        colors = sns.diverging_palette(220,15,n=2)
        ordered_colors = [colors[int(i)] for i in (np.sign(loading)+1)/2]
        if bootstrap_err is None:
            abs(loading).plot(kind='barh', ax=ax, color=ordered_colors,
                                width=.7)
        else:
            abs(loading).plot(kind='barh', ax=ax, color=ordered_colors,
                                width=.7, xerr=bootstrap_err, error_kw={'linewidth': height/10})
        # draw lines separating groups
        if grouping is not None:
            factor_breaks = np.cumsum([len(i[1]) for i in grouping])[:-1]
            for y_val in factor_breaks:
                ax.hlines(y_val-.5, 0, 1.1, lw=height/10, 
                          color='grey', linestyle='dashed')
                if separate_ticklabels:
                    if label_loc in ['left', 'leftall']:
                        ax.hlines(y_val-.5, -line_length, 0, lw=height/20, 
                                  clip_on=False)
                    elif label_loc in ['right']:
                        ax.hlines(y_val-.5, 1, 1+line_length, lw=height/20, 
                                  clip_on=False)
        # set axes properties
        ax.set_xlim(0, max(max(abs(loading)), 1.1)); 
        ax.set_yticklabels(''); 
        ax.set_xticklabels('')
        labels = ax.get_yticklabels()
        locs = ax.yaxis.get_ticklocs()
        # add factor label to plot
        if title:
            ax.set_title(title, ha='left', va='bottom', fontsize=width*8,
                          weight='bold', rotation=20, y=1, x=0)
        ax.xaxis.set_tick_params(size=height/4, width=height/10, pad=height/2,
                                 color='#666666')
        # add labels of measures to top and bottom
        tick_colors = ['#000000','#444098']
        ax.set_facecolor('#DBDCE7')
        for location in locs[2::3]:
            ax.axhline(y=location, xmin=0, xmax=1, color='w', 
                       zorder=-1, lw=height/10)
        # if right or both given, plot half labels on the right side
        if label_loc in ['right', 'both']:
            for i, label in enumerate(labels):
                label.set_text('%s  %s' % (i+1, label.get_text()))
            ax_copy = ax.twinx()
            ax_copy.set_ybound(ax.get_ybound())
            ax_copy.set_yticks(locs[::2])
            right_labels = ax_copy.set_yticklabels(labels[::2], 
                                                   fontsize=DV_fontsize)
            ax_copy.yaxis.set_tick_params(size=height/4, width=height/10, pad=width)
            if grouping is not None and color_grouping:
                # change colors of ticks based on factor group
                color_i = 1
                last_group = None
                for j, label in enumerate(right_labels):
                    group = np.digitize(locs[::2][j], factor_breaks)
                    if last_group is None or group != last_group:
                        color_i = 1-color_i
                        last_group = group
                    color = tick_colors[color_i]
                    label.set_color(color) 
        # if left or both given, plot half labels on left size
        if label_loc in ['left', 'both']:
            for i, label in enumerate(labels):
                label.set_text('%s  %s' % (label.get_text(), i+1))
            # and other half on bottom
            ax.set_yticks(locs[1::2])
            left_labels=ax.set_yticklabels(labels[1::2], 
                                           fontsize=DV_fontsize)
            ax.yaxis.set_tick_params(size=height/4, width=height/10, pad=width)
            if grouping is not None and color_grouping:
                # change colors of ticks based on factor group
                color_i = 1
                last_group = None
                for j, label in enumerate(left_labels):
                    group = np.digitize(locs[1::2][j], factor_breaks)
                    if last_group is None or group != last_group:
                        color_i = 1-color_i
                        last_group = group
                    color = tick_colors[color_i]
                    label.set_color(color) 
        # if leftall given, plot all labels on left
        if label_loc == 'leftall':
            for i, label in enumerate(labels):
                label.set_text('%s  %s' % (label.get_text(), i+1))
            # and other half on bottom
            ax.set_yticks(locs)
            left_labels=ax.set_yticklabels(labels,fontsize=DV_fontsize)
            ax.yaxis.set_tick_params(size=height/4, width=height/10, pad=width)
            if grouping is not None and color_grouping:
                # change colors of ticks based on factor group
                color_i = 1
                last_group = None
                for j, label in enumerate(left_labels):
                    group = np.digitize(locs[j], factor_breaks)
                    if last_group is None or group != last_group:
                        color_i = 1-color_i
                        last_group = group
                    color = tick_colors[color_i]
                    label.set_color(color)             
        else:
            ax.set_yticklabels('')
            ax.yaxis.set_tick_params(size=0)
    if ax is None:
        return f
                
def plot_bar_factors(results, c, size=4.6, thresh=75, rotate='oblimin',
                     dpi=300, ext='png', plot_dir=None):
    """ Plots factor analytic results as bars
    
    Args:
        results: a dimensional structure results object
        c: the number of components to use
        dpi: the final dpi for the image
        size: scalar - the width of the plot. The height is determined
            by the number of factors
        thresh: proportion of factor loadings to remove
        ext: the extension for the saved figure
        plot_dir: the directory to save the figure. If none, do not save
    """
    # set up plot variables
    
    EFA = results.EFA
    loadings = EFA.reorder_factors(EFA.get_loading(c, rotate=rotate))           
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
    # change variable names to make them more readable
    loadings.index = format_variable_names(loadings.index)
    if bootstrap_CI is not None:
        bootstrap_CI.index = format_variable_names(bootstrap_CI.index)
    # plot
    n_factors = len(loadings.columns)
    f, axes = plt.subplots(1, n_factors, figsize=(size, size*2))
    for i, k in enumerate(loadings.columns):
        loading = loadings[k]
        ax = axes[i]
        if bootstrap_CI is not None:
            bootstrap_err = bootstrap_CI[k]
        else:
            bootstrap_err = None
        label_loc=None
        if i==0:
            label_loc = 'leftall'
        plot_bar_factor(loading, 
                        ax,
                        bootstrap_err, 
                        width=size/n_factors,
                        height=size*2,
                        grouping=grouping,
                        label_loc=label_loc,
                        title=k
                        )
    if plot_dir:
        filename = 'factor_bars_EFA%s.%s' % (c, ext)
        save_figure(f, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()

def plot_heatmap_factors(results, c, size=4.6, thresh=75, rotate='oblimin',
                     DA=False, dpi=300, ext='png', plot_dir=None):
    """ Plots factor analytic results as bars
    
    Args:
        results: a dimensional structure results object
        c: the number of components to use
        dpi: the final dpi for the image
        size: scalar - the width of the plot. The height is determined
            by the number of factors
        thresh: proportion of factor loadings to remove
        ext: the extension for the saved figure
        plot_dir: the directory to save the figure. If none, do not save
    """
    if DA:
        EFA = results.DA
    else:
        EFA = results.EFA
    loading = EFA.get_loading(c, rotate=rotate)
    loadings = EFA.reorder_factors(loading, rotate=rotate)           
    grouping = get_factor_groups(loadings)
    flattened_factor_order = []
    for sublist in [i[1] for i in grouping]:
        flattened_factor_order += sublist
    loadings = loadings.loc[flattened_factor_order]
    # get threshold for loadings
    if thresh>0:
        thresh_val = np.percentile(abs(loadings).values, thresh)
        print('Thresholding all loadings less than %s' % np.round(thresh_val, 3))
        loadings = loadings.mask(abs(loadings) <= thresh_val, 0)
        # remove variables that don't cross the threshold for any factor
        kept_vars = list(loadings.index[loadings.mean(1)!=0])
        print('%s Variables out of %s are kept after threshold' % (len(kept_vars), loadings.shape[0]))
        loadings = loadings.loc[kept_vars]
        # remove masked variabled from grouping
        threshed_groups = []
        for factor, group in grouping:
            group = [x for x in group if x in kept_vars]
            threshed_groups.append([factor,group])
        grouping = threshed_groups
    # change variable names to make them more readable
    loadings.index = format_variable_names(loadings.index)
    # set up plot variables
    DV_fontsize = size*2/(loadings.shape[0]//2)*30
    figsize = (size,size*2)
    
    f = plt.figure(figsize=figsize)
    ax = f.add_axes([0, 0, .08*loadings.shape[1], 1]) 
    cbar_ax = f.add_axes([.08*loadings.shape[1]+.02,0,.04,1]) 

    max_val = abs(loadings).max().max()
    sns.heatmap(loadings, ax=ax, cbar_ax=cbar_ax,
                vmax =  max_val, vmin = -max_val,
                cbar_kws={'ticks': [-max_val, -max_val/2, 0, max_val/2, max_val]},
                linecolor='white', linewidth=.01,
                cmap=sns.diverging_palette(220,15,n=100,as_cmap=True))
    ax.set_yticks(np.arange(.5,loadings.shape[0]+.5,1))
    ax.set_yticklabels(loadings.index, fontsize=DV_fontsize, rotation=0)
    ax.set_xticklabels(loadings.columns, 
                       fontsize=min(size*3, DV_fontsize*1.5),
                       ha='center',
                       rotation=90)
    ax.tick_params(length=size*.5, width=size/10)
    # format cbar
    cbar_ax.set_yticklabels([format_num(-max_val, 2), 
                             format_num(-max_val/2, 2),
                             0, 
                             format_num(-max_val/2, 2),
                             format_num(max_val, 2)])
    cbar_ax.tick_params(axis='y', length=0)
    cbar_ax.tick_params(labelsize=DV_fontsize*1.5)
    cbar_ax.set_ylabel('Factor Loading', rotation=-90, fontsize=DV_fontsize*2)
    
    # draw lines separating groups
    if grouping is not None:
        factor_breaks = np.cumsum([len(i[1]) for i in grouping])[:-1]
        for y_val in factor_breaks:
            ax.hlines(y_val, 0, loadings.shape[1], lw=size/5, 
                      color='grey', linestyle='dashed')
                
    if plot_dir:
        filename = 'factor_heatmap_EFA%s.%s' % (c, ext)
        save_figure(f, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
        
def plot_polar_factors(results, c, color_by_group=True, rotate='oblimin',
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
    loadings = EFA.get_loading(c, rotate=rotate)
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
        plt.close()

    
def plot_task_factors(results, c, rotate='oblimin',
                      task_sublists=None, normalize_loadings=False,
                      figsize=10,  dpi=300, ext='png', plot_dir=None):
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
    loadings = EFA.get_loading(c, rotate=rotate)
    max_loading = abs(loadings).max().max()
    tasks = np.unique([i.split('.')[0] for i in loadings.index])
    
    if task_sublists is None:
        task_sublists = {'surveys': [t for t in tasks if 'survey' in t],
                        'tasks': [t for t in tasks if 'survey' not in t]}

    for sublist_name, task_sublist in task_sublists.items():
        for i, task in enumerate(task_sublist):
            # plot loading distributions. Each measure is scaled so absolute
            # comparisons are impossible. Only the distributions can be compared
            f, ax = plt.subplots(1,1, 
                                 figsize=(figsize, figsize), subplot_kw={'projection': 'polar'})
            task_loadings = loadings.filter(regex='^%s' % task, axis=0)
            task_loadings.index = format_variable_names(task_loadings.index)
            if normalize_loadings:
                task_loadings = task_loadings = (task_loadings.T/abs(task_loadings).max(1)).T
            # format variable names
            task_loadings.index = format_variable_names(task_loadings.index)
            # plot
            visualize_task_factors(task_loadings, ax, ymax=max_loading,
                                   xticklabels=True, label_size=figsize*2)
            ax.set_title(' '.join(task.split('_')), 
                              y=1.14, fontsize=25)
            
            if plot_dir is not None:
                if normalize_loadings:
                    function_directory = 'factor_DVnormdist_EFA%s_subset-%s' % (c, sublist_name)
                else:
                    function_directory = 'factor_DVdist_EFA%s_subset-%s' % (c, sublist_name)
                makedirs(path.join(plot_dir, function_directory), exist_ok=True)
                filename = '%s.%s' % (task, ext)
                save_figure(f, path.join(plot_dir, function_directory, filename),
                            {'bbox_inches': 'tight', 'dpi': dpi})
                plt.close()
            
def plot_entropies(results, rotate='oblimin', 
                   dpi=300, figsize=(20,8), ext='png', plot_dir=None): 
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
    entropies = EFA.results['entropies_%s' % rotate].copy()
    null_entropies = EFA.results['null_entropies_%s' % rotate].copy()
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
            plt.close()
    
# plot specific variable groups
def plot_DDM(results, c, rotate='oblimin', 
             dpi=300, figsize=(20,8), ext='png', plot_dir=None): 
    EFA = results.EFA
    loading = abs(EFA.get_loading(c, rotate=rotate))
    cats = []
    for i in loading.index:
        if 'drift' in i:
            cats.append('Drift')
        elif 'thresh' in i:
            cats.append('Thresh')
        elif 'non_decision' in i:
            cats.append('Non-Decision')
        else:
            cats.append('Misc')
    loading.insert(0,'category', cats)
    # plotting
    colors = sns.color_palette("Set1", 8, .75)
    color_map = {v:i for i,v in enumerate(loading.category.unique())}
    
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111, projection='3d')
    for name, group in loading.groupby('category'):
        ax.scatter(group['Speeded IP'],
                   group['Caution'],
                   group['Perc / Resp'],
                   marker='o',
                   s=150,
                   c=colors[color_map[name]],
                   label=name)
    ax.tick_params(labelsize=0, length=0)
    ax.set_xlabel('Speeded IP', fontsize=20)
    ax.set_ylabel('Caution', fontsize=20)
    ax.set_zlabel('Perc / Resp', fontsize=20)
    ax.view_init(30, 30)
    leg = plt.legend(fontsize=20)
    beautify_legend(leg, colors)      
    if plot_dir is not None:
        fig.savefig(path.join(plot_dir, 'DDM_factors.%s' % ext), 
                  bbox_inches='tight', dpi=dpi)
        plt.close()



        
def plot_EFA(results, plot_dir=None, verbose=False, size=4.6, dpi=300, ext='png',
             plot_task_kws={}):

    c = results.EFA.results['num_factors']
    #if verbose: print("Plotting BIC/SABIC")
    #plot_BIC_SABIC(EFA, plot_dir)
    if verbose: print("Plotting communality")
    plot_communality(results, c, size=size, plot_dir=plot_dir, dpi=dpi,  ext=ext)
#    if verbose: print("Plotting entropies")
#    plot_entropies(results, plot_dir=plot_dir, dpi=dpi,  ext=ext)
    if verbose: print("Plotting factor bars")
    plot_bar_factors(results, c, size=size, plot_dir=plot_dir, dpi=dpi,  ext=ext)
    if verbose: print("Plotting factor heatmap")
    plot_heatmap_factors(results, c=c, thresh=0, size=size, plot_dir=plot_dir, dpi=dpi, ext=ext)
#    if verbose: print("Plotting task factors")
#    plot_task_factors(results, c, plot_dir=plot_dir, dpi=dpi,  ext=ext, **plot_task_kws)
#    plot_task_factors(results, c, normalize_loadings=True, plot_dir=plot_dir, dpi=dpi,  ext=ext, **plot_task_kws)
    if verbose: print("Plotting factor correlations")
    plot_factor_correlation(results, c, title=False, plot_dir=plot_dir, dpi=dpi,  ext=ext)
    if verbose: print("Plotting DDM factors")
    if 'task' in results.ID:
        plot_DDM(results, c, plot_dir=plot_dir, dpi=dpi,  ext=ext)
    