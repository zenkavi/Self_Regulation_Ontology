# imports
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from os import makedirs, path
import pandas as pd
import seaborn as sns

from dimensional_structure.plot_utils import save_figure, visualize_factors, visualize_task_factors
from dimensional_structure.utils import get_factor_groups
from selfregulation.utils.plot_utils import beautify_legend, format_variable_names
from selfregulation.utils.r_to_py_utils import get_attr
from selfregulation.utils.utils import get_behav_data

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
            plt.close()

def plot_communality(results, c, figsize=20, dpi=300, ext='png', plot_dir=None):
    EFA = results.EFA
    loading = EFA.get_loading(c)
    communality = (loading**2).sum(1).sort_values()
    communality.index = [i.replace('.logTr','') for i in communality.index]
    # load retest data
    retest_data = get_behav_data(dataset='Retest_02-03-2018', file='bootstrap_merged.csv.gz')
    retest_data = retest_data.groupby('dv').mean()    
    retest_data.rename({'dot_pattern_expectancy.BX.BY_hddm_drift': 'dot_pattern_expectancy.BX-BY_hddm_drift',
                        'dot_pattern_expectancy.AY.BY_hddm_drift': 'dot_pattern_expectancy.AY-BY_hddm_drift'},
                        axis='index',
                        inplace=True)
    # reorder data in line with communality
    retest_data = retest_data.loc[communality.index]
    # reformat variable names
    communality.index = format_variable_names(communality.index)
    retest_data.index = format_variable_names(retest_data.index)
    if len(retest_data) > 0:
        # noise ceiling
        noise_ceiling = retest_data.pearson
        # remove very low reliabilities
        noise_ceiling[noise_ceiling<.2]= np.nan
        # adjust
        adjusted_communality = communality/noise_ceiling
        # correlation
        correlation = pd.concat([communality, noise_ceiling], axis=1).corr().iloc[0,1]
        noise_ceiling.replace(np.nan, 0, inplace=True)
        adjusted_communality.replace(np.nan, 0, inplace=True)
        
    # plot communality bars woo!
    if len(retest_data)>0:
        f, axes = plt.subplots(1, 3, figsize=(3*(figsize/10), figsize))
    
        plot_bar_factor(communality, axes[0], figsize=figsize,
                        label_loc='left',  title='Communality')
        plot_bar_factor(noise_ceiling, axes[1], figsize=figsize,
                        label_loc=None,  title_loc='bottom', title='Test-Retest')
        plot_bar_factor(adjusted_communality, axes[2], figsize=figsize,
                        label_loc='right',  title='Adjusted Communality')
    else:
        f = plot_bar_factor(communality, label_loc='both', 
                            figsize=figsize, title='Communality')
    if plot_dir:
        filename = 'communality_bars-EFA%s.%s' % (c, ext)
        save_figure(f, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})
    
    # plot communality histogram
    if len(retest_data) > 0:
        with sns.axes_style('white'):
            colors = sns.color_palette(n_colors=2, desat=.75)
            f, ax = plt.subplots(1,1,figsize=(figsize,figsize))
            sns.kdeplot(communality, linewidth=3, 
                        shade=True, label='Communality', color=colors[0])
            sns.kdeplot(adjusted_communality, linewidth=3, 
                        shade=True, label='Adjusted Communality', color=colors[1])
            leg=ax.legend(fontsize=figsize*2, loc='upper right')
            beautify_legend(leg, colors)
            plt.xlabel('Communality', fontsize=figsize*2)
            ax.set_yticks([])
            ax.set_ylim(0, ax.get_ylim()[1])
            ax.set_xlim(0, ax.get_xlim()[1])
            ax.spines['right'].set_visible(False)
            #ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            # add correlation
            correlation = "{0:0.2f}".format(np.mean(correlation))
            ax.text(1, 1.25, 'Correlation Between Communality \nand Test-Retest: %s' % correlation,
                    size=figsize*2)
        if plot_dir:
            filename = 'communality_dist-EFA%s.%s' % (c, ext)
            save_figure(f, path.join(plot_dir, filename), 
                        {'bbox_inches': 'tight', 'dpi': dpi})
            plt.close()
        
    
        
    
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
        plt.close()
        
def plot_factor_correlation(results, c, figsize=12, dpi=300, ext='png', plot_dir=None):
    EFA = results.EFA
    loading = EFA.get_loading(c)
    # get factor correlation matrix
    reorder_vec = EFA._get_factor_reorder(c)
    phi = get_attr(EFA.results['factor_tree_Rout'][c],'Phi')
    phi = pd.DataFrame(phi, columns=loading.columns, index=loading.columns)
    phi = phi.iloc[reorder_vec, reorder_vec]
    with sns.plotting_context('notebook', font_scale=2):
        f = plt.figure(figsize=(figsize*5/4, figsize))
        ax1 = f.add_axes([0,0,.75,.75])
        ax1_cbar = f.add_axes([.7, .05, .03, .65])
        sns.heatmap(phi, ax=ax1, square=True, vmax=.5, vmin=-.5,
                    cbar_ax=ax1_cbar,
                    cmap=sns.diverging_palette(220,15,n=100,as_cmap=True))
        yticklabels = ax1.get_yticklabels()
        ax1.set_yticklabels(yticklabels, rotation = 0, ha="right")
        ax1.set_title('%s 1st-Level Factor Correlations' % results.ID.split('_')[0],
                  weight='bold', y=1.05)
    # get higher order correlations
    if 'factor2_tree' in EFA.results.keys() and c in EFA.results['factor2_tree'].keys():
        higher_loading = EFA.results['factor2_tree'][c].iloc[reorder_vec]
        max_val = np.max(np.max(abs(higher_loading)))
        ax2 = f.add_axes([.85,0,.04*higher_loading.shape[1],.75])
        sns.heatmap(higher_loading, ax=ax2, cbar=True,
                    yticklabels=False, vmax=max_val, vmin=-max_val,
                    cmap=sns.diverging_palette(220,15,n=100,as_cmap=True))
        ax2.set_title('2nd-Order Factor Loadings', weight='bold', y=1.05)
        ax2.yaxis.set_label_position('right')
    if plot_dir:
        filename = 'factor_correlations_EFA%s.%s' % (c, ext)
        save_figure(f, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})
        

def plot_bar_factor(loading, ax=None, bootstrap_err=None, grouping=None,
                    figsize=20, label_loc='left', title=None, title_loc='top'):
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
        title_loc: 'top', 'bottom', or None
    """
    if ax is None:
        f, ax = plt.subplots(1,1, figsize=(figsize/12, figsize))
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
                                width=.7, xerr=bootstrap_err)
        # draw lines separating groups
        if grouping is not None:
            factor_breaks = np.cumsum([len(i[1]) for i in grouping])[:-1]
            for y_val in factor_breaks:
                ax.hlines(y_val-.5, 0, 1.1, lw=2, color='grey', linestyle='dashed')
        # set axes properties
        ax.set_xlim(0, max(max(abs(loading)), 1.1)); 
        ax.set_yticklabels(''); 
        ax.set_xticklabels('')
        labels = ax.get_yticklabels()
        locs = ax.yaxis.get_ticklocs()
        # add factor label to plot
        DV_fontsize = figsize/(len(labels)//2)*45
        if title and title_loc == 'top':
            ax.set_title(title, ha='center', fontsize=figsize*.75,
                          weight='bold')
        elif title and title_loc == 'bottom':
            ax.set_xlabel(title, ha='center', fontsize=figsize*.75,
                           weight='bold')
        # add labels of measures to top and bottom
        tick_colors = ['#000000','#444098']
        ax.set_facecolor('#DBDCE7')
        for location in locs[2::3]:
            ax.axhline(y=location, xmin=0, xmax=1, color='w', zorder=-1)
        if label_loc in ['right', 'both']:
            for i, label in enumerate(labels):
                label.set_text('%s  %s' % (i+1, label.get_text()))
            ax_copy = ax.twinx()
            ax_copy.set_ybound(ax.get_ybound())
            ax_copy.set_yticks(locs[::2])
            right_labels = ax_copy.set_yticklabels(labels[::2], 
                                                   fontsize=DV_fontsize)
            ax_copy.yaxis.set_tick_params(size=5, width=2, color='#666666')
            if grouping is not None:
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
        if label_loc in ['left', 'both']:
            for i, label in enumerate(labels):
                label.set_text('%s  %s' % (label.get_text(), i+1))
            # and other half on bottom
            ax.set_yticks(locs[1::2])
            left_labels=ax.set_yticklabels(labels[1::2], 
                                           fontsize=DV_fontsize)
            ax.yaxis.set_tick_params(size=5, width=2, color='#666666')
            if grouping is not None:
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
        else:
            ax.set_yticklabels('')
            ax.yaxis.set_tick_params(size=0)
    if ax is None:
        return f
                
def plot_bar_factors(results, c, figsize=20, thresh=75,
                     dpi=300, ext='png', plot_dir=None):
    """ Plots factor analytic results as bars
    
    Args:
        results: a dimensional structure results object
        c: the number of components to use
        dpi: the final dpi for the image
        figsize: scalar - the width of the plot. The height is determined
            by the number of factors
        thresh: proportion of factor loadings to remove
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
    # change variable names to make them more readable
    loadings.index = format_variable_names(loadings.index)
    if bootstrap_CI is not None:
        bootstrap_CI.index = format_variable_names(bootstrap_CI.index)
    # plot
    n_factors = len(loadings.columns)
    f, axes = plt.subplots(1, n_factors, figsize=(n_factors*(figsize/12), figsize))
    for i, k in enumerate(loadings.columns):
        loading = loadings[k]
        ax = axes[i]
        if bootstrap_CI is not None:
            bootstrap_err = bootstrap_CI[k]
        else:
            bootstrap_err = None
        label_loc=None
        title_loc = 'top'
        if i==0:
            label_loc = 'left'
        elif i==n_factors-1:
            label_loc='right'
        if i%2:
            title_loc = 'bottom'
        plot_bar_factor(loading, 
                        ax,
                        bootstrap_err, 
                        figsize=figsize,
                        grouping=grouping,
                        label_loc=label_loc,
                        title_loc=title_loc,
                        title=k
                        )
                
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
        plt.close()

    
def plot_task_factors(results, c, task_sublists=None, normalize_loadings = False,
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
            plt.close()
    
# plot specific variable groups
def plot_DDM(results, c, dpi=300, figsize=(20,8), ext='png', plot_dir=None): 
    EFA = results.EFA
    loading = abs(EFA.get_loading(c))
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



        
def plot_EFA(results, plot_dir=None, verbose=False, dpi=300, ext='png',
             plot_task_kws={}):

    c = results.EFA.results['num_factors']
    #if verbose: print("Plotting BIC/SABIC")
    #plot_BIC_SABIC(EFA, plot_dir)
    if verbose: print("Plotting communality")
    plot_communality(results, c, plot_dir=plot_dir, dpi=dpi,  ext=ext)
    if verbose: print("Plotting entropies")
    plot_entropies(results, plot_dir=plot_dir, dpi=dpi,  ext=ext)
    if verbose: print("Plotting factor bars")
    plot_bar_factors(results, c, plot_dir=plot_dir, dpi=dpi,  ext=ext)
    if verbose: print("Plotting factor polar")
    plot_polar_factors(results, c=c, plot_dir=plot_dir, dpi=dpi,  ext=ext)
    if verbose: print("Plotting task factors")
    plot_task_factors(results, c, plot_dir=plot_dir, dpi=dpi,  ext=ext, **plot_task_kws)
    plot_task_factors(results, c, normalize_loadings=True, plot_dir=plot_dir, dpi=dpi,  ext=ext, **plot_task_kws)
    if verbose: print("Plotting factor correlations")
    plot_factor_correlation(results, c, plot_dir=plot_dir, dpi=dpi,  ext=ext)
    if verbose: print("Plotting DDM factors")
    if 'task' in results.ID:
        plot_DDM(results, c, plot_dir=plot_dir, dpi=dpi,  ext=ext)
    