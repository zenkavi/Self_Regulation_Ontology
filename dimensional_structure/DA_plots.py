import matplotlib.pyplot as plt
import numpy as np
from os import  path
import pandas as pd
import seaborn as sns

from dimensional_structure.EFA_plots import plot_bar_factor
from dimensional_structure.plot_utils import save_figure
from dimensional_structure.utils import get_factor_groups
from selfregulation.utils.plot_utils import format_variable_names
from selfregulation.utils.r_to_py_utils import get_attr


def plot_demo_factor_dist(results, c, figsize=12, dpi=300, ext='png', plot_dir=None):
    DA = results.DA
    sex = DA.raw_data['Sex']
    sex_percent = "{0:0.1f}%".format(np.mean(sex)*100)
    scores = DA.get_scores(c)
    axes = scores.hist(bins=40, grid=False, figsize=(figsize*1.3,figsize))
    axes = axes.flatten()
    f = plt.gcf()
    for ax in axes:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    axes[-1].set_xlabel('N: %s, Female Percent: %s' % (len(scores), sex_percent), 
        labelpad=20)
    if plot_dir:
        filename = 'factor_correlations_DA%s.%s' % (c, ext)
        save_figure(f, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
        
def plot_factor_correlation(results, c, figsize=12, dpi=300, ext='png', plot_dir=None):
    DA = results.DA
    loading = DA.get_loading(c)
    # get factor correlation matrix
    reorder_vec = DA._get_factor_reorder(c)
    phi = get_attr(DA.results['factor_tree_Rout'][c],'Phi')
    phi = pd.DataFrame(phi, columns=loading.columns, index=loading.columns)
    phi = phi.iloc[reorder_vec, reorder_vec]
    with sns.plotting_context('notebook', font_scale=2):
        f = plt.figure(figsize=(figsize*5/4, figsize))
        ax1 = f.add_axes([0,0,.9,.9])
        ax1_cbar = f.add_axes([.92, .1, .03, .7])
        sns.heatmap(phi, ax=ax1, square=True, vmax=.5, vmin=-.5,
                    cbar_ax = ax1_cbar,
                    cmap=sns.diverging_palette(220,15,n=100,as_cmap=True))
        yticklabels = ax1.get_yticklabels()
        ax1.set_yticklabels(yticklabels, rotation = 0, ha="right")
        ax1.set_title('Demographic Factor Correlations', weight='bold', y=1.05)
    if plot_dir:
        filename = 'factor_correlations_DA%s.%s' % (c, ext)
        save_figure(f, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()

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
    DA = results.DA
    loadings = DA.reorder_factors(DA.get_loading(c))           
    grouping = get_factor_groups(loadings)
    flattened_factor_order = []
    for sublist in [i[1] for i in grouping]:
        flattened_factor_order += sublist
    loadings = loadings.loc[flattened_factor_order]
    # bootstrap CI
    bootstrap_CI = DA.get_boot_stats(c)
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
        filename = 'factor_bars_DA%s.%s' % (c, ext)
        save_figure(f, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
        
def plot_DA(results, plot_dir=None, verbose=False, dpi=300, ext='png',
             plot_task_kws={}):
    c = results.DA.results['num_factors']
    #if verbose: print("Plotting BIC/SABIC")
    #plot_BIC_SABIC(EFA, plot_dir)
    if verbose: print("Plotting Distributions")
    plot_demo_factor_dist(results, c, plot_dir=plot_dir, dpi=dpi,  ext=ext)
    if verbose: print("Plotting factor correlations")
    plot_factor_correlation(results, c, plot_dir=plot_dir, dpi=dpi,  ext=ext)
    if verbose: print("Plotting factor bars")
    plot_bar_factors(results, c, plot_dir=plot_dir, dpi=dpi,  ext=ext)