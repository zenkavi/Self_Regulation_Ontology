import matplotlib.pyplot as plt
import numpy as np
from os import  path
import pandas as pd
import seaborn as sns

from dimensional_structure.EFA_plots import plot_bar_factor
from dimensional_structure.utils import get_factor_groups
from selfregulation.utils.plot_utils import format_num, format_variable_names, save_figure
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
        
def plot_factor_correlation(results, c, size=4.6, dpi=300, ext='png', plot_dir=None):
    DA = results.DA
    loading = DA.get_loading(c)
    # get factor correlation matrix
    reorder_vec = DA._get_factor_reorder(c)
    phi = get_attr(DA.results['factor_tree_Rout'][c],'Phi')
    phi = pd.DataFrame(phi, columns=loading.columns, index=loading.columns)
    phi = phi.iloc[reorder_vec, reorder_vec]
    with sns.plotting_context('notebook', font_scale=2):
        f = plt.figure(figsize=(size*5/4, size))
        ax1 = f.add_axes([0,0,.9,.9])
        cbar_ax = f.add_axes([.91, .05, .03, .8])
        sns.heatmap(phi, ax=ax1, square=True, vmax=.5, vmin=-.5,
                    cbar_ax=cbar_ax,
                    cmap=sns.diverging_palette(220,15,n=100,as_cmap=True))
        yticklabels = ax1.get_yticklabels()
        ax1.set_yticklabels(yticklabels, rotation = 0, ha="right")
        ax1.set_title('%s Factor Correlations' % results.ID.split('_')[0].title(),
                  weight='bold', y=1.05, fontsize=size*3)
        ax1.tick_params(labelsize=size*3)
        # format cbar
        cbar_ax.set_yticklabels([-.5, -.25, 0, .25, .5])
        cbar_ax.tick_params(axis='y', length=0)
        cbar_ax.tick_params(labelsize=size*2)
        cbar_ax.set_ylabel('Pearson Correlation', rotation=-90, labelpad=size*4, fontsize=size*3)
    
    if plot_dir:
        filename = 'factor_correlations_DA%s.%s' % (c, ext)
        save_figure(f, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
        
def plot_heatmap_factors(results, c, size=4.6, thresh=75,
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
    
    
    DA = results.DA
    loadings = DA.reorder_factors(DA.get_loading(c))           
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
    ax.set_yticklabels(loadings.index, fontsize=DV_fontsize)
    ax.set_xticklabels(loadings.columns, 
                                fontsize=size*.08*20,
                                ha='left',
                                rotation=-30)
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
        filename = 'factor_heatmap_DA%s.%s' % (c, ext)
        save_figure(f, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
        
def plot_DA(results, plot_dir=None, verbose=False, size=10, dpi=300, ext='png',
             plot_task_kws={}):
    c = results.DA.results['num_factors']
    #if verbose: print("Plotting BIC/SABIC")
    #plot_BIC_SABIC(EFA, plot_dir)
    if verbose: print("Plotting Distributions")
    plot_demo_factor_dist(results, c, plot_dir=plot_dir, dpi=dpi,  ext=ext)
    if verbose: print("Plotting factor correlations")
    plot_factor_correlation(results, c, size=size, plot_dir=plot_dir, dpi=dpi,  ext=ext)
    if verbose: print("Plotting factor bars")
    plot_heatmap_factors(results, c, size=size, plot_dir=plot_dir, dpi=dpi,  ext=ext)