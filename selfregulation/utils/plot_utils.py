from math import ceil
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
import seaborn as sns

#***************************************************
# ********* Plotting Functions **********************
#**************************************************
def beautify_legend(legend, colors):
    for i, text in enumerate(legend.get_texts()):
        text.set_color(colors[i])
    for item in legend.legendHandles:
        item.set_visible(False)
        
def DDM_plot(v,t,a, sigma = .1, n = 10, plot_n = 15, file = None):
    """ Make a plot of trajectories using ddm parameters (in seconds)
    
    """
    # generate trajectory
    v = v/1000
    t =  t*1000
    timesteps = np.arange(2000)
    trajectories = []
    while len(trajectories) < n:
        y = [0]
        for step in timesteps[1:]:
            if step < t:
                y += [0]
            else:
                y += [y[-1]+v+np.random.normal(0,sigma)]
            if y[-1] > a:
                trajectories.append((y,'correct'))
                break
            elif y[-1] < -a:
                trajectories.append((y,'fail'))
                break
    # rts
    p_correct = np.sum([1 for i in range(n) if trajectories[i][1] == 'correct'])/n
    correct_rts = []
    incorrect_rts = []
    plot_trajectories = []
    trajectory_count = [0,0]
    positive_trace_num = np.round(p_correct*plot_n)
    for y, valence in trajectories:
        if valence == 'correct':
            correct_rts.append(len(y))
            if trajectory_count[1] < positive_trace_num:
                plot_trajectories.append((y,valence))
                trajectory_count[1]+=1
        else:
            incorrect_rts.append(len(y))
            if trajectory_count[0] < (plot_n - positive_trace_num):
                plot_trajectories.append((y,valence))
                trajectory_count[0]+=1
    
    # plot
    sns.set_context('talk')
    plot_start = int(max(0,t-50))
    fig = plt.figure(figsize = [10,6])
    ax = fig.add_axes([0,.2,1,.6]) 
    ax.set_xticklabels([])
    plt.hold(True)
    max_y = 0
    for trajectory in plot_trajectories:
        y = trajectory[0]
        color = ['red','green'][trajectory[1] == 'correct']
        plt.plot(timesteps[plot_start:len(y)],y[plot_start:], c = color)
        if len(y) > max_y:
            max_y = len(y)
    plt.hlines([a,-a],0,max_y+50,linestyles = 'dashed')
    plt.xlim([plot_start,max_y+50])
    plt.ylim([-a*1.01,a*1.01])
    plt.ylabel('Decision Variable', fontsize = 20)
    with sns.axes_style("dark"):
        ax2 = fig.add_axes([0,.8,1,.2]) 
        sns.kdeplot(pd.Series(correct_rts), color = 'g', ax = ax2, shade = True)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax3 = fig.add_axes([0,0,1,.2])
        ax3.invert_yaxis()
        if len(incorrect_rts) > 0:
            sns.kdeplot(pd.Series(incorrect_rts), color = 'r', ax = ax3, shade = True)
            ax3.set_ylim(ax3.get_ylim()[0]/p_correct,0)
            ax3.set_yticklabels([])
            plt.xlabel('Time Step (ms)', fontsize = 20)
    
    if file:
        fig.savefig(file, dpi = 300)
    return fig, trajectories


def dendroheatmap(link, dist_df, clusters=None,
                  label_fontsize=None, labels=True,
                  figsize=None, title=None, filename=None):
    """Take linkage and distance matrices and plot
    
    Args:
        link: linkage matrix
        dist_df: distance dataframe where index/columns are in the same order
                 as the input to link
        clusters: (optional) list of cluster labels created from the linkage 
                   used to parse the dendrogram heatmap
                   Assumes that clusters are contiguous along the dendrogram
        label_fontsize: int, fontsize for labels
        labels: (optional) bool, whether to show labels on heatmap
        figsize: figure size
        filename: string. If given, save to this location
    """
    row_dendr = dendrogram(link, labels=dist_df.index, no_plot = True)
    rowclust_df = dist_df.iloc[row_dendr['leaves'], row_dendr['leaves']]
    # plot
    if figsize is None:
        figsize=(16,16)
    if label_fontsize == None:
            label_fontsize = figsize[1]*.27
    sns.set_style("white")
    fig = plt.figure(figsize = figsize)
    ax1 = fig.add_axes([.16,.3,.62,.62]) 
    cax = fig.add_axes([0.21,0.25,0.5,0.02]) 
    sns.heatmap(rowclust_df, ax=ax1, xticklabels = False,
                cbar_ax=cax, 
                cbar_kws={'orientation': 'horizontal'})
    # update colorbar ticks
    cbar = ax1.collections[0].colorbar
    cbar.set_ticks([0, .5, .99])
    cbar.set_ticklabels([0, .5, ceil(dist_df.max().max())])
    cax.tick_params(labelsize=20)
    # reorient axis labels
    ax1.yaxis.tick_right()
    ax1.set_yticklabels(rowclust_df.columns[::-1], rotation=0, 
                       rotation_mode="anchor", fontsize=label_fontsize, 
                       visible=labels)
    ax1.set_xticklabels(rowclust_df.columns, rotation=-90, 
                       rotation_mode = "anchor", ha = 'left')
    ax2 = fig.add_axes([.01,.3,.15,.62])
    plt.axis('off')
    # plot dendrogram
    row_dendr = dendrogram(link, orientation='left',  ax = ax2, 
                           color_threshold=-1,
                           above_threshold_color='gray') 
    ax2.invert_yaxis()
    if title is not None:
        ax1.set_title(title, fontsize=40)
    
    # add parse lines between trees 
    if clusters is not None:
        groups = clusters[row_dendr['leaves']][::-1]
        cuts = []
        curr = groups[0]
        for i,label in enumerate(groups[1:]):
            if label!=curr:
                cuts.append(i+1)
                curr=label
        
        for ax, color in [(ax1, 'w')]:
            y_min, y_max = ax.get_ylim()
            ticks = [(tick - y_min)/(y_max - y_min) for tick in ax.get_yticks()]
            pad = (ticks[0]-ticks[1])/2
            separations = (ticks+pad)*max(y_min, y_max)
            for c in cuts:
                ax.hlines(separations[c], 0, len(rowclust_df), colors=color,
                          linestyles='dashed') 
                
    if filename:
        fig.savefig(filename, bbox_inches='tight')
    return fig

def get_dendrogram_color_fun(Z, labels, clusters, color_palette=sns.hls_palette):
    """ return the color function for a dendrogram
    
    ref: https://stackoverflow.com/questions/38153829/custom-cluster-colors-of-scipy-dendrogram-in-python-link-color-func
    Args:
        Z: linkage 
        Labels: list of labels in the order of the dendrogram. They should be
            the index of the original clustered list. I.E. [0,3,1,2] would
            be the labels list - the original list reordered to the order of the leaves
        clusters: cluster assignments for the labels in the original order
    
    """
    dflt_col = "#808080"   # Unclustered gray
    color_palette = color_palette(len(np.unique(clusters)))
    D_leaf_colors = {i: to_hex(color_palette[clusters[i]-1]) for i in labels}
    # notes:
    # * rows in Z correspond to "inverted U" links that connect clusters
    # * rows are ordered by increasing distance
    # * if the colors of the connected clusters match, use that color for link
    link_cols = {}
    for i, i12 in enumerate(Z[:,:2].astype(int)):
      c1, c2 = (link_cols[x] if x > len(Z) else D_leaf_colors[x]
        for x in i12)
      link_cols[i+1+len(Z)] = c1 if c1 == c2 else dflt_col
    
    return lambda x: link_cols[x], color_palette

def heatmap(df):
    """
    :df: plot heatmap
    """
    plt.Figure(figsize = [16,16])
    sns.set_style("white")
    fig = plt.figure(figsize = [12,12])
    ax = fig.add_axes([.1,.2,.6,.6]) 
    cax = fig.add_axes([0.02,0.3,0.02,0.4]) 
    sns.heatmap(df, ax = ax, cbar_ax = cax, xticklabels = False)
    ax.yaxis.tick_right()
    ax.set_yticklabels(df.columns[::-1], rotation=0, rotation_mode="anchor", fontsize = 'large')
    ax.set_xticklabels(df.columns, rotation=-90, rotation_mode = "anchor", ha = 'left') 
    return fig