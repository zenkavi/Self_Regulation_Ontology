from dynamicTreeCut import cutreeHybrid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import squareform
import seaborn as sns

#***************************************************
# ********* Plotting Functions **********************
#**************************************************

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
                  figsize=None, filename=None):
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
    ax = fig.add_axes([.16,.3,.62,.62]) 
    cax = fig.add_axes([0.21,0.25,0.5,0.02]) 
    sns.heatmap(rowclust_df, ax=ax, xticklabels = False,
                cbar_ax=cax, 
                cbar_kws={'orientation': 'horizontal'})
    ax.yaxis.tick_right()
    ax.set_yticklabels(rowclust_df.columns[::-1], rotation=0, 
                       rotation_mode="anchor", fontsize=label_fontsize, 
                       visible=labels)
    ax.set_xticklabels(rowclust_df.columns, rotation=-90, 
                       rotation_mode = "anchor", ha = 'left')
    ax1 = fig.add_axes([.01,.3,.15,.62])
    plt.axis('off')
    row_dendr = dendrogram(link, orientation='left',  ax = ax1, 
                           color_threshold=-1,
                           above_threshold_color='gray') 
    ax1.invert_yaxis()
    
    # add parse lines between trees 
    if clusters is not None:
        groups = clusters[row_dendr['leaves']][::-1]
        cuts = []
        curr = groups[0]
        for i,label in enumerate(groups[1:]):
            if label!=curr:
                cuts.append(i+1)
                curr=label
        y_min, y_max = ax.get_ylim()
        ticks = [(tick - y_min)/(y_max - y_min) for tick in ax.get_yticks()]
        pad = (ticks[0]-ticks[1])/2
        separations = (ticks+pad)*len(rowclust_df)
        for c in cuts:
            ax.hlines(separations[c], 0, len(rowclust_df), colors='w') 
    if filename:
        fig.savefig(filename, bbox_inches='tight')
    return fig
    
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