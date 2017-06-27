import matplotlib.pyplot as plt
from os import path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
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


def dendroheatmap(df, labels = True, label_fontsize = None):
    """
    plot hierarchical clustering and heatmap
    :df: a correlation matrix
    """
    # ensure this is a similarity matrix of some kind
    assert (df.shape[0]==df.shape[1] and df.iloc[0,1]==df.iloc[1,0]), \
            "df must be a correlation matrix"
    #clustering
    corr_vec = 1-df.values[np.triu_indices_from(df,k=1)]
    row_clusters = linkage(corr_vec, method='ward', metric='euclidean')    
    #dendrogram
    row_dendr = dendrogram(row_clusters, labels=df.columns, no_plot = True)
    df_rowclust = df.ix[row_dendr['leaves'],row_dendr['leaves']]
    #plotting
    if label_fontsize == None:
        label_fontsize = len(df_rowclust)/22
    sns.set_style("white")
    fig = plt.figure(figsize = [16,16])
    ax = fig.add_axes([.1,.2,.6,.6]) 
    cax = fig.add_axes([0.02,0.3,0.02,0.4]) 
    sns.heatmap(df_rowclust, ax = ax, cbar_ax = cax, xticklabels = False)
    ax.yaxis.tick_right()
    ax.set_yticklabels(df_rowclust.columns[::-1], rotation=0, rotation_mode="anchor", fontsize = label_fontsize, visible = labels)
    ax.set_xticklabels(df_rowclust.columns, rotation=-90, rotation_mode = "anchor", ha = 'left')
    ax1 = fig.add_axes([.1,.8,.6,.2])
    plt.axis('off')
    row_dendr = dendrogram(row_clusters, orientation='top',  
                           count_sort='ascending', ax = ax1) 
    return fig, row_dendr['leaves']

def dendroheatmap_left(df, labels = True, label_fontsize = 'large'):
    """
    :df: plot hierarchical clustering and heatmap, dendrogram on left
    """
    # ensure this is a similarity matrix of some kind
    assert (df.shape[0]==df.shape[1] and df.iloc[0,1]==df.iloc[1,0]), \
            "df must be a correlation matrix"
    #clustering
    corr_vec = 1-df.values[np.triu_indices_from(df,k=1)]
    row_clusters = linkage(corr_vec, method='ward', metric='euclidean')   
    #dendrogram
    row_dendr = dendrogram(row_clusters, labels=df.columns, no_plot = True)
    df_rowclust = df.ix[row_dendr['leaves'],row_dendr['leaves']]
    sns.set_style("white")
    fig = plt.figure(figsize = [16,16])
    ax = fig.add_axes([.16,.3,.62,.62]) 
    cax = fig.add_axes([0.21,0.25,0.5,0.02]) 
    sns.heatmap(df_rowclust, ax = ax, cbar_ax = cax, cbar_kws = {'orientation': 'horizontal'}, xticklabels = False)
    ax.yaxis.tick_right()
    ax.set_yticklabels(df_rowclust.columns[::-1], rotation=0, rotation_mode="anchor", fontsize = label_fontsize, visible = labels)
    ax.set_xticklabels(df_rowclust.columns, rotation=-90, rotation_mode = "anchor", ha = 'left')
    ax1 = fig.add_axes([.01,.3,.15,.62])
    plt.axis('off')
    row_dendr = dendrogram(row_clusters, orientation='left',  
                           count_sort='descending', ax = ax1) 
    return fig, row_dendr['leaves']
    
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