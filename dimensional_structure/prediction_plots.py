import math
import matplotlib.pyplot as plt
import numpy as np
from os import path
import pandas as pd
import seaborn as sns
from plot_utils import plot_loadings, save_figure
from utils import shorten_labels
from selfregulation.utils.plot_utils import CurvedText
sns.set_palette("Set1", 8, .75)



def visualize_importance(importance, ax, xticklabels=True, yticklabels=True, 
                         label_size=10, pad=0, label_scale=0, title=None, ymax=None):
    importance_vars = importance[0]
    if importance[1] is not None:
        importance_vals = [abs(i)+pad for i in importance[1].T]
        plot_loadings(ax, importance_vals, kind='line', offset=.5,
                      plot_kws={'alpha': 1})
    else:
        ax.set_yticks([])
    # set up x ticks
    xtick_locs = np.arange(0.0, 2*np.pi, 2*np.pi/len(importance_vars))
    ax.set_xticks(xtick_locs)
    ax.set_xticks(xtick_locs+np.pi/len(importance_vars), minor=True)
    if xticklabels:
        if type(importance_vars[0]) != str:
            importance_vars[0] = ['Fac %s' % str(i+1) for i in importance_vars]
        offset=-.1
        scale = 1+label_scale
        size = ax.get_position().expanded(scale, scale)
        ax2=ax.get_figure().add_axes(size,zorder=2)
        
        max_var_length = max([len(v) for v in importance_vars])
        for i, var in enumerate(importance_vars):
            offset=.25
            start = (i-offset)*2*np.pi/len(importance_vars)
            end = (i+(1-offset))*2*np.pi/len(importance_vars)
            curve = [
                np.cos(np.linspace(start,end,100)),
                np.sin(np.linspace(start,end,100))
            ]  
            plt.plot(*curve, alpha=1)
            # pad strings to longest length
            num_spaces = (max_var_length-len(var))
            var = ' '*(num_spaces//2) + var + ' '*(num_spaces-num_spaces//2)
            ax2.plot(*curve, alpha=0)
            curvetext = CurvedText(
                x = curve[0][::-1],
                y = curve[1][::-1],
                text=var, #'this this is a very, very long text',
                va = 'bottom',
                axes = ax2,
                fontsize=label_size*(14/len(importance_vars))##calls ax.add_artist in __init__
            )
            ax2.axis('off')
        
    if title:
        ax.set_title(title, fontsize=label_size*1.5)
    # set up yticks
    if importance[1] is not None:
        if ymax:
            ax.set_ylim(top=ymax)
        ytick_locs = ax.yaxis.get_ticklocs()
        new_yticks = np.linspace(0, ytick_locs[-1], 7)
        ax.set_yticks(new_yticks)
        if yticklabels:
            labels = np.round(new_yticks,2)
            replace_dict = {i:'' for i in labels[::2]}
            labels = [replace_dict.get(i, i) for i in labels]
            ax.set_yticklabels(labels)



def plot_prediction(results, target_order=None, include_shuffle=False, 
                    plot_heights=None,
                    figsize=(20,16),  dpi=300, plot_dir=None):
    predictions = results.load_prediction_object()['data']
    shuffled_predictions = results.load_prediction_object(shuffle=True)['data']
    
    if target_order is None:
        target_order = predictions.keys()
    # get prediction success
    r2s = [(k,predictions[k]['scores_cv'][0]) for k in target_order]
    insample_r2s = [(k, predictions[k]['scores_insample'][0]) for k in target_order]
    shuffled_r2s = [(k, shuffled_predictions[k]['scores_cv'][0]) for k in target_order]
    # plot
    fig = plt.figure(figsize=figsize)
    # plot bars
    ind = np.arange(len(r2s))
    width=.25
    ax1 = fig.add_axes([0,0,1,.5]) 
    ax1.bar(ind, [i[1] for i in r2s], width, label='CV Prediction')
    ax1.bar(ind+width, [i[1] for i in insample_r2s], width, label='Insample Prediction')
    if include_shuffle:
        ax1.bar(ind+width*2, [i[1] for i in shuffled_r2s], width, label='Shuffled Prediction')
    ax1.set_xticks(np.arange(0,len(r2s))+width/2)
    ax1.set_xticklabels([i[0] for i in r2s], rotation=15, fontsize=figsize[0]*.9)
    ax1.set_ylabel('R2', fontsize=20)
    ylim = ax1.get_ylim()
    ax1.set_ylim(ylim[0], max(.1, ylim[1]+.15))
    ax1.legend(fontsize=20, loc='upper left')
    
    # get importances
    vals = [predictions[i] for i in target_order]
    importances = [(i['predvars'], i['importances']) for i in vals]
    # plot
    axes=[]
    N = len(importances)
    #if plot_heights is None:
    ylim = ax1.get_ylim()[1]
    plot_heights = [predictions[k]['scores_insample'][0]/ylim*.5+.05 for k in target_order]
    xlow, xhigh = ax1.get_xlim()
    plot_x = (ax1.get_xticks()-xlow)/(xhigh-xlow)-(1/N/2)
    for i, importance in enumerate(importances):
        axes.append(fig.add_axes([plot_x[i],plot_heights[i], 1/N,1/N], projection='polar'))
        visualize_importance(importance, axes[i],
                             yticklabels=False, xticklabels=False)
    # plot top prediction, labeled
    top_prediction = max(enumerate(r2s), key=lambda x: x[1][1])
    label_importance = importances[top_prediction[0]]
    axes.append(fig.add_axes([.25,.5,.5,.5], projection='polar'))
    visualize_importance(label_importance, axes[-1], yticklabels=False,
                         xticklabels=True,
                         label_size=figsize[1]*.9,
                         label_scale=.2)
    
    
    if plot_dir is not None:
        filename = 'prediction_output.png'
        save_figure(fig, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})

    




