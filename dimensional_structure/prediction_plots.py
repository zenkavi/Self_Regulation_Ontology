import matplotlib.pyplot as plt
import numpy as np
from os import path
import pandas as pd
import seaborn as sns
from plot_utils import plot_loadings, save_figure
from utils import shorten_labels
sns.set_palette("Set1", 8, .75)

def visualize_importance(importance, ax, xticklabels=True, yticklabels=True, 
                         label_size=10, pad=0, ymax=None, legend=False):
    """Plot task loadings on one axis"""
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
        if type(importance_vars[0]) == str:
            ax.set_xticklabels(importance_vars, y=.5)
        else:
            ax.set_xticklabels(['Fac %s' % str(i+1) for i in importance_vars], 
                               y=-.2, minor=True)
        plt.gcf().canvas.draw()
        N = len(ax.get_xticklabels())
        angles = np.linspace(0,2*np.pi,N+1) 
        angles[np.cos(angles) < 0] = angles[np.cos(angles) < 0] + np.pi
        angles = np.rad2deg(angles) + 360/N/2
        labels = []
        for label, angle in zip(ax.get_xticklabels(), angles):
            x,y = label.get_position()
            x+= xtick_locs[1]/2
            lab = ax.text(x,y, label.get_text(), transform=label.get_transform(),
                          ha=label.get_ha(), va=label.get_va(), size=label_size)
            lab.set_rotation(angle)
            labels.append(lab)
        ax.set_xticklabels([])

    # set up yticks
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
    if legend:
        ax.legend(loc='upper center', bbox_to_anchor=(.5,-.15))


def plot_prediction(results, target_order=None, include_shuffle=False,
                    figsize=(20,16),  plot_dir=None):
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
    ax1.set_xticks(range(len(r2s)))
    ax1.set_xticklabels([i[0] for i in r2s], rotation=15, fontsize=figsize[0]*.9)
    ax1.set_ylabel('R2', fontsize=20)
    ax1.legend(fontsize=20)
    
    # get importances
    vals = [predictions[i] for i in target_order]
    importances = [(i['predvars'], i['importances']) for i in vals]
    # plot
    axes=[]
    N = len(importances)
    for i, importance in enumerate(importances):
        axes.append(fig.add_axes([1/N*i,.53, 1/N,1/N], projection='polar'))
        visualize_importance(importance, axes[i],
                             yticklabels=False, xticklabels=False)
    axes.append(fig.add_axes([.33, .65, .34, .34], projection='polar'))
    label_importance = list(importance)
    label_importance[1] = None
    visualize_importance(label_importance, axes[-1], yticklabels=False,
                         label_size=figsize[1]*.9)
    
    
    if plot_dir is not None:
        filename = 'prediction_output.png'
        save_figure(fig, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight'})

    




