import matplotlib.pyplot as plt
import numpy as np
from os import path
import pandas as pd
import seaborn as sns
from utils import plot_loadings, save_figure
sns.set_palette("Set1", 8, .75)

def visualize_importance(importance, ax, xticklabels=True, 
                           yticklabels=True, pad=0, ymax=None, legend=True):
    """Plot task loadings on one axis"""
    importance_vars = importance[0]
    importance_vals = [abs(i)+pad for i in importance[1].T]
    plot_loadings(ax, importance_vals, kind='line', offset=.5,
                  plot_kws={'alpha': 1})
    # set up x ticks
    xtick_locs = np.arange(0.0, 2*np.pi, 2*np.pi/len(importance_vars))
    ax.set_xticks(xtick_locs)
    ax.set_xticks(xtick_locs+np.pi/len(importance_vars), minor=True)
    if xticklabels:
        if type(importance_vars[0]) == str:
            ax.set_xticklabels(importance_vars, 
                               y=.08, minor=True)
        else:
            ax.set_xticklabels(['Fac %s' % str(i+1) for i in importance_vars], 
                               y=.08, minor=True)
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


def plot_prediction(results, plot_dir=None):
    predictions = results.load_prediction_object()['data']
    shuffled_predictions = results.load_prediction_object(shuffle=True)['data']
    # get prediction success
    r2s = [(k,v['scores_cv'][0]) for k,v in predictions.items()]
    insample_r2s = [(k,v['scores_insample'][0]) for k,v in predictions.items()]
    shuffled_r2s = [(k,v['scores_cv'][0]) for k,v in shuffled_predictions.items()]
    # get importances
    importances = [(i['predvars'], i['importances']) for i in predictions.values()]
    # plot
    fig = plt.figure(figsize=(20,12))
    # plot bars
    ind = np.arange(len(r2s))
    width=.25
    ax1 = fig.add_axes([0,0,1,.6]) 
    ax1.bar(ind, [i[1] for i in r2s], width, label='Prediction')
    ax1.bar(ind+width, [i[1] for i in insample_r2s], width, label='Insample Prediction')
    ax1.bar(ind+width*2, [i[1] for i in shuffled_r2s], width, label='Shuffled Prediction')
    ax1.set_xticks(range(len(r2s)))
    ax1.set_xticklabels([i[0] for i in r2s], rotation=0, fontsize=20,y=-.05)
    ax1.set_ylabel('R2', fontsize=20)
    ax1.legend(fontsize=20)
    
    axes=[]
    for i, importance in enumerate(importances):
        if i!=0:
            axes.append(fig.add_axes([.05+.2*i,.6,.1,.4], projection='polar'))
            visualize_importance(importance, axes[i],
                                 yticklabels=False, xticklabels=False)
        elif i==0:
            axes.append(fig.add_axes([0,.63,.2,.4], projection='polar'))
            visualize_importance(importance, axes[i],
                                 yticklabels=False)
    if plot_dir is not None:
        filename = 'prediction_output.png'
        save_figure(fig, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight'})

    




