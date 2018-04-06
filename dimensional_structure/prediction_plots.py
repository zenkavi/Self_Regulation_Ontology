import matplotlib.pyplot as plt
import numpy as np
from os import path
import pandas as pd
import pickle
import seaborn as sns
from dimensional_structure.plot_utils import get_short_names, plot_loadings, save_figure
from selfregulation.utils.plot_utils import beautify_legend, CurvedText

colors = sns.color_palette('Blues_d',3) + sns.color_palette('Reds_d',2)[:1]


shortened_factors = get_short_names()

def visualize_importance(importance, ax, xticklabels=True, yticklabels=True, 
                         label_size=10, pad=0, label_scale=0, title=None, 
                         ymax=None, color=colors[0]):
    importance_vars = importance[0]
    importance_vars = [shortened_factors.get(v,v) for v in importance_vars]
    if importance[1] is not None:
        importance_vals = [abs(i)+pad for i in importance[1].T]
        plot_loadings(ax, importance_vals, kind='line', offset=.5, 
                      colors=[color], plot_kws={'alpha': 1,})
    else:
        ax.set_yticks([])
    # set up x ticks
    xtick_locs = np.arange(0.0, 2*np.pi, 2*np.pi/len(importance_vars))
    ax.set_xticks(xtick_locs)
    ax.set_xticks(xtick_locs+np.pi/len(importance_vars), minor=True)
    if xticklabels:
        if type(importance_vars[0]) != str:
            importance_vars = ['Fac %s' % str(i+1) for i in importance_vars]
        scale = 1+label_scale
        size = ax.get_position().expanded(scale, scale)
        ax2=ax.get_figure().add_axes(size,zorder=2)
        max_var_length = max([len(v) for v in importance_vars])
        for i, var in enumerate(importance_vars):
            offset=.3*25/len(importance_vars)**2
            start = (i-offset)*2*np.pi/len(importance_vars)
            end = (i+(1-offset))*2*np.pi/len(importance_vars)
            curve = [
                np.cos(np.linspace(start,end,100)),
                np.sin(np.linspace(start,end,100))
            ]  
            plt.plot(*curve, alpha=0)
            # pad strings to longest length
            num_spaces = (max_var_length-len(var))
            var = ' '*(num_spaces//2) + var + ' '*(num_spaces-num_spaces//2)
            curvetext = CurvedText(
                x = curve[0][::-1],
                y = curve[1][::-1],
                text=var, #'this this is a very, very long text',
                va = 'top',
                axes = ax2,
                fontsize=label_size##calls ax.add_artist in __init__
            )
            ax2.axis('off')
        
    if title:
        ax.set_title(title, fontsize=label_size*1.5, y=1.1)
    # set up yticks
    if importance[1] is not None:
        ax.set_ylim(bottom=0)
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

def plot_prediction(results, target_order=None, EFA=True, classifier='lasso',
                    include_shuffle=False,  ymax=None, figsize=(20,16),  
                    dpi=300, plot_dir=None):
    predictions = results.load_prediction_object(EFA=EFA, classifier=classifier)
    if predictions is None:
        print('No prediction object found!')
        return
    else:
        predictions = predictions['data']
    shuffled_predictions = results.load_prediction_object(EFA=EFA, classifier=classifier, shuffle=True)['data']
    
    if target_order is None:
        target_order = predictions.keys()
    # get prediction success
    r2s = [[k,predictions[k]['scores_cv'][0]] for k in target_order]
    insample_r2s = [[k, predictions[k]['scores_insample'][0]] for k in target_order]
    shuffled_r2s = [[k, shuffled_predictions[k]['scores_cv'][0]] for k in target_order]
    # convert nans to 0
    r2s = [(i, k) if k==k else (i,0) for i, k in r2s]
    insample_r2s = [(i, k) if k==k else (i,0) for i, k in insample_r2s]
    shuffled_r2s = [(i, k) if k==k else (i,0) for i, k in shuffled_r2s]
    # plot
    fig = plt.figure(figsize=figsize)
    # plot bars
    ind = np.arange(len(r2s))
    width=.25
    ax1 = fig.add_axes([0,0,1,.5]) 
    ax1.bar(ind, [i[1] for i in r2s], width, 
            label='Cross-Validated Prediction', color=colors[0])
    ax1.bar(ind+width, [i[1] for i in insample_r2s], width, 
            label='Insample Prediction', color=colors[1])
    if include_shuffle:
        ax1.bar(ind+width*2, [i[1] for i in shuffled_r2s], width, 
                label='Shuffled Prediction', color=colors[2])
        ax1.set_xticks(np.arange(0,len(r2s))+width)
    else:
        ax1.set_xticks(np.arange(0,len(r2s))+width/2)
        tick_locs = ax1.get_xticks()
        xmin, xmax = ax1.get_xlim()
        xrange = xmax-xmin
        for i, (name, val) in enumerate(shuffled_r2s):
            ax1.axhline(val, 
                        (tick_locs[i]-width+abs(xmin))/xrange, 
                        (tick_locs[i]+width+abs(xmin))/xrange,
                        color='w',
                        linestyle='--')
    ax1.set_xticklabels([i[0] for i in r2s], rotation=15, fontsize=figsize[0]*1.15)
    ax1.set_ylabel('R2', fontsize=30, labelpad=10)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.tick_params(length=5, width=2)
    ylim = ax1.get_ylim()
    if ymax is None:
        ymax = max(.15, ylim[1]+.1)
    ax1.set_ylim(ylim[0], ymax)
    leg = ax1.legend(fontsize=24, loc='upper left')
    beautify_legend(leg, colors[:3])

    # draw grid
    ax1.set_axisbelow(True)
    plt.grid(axis='y', linestyle='dotted')
    # Plot Polar Plots for importances
    if EFA == True:
        # get importances
        vals = [predictions[i] for i in target_order]
        importances = [(i['predvars'], i['importances']) for i in vals]
        # plot
        axes=[]
        N = len(importances)
        best_predictors = sorted(enumerate(r2s), key = lambda x: x[1][1])
        #if plot_heights is None:
        ylim = ax1.get_ylim()[1]
        plot_heights = [max(predictions[k]['scores_cv'][0], 
                            predictions[k]['scores_insample'][0])/ylim*.5+.018 
                        for k in target_order]
        xlow, xhigh = ax1.get_xlim()
        plot_x = (ax1.get_xticks()-xlow)/(xhigh-xlow)-(1/N/2)
        for i, importance in enumerate(importances):
            axes.append(fig.add_axes([plot_x[i], plot_heights[i], 1/N,1/N], projection='polar'))
            if i in [best_predictors[-1][0], best_predictors[-2][0]]:
                color = colors[3]
            else:
                color = colors[0]
            visualize_importance(importance, axes[i],
                                 yticklabels=False, xticklabels=False,
                                 color=color)
        # plot top 2 predictions, labeled  
        if best_predictors[-1][0] < best_predictors[-2][0]:
            locs = [.25, .75]
        else:
            locs = [.75, .25]
        label_importance = importances[best_predictors[-1][0]]
        ratio = figsize[1]/figsize[0]
        axes.append(fig.add_axes([locs[0]-.2*ratio,.56,.4*ratio,.4], projection='polar'))
        visualize_importance(label_importance, axes[-1], yticklabels=False,
                             xticklabels=True,
                             label_size=figsize[1]*1.5,
                             label_scale=.22,
                             title=best_predictors[-1][1][0],
                             color=colors[3])
        # 2nd top
        label_importance = importances[best_predictors[-2][0]]
        ratio = figsize[1]/figsize[0]
        axes.append(fig.add_axes([locs[1]-.2*ratio,.56,.4*ratio,.4], projection='polar'))
        visualize_importance(label_importance, axes[-1], yticklabels=False,
                             xticklabels=True,
                             label_size=figsize[1]*1.5,
                             label_scale=.23,
                             title=best_predictors[-2][1][0],
                             color=colors[3])
    if plot_dir is not None:
        if EFA:
            filename = 'EFA_%s_prediction_output.png' % classifier
        else:
            filename = 'IDM_%s_prediction_output.png' % classifier
        save_figure(fig, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()

    

def plot_prediction_comparison(results, figsize=(14,8), dpi=300, plot_dir=None):
    R2s = {}
    for EFA in [False, True]:
        predictions = results.get_prediction_files(EFA=EFA, shuffle=False)
        for filey in predictions:
            feature = 'EFA' if EFA else 'IDM'
            prediction_object = pickle.load(open(filey, 'rb'))
            name = prediction_object['info']['classifier']
            R2 = [i['scores_cv'][0] for i in prediction_object['data'].values()]
            R2 = np.nan_to_num(R2)
            R2s[feature+'_'+name] = R2
        

    R2s = pd.DataFrame(R2s).melt(var_name='Classifier', value_name='R2')
    R2s['Feature'], R2s['Classifier'] = R2s.Classifier.str.split('_', 1).str
    f = plt.figure(figsize=figsize)
    sns.barplot(x='Classifier', y='R2', data=R2s, hue='Feature',
                palette=colors[:2])
    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='x', labelsize=18)
    leg = ax.legend(fontsize=24, loc='upper right')
    beautify_legend(leg, colors[:2])
    plt.xlabel('Classifier', fontsize=20, labelpad=10)
    plt.ylabel('R2', fontsize=20, labelpad=10)
    plt.title('Comparison of Prediction Methods', fontsize=24, y=1.05)
    
    if plot_dir is not None:
        filename = 'prediction_comparison'
        save_figure(f, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
    


