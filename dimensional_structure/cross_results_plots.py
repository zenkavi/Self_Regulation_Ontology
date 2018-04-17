#!/usr/bin/env python

# Script to generate all_results or plots across all_results objects
from itertools import combinations, product
import matplotlib.pyplot as plt
import numpy as np
from os import path
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import cross_val_score
from selfregulation.utils.plot_utils import beautify_legend, save_figure
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_recent_dataset

def extract_tril(mat, k=0):
    return mat[np.tril_indices_from(mat, k=k)]


def plot_corr_hist(all_results, reps=100, size=4.6, 
                   dpi=300, ext='png', plot_dir=None):
    colors = sns.color_palette('Blues_d',3)[0:2] + sns.color_palette('Reds_d',2)[:1]
    survey_corr = abs(all_results['survey'].data.corr())
    task_corr = abs(all_results['task'].data.corr())
    all_data = pd.concat([all_results['task'].data, all_results['survey'].data], axis=1)
    datasets = [('survey', all_results['survey'].data), 
                ('task', all_results['task'].data), 
                ('all', all_data)]
    # get cross corr
    cross_corr = abs(all_data.corr()).loc[survey_corr.columns,
                                                    task_corr.columns]
    
    plot_elements = [(extract_tril(survey_corr.values,-1), 'Within Surveys'),
                     (extract_tril(task_corr.values,-1), 'Within Tasks'),
                     (cross_corr.values.flatten(), 'Surveys x Tasks')]
    
    # get shuffled 95% correlation
    shuffled_95 = []
    for label, df in datasets:
        shuffled_corr = np.array([])
        for _ in range(reps):
            # create shuffled
            shuffled = df.copy()
            for i in shuffled:
                shuffle_vec = shuffled[i].sample(len(shuffled)).tolist()
                shuffled.loc[:,i] = shuffle_vec
            if label == 'all':
                shuffled_corr = abs(shuffled.corr()).loc[survey_corr.columns,
                                                    task_corr.columns]
            else:
                shuffled_corr = abs(shuffled.corr())
            np.append(shuffled_corr, extract_tril(shuffled_corr.values,-1))
        shuffled_95.append(np.percentile(shuffled_corr,95))
    
    # get cross_validated r2
    average_r2 = {}
    for (slabel, source), (tlabel, target) in product(datasets[:-1], repeat=2):
        scores = []
        for var, values in target.iteritems():
            if var in source.columns:
                predictors = source.drop(var, axis=1)
            else:
                predictors = source
            lr = RidgeCV()  
            cv_score = np.mean(cross_val_score(lr, predictors, values, cv=10))
            scores.append(cv_score)
        average_r2[(slabel, tlabel)] = np.mean(scores)

                
    # bring everything together
    plot_elements = [(extract_tril(survey_corr.values,-1), 'Within Surveys', 
                      average_r2[('survey','survey')]),
                     (extract_tril(task_corr.values,-1), 'Within Tasks',
                      average_r2[('task','task')]),
                     (cross_corr.values.flatten(), 'Surveys x Tasks',
                      average_r2[('survey', 'task')])]
    
    with sns.axes_style('white'):
        f, axes = plt.subplots(1,3, figsize=(10,4))
        plt.subplots_adjust(wspace=.3)
        for i, (corr, label, r2) in enumerate(plot_elements):
            #h = axes[i].hist(corr, normed=True, color=colors[i], 
            #         bins=12, label=label, rwidth=1, alpha=.4)
            sns.kdeplot(corr, ax=axes[i], color=colors[i], shade=True,
                        label=label, linewidth=3)
            axes[i].text(.4, axes[i].get_ylim()[1]*.5, 'CV-R2: {0:.2f}'.format(r2))
        for i, ax in enumerate(axes):
            ax.vlines(shuffled_95[i], *ax.get_ylim(), color=[.2,.2,.2], 
                      linewidth=2, linestyle='dashed', zorder=10)
            ax.set_xlim(0,1)
            ax.set_ylim(0, ax.get_ylim()[1])
            ax.set_xticks([0,.5,1])
            ax.set_xticklabels([0,.5,1], fontsize=16)
            ax.set_yticks([])
            ax.spines['right'].set_visible(False)
            #ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            leg=ax.legend(fontsize=14, loc='upper center')
            beautify_legend(leg, [colors[i]])
        axes[1].set_xlabel('Pearson Correlation', fontsize=20, labelpad=10)
        axes[0].set_ylabel('Normalized Density', fontsize=20, labelpad=10)
    
    # save
    if plot_dir is not None:
        # make histogram plot
        save_figure(f, path.join(plot_dir, 'within-across_correlations.pdf'),
                                {'bbox_inches': 'tight', 'dpi': 300})
        
    
    


from sklearn.decomposition import PCA
def plot_EFA_relationships(all_results):
    EFA_all_results = {k:v.EFA for k,v in all_results.items()}
    scores = {k:v.get_scores() for k,v in EFA_all_results.items()}
    # quantify relationships using linear regression
    for name1, name2 in combinations(scores.keys(), 2):
        scores1 = scores[name1]
        scores2 = scores[name2]
        lr = LinearRegression()  
        cv_score = np.mean(cross_val_score(lr, scores1, scores2, cv=10))
        print(name1, name2, cv_score)
    # plot
    # plot task factors in task PCA space
    pca = PCA(2)
    task_pca = pca.fit_transform(scores['task'])
    palettes = ['Reds', 'Blues', 'Greens']
    all_colors = []
    # plot scores in task PCA space
    f, ax = plt.subplots(figsize=[12,8])
    ax.set_facecolor('white')

    for k,v in scores.items():
        palette = sns.color_palette(palettes.pop(), n_colors = len(v.columns))
        all_colors += palette
        lr = LinearRegression()
        lr.fit(task_pca, v)
        for i, coef in enumerate(lr.coef_):
            plt.plot([0,coef[0]], [0, coef[1]], linewidth=3, 
                     c=palette[i], label=k+'_'+str(v.columns[i]))
    leg = plt.legend(bbox_to_anchor=(.8, .5))
    frame = leg.get_frame()
    frame.set_color('black')
    beautify_legend(leg, all_colors)

def plot_BIC(all_results, size=4.6, dpi=300, ext='png', plot_dir=None):
    """ Plots BIC and SABIC curves
    
    Args:
        all_results: a dimensional structure all_results object
        dpi: the final dpi for the image
        ext: the extension for the saved figure
        plot_dir: the directory to save the figure. If none, do not save
    """
    all_colors = [sns.color_palette('Blues_d',3)[0:3],
              sns.color_palette('Reds_d',3)[0:3],
              sns.color_palette('Greens_d',3)[0:3],
              sns.color_palette('Oranges_d',3)[0:3]]
    height= size*.75/len(all_results)
    fig, axes = plt.subplots(1, len(all_results), figsize=(size, height))
    for i, results in enumerate(all_results.values()):
        ax1 = axes[i]
        name = results.ID.split('_')[0].title()
        EFA = results.EFA
        # Plot BIC and SABIC curves
        colors = all_colors[i]
        with sns.axes_style('white'):
            x = list(EFA.results['cscores_metric-BIC'].keys())
            # score keys
            keys = [k for k in EFA.results.keys() if 'cscores' in k]
            for key in keys:
                metric = key.split('-')[-1]
                BIC_scores = [EFA.results[key][i] for i in x]
                BIC_c = EFA.results['c_metric-%s' % metric]
                ax1.plot(x, BIC_scores,  'o-', c=colors[0], lw=size/6, label=metric,
                         markersize=height*2)
                ax1.plot(BIC_c, BIC_scores[BIC_c-1], '.', color='white',
                         markeredgecolor=colors[0], markeredgewidth=height/2, 
                         markersize=height*4)
            if i==0:
                ax1.set_ylabel('Score', fontsize=height*3)
                leg = ax1.legend(loc='center right',
                                 fontsize=height*3, markerscale=0)
                beautify_legend(leg, colors=colors)
            ax1.set_xlabel('# Factors', fontsize=height*3)
            ax1.tick_params(labelsize=height*2)
            ax1.set_title(name, fontsize=height*4)
    if plot_dir is not None:
        save_figure(fig, path.join(plot_dir, 'BIC_SABIC_curves.%s' % ext),
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
            


