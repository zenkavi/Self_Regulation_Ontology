#!/usr/bin/env python

# Script to generate results or plots across results objects

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from selfregulation.utils.plot_utils import beautify_legend


def extract_tril(mat, k=0):
    return mat[np.tril_indices_from(mat, k=k)]


def plot_corr_hist(results, colors, reps=100):
    survey_corr = abs(results['survey'].data.corr())
    task_corr = abs(results['task'].data.corr())
    all_data = pd.concat([results['task'].data, results['survey'].data], axis=1)
    cross_corr = abs(all_data.corr()).loc[survey_corr.columns,
                                                    task_corr.columns]
    
    plot_elements = [(extract_tril(survey_corr.values,-1), 'Within Surveys'),
                     (extract_tril(task_corr.values,-1), 'Within Tasks'),
                     (cross_corr.values.flatten(), 'Surveys x Tasks')]
    
    shuffled_95 = []
    for df in [results[key].data for key in ['survey','task','all']]:
        shuffled_corr = np.array([])
        for _ in range(reps):
            # create shuffled
            shuffled = df.copy()
            for i in shuffled:
                shuffle_vec = shuffled[i].sample(len(shuffled)).tolist()
                shuffled.loc[:,i] = shuffle_vec
            shuffled_corr = abs(shuffled.corr())
            np.append(shuffled_corr, extract_tril(shuffled_corr.values,-1))
        shuffled_95.append(np.percentile(shuffled_corr,95))
    
    with sns.axes_style('white'):
        f, axes = plt.subplots(1,3, figsize=(10,4))
        plt.subplots_adjust(wspace=.3)
        for i, (corr, label) in enumerate(plot_elements):
            #h = axes[i].hist(corr, normed=True, color=colors[i], 
            #         bins=12, label=label, rwidth=1, alpha=.4)
            sns.kdeplot(corr, ax=axes[i], color=colors[i], shade=True,
                        label=label, linewidth=3)
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
        axes[1].set_xlabel('Pearson Correlation', fontsize=20, y=-.05)
    return f
    
    



    
