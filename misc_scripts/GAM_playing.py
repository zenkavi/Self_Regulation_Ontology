#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:00:23 2018

@author: ian
"""
import matplotlib.pyplot as plt
import numpy as np
from os import path
import pickle
from pygam import s, f, l, LinearGAM
import seaborn as sns
from sklearn.metrics import mean_absolute_error

from selfregulation.utils.get_balanced_folds import BalancedKFold
from selfregulation.utils.plot_utils import format_num, save_figure
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_recent_dataset

# reference: https://pygam.readthedocs.io/en/latest/notebooks/tour_of_pygam.html#Terms-and-Interactions
# ********************************************************
# helper functions
# ********************************************************

def get_avg_score(scores, score_type='R2'):
    return np.mean([i[score_type] for i in scores])

def run_GAM(X, Y, n_splines=20):
    # set up GAM
    formula = s(0, n_splines)
    for i in range(1, X.shape[1]):
        formula = formula + s(i, n_splines)
    gam = LinearGAM(formula)
    gam.fit(X, X.iloc[:,0])

    GAM_results = {}
    for name, y in Y.iteritems():
        print("\nFitting for %s\n" % name)
        CV = BalancedKFold(10)
        in_scores = []
        cv_scores = []
        for train,test in CV.split(X,y):
            Xtrain = X.iloc[train,:]
            ytrain = y.iloc[train]
            Xtest = X.iloc[test,:]
            ytest = y.iloc[test]
            gam = LinearGAM(formula)
            gam.gridsearch(X, y)
            # insample
            in_pred = gam.predict(Xtrain)
            in_scores.append({'r': np.corrcoef(ytrain,in_pred)[0,1],
                              'R2': np.corrcoef(ytrain,in_pred)[0,1]**2,
                              'MAE': mean_absolute_error(ytrain,in_pred)})
            # out of fold
            pred = gam.predict(Xtest)
            cv_scores.append({'r': np.corrcoef(ytest,pred)[0,1],
                              'R2': np.corrcoef(ytest,pred)[0,1]**2,
                              'MAE': mean_absolute_error(ytest,pred)})
        gam.gridsearch(X, y)
        GAM_results[name] = {'cv_scores': cv_scores,
                              'insample_scores': in_scores,
                              'model': gam}
    return GAM_results

def plot_GAM(gams, X, Y, size=4, dpi=300, ext='png', filename=None):
    cols = X.shape[1]
    rows = Y.shape[1]
    colors = sns.color_palette(n_colors=rows)
    plt.rcParams['figure.figsize'] = (cols*size, rows*size)
    fig, mat_axs = plt.subplots(rows, cols)
    titles = X.columns
    for j, (name, out) in enumerate(gams.items()):
        axs = mat_axs[j]
        gam = out['model']
        R2 = get_avg_score(out['cv_scores'])
        p_vals = gam.statistics_['p_values']
        for i, ax in enumerate(axs):
            XX = gam.generate_X_grid(i)
            pdep, confi = gam.partial_dependence(i, X=XX, width=.95)
            ax.plot(XX[:, i], pdep, c=colors[j], lw=size)
            ax.plot(XX[:, i], confi[:, 0], c='grey', ls='--', lw=size/2)
            ax.plot(XX[:, i], confi[:, 1], c='grey', ls='--', lw=size/2)
            ax.text(.5, .95, 'p< %s' % format_num(p_vals[i]), va='center', 
                    fontsize=size*3, transform=ax.transAxes)
            if j==0:
                ax.set_title(titles[i], fontsize=size*4)
            if i==0:
                ax.set_ylabel(name + ' (%s)' % format_num(R2), 
                              fontsize=size*4)
    plt.subplots_adjust(hspace=.4)
    if filename is not None:
        save_figure(fig, '%s.%s' % (filename,ext),
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()

# ********************************************************

# ********************************************************
# Load Data
# ********************************************************
results = load_results(get_recent_dataset())
Y = results['task'].DA.get_scores()


# ********************************************************
# Fitting
# ********************************************************

GAM_results = {}
GAM_results['task'] = run_GAM(results['task'].EFA.get_scores(), Y)
GAM_results['survey'] = run_GAM(results['survey'].EFA.get_scores(), Y)

output_dir = path.dirname(results['task'].get_output_dir())
pickle.dump(GAM_results, open(path.join(output_dir, 'GAM_results.pkl'), 'wb'))
# ********************************************************
# Inspect
# ********************************************************
gams = GAM_results['task']
X = results['task'].EFA.get_scores()

for k,v in gams.items():
    print('*'*79)
    print(k)
    print('CV', get_avg_score(v['cv_scores']))
    print('Insample', get_avg_score(v['insample_scores']))
    print('*'*79)
    

# plot full matrix
plot_dir = path.dirname(results['task'].get_plot_dir())
plot_GAM(GAM_results['task'], 
         results['task'].EFA.get_scores(), 
         Y, 
         filename=path.join(plot_dir, 'task_GAM'))

plot_GAM(GAM_results['survey'], 
         results['survey'].EFA.get_scores(), 
         Y, 
         filename=path.join(plot_dir, 'survey_GAM'))