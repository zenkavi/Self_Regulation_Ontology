#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:00:23 2018

@author: ian
"""
import matplotlib.pyplot as plt
import numpy as np
from pygam import s, f, LinearGAM
from selfregulation.utils.get_balanced_folds import BalancedKFold
from sklearn.metrics import mean_absolute_error

X = results['survey'].EFA.get_scores()
Y = results['survey'].DA.get_scores()
y = Y.iloc[:, 4]


n_splines = 20
formula = s(0, n_splines)
for i in range(1, X.shape[1]):
    formula = formula + s(i, n_splines)
gam = LinearGAM(formula)
gam.fit(X, y)



CV = BalancedKFold(10)
in_scores = []
cv_scores = []
for train,test in CV.split(X,y):
    Xtrain = X.iloc[train,:]
    ytrain = y.iloc[train]
    Xtest = X.iloc[test,:]
    ytest = y.iloc[test]
    gam = LinearGAM(formula)
    gam.gridsearch(Xtrain, ytrain)
    gam.fit(Xtrain,ytrain)
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

def get_avg_score(scores, score_type='R2'):
    return np.mean([i[score_type] for i in scores])
    
    

# plot full data predictions
gam.fit(X,y)
gam.summary()
plt.rcParams['figure.figsize'] = (28, 8)
fig, axs = plt.subplots(1, X.shape[1])
titles = X.columns
for i, ax in enumerate(axs):
    XX = gam.generate_X_grid(i)
    pdep, confi = gam.partial_dependence(i, X=XX, width=.95)
    ax.plot(XX[:, i], pdep)
    ax.plot(XX[:, i], confi[:, 0], c='grey', ls='--')
    ax.plot(XX[:, i], confi[:, 1], c='grey', ls='--')
    ax.set_title(titles[i])
plt.show()



i=4
XX = gam.generate_X_grid(term=i, n=500)
f = plt.figure(figsize=(8,6))
plt.plot(XX, gam.predict(XX), 'r--')
conf = gam.prediction_intervals(XX, width=.95)
plt.plot(XX, conf[:,0], color='b', ls='--')
plt.plot(XX, conf[:,1], color='b', ls='--')

plt.scatter(X.iloc[:,i], y, facecolor='gray', edgecolors='none')
plt.title('95% prediction interval');


