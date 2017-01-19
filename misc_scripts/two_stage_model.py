#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 20:35:26 2017

@author: ian
"""
import numpy
from math import exp
from selfregulation.utils.utils import get_behav_data

class Two_Stage_Model(object):
    def __init__(self,alpha1,alpha2,lam,B1,B2,W,p):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.lam = lam
        self.B1= B1
        self.B2 = B2
        self.W = W
        self.p = p
        # stage action possibilities
        self.stage_action_list = {0: (0,1), 1: (2,3), 2: (4,5)}
        # transition counts
        self.transition_counts = {(0,1):0, (0,2):0, (1,1):0, (1,2):0}
        # initialize Q values
        self.Q_TD_values = numpy.ones((3,6))*.5
        self.Q_MB_values = numpy.ones((3,6))*.5
        self.sum_neg_ll = None

    def updateQTD(self,r,s1,a1,s2=None,a2=None,alpha=.05):
        if s2 == None:
            delta = r - self.Q_TD_values[s1,a1]
        else:
            delta = r + self.Q_TD_values[s2,a2] - self.Q_TD_values[s1,a1]
        self.Q_TD_values[s1,a1] += alpha*delta
        return delta
    
    def updateQMB(self,T):
        self.Q_MB_values[1:3,:] = self.Q_TD_values[1:3,:]
        for a in self.stage_action_list[0]:
            self.Q_MB_values[0,a] = T[(a,1)] * numpy.max(self.Q_TD_values[1,2:3]) + \
                                T[(a,2)] * numpy.max(self.Q_TD_values[2,4:5])
        
    def trialUpdate(self,trial,alpha1,alpha2,lam):
        s1 = int(trial.stage); s2 = int(trial.stage_second)
        a1 = int(trial.stim_selected_first); a2 = int(trial.stim_selected_second)
        self.transition_counts[(a1,s2)] += 1
        delta1 = self.updateQTD(0,s1, a1, s2, a2, alpha1)
        delta2 = self.updateQTD(trial.feedback, s2, a2, alpha=alpha2)
        print(s2, a2, delta2)
        self.Q_TD_values[(s1, a1)] += alpha1*lam*delta2
    
    def get_choices(self,trial,last_choice):
        s1 = int(trial.stage); s2 = int(trial.stage_second)
        a1 = int(trial.stim_selected_first); a2 = int(trial.stim_selected_second)
        W = self.W
        # stage one and two choices
        P_action = numpy.zeros(2)
        # choice probabilities
        choice_probabilities = []
        for stage in [s1,s2]:
            for a in self.stage_action[stage]:
                Qnet = (W)*self.Q_MB_values[s1,a] + (1-W)*self.Q_TD_values[s1,a]
                repeat = (self.p*(a==last_choice))
                P_action[a] = exp(self.B1*(Qnet+repeat))
            P_action/=numpy.sum(P_action)
            choice_probabilities.append(P_action)

        return choice_probabilities[0][a1], choice_probabilities[1][self.stage_action[s2].index(a2)]
    
    def run_trials(self,df):
        # run trials
        last_choice = -1
        action_probs = []
        Q_vals = []
        MB_vals = []
        for i, trial in df.iterrows():
            Q_vals.append(self.Q_TD_values.copy())
            MB_vals.append(self.Q_MB_values.copy())
            Pa1, Pa2 = self.get_choices(trial, last_choice)
            action_probs.append((Pa1,Pa2))
            self.trialUpdate(trial,self.alpha1,self.alpha2,self.lam)
            # define T:
            if (self.transition_counts[(0,1)]+self.transition_counts[(1,2)]) > \
                (self.transition_counts[(0,2)]+self.transition_counts[(1,1)]):
                T = {(0,1):.7, (0,2):.3, (1,1):.3, (1,2):.7}
            else: 
                T = {(0,1):.3, (0,2):.7, (1,1):.7, (1,2):.3}
            self.updateQMB(T)
            last_choice = trial.stim_selected_first
        self.sum_neg_ll = numpy.sum(-numpy.log(list(zip(*action_probs))[0])) + numpy.sum(-numpy.log(list(zip(*action_probs))[1]))   
        return Q_vals, MB_vals
        
    def get_neg_ll(self):
        return self.sum_neg_ll
        
        
        
def get_likelihood(params, df):
        # set initial parameters
        alpha1 = params['alpha1']
        alpha2 = params['alpha2']
        lam = params['lam']
        B1 = params['B1']
        B2 = params['B2']
        W = params['W']
        p = params['p']
        model = Two_Stage_Model(alpha1,alpha2,lam,B1,B2,W,p)
        model.run_trials(df)
        return model.get_neg_ll
        
def fit_decision_model(df):
    import lmfit
    fit_params = lmfit.Parameters()
    fit_params.add('alpha1', value=.5, min=0, max=1)
    fit_params.add('alpha2', value=.5, min=0, max=1)
    fit_params.add('lam', value = .5, min=0, max=1)
    fit_params.add('W', value = .5, min=0, max=1)
    fit_params.add('p', value = 0)
    fit_params.add('B1', value = 3)
    fit_params.add('B2', value = 3)
    
    out = lmfit.minimize(get_likelihood, fit_params, method = 'lbfgsb', kws={'df': df})
    lmfit.report_fit(out)
    return out.params.valuesdict()