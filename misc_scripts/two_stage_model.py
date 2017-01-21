#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 20:35:26 2017

@author: ian
"""
import lmfit
import numpy
from math import exp
import pandas as pd
from selfregulation.utils.utils import get_behav_data
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm

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
        self.Q_TD_values = numpy.ones((3,6))*0
        self.Q_MB_values = numpy.ones((3,6))*0
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
            self.Q_MB_values[0,a] = T[(a,1)] * numpy.max(self.Q_TD_values[1,2:4]) + \
                                T[(a,2)] * numpy.max(self.Q_TD_values[2,4:6])
        
    def trialUpdate(self,s1,s2,a1,a2,r,alpha1,alpha2,lam):
        # update TD values
        delta1 = self.updateQTD(0,s1, a1, s2, a2, alpha1)
        delta2 = self.updateQTD(r, s2, a2, alpha=alpha2)
        self.Q_TD_values[(s1, a1)] += alpha1*lam*delta2
        # update MB values
        self.transition_counts[(a1,s2)] += 1
        # define T:
        if (self.transition_counts[(0,1)]+self.transition_counts[(1,2)]) > \
            (self.transition_counts[(0,2)]+self.transition_counts[(1,1)]):
            T = {(0,1):.7, (0,2):.3, (1,1):.3, (1,2):.7}
        else: 
            T = {(0,1):.3, (0,2):.7, (1,1):.7, (1,2):.3}
        self.updateQMB(T)
    
    def get_softmax_probs(self,stages,last_choice):
        W = self.W
        if type(stages) != list:
            stages = [stages]
        # stage one and two choices
        P_action = numpy.zeros(2)
        # choice probabilities
        choice_probabilities = []
        for stage in stages:
            for i,a in enumerate(self.stage_action_list[stage]):
                Qnet = (W)*self.Q_MB_values[stage,a] + (1-W)*self.Q_TD_values[stage,a]
                repeat = (self.p*(a==last_choice))
                P_action[i] = exp(self.B1*(Qnet+repeat))
            P_action/=numpy.sum(P_action)
            choice_probabilities.append(P_action.copy())
        return choice_probabilities
    
    def run_trial(self,trial,last_choice):
        s1 = int(trial['stage']); s2 = int(trial['stage_second'])
        a1 = int(trial['stim_selected_first']); a2 = int(trial['stim_selected_second'])
        r = int(trial['feedback'])
        # return probability of all actions
        probs1, probs2 = self.get_softmax_probs([s1,s2],last_choice)
        # get probability of selected actions
        Pa1 = probs1[a1]
        Pa2 = probs2[self.stage_action_list[s2].index(a2)]
        self.trialUpdate(s1,s2,a1,a2,r,self.alpha1,self.alpha2,self.lam)
        return Pa1,Pa2
        
    def run_trials(self, df):
        # run trials
        last_choice = -1
        action_probs = []
        Q_vals = []
        MB_vals = []
        for i,trial in df.iterrows():
            Q_vals.append(self.Q_TD_values.copy())
            MB_vals.append(self.Q_MB_values.copy())
            Pa1, Pa2 = self.run_trial(trial,last_choice)
            action_probs.append((Pa1,Pa2))
            last_choice = trial['stim_selected_first']
        self.sum_neg_ll = numpy.sum(-numpy.log(list(zip(*action_probs))[0])) + numpy.sum(-numpy.log(list(zip(*action_probs))[1]))   
    
    def simulate(self, ntrials=10):
        trials = []
        reward_probs = numpy.random.rand(6)*.5+.25 #rewards for each action
        reward_probs[0:2] = 0
        transition_probs = [.7,.3] #transition to new stages (probability to go to stage 2)
        # initial conditions
        last_choice = -1
        for trial in range(ntrials):
            s1 = 0
            # get first choice without knowing the second choice
            first_action_probs = self.get_softmax_probs(s1,last_choice)[0]
            a1 = numpy.random.choice(self.stage_action_list[s1], p=first_action_probs)
            # get second stage
            s2 = numpy.random.binomial(1,transition_probs[a1])+1
            second_action_probs = self.get_softmax_probs(s2,last_choice)[0]
            a2 = numpy.random.choice(self.stage_action_list[s2], p=second_action_probs)
            feedback = numpy.random.binomial(1,reward_probs[a2]) 
            trials.append({'stage':s1, 'stage_second':s2,
                           'stim_selected_first':a1,'stim_selected_second':a2,
                           'feedback':feedback,
                           'first_action_prob': first_action_probs[a1],
                            'second_action_prob': second_action_probs[self.stage_action_list[s2].index(a2)]})
            self.trialUpdate(s1,s2,a1,a2,feedback,self.alpha1,self.alpha2,self.lam)
            last_choice = a1
            reward_probs[2:]+=numpy.random.randn(4)*.025
            reward_probs[2:] = numpy.maximum(numpy.minimum(reward_probs[2:],.75),.25)
        return trials
    
    def get_neg_ll(self):
        return self.sum_neg_ll

def get_likelihood(params, df):
        # set initial parameters
        alpha1 = params['alpha1']
        if 'alpha2' in params.keys():
            alpha2 = params['alpha2']
        else:
            alpha2 = params['alpha1']
        lam = params['lam']
        B1 = params['B1']
        if 'B2' in params.keys():
            B2 = params['B2']
        else:
            B2 = params['B1']
        W = params['W']
        p = params['p']
        model = Two_Stage_Model(alpha1,alpha2,lam,B1,B2,W,p)
        model.run_trials(df)
        return model.get_neg_ll()
        
def fit_decision_model(df):
    import lmfit
    fit_params = lmfit.Parameters()
    fit_params.add('alpha1', value=.5, min=0, max=1)
    fit_params.add('alpha2', value=.5, min=0, max=1)
    fit_params.add('lam', value = .5)
    fit_params.add('W', value = .5)
    fit_params.add('p', value = 0)
    fit_params.add('B1', value = 3)
    fit_params.add('B2', value = 3)
    
    out = lmfit.minimize(get_likelihood, fit_params, method = 'lbfgsb', kws={'df': df})
    lmfit.report_fit(out)
    return out.params.valuesdict()

def logistic_analysis(df):
    df.loc[:,'stage_transition'] = 'infrequent'
    df.loc[abs(2-(df.stim_selected_first + df.stage_second))==0, 'stage_transition'] = 'frequent'
    df.insert(0, 'stay', (df['stim_selected_first'].diff()==0).astype(int))
    df.insert(0, 'stage_transition_last', df['stage_transition'].shift(1))
    df.insert(0, 'feedback_last', df['feedback'].shift(1))
    rs = smf.glm(formula = 'stay ~ feedback_last * C(stage_transition_last, Treatment(reference = "infrequent"))', data = df, family = sm.families.Binomial()).fit()
    return {'model_free': rs.params[2], 'model_based': rs.params[3]}

df = get_behav_data(file = 'Individual_Measures/two_stage_decision.csv.gz')     
params = {'alpha1':.7,
          'alpha2': .4,
          'lam':.63,
          'B1':4.23,
          'B2':2.95,
          'W':.51,
          'p':.17} 

n_subjects = 50
W_space = numpy.linspace(0,1,5)
p_space = numpy.linspace(0,1,5)
# generate data
data = pd.DataFrame()
sub_id = 1
for p in p_space:
    for W in W_space:
        params['W'] = W
        for sub in range(n_subjects):
            model = Two_Stage_Model(**params)
            trials = model.simulate(200)
            simulate_df = pd.DataFrame(trials)
            simulate_df.loc[:,'id'] = sub_id
            simulate_df.loc[:,'W'], simulate_df.loc[:,'p'] = [W,p]
            data = pd.concat([data, simulate_df])
            sub_id += 1

subject_params = []
for name, subj_data in data.groupby('id'):
    recovered_params = fit_decision_model(subj_data)
    recovered_params['actual_W'] = W
    subject_params.append(recovered_params)



#get_likelihood(params,simulate_df)
recovered_params = fit_decision_model(simulate_df)
recovered_params['actual_W'] = W
subject_params.append(recovered_params)
logistic_output = logistic_analysis(simulate_df)
logistic_output['actual_W'] = W
logistic_vals.append(logistic_output)

    
subject_params = pd.DataFrame(subject_params)
subject_params.loc[:,['W','p']].hist()

logistic_vals = pd.DataFrame(logistic_vals)
logistic_vals.hist()


        
