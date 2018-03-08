import hddm
import numpy as np
import os
import pandas as pd
import pickle
from scipy.stats import linregress
import statsmodels.formula.api as sm
import sys

sys.path.append(os.getcwd())
from post_pred_gen_debug import post_pred_gen


#Read model in
m = pickle.load(open('path_to_model.model', 'rb'))

#Or simulate data
data, params = hddm.generate.gen_rand_data(params={'easy': {'v': 1, 'a': 2, 't': .3},
                                                   'hard': {'v': 1, 'a': 2, 't': .3}}, subjs=2)

#Generate posterior predictive data. This command below will have 500 samples, two nodes per subject (for each condition - easy and hard)
#n samples (in this case 100) for each trial
ppc_data = post_pred_gen(m) 

#We are interested in how similar the observed and sampled rt distributions are. To get at this we will regress the observed rt distributions over the rt distributions generated by sampling from the posterior and creating a posterior predictive. Then we will summarize the results of the similarity between each posterioripredictive sample and the observed score per subject.

#The hddm package has some convenience functions to calculate statistics on the posterior predictive.
#For example this function returnd the mean of rt's for each posterior predictive sample for each condition for each subject [4(subject x condition)*100 (samples)]
hddm.utils.post_pred_stats(data, ppc_data, stats=lambda x: np.mean(x), call_compare=False)

#The same grouped mean operation can be done on the posterior predictive data with the following syntax as well 
ppc_data.groupby(level=(0, 1)).mean()

#The hddm documentation presents an example of getting summary statistics relating model parameters to other trial by trial variables (e.g. eeg-response)
#For this we create posterior predictive data with appending the observed data to it
ppc_data_append = post_pred_gen(m, samples = 100, append_data = True)

#Here we relate the posterior predictive rt to the empirical trial by trial rt by looping through each generated sample and extracting the slope from regressing the empirical rt on the sampled rt
ppc_regression_append = []
for (node, sample), sim_data in ppc_data_append.groupby(level=(0, 1)):
    ppc_regression_append.append(linregress(sim_data.rt, sim_data.rt_sampled)[0])

#the above for loop is the (almost) same thing as getting the correlation between the observed rt distribution and the sampled rt distribution using the following shorter syntax
ppc_data_append.groupby(level=(0, 1))[['rt','rt_sampled']].corr().ix[0::2,'rt_sampled']

#The shorter syntax and the linregress function are limited, however, in the details of the output we obtain from them
#So I've chosen to use a loop and the statsmodel sm function instead
#The general strategy then is:
#For each linear regression of rt on rt_sampled get intercept, slope, p-val for intercept, p-val for slope and r-squared
#Output this subject_n*sample length df as *_fitstat_samples.csv
#then summarize this for each subject and get average intercept, sd of intercept, average slope, sd of slope, average p value intercept, sd of p value of intercept, average p value of slope, sd of p value of slope and (most importantly) average r squared
#output this summary as *_fitstat_summary.csv

out_dir='/Users/zeynepenkavi/Downloads/'
task = 'test'

#output nested dictionary for each sub with model details
ppc_regression_samples = {}
#This loop should output n*condition*sample regression (e.g. 2*2*100)
for (node, sample), sim_data in ppc_data_append.groupby(level=(0,1)):
    sample_out = {}
    model = sm.ols(formula='rt ~ rt_sampled', data=sim_data)
    fitted = model.fit()
    sample_out['int_val'] = fitted.params[0]
    sample_out['int_pval'] = fitted.pvalues[0]
    sample_out['slope_val'] = fitted.params[1]
    sample_out['slope_pval'] = fitted.pvalues[1]
    sample_out['rsq'] = fitted.rsquared
    sample_out['rsq_adj'] = fitted.rsquared_adj
    ppc_regression_samples.update({node+'_'+str(sample): sample_out})

#Convert sample*subject length dict to dataframe
ppc_regression_samples = pd.DataFrame.from_dict(ppc_regression_samples, orient="index")

#Add subj_id and condition columns
if ppc_regression_samples.index.tolist()[0].find("(") != -1:
    ppc_regression_samples['condition'] = [s[s.find("(")+1:s.find(")")] for s in ppc_regression_samples.index.tolist()]

if ppc_regression_samples.index.tolist()[0].find(".") != -1:
    ppc_regression_samples['subj_id'] = [s[s.find(".")+1:s.find("_")] for s in ppc_regression_samples.index.tolist()]
else:
    ppc_regression_samples['subj_id'] = 0
                       
#Save sample level output
#ppc_regression_samples.to_csv(out_dir+task+'_fitstat_samples.csv')

#Summarize on subject*condition level
if 'condition' in ppc_regression_samples.columns:
    means = ppc_regression_samples.groupby(['condition', 'subj_id']).mean().reset_index(level=['condition', 'subj_id'])
    stds = ppc_regression_samples.groupby(['condition', 'subj_id']).std().reset_index(level=['condition', 'subj_id'])
else:
    means = ppc_regression_samples.groupby(['subj_id']).mean().reset_index(level=['subj_id'])
    stds = ppc_regression_samples.groupby(['subj_id']).mean().reset_index(level=['subj_id'])

ppc_regression_subj = means.merge(stds, on = ['condition', 'subj_id'], suffixes = ('_mean', '_std'))
    
#Save summarized output
#ppc_regression_subj.to_csv(out_dir+task+'_fitstat_summary.csv')

###Add correct groupby's to ppc_data generations for each task
