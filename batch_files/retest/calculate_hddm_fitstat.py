from glob import glob
import inspect
from kabuki.utils import concat_models
import numpy as np
import os
from os import path
import pandas as pd
import pickle
import re
import statsmodels.formula.api as sm
import sys

sys.path.append(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
from post_pred_gen_debug import post_pred_gen

model_dir = sys.argv[1]
task = sys.argv[2]
subset = sys.argv[3] +'_'
output_dir = sys.argv[4]
hddm_type = sys.argv[5] #(flat or hierarhical)
parallel = sys.argv[6]
sub_id_dir = sys.argv[7]
samples = int(float(sys.argv[8]))

##############################################
############ HELPER FUNCTIONS ################
##############################################

##############################################
############### For Fitstats #################
##############################################

# Define helper function to get fitstat
def get_fitstats(m, samples = samples, groupby=None, append_data = True, output_dir = output_dir, task = task, subset = subset, hddm_type=hddm_type):
    
    #Sample from posterior predictive and generate data
    ppc_data_append = post_pred_gen(m, samples = samples, append_data = append_data, groupby=groupby)
    
    ppc_data_append.reset_index(inplace=True) 
    
    if(all(v is not None for v in [output_dir, task, hddm_type]) and subset != 'flat'):
        ppc_data_append.to_csv(path.join(output_dir, task+ '_'+subset+hddm_type+'_ppc_data.csv'))
    
    #Adding 3 to avoid negatives
    ppc_data_append['log_rt'] = np.log(ppc_data_append.rt+3)
    ppc_data_append['log_rt_sampled'] = np.log(ppc_data_append.rt_sampled+3)
    
    #This loop should output n*condition*sample regression (e.g. 2*2*100)
    #level 0 = 'node' referring to subject
    #level 1 = 'sample' referring to correct vs incorrect (?)
    #will this work with flat?
    ppc_regression_samples = {}
    for (node), sim_data in ppc_data_append.groupby(['node']):        
        sub_out = {}
        model = sm.ols(formula='rt ~ rt_sampled', data=sim_data)
        fitted = model.fit()
        sub_out['int_val'] = fitted.params[0]
        sub_out['int_pval'] = fitted.pvalues[0]
        sub_out['slope_val'] = fitted.params[1]
        sub_out['slope_pval'] = fitted.pvalues[1]
        sub_out['rsq'] = fitted.rsquared
        sub_out['rsq_adj'] = fitted.rsquared_adj
        
        try:
            log_model = sm.ols(formula='log_rt ~ log_rt_sampled', data=sim_data)
            log_fitted = log_model.fit()
            sub_out['log_int_val'] = log_fitted.params[0]
            sub_out['log_int_pval'] = log_fitted.pvalues[0]
            sub_out['log_slope_val'] = log_fitted.params[1]
            sub_out['log_slope_pval'] = log_fitted.pvalues[1]
            sub_out['log_rsq'] = log_fitted.rsquared
            sub_out['log_rsq_adj'] = log_fitted.rsquared_adj
        except:
            sub_out['log_int_val'] = np.nan
            sub_out['log_int_pval'] = np.nan
            sub_out['log_slope_val'] = np.nan
            sub_out['log_slope_pval'] = np.nan
            sub_out['log_rsq'] = np.nan
            sub_out['log_rsq_adj'] = np.nan
        
        ppc_regression_samples.update({node: sub_out})
    
    #Convert sample*subject length dict to dataframe
    ppc_regression_samples = pd.DataFrame.from_dict(ppc_regression_samples, orient="index")
    
    ppc_regression_samples.reset_index(inplace=True)
    
    if ppc_regression_samples['index'].str.contains("\\.")[0]:
        ppc_regression_samples['subj_id'] = [s[s.find(".")+1:s.find(")")] for s in ppc_regression_samples['index']]
    else:
        ppc_regression_samples['subj_id'] = 0
    
    if(all(v is not None for v in [output_dir, task, hddm_type]) and subset != 'flat'):
        ppc_data_append.to_csv(path.join(output_dir, task+ '_'+subset+hddm_type+'_ppc_regression_samples.csv'))
    
    return ppc_regression_samples

##############################################
############# For Model Loading ##############
##############################################

def get_dbpath():
    db_path = path.join(model_dir, task+'_parallel_output/')
    return db_path


def load_parallel_models(model_path):
    loadfile = sorted(glob(model_path))
    models = []
    for l in loadfile:
        
        #For some reason the refit model db names are stored in path names different that where they actually are
        #So loading the model directly doesn't work
        #Instead I create a temporary copy of the traces.db files in the path where the model object thinks it should be
        #Thought about loading the traces and regenerating the model but couldn't be sure if it would be the same
        
        m = pickle.load(open(l, 'rb'))
        models.append(m)
    return models    

##############################################
############### Groupby lookup ################
##############################################

def get_groupby_array(task=None):
    groupby_array_dict = \
    {
        'adaptive_n_back': None, 
        'attention_network_task': ['flanker_type', 'cue'],
        'choice_reaction_time': None,
        'directed_forgetting': ['probe_type'],
        'dot_pattern_expectancy' : ['condition'],
        'local_global_letter' : ['condition', 'conflict_condition', 'switch'],
        'motor_selective_stop_signal': ['critical_key'],
        'recent_probes': ['probeType'],
        'shape_matching': ['condition'],
        'simon': ['condition'],
        'stim_selective_stop_signal': ['condition'],
        'stop_signal': ['condition'],
        'stroop': ['condition'],
        'threebytwo': ['cue_switch_binary', 'task_switch_binary', 'CTI']
    }
    if task is None:
        return groupby_array_dict
    else:
        return groupby_array_dict[task]

##############################################
############### Sub Id lookup ################
##############################################

def get_hddm_subids(df):
    # set up data
    data = (df.loc[:,'rt']/1000).astype(float).to_frame()
    # add subject ids 
    data.insert(0,'subj_idx', df['worker_id'])
    # remove missed responses and extremely short response
    data = data.query('rt > .05')
    subj_ids = data.subj_idx.unique()
    ids = {int(i):subj_ids[i] for i in range(len(subj_ids))}
    return ids

def directed_subids(df):
    n_responded_conds = df.query('rt>.05').groupby('worker_id').probe_type.unique().apply(len)
    complete_subjs = list(n_responded_conds.index[n_responded_conds==3])
    missing_subjs = set(n_responded_conds.index)-set(complete_subjs)
    if len(missing_subjs) > 0:
        print('Subjects without full design matrix: %s' % missing_subjs)
    df = df.query('worker_id in %s' % complete_subjs)
    subids = get_hddm_subids(df.query('trial_id == "probe"'))
    return subids

def DPX_subids(df):
    n_responded_conds = df.query('rt>0').groupby('worker_id').condition.unique().apply(len)
    complete_subjs = list(n_responded_conds.index[n_responded_conds==4])
    missing_subjs = set(n_responded_conds.index)-set(complete_subjs)
    if len(missing_subjs) > 0:
        print('Subjects without full design matrix: %s' % missing_subjs)
    df = df.query('worker_id in %s' % complete_subjs)
    subids = get_hddm_subids(df)
    return subids

def motor_SS_subids(df):
    df = df.copy()
    critical_key = (df.correct_response == df.stop_response).map({True: 'critical', False: 'non-critical'})
    df.insert(0, 'critical_key', critical_key)
    df = df.query('SS_trial_type == "go" and \
                 exp_stage not in ["practice","NoSS_practice"]')
    subids = get_hddm_subids(df)
    return subids


def recent_subids(df):
    n_responded_conds = df.query('rt>.05').groupby('worker_id').probeType.unique().apply(len)
    complete_subjs = list(n_responded_conds.index[n_responded_conds==4])
    missing_subjs = set(n_responded_conds.index)-set(complete_subjs)
    if len(missing_subjs) > 0:
        print('Subjects without full design matrix: %s' % missing_subjs)
    df = df.query('worker_id in %s' % complete_subjs)
    subids = get_hddm_subids(df)
    return subids

def shape_matching_subids(df):
    # restrict to the conditions of interest
    df = df.query('condition in %s' % ['SDD', 'SNN'])
    n_responded_conds = df.query('rt>.05').groupby('worker_id').condition.unique().apply(len)
    complete_subjs = list(n_responded_conds.index[n_responded_conds==2])
    missing_subjs = set(n_responded_conds.index)-set(complete_subjs)
    if len(missing_subjs) > 0:
        print('Subjects without full design matrix: %s' % missing_subjs)
    df = df.query('worker_id in %s' % complete_subjs)
    subids = get_hddm_subids(df)
    return subids

def stim_SS_subids(df):
    df = df.query('condition != "stop" and \
                 exp_stage not in ["practice","NoSS_practice"]')
    subids = get_hddm_subids(df)
    return subids

def SS_subids(df):
    df = df.query('SS_trial_type == "go" \
                 and exp_stage not in ["practice","NoSS_practice"]')
    subids = get_hddm_subids(df)
    return subids

def threebytwo_subids(df):
    df = df.copy()  
    df.loc[:,'cue_switch_binary'] = df.cue_switch.map(lambda x: ['cue_stay','cue_switch'][x!='stay'])
    df.loc[:,'task_switch_binary'] = df.task_switch.map(lambda x: ['task_stay','task_switch'][x!='stay'])
    subids = get_hddm_subids(df)
    return subids

def twobytwo_subids(df):
    df = df.copy()
    df.loc[:,'cue_switch_binary'] = df.cue_switch.map(lambda x: ['cue_stay','cue_switch'][x!='stay'])
    df.loc[:,'task_switch_binary'] = df.task_switch.map(lambda x: ['task_stay','task_switch'][x!='stay'])
    subids = get_hddm_subids(df)
    return subids


def get_subids_fun(task=None):
    subids_fun_dict = \
    {
        'adaptive_n_back': lambda df: get_hddm_subids(df.query('exp_stage == "adaptive"')),
        'attention_network_task': lambda df: get_hddm_subids(df),
        'choice_reaction_time': lambda df: get_hddm_subids(df),
        'directed_forgetting': lambda df: directed_subids(df),
        'dot_pattern_expectancy': lambda df: DPX_subids(df), 
                                                      
        'local_global_letter': lambda df: get_hddm_subids(df),
        'motor_selective_stop_signal': lambda df: motor_SS_subids(df),
        'recent_probes': lambda df: recent_subids(df),
        'shape_matching': lambda df: shape_matching_subids(df), 
        'simon': lambda df: get_hddm_subids(df), 
        'stim_selective_stop_signal': lambda df: stim_SS_subids(df),
        'stop_signal': lambda df: SS_subids(df),
        'stroop': lambda df: get_hddm_subids(df), 
        'threebytwo': lambda df: threebytwo_subids(df),
        'twobytwo': lambda df: twobytwo_subids(df)
    }
    if task is None:
        return subids_fun_dict
    else:
        return subids_fun_dict[task]

##############################################
############### GET FITSTATS #################
##############################################

# Case 1: fitstat for all subjects of flat models (no hierarchy)
if hddm_type == 'flat':

    ## Strategy: looping through all model files for task, subset
    model_path = path.join(model_dir, task+'_'+ subset+ '*_flat.model')
    models_list = sorted(glob(model_path))
    fitstats = pd.DataFrame()

    ### Step 1: Read model in for a given subject
    for model in models_list:
        m = pickle.load(open(model, 'rb'))
        
        ### Step 2: Get fitstat for read in model
        fitstat = get_fitstats(m)
        
        ### Step 3: Extract sub id from file name
        sub_id = re.search(model_dir+task+ '_'+subset+'(.+?)_flat.model', model).group(1)
        
        ### Step 4: Update flat fitstat dict with sub_id
        fitstat['sub_id'] = sub_id
        
        ### Step 5: Add individual output to dict with all subjects
        fitstats = fitstats.append(fitstat)

    ### Step 6: Output df with task, subset, model type (flat or hierarchical)
    fitstats.to_csv(path.join(output_dir, task+'_'+subset+hddm_type+'_fitstats.csv'))


# Case 2: fitstat for all subjects for hierarchical models
if hddm_type == 'hierarchical':

    ## Case 2a: with parallelization
    if parallel == 'yes':

        ### Step 1a: Concatenate all model outputs from parallelization
        model_path = path.join(model_dir, task+'_parallel_output','*.model')
        loaded_models = load_parallel_models(model_path)
        m_concat = concat_models(loaded_models)

        ### Step 2a: Get fitstat for all subjects from concatenated model
        fitstats = get_fitstats(m_concat)

    ## Case 2b: without parallelization
    elif parallel == 'no':

        ### Step 1b: Read model in
        m = pickle.load(open(path.join(model_dir,task+'.model'), 'rb'))

        ### Step 2b: Get fitstats
        fitstats = get_fitstats(m)

    ### Step 3: Extract sub id from correct df that was used for hddm
    subid_fun = get_subids_fun(task)
    sub_df = pd.read_csv(path.join(sub_id_dir, task+'.csv.gz'), compression='gzip')
    subids = subid_fun(sub_df)

    ### Step 4: Change keys in fitstats dic
    fitstats[['subj_id']] = fitstats[['subj_id']].apply(pd.to_numeric, errors='coerce')
    fitstats = fitstats.replace({'subj_id': subids})

    ### Step 5: Output df with task, subset, model type (flat or hierarchical)
    fitstats.to_csv(path.join(output_dir, task+ '_'+subset+hddm_type+'_fitstats.csv'))