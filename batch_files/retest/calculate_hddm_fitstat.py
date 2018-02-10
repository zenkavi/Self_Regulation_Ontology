from glob import glob
import hddm
from kabuki.analyze import _parents_to_random_posterior_sample
from kabuki.utils import concat_models
import numpy as np
from os import path
import pandas as pd
import pickle
import re
import statsmodels.formula.api as sm
import sys

from expanalysis.experiments.ddm_utils import load_concat_models
import post_pred_gen_debug

model_dir = sys.argv[1]
task = sys.argv[2]
subset = sys.argv[3] +'_'
output_dir = sys.argv[4]
hddm_type = sys.argv[5] #(flat or hierarhical)
parallel = sys.argv[6]
sub_id_dir = sys.argv[7]

##############################################
############ HELPER FUNCTIONS ################
##############################################

##############################################
############### For Fitstats #################
##############################################

# Define helper function to get fitstat
def get_likelihood(m, samples=100):
    value_range = np.linspace(-5,5,100)
    observeds = m.get_observeds()
    like = np.empty((samples, len(value_range)), dtype=np.float32)   
    
    #we have come up with our own way of doing a posterior predictive check
    #for each subject we sample from the posterior predictive and compare it to the data
    #we do this n=samples times and calculate the KL divergence between the posterior predictive and the actual data
    def KL_loop(obs):
        KLs = {}
        for subj_i, (node_name, bottom_node) in enumerate(obs.iterrows()):
            node = bottom_node['node']
            for sample in range(samples):
                _parents_to_random_posterior_sample(node)
                # Generate likelihood for parents parameters
                like[sample,:] = node.pdf(value_range)
                y = like.mean(axis=0)
                data_bins = np.histogram(node.value, value_range, density=True)[0]
            KL_divergence = entropy(y[1:]+1E-10, data_bins+1E-10)
            KLs[subj_i] = KL_divergence
        return KLs
    
    out = KL_loop(observeds)
    #KLs = pd.DataFrame.from_dict(KLs, orient="index").rename(index=str, columns={0: model.replace(".model",".KL")})
    return out

##############################################
############# For Model Loading ##############
##############################################

def load_parallel_models(model_path):
    loadfile = sorted(glob(model_path))
    models = []
    for l in loadfile:
        m = hddm.load(l)
        #m.load_db(l, db='pickle')
        models.append(m)
    m = load_concat_models(models)
    return m, models

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

#Testing parameters for Case 1:
#model_dir = '/oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/hddm_flat/subject_fits/'
#task = 'choice_reaction_time_'
#subset = 'retest'
#output_dir = '/oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/hddm_fitstat/'
#hddm_type = 'flat'

# Case 1: fitstat for all subjects of flat models (no hierarchy)
if hddm_type == 'flat':
## Strategy: looping through all model files for task, subset
    task = task+'_'
    model_path = path.join(model_dir, task+ subset+ '*_flat.model')
    models_list = sorted(glob(model_path))
    fitstats = {}
### Step 1: Read model in for a given subject
    for model in models_list:
        m = pickle.load(open(model, 'rb'))
### Step 2: Get fitstat for read in model
        fitstat = get_likelihood(m)
### Step 3: Extract sub id from file name
        sub_id = re.search(model_dir+task+ subset+'(.+?)_flat.model', model).group(1)
### Step 4: Update flat fitstat dict with sub_id
        fitstat[sub_id] = fitstat.pop(0)
### Step 5: Add individual output to dict with all subjects
        fitstats.update(fitstat)
### Step 6: Convert list to df
    fitstats = pd.DataFrame.from_dict(fitstats, orient='index').rename(index=str, columns={0:"KL"})
### Step 7: Output df with task, subset, model type (flat or hierarchical)
    fitstats.to_csv(path.join(output_dir, task+subset+hddm_type+'_fitstats.csv'))

#Testing parameters for Case 2a:
#model_dir = '/oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/hddm_refits/'
#task = 'choice_reaction_time'
#subset = 'refits_'
#output_dir = '/oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/hddm_fitstat/'
#hddm_type = 'hierarchical'
#parallel = 'yes'
#sub_id_dir = '/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_01-23-2018/t1_data/Individual_Measures'

# Case 2: fitstat for all subjects for hierarchical models
if hddm_type == 'hierarchical':
## Case 2a: with parallelization
    if parallel == 'yes':
### Step 1a: Concatenate all model outputs from parallelization
        model_path = path.join(model_dir, task+'_parallel_output','*.model')
        loaded_models = load_parallel_models(model_path)
        m_concat = concat_models(loaded_models[1])
### Step 2a: Get fitstat for all subjects from concatenated model
        fitstats = get_likelihood(m_concat)
## Case 2b: without parallelization
    elif parallel == 'no':
### Step 1b: Read model in
        m = hddm.load(path.join(model_dir,task+'.model'))
### Step 2b: Get fitstats
        fitstats = get_likelihood(m)
### Step 3: Extract sub id from correct df that was used for hddm
    subid_fun = get_subids_fun(task)
    sub_df = pd.read_csv(path.join(sub_id_dir, task+'.csv.gz'), compression='gzip')
    subids = subid_fun(sub_df)
### Step 4: Change keys in fitstats dic
    fitstats = dict((subids[key], value) for (key, value) in fitstats.items())
### Step 5: Convert list to df
    fitstats = pd.DataFrame.from_dict(fitstats, orient='index').rename(index=str, columns={0:"KL"})
### Step 6: Output df with task, subset, model type (flat or hierarchical)
    fitstats.to_csv(path.join(output_dir, task+ '_'+subset+hddm_type+'_fitstats.csv'))

##############################################
############### Test Commands ################
##############################################

#test
#python calculate_hddm_fitstat.py /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_complete_output/ choice_reaction_time t1 /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/hddm_fitstat/ hierarchical yes /oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Complete_01-22-2018/Individual_Measures/

#retest
#python calculate_hddm_fitstat.py /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/ choice_reaction_time retest /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/hddm_fitstat/ hierarchical yes /oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_01-23-2018/Individual_Measures/

#refits
#python calculate_hddm_fitstat.py /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/hddm_refits/ choice_reaction_time refit /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/hddm_fitstat/ hierarchical yes /oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_01-23-2018/t1_data/Individual_Measures/

#test flat
#python calculate_hddm_fitstat.py /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/hddm_flat/subject_fits/ choice_reaction_time t1 /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/hddm_fitstat/ flat no /oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_01-23-2018/t1_data/Individual_Measures/

#retest flat
#python calculate_hddm_fitstat.py /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/hddm_flat/subject_fits/ choice_reaction_time retest /oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/hddm_fitstat/ flat no /oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_01-23-2018/Individual_Measures/