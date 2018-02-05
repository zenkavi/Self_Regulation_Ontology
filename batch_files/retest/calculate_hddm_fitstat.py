from glob import glob
import hddm
from kabuki.analyze import _parents_to_random_posterior_sample
import numpy as np
import os
from os import path
import pandas as pd
import pickle
import re
from scipy.stats import entropy
import sys

from expanalysis.experiments.ddm_utils import load_concat_models, load_model

model_dir = sys.argv[1]
task = sys.argv[2]+'_'
subset = sys.argv[3] +'_'
output_dir = sys.argv[4]
hddm_type = sys.argv[5] #(flat or hierarhical)

# Define helper function to get fitstat
def get_likelihood(m, samples=10):
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

#Testing parameters for Case 1:
#model_dir = '/oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/hddm_flat/subject_fits/'
#task = 'choice_reaction_time_'
#subset = 'retest_'
#output_dir = '/oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/hddm_fitstat/'

if hddm_type == 'flat':
# Case 1: fitstat for all subjects of flat models (no hierarchy)
## Strategy: looping through all model files for task, subset
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
    fitstats.to_csv(path.join(output_dir, task+subset+'flat_fitstats.csv'))

# Case 2: fitstat for all subjects for hierarchical models

## Case 2a: without parallelization

## Case 2a: with parallelization

#########
                   

KLs = get_likelihood(m)

KLs.to_csv(out_dir + model.replace('.model', '_'+sample+'_KLs.csv'))

ddm_task_lookup = {'adaptive_n_back_base.model':'adaptive_n_back',
                   'ANT_cue_condition.model': 'attention_network_task',
                   'ANT_flanker_base.model': 'attention_network_task',
                   'ANT_flanker_condition.model': 'attention_network_task',
                   'attention_network_task_flanker_base.model': 'attention_network_task',
                   'attention_network_task_cue_condition.model': 'attention_network_task',
                   'attention_network_task_flanker_condition.model' : 'attention_network_task',
                   'choice_RT_base.model': 'choice_reaction_time',
                   'choice_reaction_time_base.model': 'choice_reaction_time',
                   'directed_forgetting_base.model': 'directed_forgetting',
                   'directed_forgetting_condition.model': 'directed_forgetting',
                   'dot_pattern_expectancy_base.model': 'dot_pattern_expectancy',
                   'dot_pattern_expectancy_condition.model':'dot_pattern_expectancy',
                   'DPX_base.model':'dot_pattern_expectancy',
                   'DPX_condition.model':'dot_pattern_expectancy',
                   'local_global_base.model': 'local_global_letter',
                   'local_global_conflict_condition.model':'local_global_letter',
                   'local_global_switch_condition.model':'local_global_letter',
                   'local_global_letter_base.model': 'local_global_letter',
                   'local_global_letter_conflict_condition.model':'local_global_letter',
                   'local_global_letter_switch_condition.model':'local_global_letter',
                   'motor_SS_base.model':'motor_selective_stop_signal',
                   'motor_selective_stop_signal_base.model': 'motor_selective_stop_signal',
                   'recent_probes_base.model':'recent_probes',
                   'recent_probes_condition.model':'recent_probes',
                   'shape_matching_base.model':'shape_matching',
                   'shape_matching_condition.model':'shape_matching',
                   'simon_base.model':'simon',
                   'simon_condition.model':'simon',
                   'SS_base.model':'stop_signal',
                   'stim_SS_base.model':'stim_selective_stop_signal',
                   'stop_signal_base.model': 'stop_signal', 
                   'stroop_base.model':'stroop',
                   'stroop_condition.model':'stroop',
                   'threebytwo_base.model':'threebytwo',
                   'threebytwo_cue_condition.model':'threebytwo',
                   'threebytwo_task_condition.model':'threebytwo'}
