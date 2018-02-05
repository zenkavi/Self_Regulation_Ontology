from glob import glob
import hddm
from kabuki.analyze import _parents_to_random_posterior_sample
from kabuki.utils import concat_models
import numpy as np
from os import path
import pandas as pd
import pickle
import re
from scipy.stats import entropy
import sys

from expanalysis.experiments.ddm_utils import load_concat_models

model_dir = sys.argv[1]
task = sys.argv[2]+'_'
subset = sys.argv[3] +'_'
output_dir = sys.argv[4]
hddm_type = sys.argv[5] #(flat or hierarhical)
parallel = sys.argv[6]

##############################################
############ HELPER FUNCTIONS ################
##############################################

##############################################
############### For Fitstats #################
##############################################

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

##############################################
############### GET FITSTATS #################
##############################################

#Testing parameters for Case 1:
#model_dir = '/oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/hddm_flat/subject_fits/'
#task = 'choice_reaction_time_'
#subset = 'retest_'
#output_dir = '/oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/hddm_fitstat/'
#hddm_type = 'flat'

# Case 1: fitstat for all subjects of flat models (no hierarchy)
if hddm_type == 'flat':
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

#Testing parameters for Case 2a:
#model_dir = '/oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/hddm_refits/'
#task = 'choice_reaction_time_'
#subset = 'refits'
#output_dir = '/oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_retest_output/hddm_fitstat/'
#hddm_type = 'hierarchical'
#parallel = 'yes'

# Case 2: fitstat for all subjects for hierarchical models
if hddm_type == 'hierarchical':
## Case 2a: with parallelization
    if parallel == 'yes':
### Step 1a: Concatenate all model outputs from parallelization
        model_path = path.join(model_dir, task+'parallel_output','*.model')
        loaded_models = load_parallel_models(model_path)
        m_concat = concat_models(loaded_models[1])
### Step 2a: Get fitstat for all subjects from concatenated model
        fitstats = get_likelihood(m_concat)
## Case 2b: without parallelization
    elif parallel == 'no':
### Step 1b: Read model in
        m = hddm.load(path.join(model_dir+task+'.model'))
### Step 2b: Get fitstats
        fitstats = get_likelihood(m)
### Step 3: Extract sub id from correct df that was used for hddm
### Step 4: Change keys in fitstats dic

