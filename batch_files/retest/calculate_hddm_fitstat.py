import sys
from os import path
import pandas as pd
import pickle
import numpy as np
from kabuki.analyze import _parents_to_random_posterior_sample
from kabuki.analyze import _plot_posterior_pdf_node
from scipy.stats import entropy


model = sys.argv[1]
sub_id_dir = sys.argv[2]

out_dir = '/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_11-27-2017/batch_output/hddm_models'

m = pickle.load(open('/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/batch_files/retest/stroop_base.model', 'rb'))

ddm_tasks = ['adaptive_n_back', 'attention_network_task', 'choice_reaction_time', 'directed_forgetting', 'dot_pattern_expectancy', 'local_global_letter', 'motor_selective_stop_signal', 'recent_probes', 'shape_matching', 'simon', 'stim_selective_stop_signal','stop_signal', 'stroop', 'threebytwo']

task_name = ...

sub_ids = pd.read_csv(sub_id_dir + task_name +'csv.gz' , compression = 'gzip')

sub_ids = sub_ids.worker_id.unique()

def get_likelihood(model, samples=10):
    value_range = np.linspace(-5,5,100)
    observeds = m.get_observeds()
    like = np.empty((samples, len(value_range)), dtype=np.float32)
    KLs = {}
    for subj_i, (node_name, bottom_node) in enumerate(observeds.iterrows()):
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

KLs = get_likelihood(m)

