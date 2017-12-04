from kabuki.analyze import _parents_to_random_posterior_sample
import numpy as np
import os
import pandas as pd
import pickle
from scipy.stats import entropy
import sys


model = sys.argv[1]
sub_id_dir = sys.argv[2]
release = sys.argv[3]

out_dir = '/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/'+release+'/batch_output/hddm_models/'

os.chdir(out_dir)

m = pickle.load(open('/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/'+release+'/batch_output/hddm_models/'+model, 'rb'))

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
                   
task_name = ddm_task_lookup.get(model)

sub_ids = pd.read_csv(sub_id_dir + task_name +'.csv.gz' , compression = 'gzip')

sub_ids = sub_ids.worker_id.unique()

def get_likelihood(model, samples=10):
    value_range = np.linspace(-5,5,100)
    observeds = model.get_observeds()
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
    tr = dict(zip(KLs.keys(), sub_ids))
    KLs = {tr[k]: v for k, v in KLs.items()}
    return KLs

KLs = get_likelihood(m)

KLs = pd.DataFrame.from_dict(KLs, orient="index").rename(index=str, columns={0: model.replace(".model",".KL")})

KLs.to_csv(out_dir + model.replace('.model', '_KLs.csv'))