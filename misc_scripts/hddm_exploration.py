from expanalysis.experiments.processing import  extract_experiment, extract_DVs, get_DV
from os import path
import pandas as pd
from scipy.stats import ttest_ind, ttest_1samp
import sys
sys.path.append('../utils')
from utils import get_info


data_dir=path.join(get_info('base_directory'),'Data')
    
discovery_directory = path.join(data_dir, 'Discovery_09-26-2016')
failed_directory = path.join(data_dir, 'Failed_09-26-2016')
local_dir = path.join(data_dir,'Local')

discovery_data = pd.read_json(path.join(local_dir,'mturk_discovery_data_post.json')).reset_index(drop = True)



exps_of_interest = ['simon','stroop','attention_network_task','dot_pattern_expectancy','choice_reaction_time', 
                    'threebytwo', 'recent_probes', 'directed_forgetting', 'local_global_letter', 'shape_matching_task']
data = discovery_data.query('experiment_exp_id in %s' %exps_of_interest)

DV_df, valence = extract_DVs(data.query('experiment_exp_id in %s' % list(np.sort(data.experiment_exp_id.unique())[37:40])))

DV_df.filter(regex = '/..(hddm|EZ)_thresh$').apply(lambda x: ttest_1samp(x, 0))
