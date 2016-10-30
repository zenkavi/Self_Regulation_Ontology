from expanalysis.experiments.processing import  extract_experiment, get_DV
from os import path
import pandas as pd
from scipy.stats import ttest_ind
import sys
sys.path.append('../utils')
from utils import get_info


data_dir=path.join(get_info('base_directory'),'Data')
    
discovery_directory = path.join(data_dir, 'Discovery_09-26-2016')
failed_directory = path.join(data_dir, 'Failed_09-26-2016')
local_dir = path.join(data_dir,'Local')

discovery_data = pd.read_json(path.join(local_dir,'mturk_discovery_data_post.json')).reset_index(drop = True)

a,b = get_DV(discovery_data, 'simon')
