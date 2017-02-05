import json
import numpy as np
from os import makedirs, path
import pandas as pd
import seaborn as sns

from selfregulation.utils.utils import get_behav_data, get_info

from expanalysis.experiments.processing import calc_exp_DVs
#work around for spyder bug in python 3
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)



#***************************************************
# ********* Load Data **********************
#************************************************** 
try:
    worker_lookup = json.load(open("../Data/Local/worker_lookup.json",'r'))
    inverse_lookup = {v: k for k, v in worker_lookup.items()}
except IOError:
    print('no worker lookup found!')

try:
    worker_counts = json.load(open("../Data/Local/worker_counts.json",'r'))
except IOError:
    print('no worker counts found!')
    
try:
    worker_pay = pd.read_json("../Data/Local/worker_pay.json",'r')
except IOError:
    print('no worker pay found!')

# plot number of tasks completed
sns.plt.figure(figsize = (16,12))
sns.plt.hist(list(worker_counts.values()), bins = 30, color = 'c')
sns.plt.xlabel('Number of Tasks', size = 30)
sns.plt.ylabel('Number of Subjects', size = 30)
sns.plt.title('Histogram of MTurk Task Completion', size = 40)
sns.plt.tick_params(labelsize = 20)

#get pay
pay = worker_pay
workers = []
pay_list = [pay.total.get(inverse_lookup.get(w,'not found'),'not_found') if pay.base.get(inverse_lookup.get(w,'not found'),'not_found') != 60 else pay.bonuses.get(inverse_lookup.get(w,'not found'),'not_found') for w in workers]

#load Data
token = get_info('expfactory_token')
try:
    data_dir=get_info('data_directory')
except Exception:
    data_dir=path.join(get_info('base_directory'),'Data')
    
discovery_directory = path.join(data_dir, 'Discovery_09-26-2016')
failed_directory = path.join(data_dir, 'Failed_09-26-2016')
local_dir = path.join(data_dir,'Local')
if not path.exists(local_dir):
    makedirs(local_dir)

# read preprocessed data
discovery_data = pd.read_json(path.join(local_dir,'mturk_discovery_data_post.json')).reset_index(drop = True)
failed_data = pd.read_json(path.join(local_dir,'mturk_failed_data_post.json')).reset_index(drop = True)

# plot QC
# plot number of tasks completed
sns.plt.figure(figsize = (16,12))
sns.plt.hist(list(worker_counts.values()), bins = 30, color = 'c')
sns.plt.xlabel('Number of Tasks', size = 30)
sns.plt.ylabel('Number of Subjects', size = 30)
sns.plt.title('Histogram of MTurk Task Completion', size = 40)
sns.plt.tick_params(labelsize = 20)


#calculate DVs
DV_df, valence_df = extract_DVs(discovery_data, use_group_fun = False)
DV_df.to_json(path.join(local_dir, 'mturk_discovery_DV.json'))
valence_df.to_json(path.join(local_dir, 'mturk_discovery_DV_valence.json'))