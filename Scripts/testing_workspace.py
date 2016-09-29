from expanalysis.experiments.processing import extract_row, post_process_data, post_process_exp, extract_experiment, calc_DVs, extract_DVs,flag_data,  get_DV, generate_reference
import json
import numpy as np
import pandas as pd
import seaborn as sns

#work around for spyder bug in python 3
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)



#***************************************************
# ********* Load Data **********************
#************************************************** 
try:
    worker_lookup = json.load(open("../Data/worker_lookup.json",'r'))
    inverse_lookup = {v: k for k, v in worker_lookup.items()}
except IOError:
    print('no worker lookup found!')

try:
    worker_counts = json.load(open("../Data/worker_counts.json",'r'))
except IOError:
    print('no worker counts found!')
    
try:
    worker_pay = pd.read_json("../Data/worker_pay.json",'r')
except IOError:
    print('no worker pay found!')

#get pay
pay = worker_pay
workers = []
pay_list = [pay.total.get(inverse_lookup.get(w,'not found'),'not_found') if pay.base.get(inverse_lookup.get(w,'not found'),'not_found') != 60 else pay.bonuses.get(inverse_lookup.get(w,'not found'),'not_found') for w in workers]

#load Data
token = get_info('expfactory_token')
try:
    data_dir=get_info('data_directory')
except Exception:
    data_dir=get_info('base_directory') + 'Data/'


# read preprocessed data
data = pd.read_json(data_dir + 'mturk_discovery_data_post.json').reset_index(drop = True)
failed_data = pd.read_json(data_dir + 'mturk_failed_data_post.json').reset_index(drop = True)
failed_data = failed_data[np.logical_not(failed_data.worker_id.str.contains('s5'))]

# get DV df
DV_df = pd.read_json(data_dir + 'mturk_discovery_DV.json')
valence_df = pd.read_json(data_dir + 'mturk_discovery_DV_valence.json')

#save data
save_task_data(data_dir, data)

    

# ************************************
# ********* Save Components of Data **
# ************************************
items_df = get_items(data)
items_pivot_df = items_df.pivot('worker','item_ID','coded_response')





