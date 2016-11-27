import sys
sys.path.append('/corral-repl/utexas/poldracklab/users/zenkavi/expfactory-analysis')
from expanalysis.experiments.jspsych import calc_time_taken, get_post_task_responses
from expanalysis.experiments.processing import post_process_data, extract_DVs
from expanalysis.results import get_filters
import json
import numpy as np
from os import path
import os
import pandas as pd
import time

sys.path.append('/corral-repl/utexas/poldracklab/users/zenkavi/Self_Regulation_Ontology/utils')
from data_preparation_utils import anonymize_data, calc_trial_order, convert_date, download_data, get_bonuses, get_pay,  remove_failed_subjects
from utils import get_info

#if len(sys.argv) < 4:
#    sys.exit("Usage: concat_process_retest_data_tacc.py")


data_dir=get_info('data_directory', infile='/corral-repl/utexas/poldracklab/users/zenkavi/Self_Regulation_Ontology/Self_Regulation_Retest_Settings_Tacc.txt')


data = pd.DataFrame()

for f in os.listdir(data_dir):
    if f.find("mturk_data") != -1:
        tmp = pd.read_json(path.join(data_dir, f))
        data = pd.concat([data, tmp])

data.to_json(data_dir + 'mturk_data_all.json')
        
print('Finished loading raw data and saved concattaneted data')        
        
#anonymize data
worker_lookup = anonymize_data(data)
json.dump(worker_lookup, open(path.join(data_dir, 'worker_lookup.json'),'w'))

# record subject completion statistics
(data.groupby('worker_id').count().finishtime).to_json(path.join(data_dir, 'worker_counts.json'))

# add a few extras
convert_date(data)
bonuses = get_bonuses(data)
calc_time_taken(data)
get_post_task_responses(data)   
calc_trial_order(data)

# save data
data.to_json(path.join(data_dir, 'mturk_data_extras.json'))

# calculate pay
pay = get_pay(data)
pay.to_json(path.join(data_dir, 'worker_pay.json'))
print('Finished saving worker pay')