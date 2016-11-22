import sys
sys.path.append('/corral-repl/utexas/poldracklab/users/zenkavi/expfactory-analysis')
from expanalysis.experiments.jspsych import calc_time_taken, get_post_task_responses
from expanalysis.experiments.processing import post_process_data, extract_DVs
from expanalysis.results import get_filters
import json
import numpy as np
from os import path
import pandas as pd
import time

sys.path.append('/corral-repl/utexas/poldracklab/users/zenkavi/Self_Regulation_Ontology/utils')
from data_preparation_utils import anonymize_data, calc_trial_order, convert_date, download_data, get_bonuses, get_pay,  remove_failed_subjects
from utils import get_info

if len(sys.argv) < 2:
    sys.exit("Usage: download_retest_data_tacc.py 'all' '['retest', 'incomplete']' 'http://expfactory.org/api/results' 'http://expfactory.org/api/results/?page=3' ")

job = sys.argv[1]
sample = sys.argv[2]
url = sys.argv[3]
last_url = sys.argv[4]
        
#load Data
token = get_info('expfactory_token', infile='/corral-repl/utexas/poldracklab/users/zenkavi/Self_Regulation_Ontology/Self_Regulation_Retest_Settings_Tacc.txt')
try:
    data_dir=get_info('data_directory', infile='/corral-repl/utexas/poldracklab/users/zenkavi/Self_Regulation_Ontology/Self_Regulation_Retest_Settings_Tacc.txt')
except Exception:
    data_dir=path.join(get_info('base_directory', infile='/corral-repl/utexas/poldracklab/users/zenkavi/Self_Regulation_Ontology/Self_Regulation_Retest_Settings_Tacc.txt'),'Data')

if job == 'download' or job == "all":
    #***************************************************
    # ********* Load Data **********************
    #**************************************************        
    pd.set_option('display.width', 200)
    figsize = [16,12]
    #set up filters
    filters = get_filters()
    drop_columns = ['battery_description', 'experiment_reference', 'experiment_version', \
             'experiment_name','experiment_cognitive_atlas_task']
    for col in drop_columns:
        filters[col] = {'drop': True}
    
    #***************************************************
    # ********* Download Data**********************
    #**************************************************  
    #load Data
    f = open(token)
    access_token = f.read().strip()  
    data = download_data(data_dir, access_token, filters = filters,  battery = 'Self Regulation Retest Battery', url = url, last_url = last_url)
    data.reset_index(drop = True, inplace = True)
    
if job in ['extras', 'all']:
    #Process Data
    if job == "extras":
        #load Data
        data = pd.read_json(path.join(data_dir, 'mturk_data_'+time.strftime("%m_%d_%Y-%I_%M_%S")+'.json'))
        data.reset_index(drop = True, inplace = True)
        print('Finished loading raw data')
    
    #anonymize data
    worker_lookup = anonymize_data(data)
    json.dump(worker_lookup, open(path.join(data_dir, 'worker_lookup_'+time.strftime("%m_%d_%Y-%I_%M_%S")+'.json','w')))
    
    # record subject completion statistics
    (data.groupby('worker_id').count().finishtime).to_json(path.join(data_dir, 'worker_counts'+time.strftime("%m_%d_%Y-%I_%M_%S")+'.json'))
    
    # add a few extras
    convert_date(data)
    bonuses = get_bonuses(data)
    calc_time_taken(data)
    get_post_task_responses(data)   
    calc_trial_order(data)
    
    # save data
    data.to_json(path.join(data_dir, 'mturk_data_extras'+time.strftime("%m_%d_%Y-%I_%M_%S")+'.json'))
    
    # calculate pay
    pay = get_pay(data)
    pay.to_json(path.join(data_dir, 'worker_pay'+time.strftime("%m_%d_%Y-%I_%M_%S")+'.json'))
    print('Finished saving worker pay')

    
    
