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

#if len(sys.argv) < 5:
#    sys.exit("Usage: download_retest_data_tacc.py 'all' '['retest', 'incomplete']' 'http://expfactory.org/api/results' 'http://expfactory.org/api/results/?page=3' '1' ")

if len(sys.argv) < 4:
    sys.exit("Usage: download_retest_data_tacc.py 'Self Regulation Retest Battery' 'http://expfactory.org/api/results' 'http://expfactory.org/api/results/?page=3' '1' ")


#job = sys.argv[1]
#sample = sys.argv[2]
#url = sys.argv[3]
#last_url = sys.argv[4]
#job_num = sys.argv[5]

battery_name = sys.argv[1]
url = sys.argv[2]
last_url = sys.argv[3]
job_num = sys.argv[4]

        
#load Data
token = get_info('expfactory_token', infile='/corral-repl/utexas/poldracklab/users/zenkavi/Self_Regulation_Ontology/Self_Regulation_Retest_Settings_Tacc.txt')
try:
    data_dir=get_info('data_directory', infile='/corral-repl/utexas/poldracklab/users/zenkavi/Self_Regulation_Ontology/Self_Regulation_Retest_Settings_Tacc.txt')
except Exception:
    data_dir=path.join(get_info('base_directory', infile='/corral-repl/utexas/poldracklab/users/zenkavi/Self_Regulation_Ontology/Self_Regulation_Retest_Settings_Tacc.txt'),'Data')

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
mturk_data_file_name = 'mturk_data_'+job_num+'.json'
data = download_data(data_dir, access_token, filters = filters,  battery = battery_name, url = url, last_url = last_url, file_name = mturk_data_file_name)
data.reset_index(drop = True, inplace = True)    

#if job == 'download' or job == "all":
#    #***************************************************
#    # ********* Load Data **********************
#    #**************************************************        
#    pd.set_option('display.width', 200)
#    figsize = [16,12]
#    #set up filters
#    filters = get_filters()
#    drop_columns = ['battery_description', 'experiment_reference', 'experiment_version', \
#             'experiment_name','experiment_cognitive_atlas_task']
#    for col in drop_columns:
#        filters[col] = {'drop': True}
#    
#    #***************************************************
#    # ********* Download Data**********************
#    #**************************************************  
#    #load Data
#    f = open(token)
#    access_token = f.read().strip()
#    mturk_data_file_name = 'mturk_data_'+job_num+'.json'
#    data = download_data(data_dir, access_token, filters = filters,  battery = 'Self Regulation Retest Battery', url = url, last_url = last_url, file_name = mturk_data_file_name)
#    data.reset_index(drop = True, inplace = True)
    
#if job in ['extras', 'all']:
#    #Process Data
#    if job == "extras":
#        #load Data
#        data = pd.read_json(path.join(data_dir, 'mturk_data_',job_num,'.json'))
#        data.reset_index(drop = True, inplace = True)
#        print('Finished loading raw data')
#    
#    #anonymize data
#    worker_lookup = anonymize_data(data)
#    worker_lookup_file_name = 'worker_lookup_'+job_num+'.json'
#    json.dump(worker_lookup, open(path.join(data_dir, worker_lookup_file_name),'w'))
#    
#    # record subject completion statistics
#    worker_counts_file_name = 'worker_counts'+job_num+'.json'
#    (data.groupby('worker_id').count().finishtime).to_json(path.join(data_dir, worker_counts_file_name))
#    
#    # add a few extras
#    convert_date(data)
#    bonuses = get_bonuses(data)
#    calc_time_taken(data)
#    get_post_task_responses(data)   
#    calc_trial_order(data)
#    
#    # save data
#    mturk_data_extras_file_name = 'mturk_data_extras'+job_num+'.json'
#    data.to_json(path.join(data_dir, mturk_data_extras_file_name))
#    
#    # calculate pay
#    pay = get_pay(data)
#    worker_pay_file_name='worker_pay'+job_num+'.json'
#    pay.to_json(path.join(data_dir, worker_pay_file_name))
#    print('Finished saving worker pay')

    ##make this script to only download and save the data
    ##write another to put all downloaded data together
    ##potentially in the same script have the extra stuff calculated (lookup, countrs, extras etc)
    ##also write looping script to generate commands for the exec file
