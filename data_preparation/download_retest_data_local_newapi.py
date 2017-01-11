import sys
sys.path.append('/Users/zeynepenkavi/Dropbox/PoldrackLab/expfactory-analysis')
from expanalysis.experiments.jspsych import calc_time_taken, get_post_task_responses
from expanalysis.experiments.processing import post_process_data, extract_DVs
from expanalysis.results import get_filters
import json
import numpy as np
from os import path
import pandas as pd

from selfregulation.utils.data_preparation_utils import anonymize_data, calc_trial_order, convert_date, download_data, get_bonuses, get_pay,  remove_failed_subjects
from selfregulation.utils.utils import get_info

#set token and data directory
token = get_info('expfactory_token', infile='/Users/zeynepenkavi/Documents/PoldrackLabLocal/Self_Regulation_Ontology/Self_Regulation_Retest_Settings_Local_NewApi.txt')
data_dir=get_info('retest_data_directory', infile='/Users/zeynepenkavi/Documents/PoldrackLabLocal/Self_Regulation_Ontology/Self_Regulation_Retest_Settings_Local_NewApi.txt')

# Setup options
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
#Haven't gotten the battery selection part work on py3 so avoiding it since using the new api it only downloads the relevant battery
#data = download_data(data_dir, access_token, filters = filters,  battery = 'Self Regulation Retest Battery' , url = 'http://www.expfactory.org/new_api/results/62/', file_name = 'mturk_data_newapi.json')
data = download_data(data_dir, access_token, filters = filters, url = 'http://www.expfactory.org/new_api/results/62/', file_name = 'mturk_data_newapi.json')
data.reset_index(drop = True, inplace = True)    

#Reload in case this is what is breaking the date conversion
data = pd.read_json(path.join(data_dir, 'mturk_data_newapi.json'))
data.reset_index(drop = True, inplace = True)
print('Finished re-loading raw data')    
    
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