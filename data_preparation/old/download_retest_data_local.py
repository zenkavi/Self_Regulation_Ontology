import sys
sys.path.append('/Users/zeynepenkavi/Dropbox/PoldrackLab/expfactory-analysis')
from expanalysis.experiments.jspsych import calc_time_taken, get_post_task_responses
from expanalysis.experiments.processing import post_process_data, extract_DVs
from expanalysis.results import get_filters
import json
import numpy as np
from os import path
import pandas as pd

sys.path.append('/Users/zeynepenkavi/Dropbox/PoldrackLab/Self_Regulation_Ontology/utils')
from data_preparation_utils import anonymize_data, calc_trial_order, convert_date, download_data, get_bonuses, get_pay,  remove_failed_subjects
from utils import get_info

#load Data
token = get_info('expfactory_token', infile='/corral-repl/utexas/poldracklab/users/zenkavi/Self_Regulation_Ontology/Self_Regulation_Retest_Settings_Local.txt')
try:
    data_dir=get_info('data_directory', infile='/corral-repl/utexas/poldracklab/users/zenkavi/Self_Regulation_Ontology/Self_Regulation_Retest_Settings_Local.txt')
except Exception:
    data_dir=path.join(get_info('base_directory', infile='/corral-repl/utexas/poldracklab/users/zenkavi/Self_Regulation_Ontology/Self_Regulation_Retest_Settings_Local.txt'),'Data')

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
#mturk_data_file_name = 'mturk_data_'+job_num+'.json'
data = download_data(data_dir, access_token, filters = filters,  battery = 'Self Regulation Retest Battery' , url = 'http://expfactory.org/api/results', file_name = 'mturk_data.json')
data.reset_index(drop = True, inplace = True)    

    
    
