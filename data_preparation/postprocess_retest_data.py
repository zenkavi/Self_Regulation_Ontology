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

#Set token and directories
token = get_info('expfactory_token', infile='/Users/zeynepenkavi/Documents/PoldrackLabLocal/Self_Regulation_Ontology/Self_Regulation_Retest_Settings_Local_NewApi.txt')
try:
    data_dir=get_info('retest_data_directory', infile='/Users/zeynepenkavi/Documents/PoldrackLabLocal/Self_Regulation_Ontology/Self_Regulation_Retest_Settings_Local_NewApi.txt')
except Exception:
    data_dir=path.join(get_info('base_directory', infile='/Users/zeynepenkavi/Documents/PoldrackLabLocal/Self_Regulation_Ontology/Self_Regulation_Retest_Settings_Local_NewApi.txt'),'Data')

#load Data
retest_data = pd.read_json(path.join(data_dir, 'mturk_data_extras.json'))
retest_data.reset_index(drop = True, inplace = True)
print('Finished loading raw retest data')

# create dataframe to hold failed data
failed_data = pd.DataFrame()

post_process_data(retest_data)
failures = remove_failed_subjects(retest_data)
failed_data = pd.concat([failed_data,failures])
retest_data.to_json(path.join(data_dir,'mturk_retest_data_post.json'))
print('Finished saving post-processed retest data')

# save failed data
failed_data = failed_data.reset_index(drop = True)
failed_data.to_json(data_dir + 'mturk_retest_failed_data_post.json')
print('Finished saving post-processed retest failed data')