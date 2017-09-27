import sys
sys.path.append('/oak/stanford/groups/russpold/users/zenkavi/expfactory-analysis')
sys.path.append('/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/data_preparation')
from expanalysis.experiments.jspsych import calc_time_taken, get_post_task_responses
from expanalysis.experiments.processing import post_process_data, extract_DVs, extract_experiment
from expanalysis.experiments.utils import remove_duplicates, result_filter
from expanalysis.results import get_filters, get_result_fields
from expanalysis.results import Result
import json
import numpy as np
from os import path, makedirs
import pandas as pd
import datetime
import pickle

from selfregulation.utils.data_preparation_utils import calc_trial_order, convert_date, get_bonuses, get_pay, remove_failed_subjects
from selfregulation.utils.utils import get_info
from selfregulation.utils.retest_data_utils import anonymize_retest_data

#set token and data directory
token = get_info('expfactory_token', infile='/Users/zeynepenkavi/Documents/PoldrackLabLocal/Self_Regulation_Ontology/Self_Regulation_Retest_Settings.txt')
release_date = datetime.date.today().strftime("%m-%d-%Y")
data_dir=path.join('/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/','Retest_'+release_date, 'Local')

if not path.exists(data_dir):
    makedirs(data_dir)


# Set up filters
filters = get_filters()
drop_columns = ['battery_description', 'experiment_reference', 'experiment_version', \
         'experiment_name','experiment_cognitive_atlas_task']
for col in drop_columns:
    filters[col] = {'drop': True}

# Strip token from specified file
f = open(token)
access_token = f.read().strip()

# Set up variables for the download request
battery = 'Self Regulation Retest Battery' 
url = 'http://www.expfactory.org/new_api/results/62/'
file_name = 'mturk_retest_data.json'

fields = get_result_fields()

# Create results object
results = Result(access_token, filters = filters, url = url)

# Clean filters from results objects
results.clean_results(filters)

# Extract data from the results object
data = results.data

# Remainder of download_data
data = result_filter(data, battery = battery)
remove_duplicates(data)
data = data.query('worker_id not in ["A254JKSDNE44AM", "A1O51P5O9MC5LX"]') # Sandbox workers
data.reset_index(drop = True, inplace = True) 

# Save data
data.to_json(path.join(data_dir, file_name))

#load data (in case anything broke)
#data = pd.read_json(path.join(data_dir, 'mturk_retest_data.json'))

# In case index got messed up
data.reset_index(drop = True, inplace = True)

worker_lookup = anonymize_retest_data(data, data_dir)
json.dump(worker_lookup, open(path.join(data_dir, 'retest_worker_lookup.json'),'w'))

# record subject completion statistics
(data.groupby('worker_id').count().finishtime).to_json(path.join(data_dir, 'retest_worker_counts.json'))

# add a few extras
convert_date(data)
bonuses = get_bonuses(data)
calc_time_taken(data)
get_post_task_responses(data)   
calc_trial_order(data)

# save data (gives an error but works? - NO EMPTY FILE. FIGURE OUT!
# seems to be a memory issue; temp solution with pickling)
file_name = 'mturk_retest_data_extras'

try:
    data.to_json(path.join(data_dir, file_name+'.json'))
except:
    pickle.dump(data, open(path.join(data_dir, file_name+'.pkl'), 'wb'), -1)
    

# calculate pay
pay = get_pay(data)
pay.to_json(path.join(data_dir, 'retest_worker_pay.json'))

# create dataframe to hold failed data
failed_data = pd.DataFrame()

post_process_data(data)

failures = remove_failed_subjects(data)
failed_data = pd.concat([failed_data,failures])
data.to_json(path.join(data_dir,'mturk_retest_data_post.json'))

# save failed data
failed_data = failed_data.reset_index(drop = True)
failed_data.to_json(path.join(data_dir, 'mturk_retest_failed_data_post.json'))


