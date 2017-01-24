import sys
sys.path.append('/Users/zeynepenkavi/Dropbox/PoldrackLab/expfactory-analysis')
sys.path.append('/Users/zeynepenkavi/Documents/PoldrackLabLocal/Self_Regulation_Ontology/data_preparation')
from expanalysis.experiments.jspsych import calc_time_taken, get_post_task_responses
from expanalysis.experiments.processing import post_process_data, extract_DVs
from expanalysis.results import get_filters, get_fields
import json
import numpy as np
from os import path
import pandas as pd
from time import time

from selfregulation.utils.data_preparation_utils import anonymize_data, calc_trial_order, convert_date, download_data, get_bonuses, get_pay,  remove_failed_subjects
from selfregulation.utils.utils import get_info

#set token and data directory
token = get_info('expfactory_token', infile='/Users/zeynepenkavi/Documents/PoldrackLabLocal/Self_Regulation_Ontology/Self_Regulation_Retest_Settings_Local_NewApi.txt')
data_dir=get_info('retest_data_directory', infile='/Users/zeynepenkavi/Documents/PoldrackLabLocal/Self_Regulation_Ontology/Self_Regulation_Retest_Settings_Local_NewApi.txt')

# Setup options
pd.set_option('display.width', 200)

# Set up filters
filters = get_filters()
drop_columns = ['battery_description', 'experiment_reference', 'experiment_version', \
         'experiment_name','experiment_cognitive_atlas_task']
for col in drop_columns:
    filters[col] = {'drop': True}

# Strip token from specified file
f = open(token)
access_token = f.read().strip()

# Patch function to make sure data is unpecked correctly from the json object
def results_to_df(results,fields):
        '''results_to_df converts json result into a dataframe of json objects
        :param fields: list of (top level) fields to parse
        '''
        tmp = pandas.DataFrame(results.json)
        results.data = pandas.DataFrame()
        for field in fields:
#            if isinstance(tmp[field].values[0],dict):
            if sum([isinstance(tmp[field].values[i],dict) for i in range(0,tmp.shape[0])]) == tmp.shape[0]:
                try:
#                    field_df = pandas.concat([pandas.DataFrame.from_dict(item, orient='index').T for item in tmp[field]])
#                    field_df = pandas.DataFrame.from_dict([tmp[field].values[0]])
#                    field_df = pandas.concat([pandas.DataFrame.from_dict(item, orient='index').T for item in iter(tmp[field].values) ])
                    field_df = pandas.concat([pandas.DataFrame.from_dict([item]) for item in iter(tmp[field].values) ])
                    field_df.index = range(0,field_df.shape[0])
                    field_df.columns = ["%s_%s" %(field,x) for x in field_df.columns.tolist()]
                    results.data = pandas.concat([results.data,field_df],axis=1)
                except:
                    results.data[field] = tmp[field]                   
            else:
                 results.data[field] = tmp[field]

# Set up variables for the download request
battery = 'Self Regulation Retest Battery' 
url = 'http://www.expfactory.org/new_api/results/62/'
file_name = 'mturk_retest_data_manual_newapi.json'

fields = get_fields()

# Create results object
results = Result(access_token, filters = filters, url = url)
results_to_df(results, fields)
results.clean_results(filters)

# Extract data from the results object
data = results.data

# Remainder of download_data
data = result_filter(data, battery = battery)
remove_duplicates(data)
data = data.query('worker_id not in ["A254JKSDNE44AM", "A1O51P5O9MC5LX"]') # Sandbox workers
data.reset_index(drop = True, inplace = True) 

# Save data
data.to_json(data_dir + file_name)

# In case index got messed up
data.reset_index(drop = True, inplace = True)

# anonymize data
worker_lookup = anonymize_data(data)
json.dump(worker_lookup, open(path.join(data_dir, 'retest_worker_lookup.json'),'w'))

# record subject completion statistics
(data.groupby('worker_id').count().finishtime).to_json(path.join(data_dir, 'retest_worker_counts.json'))

# add a few extras
convert_date(data)
bonuses = get_bonuses(data)
calc_time_taken(data)
get_post_task_responses(data)   
calc_trial_order(data)

# save data (gives an error but works?)
file_name = 'mturk_retest_data_manual_extras.json'
data.to_json(data_dir + file_name)

# calculate pay
pay = get_pay(data)
pay.to_json(path.join(data_dir, 'retest_worker_pay.json'))
)

# create dataframe to hold failed data
failed_data = pd.DataFrame()

post_process_data(data)

failures = remove_failed_subjects(data)
failed_data = pd.concat([failed_data,failures])
data.to_json(path.join(data_dir,'mturk_retest_data_manual_post.json'))

# save failed data
failed_data = failed_data.reset_index(drop = True)
failed_data.to_json(data_dir + 'mturk_retest_failed_data_manual_post.json')

## DV calculations

## To be created
    
# DV_valence.csv:
from expanalysis.experiments.processing import get_battery_DVs
DV_df, valence_df = get_battery_DVs(data)
DV_df.to_json(path.join(data_dir, 'mturk_' + sample + '_DV.json'))
valence_df.to_json(path.join(data_dir, 'mturk_' + sample + '_DV_valence.json'))    


# Individual_Measures/ - done
from selfregulation.utils.data_preparation_utils import save_task_data
save_task_data(data_dir, data)

#alcohol_drugs.csv
#alcohol_drugs_ordinal.csv
#demographic_health.csv

#demographics.csv
#demographics_ordinal.csv
demog_data = extract_experiment(data,'demographics_survey')
from process_demographics import process_demographics
demog_data = process_demographics(demog_data, directory, meta_dir)
    

health.csv
health_ordinal.csv
items.csv.gz
meaningful_variables.csv
meaningful_variables_EZ.csv
meaningful_variables_clean.csv
meaningful_variables_hddm.csv
meaningful_variables_imputed.csv
meaningful_variables_imputed_for_task_selection.csv
meaningful_variables_noDDM.csv

#metadata/
#references/
meta_dir = path.join(data_dir,'metadata')
reference_dir = path.join(data_dir,'references')
from os import makedirs
if not path.exists(meta_dir):
    makedirs(meta_dir)
if not path.exists(reference_dir):
    makedirs(reference_dir)

short_DV_valence.csv
short_meaningful_variables.csv
short_meaningful_variables_EZ.csv
short_meaningful_variables_clean.csv
short_meaningful_variables_hddm.csv
short_meaningful_variables_imputed.csv
short_meaningful_variables_imputed_for_task_selection.csv
short_meaningful_variables_noDDM.csv
short_subject_x_items.csv
short_taskdata.csv
short_taskdata_clean.csv
short_taskdata_imputed.csv
short_taskdata_imputed_for_task_selection.csv
short_variables_exhaustive.csv
subject_x_items.csv
taskdata.csv
taskdata_clean.csv
taskdata_imputed.csv
taskdata_imputed_for_task_selection.csv
variables_exhaustive.csv
