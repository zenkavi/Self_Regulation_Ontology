import sys
sys.path.append('/Users/zeynepenkavi/Dropbox/PoldrackLab/expfactory-analysis')
sys.path.append('/Users/zeynepenkavi/Documents/PoldrackLabLocal/Self_Regulation_Ontology/data_preparation')
from expanalysis.experiments.jspsych import calc_time_taken, get_post_task_responses
from expanalysis.experiments.processing import post_process_data, extract_DVs, extract_experiment
from expanalysis.results import get_filters, get_result_fields
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

fields = get_result_fields()

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

#load data (in case anything broke)
#data = pd.read_json(path.join(data_dir, 'mturk_retest_data_manual_newapi.json'))

# add a few extras
convert_date(data)
bonuses = get_bonuses(data)
calc_time_taken(data)
# might get error
get_post_task_responses(data)   
calc_trial_order(data)

# save data (gives an error but works? - NO EMPTY FILE. FIGURE OUT!)
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
    
# Individual_Measures/ - done
from selfregulation.utils.data_preparation_utils import save_task_data
save_task_data(data_dir, data)

#metadata/ - done
#references/ - done
meta_dir = path.join(data_dir,'metadata')
reference_dir = path.join(data_dir,'references')
from os import makedirs
if not path.exists(meta_dir):
    makedirs(meta_dir)
if not path.exists(reference_dir):
    makedirs(reference_dir)

#demographics.csv - done
#demographics_ordinal.csv - done
demog_data = extract_experiment(data,'demographics_survey')
from process_demographics import process_demographics
demog_data = process_demographics(demog_data, data_dir, meta_dir)

#alcohol_drugs.csv - done
#alcohol_drugs_ordinal.csv - done
alcohol_drug_data = extract_experiment(data,'alcohol_drugs_survey')
from process_alcohol_drug import process_alcohol_drug

#health.csv - done
#health_ordinal.csv - done
health_data = extract_experiment(data,'k6_survey')
from process_health import process_health
health_data = process_health(health_data, data_dir, meta_dir)

#demographic_health.csv - done
target_data = pd.concat([demog_data, alcohol_drug_data, health_data], axis = 1)
target_data.to_csv(path.join(data_dir,'demographic_health.csv'))

#reference/demographic_health_reference.csv - done
np.savetxt(path.join(reference_dir,'demographic_health_reference.csv'), target_data.columns, fmt = '%s', delimiter=",")
    
#items.csv.gz - done
from selfregulation.utils.data_preparation_utils import get_items
items_df = get_items(data)
subjectsxitems = items_df.pivot('worker','item_ID','coded_response')
assert subjectsxitems.shape[1] == 594, "Wrong number of items found"
items_df.to_csv(path.join(data_dir, 'items.csv.gz'), compression = 'gzip')

#subject_x_items.csv - done
subjectsxitems.to_csv(path.join(data_dir, 'subject_x_items.csv'))

from os import chdir
chdir('/Users/zeynepenkavi/Documents/PoldrackLabLocal/Self_Regulation_Ontology/Data')
from selfregulation.utils.data_preparation_utils import convert_var_names
convert_var_names(subjectsxitems)
assert np.max([len(name) for name in subjectsxitems.columns])<=8, "Found column names longer than 8 characters in short version"

readme_lines = []
readme_lines += ["demographics_survey.csv: demographic information from expfactory-surveys\n\n"]
readme_lines += ["alcohol_drug_survey.csv: alcohol, smoking, marijuana and other drugs from expfactory-surveys\n\n"]
readme_lines += ["ky_survey.csv: mental health and neurological/health conditions from expfactory-surveys\n\n"]
readme_lines += ["items.csv.gz: gzipped csv of all item information across surveys\n\n"]
readme_lines += ["subject_x_items.csv: reshaped items.csv such that rows are subjects and columns are individual items\n\n"]
readme_lines += ["Individual Measures: directory containing gzip compressed files for each individual measures\n\n"]    

#Read in DVs and valence
label = 'retest'
DVs = pd.read_json(path.join(data_dir,'mturk_' + label + '_DV.json'))
DVs_valence = pd.read_json(path.join(data_dir,'mturk_' + label + '_DV_valence.json'))
data = pd.read_json(path.join(data_dir,'mturk_' + label + '_data_manual_post.json')).reset_index(drop = True)

DV_df = DVs
valence_df = DVs_valence

del DVs, DVs_valence

# drop failed QC vars
from selfregulation.utils.data_preparation_utils import drop_failed_QC_vars
drop_failed_QC_vars(DV_df,data)

#save valence
def get_flip_list(valence_df):
        #flip negative signed valence DVs
        valence_df = valence_df.replace(to_replace={np.nan: 'NA'})
        flip_df = np.floor(valence_df.replace(to_replace ={'Pos': 1, 'NA': 1, 'Neg': -1}).mean())
        valence_df = pd.Series(data = [col.unique()[0] for i,col in valence_df.iteritems()], index = valence_df.columns)
        return flip_df, valence_df
flip_df, valence_df = get_flip_list(valence_df)
flip_df.to_csv(path.join(data_dir, 'DV_valence.csv'))
readme_lines += ["DV_valence.csv: Subjective assessment of whether each variable's 'natural' direction implies 'better' self regulation\n\n"]

#variables_exhaustive.csv
#drop na columns
DV_df.dropna(axis = 1, how = 'all', inplace = True)
DV_df.to_csv(path.join(data_dir, 'variables_exhaustive.csv'))
readme_lines += ["variables_exhaustive.csv: all variables calculated for each measure\n\n"]

# drop other columns of no interest
from selfregulation.utils.data_preparation_utils import drop_vars
subset = drop_vars(DV_df, saved_vars = ['simple_reaction_time.avg_rt', 'shift_task.acc'])

#meaningful_variables_noDDM.csv
# make subset without EZ variables
noDDM_subset = drop_vars(DV_df, saved_vars = ["\.acc$", "\.avg_rt$"])
noDDM_subset = drop_vars(noDDM_subset, drop_vars = ['EZ', 'hddm'])
noDDM_subset.to_csv(path.join(data_dir, 'meaningful_variables_noDDM.csv'))
readme_lines += ["meaningful_variables_noDDM.csv: subset of exhaustive data to only meaningful variables with DDM parameters removed\n\n"]

#meaningful_variables_EZ.csv
# make subset without acc/rt vars and just EZ DDM
EZ_subset = drop_vars(subset, drop_vars = ['_acc', '_rt', 'hddm'], saved_vars = ['simple_reaction_time.avg_rt', 'dospert_rt_survey'])
EZ_subset.to_csv(path.join(data_dir, 'meaningful_variables_EZ.csv'))
readme_lines += ["meaningful_variables_EZ.csv: subset of exhaustive data to only meaningful variables with rt/acc parameters removed (replaced by EZ DDM params)\n\n"]
        
#meaningful_variables_hddm.csv
# make subset without acc/rt vars and just hddm DDM
hddm_subset = drop_vars(subset, drop_vars = ['_acc', '_rt', 'EZ'], saved_vars = ['simple_reaction_time.avg_rt', 'dospert_rt_survey'])
hddm_subset.to_csv(path.join(data_dir, 'meaningful_variables_hddm.csv'))
readme_lines += ["meaningful_variables_hddm.csv: subset of exhaustive data to only meaningful variables with rt/acc parameters removed (replaced by hddm DDM params)\n\n"]

#meaningful_variables.csv
# save files that are selected for use
selected_variables = hddm_subset
selected_variables.to_csv(path.join(data_dir, 'meaningful_variables.csv'))
readme_lines += ["meaningful_variables.csv: Same as meaningful_variables_hddm.csv\n\n"]

#meaningful_variables_clean.csv
# clean data
from selfregulation.utils.data_preparation_utils import remove_correlated_task_variables, remove_outliers, transform_remove_skew
selected_variables_clean = remove_outliers(selected_variables) #getting some warning
selected_variables_clean = remove_correlated_task_variables(selected_variables_clean)
selected_variables_clean = transform_remove_skew(selected_variables_clean)
selected_variables_clean.to_csv(path.join(data_dir, 'meaningful_variables_clean.csv'))
readme_lines += ["meaningful_variables_clean.csv: same as meaningful_variables.csv with outliers defined as greater than 2.5 IQR from median removed from each column\n\n"]

#save selected variables
selected_variables_reference = valence_df
selected_variables_reference.loc[selected_variables.columns].to_csv(path.join(reference_dir, 'selected_variables_reference.csv'))

# imputed data
from selfregulation.utils.r_to_py_utils import missForest
selected_variables_imputed, error = missForest(selected_variables_clean)
selected_variables_imputed.to_csv(path.join(data_dir, 'meaningful_variables_imputed.csv'))
readme_lines += ["meaningful_variables_imputed.csv: meaningful_variables_clean.csv after imputation with missForest\n\n"]

#meaningful_variables_imputed.csv
#meaningful_variables_imputed_for_task_selection.csv
#short_DV_valence.csv
#short_meaningful_variables.csv
#short_meaningful_variables_EZ.csv
#short_meaningful_variables_clean.csv
#short_meaningful_variables_hddm.csv
#short_meaningful_variables_imputed.csv
#short_meaningful_variables_imputed_for_task_selection.csv
#short_meaningful_variables_noDDM.csv
#short_subject_x_items.csv
#short_taskdata.csv
#short_taskdata_clean.csv
#short_taskdata_imputed.csv
#short_taskdata_imputed_for_task_selection.csv
#short_variables_exhaustive.csv
#taskdata.csv
#taskdata_clean.csv
#taskdata_imputed.csv
#taskdata_imputed_for_task_selection.csv


