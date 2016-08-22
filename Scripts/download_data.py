from expanalysis.experiments.jspsych import calc_time_taken, get_post_task_responses
from expanalysis.experiments.processing import post_process_data
from expanalysis.results import get_filters
import json
import pandas as pd
from util import load_data, get_bonuses, anonymize_data, quality_check

job = raw_input('Type "download", "post" or "both": ')
sample = 'discovery'
if job == 'more':
    job = raw_input('Type "download", "post" or "both": ')
    sample = raw_input('Type "discovery", "validation" or "incomplete". Use commas to separate multiple samples or "all": ')
    if sample == 'all':
        sample = ['discovery','validation','incomplete']
    else:
        sample = sample.split(',')
token, data_dir = [line.rstrip('\n').split(':')[1] for line in open('../Self_Regulation_Settings.txt')]
data_file = data_dir + 'mturk'

if job == 'download' or job == "both":
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
    action = raw_input('Overwrite data or append new data to previous file? Type "overwrite" or "append"')
    data_source = load_data(data_file, access_token, filters = filters, action = action, battery = 'Self Regulation Battery')

if job == "post" or job == "both":
    #Process Data
    if job == "post":
        #load Data
        data_source = load_data(data_file, action = 'file', battery = 'Self Regulation Battery')
        print('Finished loading raw data')
    data = data_source.query('worker_id not in ["A254JKSDNE44AM", "A1O51P5O9MC5LX"]') # Sandbox workers
    data.reset_index(drop = True, inplace = True)
    # add a few extras
    bonuses = get_bonuses(data)
    calc_time_taken(data)
    get_post_task_responses(data)
    quality_check(data)    
    
    #anonymize data and write anonymize lookup
    worker_lookup = anonymize_data(data)
    json.dump(worker_lookup, open(data_dir + 'worker_lookup.json','w'))
    subject_assignment = pd.read_csv('../subject_assignment.csv')
    discovery_sample = list(subject_assignment.query('dataset == "discovery"').iloc[:,0])
    validation_sample = list(subject_assignment.query('dataset == "validation"').iloc[:,0])
        
    # preprocess and save each sample individually
    if 'discovery' in sample:
        # only get discovery data
        discovery_data = data.query('worker_id in %s' % discovery_sample).reset_index(drop = True)
        post_process_data(discovery_data)
        discovery_data.to_json(data_file + '_discovery_data_post.json')
        print('Finished saving post-processed discovery data')
    if 'validation' in sample:
        # only get validation data
        validation_data = data.query('worker_id in %s' % validation_sample).reset_index(drop = True)
        post_process_data(validation_data)
        validation_data.to_json(data_file + '_validation_data_post.json')
        print('Finished saving post-processed validation data')
    if 'incomplete' in sample:
        # only get validation data
        incomplete_data = data.query('worker_id not in %s' % (validation_sample + discovery_sample)).reset_index(drop = True)
        post_process_data(incomplete_data)
        incomplete_data.to_json(data_file + '_incomplete_data_post.json')
        print('Finished saving post-processed incomplete data')

    
    
