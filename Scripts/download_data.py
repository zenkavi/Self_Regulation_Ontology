from expanalysis.experiments.jspsych import calc_time_taken, get_post_task_responses
from expanalysis.experiments.processing import post_process_data, extract_DVs
from expanalysis.results import get_filters
import json
import os
import pandas as pd
from util import anonymize_data, download_data, get_bonuses, get_info, get_pay, quality_check

# Fix Python 2.x.
try: input = raw_input
except NameError: pass
    
# get options
job = input('Type "download", "extras", "post" or "all": ')
sample = 'discovery'
if job == 'more':
    job = input('More: Type "download", "extras", "post" or "both": ')
    sample = input('Type "discovery", "validation" or "incomplete". Use commas to separate multiple samples or "all": ')
    if sample == 'all':
        sample = ['discovery','validation','incomplete']
    else:
        sample = sample.split(',')   
        
token = get_info('expfactory_token')
try:
    data_dir=get_info('data_directory')
except Exception:
    data_dir=get_info('base_directory') + 'Data/'

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
    data = download_data(data_dir, access_token, filters = filters,  battery = 'Self Regulation Battery')
    data.reset_index(drop = True, inplace = True)
    
if job in ['extras', 'all']:
    #Process Data
    if job == "extras":
        #load Data
        data = pd.read_json(data_dir + 'mturk_data.json')
        data.reset_index(drop = True, inplace = True)
        print('Finished loading raw data')
    
    #anonymize data
    worker_lookup = anonymize_data(data)
    json.dump(worker_lookup, open(data_dir + 'worker_lookup.json','w'))
    
    # record subject completion statistics
    (data.groupby('worker_id').count().finishtime).to_json(data_dir + 'worker_counts.json')
    
    # add a few extras
    bonuses = get_bonuses(data)
    calc_time_taken(data)
    get_post_task_responses(data)   
    quality_check(data)
    
    # save data
    os.remove(data_dir + 'mturk_data.json')
    data.to_json(data_dir + 'mturk_data_extras.json')
    
    # calculate pay
    pay = get_pay(data)
    pay.to_json(data_dir + 'worker_pay.json')
    print('Finished saving worker pay')
    
if job in ['post', 'all']:
    #Process Data
    if job == "post":
        #load Data
        try:
            data = pd.read_json(data_dir + 'mturk_data_extras.json')
        except ValueError:
            data = pd.read_json(data_dir + 'mturk_data.json')
        data.reset_index(drop = True, inplace = True)
        print('Finished loading raw data')
    
    #get subject assignment
    subject_assignment = pd.read_csv('../subject_assignment.csv')
    discovery_sample = list(subject_assignment.query('dataset == "discovery"').iloc[:,0])
    validation_sample = list(subject_assignment.query('dataset == "validation"').iloc[:,0])
        
    # preprocess and save each sample individually
    if 'discovery' in sample:
        # only get discovery data
        discovery_data = data.query('worker_id in %s' % discovery_sample).reset_index(drop = True)
        post_process_data(discovery_data)
        discovery_data.to_json(data_dir + 'mturk_discovery_data_post.json')
        print('Finished saving post-processed discovery data')
    if 'validation' in sample:
        # only get validation data
        validation_data = data.query('worker_id in %s' % validation_sample).reset_index(drop = True)
        post_process_data(validation_data)
        validation_data.to_json(data_dir + 'mturk_validation_data_post.json')
        print('Finished saving post-processed validation data')
    if 'incomplete' in sample:
        # only get validation data
        incomplete_data = data.query('worker_id not in %s' % (validation_sample + discovery_sample)).reset_index(drop = True)
        post_process_data(incomplete_data)
        incomplete_data.to_json(data_dir + 'mturk_incomplete_data_post.json')
        print('Finished saving post-processed incomplete data')
    
    if 'discovery' in sample:
        #calculate DVs
        DV_df = extract_DVs(discovery_data)
        DV_df.to_json(data_dir + 'mturk_discovery_DV.json')

    
    
