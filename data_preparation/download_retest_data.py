from expanalysis.experiments.jspsych import calc_time_taken, get_post_task_responses
from expanalysis.experiments.processing import post_process_data, extract_DVs
from expanalysis.results import get_filters
import json
import numpy as np
from os import path
import pandas as pd
import sys

sys.path.append('../utils')
from data_preparation_utils import anonymize_data, calc_trial_order, convert_date, download_data, get_bonuses, get_pay,  remove_failed_subjects
from utils import get_info

# Fix Python 2.x.
try: input = raw_input
except NameError: pass
    
# get options
job = input('Type "download", "extras", "post", "DV",  or "all": ')
# sample = 'discovery'
sample = 'retest'
if job == 'more':
    job = input('More: Type "download", "extras", "post" or "both": ')
#     sample = input('Type "discovery", "validation" or "incomplete". Use commas to separate multiple samples or "all": ')
    sample = input('Type "retest" or "incomplete". Use commas to separate multiple samples or "all": ')
    if sample == 'all':
#         sample = ['discovery','validation','incomplete']
        sample = ['retest','incomplete']
    else:
        sample = sample.split(',')   
        
#load Data
token = get_info('expfactory_token', infile='../Self_Regulation_Settings_Retest.txt')
try:
    data_dir=get_info('data_directory', infile='../Self_Regulation_Settings_Retest.txt')
except Exception:
    data_dir=path.join(get_info('base_directory', infile='../Self_Regulation_Settings_Retest.txt'),'Data')

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
    data = download_data(data_dir, access_token, filters = filters,  battery = 'Self Regulation Retest Battery')
    data.reset_index(drop = True, inplace = True)
    
if job in ['extras', 'all']:
    #Process Data
    if job == "extras":
        #load Data
        data = pd.read_json(path.join(data_dir,  + 'mturk_data.json'))
        data.reset_index(drop = True, inplace = True)
        print('Finished loading raw data')
    
    #anonymize data
    worker_lookup = anonymize_data(data)
    json.dump(worker_lookup, open(path.join(data_dir,  + 'worker_lookup.json','w')))
    
    # record subject completion statistics
    (data.groupby('worker_id').count().finishtime).to_json(path.join(data_dir,  + 'worker_counts.json'))
    
    # add a few extras
    convert_date(data)
    bonuses = get_bonuses(data)
    calc_time_taken(data)
    get_post_task_responses(data)   
    calc_trial_order(data)
    
    # save data
    data.to_json(path.join(data_dir,  + 'mturk_data_extras.json'))
    
    # calculate pay
    pay = get_pay(data)
    pay.to_json(path.join(data_dir,  + 'worker_pay.json'))
    print('Finished saving worker pay')
    
if job in ['post', 'all']:
    #Process Data
    if job == "post":
        #load Data
        try:
            data = pd.read_json(path.join(data_dir, 'mturk_data_extras.json'))
        except ValueError:
            data = pd.read_json(path.join(data_dir,  + 'mturk_data.json'))
        data.reset_index(drop = True, inplace = True)
        print('Finished loading raw data')
    
    #get subject assignment
    # subject_assignment = pd.read_csv('samples/subject_assignment.csv', index_col = 0)
    subject_assignment_retest = pd.read_csv('samples/subject_assignment_retest.csv', index_col = 0)
    # discovery_sample = list(subject_assignment.query('dataset == "discovery"').index)
    # validation_sample = list(subject_assignment.query('dataset == "validation"').index)
    retest_sample = list(subject_assignment_retest.query('dataset == "retest"').index)
    # extra_sample =  ['s' + str(i) for i in range(501,600)]
    extra_sample =  ['s' + str(i) for i in range(51,100)]
    
    # create dataframe to hold failed data
    failed_data = pd.DataFrame()
    # preprocess extras
    # only get extra data
    extra_data = data.query('worker_id in %s' % extra_sample).reset_index(drop = True)
    post_process_data(extra_data)
    failures = remove_failed_subjects(extra_data)
    failed_data = pd.concat([failed_data,failures])
    extra_workers = np.sort(extra_data.worker_id.unique())
    print('Finished processing extra data')    
    
    if 'retest' in sample:
        # only get retest data
        retest_data = data.query('worker_id in %s' % retest_sample).reset_index(drop = True)
        post_process_data(retest_data)
        failures = remove_failed_subjects(retest_data)
        failed_data = pd.concat([failed_data,failures])
        # add extra workers if necessary
        num_failures = len(failures.worker_id.unique())
        if num_failures > 0:
            makeup_workers = extra_workers[0:num_failures]
            new_data = extra_data[extra_data['worker_id'].isin(makeup_workers)]
            retest_data = pd.concat([retest_data, new_data]).reset_index(drop = True)
            extra_data.drop(new_data.index, inplace = True)
        retest_data.to_json(path.join(data_dir,'mturk_retest_data_post.json'))
        print('Finished saving post-processed retest data')

    # preprocess and save each sample individually
    # if 'discovery' in sample:
    #     # only get discovery data
    #     discovery_data = data.query('worker_id in %s' % discovery_sample).reset_index(drop = True)
    #     post_process_data(discovery_data)
    #     failures = remove_failed_subjects(discovery_data)
    #     failed_data = pd.concat([failed_data,failures])
    #     # add extra workers if necessary
    #     num_failures = len(failures.worker_id.unique())
    #     if num_failures > 0:
    #         makeup_workers = extra_workers[0:num_failures]
    #         new_data = extra_data[extra_data['worker_id'].isin(makeup_workers)]
    #         discovery_data = pd.concat([discovery_data, new_data]).reset_index(drop = True)
    #         extra_data.drop(new_data.index, inplace = True)
    #     discovery_data.to_json(path.join(data_dir,'mturk_discovery_data_post.json'))
    #     print('Finished saving post-processed discovery data')
        
    # if 'validation' in sample:
    #     # only get validation data
    #     validation_data = data.query('worker_id in %s' % validation_sample).reset_index(drop = True)
    #     post_process_data(validation_data)
    #     failures = remove_failed_subjects(validation_data)
    #     failed_data = pd.concat([failed_data,failures])
    #     # add extra workers if necessary
    #     num_failures = len(failures.worker_id.unique())
    #     if num_failures > 0:
    #         makeup_workers = extra_workers[0:num_failures]
    #         new_data = extra_data[extra_data['worker_id'].isin(makeup_workers)]
    #         validation_data = pd.concat([validation_data, new_data]).reset_index(drop = True)
    #         extra_data.drop(new_data.index, inplace = True)
    #     validation_data.to_json(path.join(data_dir,'mturk_validation_data_post.json'))
    #     print('Finished saving post-processed validation data')
        
    if 'incomplete' in sample:
        # only get incomplete data
        # incomplete_data = data.query('worker_id not in %s' % (validation_sample + discovery_sample + extra_sample)).reset_index(drop = True)
        incomplete_data = data.query('worker_id not in %s' % (retest_sample + extra_sample)).reset_index(drop = True)
        post_process_data(incomplete_data)
        remove_failed_subjects(incomplete_data)
        incomplete_data.to_json(data_dir + 'mturk_incomplete_data_post.json')
        print('Finished saving post-processed incomplete data')
    # save failed data
    failed_data = failed_data.reset_index(drop = True)
    failed_data.to_json(data_dir + 'mturk_failed_data_post.json')
    print('Finished saving post-processed failed data')
    
if job in ['DV', 'all']:
    # for sample in ['discovery', 'failed']:
    for sample in ['retest', 'failed']:
        data = pd.read_json(data_dir + 'mturk_' + sample + '_data_post.json')
        #calculate DVs
        DV_df, valence_df = extract_DVs(data)
        DV_df.to_json(path.join(data_dir, 'mturk_' + sample + '_DV.json'))
        valence_df.to_json(path.join(data_dir, 'mturk_' + sample + '_DV_valence.json'))

    
    
