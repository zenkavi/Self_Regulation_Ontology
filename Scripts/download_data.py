from expanalysis.results import get_filters
import pandas as pd
from util import load_data, get_bonuses
from expanalysis.experiments.jspsych import calc_time_taken, get_post_task_responses
from expanalysis.experiments.processing import post_process_data

job = raw_input('Type "download", "post" or "both": ')

token, data_dir = [line.rstrip('\n').split()[1] for line in open('../Self_Regulation_Settings.txt')]
data_file = data_dir + 'Battery_Results'

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
    data_source = load_data(data_file, access_token, filters = filters, source = 'web', battery = 'Self Regulation Battery')
if job == "post" or job == "both":
    #Process Data
    if job == "post":
        #load Data
        data_source = load_data(data_file, source = 'file', battery = 'Self Regulation Battery')
        print('Finished loading raw data')
    data = data_source.query('worker_id not in ["A254JKSDNE44AM", "A1O51P5O9MC5LX"]') # Sandbox workers
    data.reset_index(drop = False, inplace = True)
    # add a few extras
    bonuses = get_bonuses(data)
    calc_time_taken(data)
    get_post_task_responses(data)
    
    # preprocess and save
    post_process_data(data)
    data.to_json(data_file + '_data_post.json')
    print('Finished saving post-processed data')
