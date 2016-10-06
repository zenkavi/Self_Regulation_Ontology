from expanalysis.experiments.processing import  extract_experiment
from os import makedirs, path
import pandas as pd
import sys

sys.path.append('../utils')
from data_preparation_utils import drop_vars
from utils import get_info

#******************************
#*** Helper Functions *********
#******************************

def get_items(data):
    excluded_surveys = ['holt_laury_survey', 'selection_optimization_compensation_survey', 'sensation_seeking_survey']
    items = []
    responses = []
    responses_text = []
    options = []
    workers = []
    item_nums = []
    exps = []
    for exp in data.experiment_exp_id.unique():
        if 'survey' in exp and exp not in excluded_surveys:
            survey = extract_experiment(data,exp)
            try:
                responses += list(survey.response.map(lambda x: float(x)))
            except ValueError:
                continue
            items += list(survey.text)
            responses_text += list(survey.response_text)
            options += list(survey.options)
            workers += list(survey.worker_id)
            item_nums += list(survey.question_num)
            exps += [exp] * len(survey.text)
    
    items_df = pd.DataFrame({'survey': exps, 'worker': workers, 'item_text': items, 'coded_response': responses,
                             'response_text': responses_text, 'options': options}, dtype = float)
    items_df.loc[:,'item_num'] = [str(i).zfill(3) for i in item_nums]
    items_df.loc[:,'item_ID'] = items_df['survey'] + '_' + items_df['item_num'].astype(str)
    return items_df
    
def save_task_data(data_loc, data):
    save_path = path.join(data_loc,'Individual_Measures')
    if not path.exists(save_path):
        makedirs(save_path)
    for exp_id in data.experiment_exp_id.unique():
        print('Saving %s...' % exp_id)
        extract_experiment(data,exp_id).to_csv(path.join(save_path, exp_id + '.csv'))
    
#******************************
#*** Save Data *********
#******************************

#load Data
try:
    data_dir=get_info('data_directory')
except Exception:
    data_dir=path.join(get_info('base_directory'),'Data')

discovery_directory = path.join(data_dir, 'Discovery_9-26-16')
failed_directory = path.join(data_dir, 'Failed_9-26-16')
local_dir = path.join(data_dir,'Local')
if not path.exists(local_dir):
    makedirs(local_dir)

# read preprocessed data
discovery_data = pd.read_json(path.join(local_dir,'mturk_discovery_data_post.json')).reset_index(drop = True)
failed_data = pd.read_json(path.join(local_dir,'mturk_failed_data_post.json')).reset_index(drop = True)

for data,directory in [(discovery_data, discovery_directory), (failed_data, failed_directory)]:
    # save target datasets
    print('Saving to %s...' % directory)
    print('Saving target measures...')
    extract_experiment(data,'demographics_survey').to_csv(path.join(directory, 'demographics.csv'))
    extract_experiment(data, 'alcohol_drugs_survey').to_csv(path.join(directory, 'alcohol_drugs.csv'))
    extract_experiment(data, 'k6_survey').to_csv(path.join(directory, 'k6_health.csv'))
    # save items
    items_df = get_items(data)
    print('Saving items...')
    items_df.to_csv(path.join(directory, 'items.csv'))
    subjectsxitems = items_df.pivot('worker','item_ID','coded_response')
    subjectsxitems.to_csv(path.join(directory, 'subject_x_items.csv'))
    
# save Individual Measures
save_task_data(path.join(local_dir,'discovery'), data)


# ************************************
# ********* Save DV dataframes **
# ************************************
directory = discovery_directory

# get DV df
DV_df = pd.read_json(path.join(local_dir,'mturk_discovery_DV.json'))
valence_df = pd.read_json(path.join(local_dir,'mturk_discovery_DV_valence.json'))

#flip negative signed valence DVs
flip_df = valence_df.replace(to_replace ={'Pos': 1, 'NA': 1, 'Neg': -1}).mean()
for c in DV_df.columns:
    try:
        DV_df.loc[:,c] = DV_df.loc[:,c] * flip_df.loc[c]
    except TypeError:
        continue
#save valence
flip_df.to_csv(path.join(directory, 'DV_valence.csv'))
   
#drop na columns
DV_df.dropna(axis = 1, how = 'all', inplace = True)
# drop other columns of no interest
subset = drop_vars(DV_df)
subset.to_csv(path.join(directory, 'meaningful_variables.csv'))
# make subset without EZ variables
noEZ_subset = drop_vars(subset, drop_vars = ['_EZ'])
noEZ_subset.to_csv(path.join(directory, 'meaningful_variables_noEZ_contrasts.csv'))
# make subset without acc/rt vars
EZ_subset = drop_vars(subset, drop_vars = ['_acc', '_rt'])
EZ_subset.to_csv(path.join(directory, 'meaningful_variables_EZ_contrasts.csv'))
# make survey subset
survey_subset = subset.filter(regex = 'survey')
survey_subset.to_csv(path.join(directory, 'meaningful_variables_surveys.csv'))