import datetime
from expanalysis.experiments.processing import  extract_experiment
from os import makedirs, path
import numpy as np
import pandas as pd
import sys
sys.path.append('../utils')
from data_preparation_utils import convert_var_names, drop_vars, get_items
from utils import get_info

#******************************
#*** Save Data *********
#******************************
date = datetime.date.today().strftime("%m-%d-%Y")

#load Data
try:
    data_dir=get_info('data_directory')
except Exception:
    data_dir=path.join(get_info('base_directory'),'Data')

discovery_directory = path.join(data_dir, 'Discovery_' + date)
failed_directory = path.join(data_dir, 'Failed_' + date)
local_dir = path.join(data_dir,'Local')
for directory in [discovery_directory, failed_directory, local_dir]:
    if not path.exists(directory):
        makedirs(directory)

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
    subjectsxitems = items_df.pivot('worker','item_ID','coded_response')
    # ensure there are the correct number of items
    assert subjectsxitems.shape[1] == 594, "Wrong number of items found"
    # save items
    items_df.to_csv(path.join(directory, 'items.csv.gz'), compression = 'gzip')
    subjectsxitems.to_csv(path.join(directory, 'subject_x_items.csv'))
    convert_var_names(subjectsxitems)
    assert np.max([len(name) for name in subjectsxitems.columns])<=8, \
        "Found column names longer than 8 characters in short version"
    subjectsxitems.to_csv(path.join(directory, 'short_subject_x_items.csv'))

   
# save Individual Measures
# save_task_data(path.join(local_dir,'discovery_' + date), data)


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
subset.to_csv(path.join(directory, 'meaningful_variables_exhaustive.csv'))
# make subset without EZ variables
noEZ_subset = drop_vars(subset, drop_vars = ['_EZ'])
noEZ_subset.to_csv(path.join(directory, 'meaningful_variables_noEZ_contrasts.csv'))
# make subset without acc/rt vars
EZ_subset = drop_vars(subset, drop_vars = ['_acc', '_rt'])
EZ_subset.to_csv(path.join(directory, 'meaningful_variables_EZ_contrasts.csv'))
EZ_subset.to_csv(path.join(directory, 'meaningful_variables.csv'))

# drop other columns of no interest
convert_var_names(subset)
subset.to_csv(path.join(directory, 'short_meaningful_variables_exhaustive.csv'))
# make subset without EZ variables
convert_var_names(noEZ_subset)
noEZ_subset.to_csv(path.join(directory, 'short_meaningful_variables_noEZ_contrasts.csv'))
# make subset without acc/rt vars
convert_var_names(EZ_subset)
EZ_subset.to_csv(path.join(directory, 'short_meaningful_variables_EZ_contrasts.csv'))
EZ_subset.to_csv(path.join(directory, 'short_meaningful_variables.csv'))






