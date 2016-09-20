from expanalysis.experiments.processing import extract_row, post_process_data, post_process_exp, extract_experiment, calc_DVs, extract_DVs,flag_data,  get_DV, generate_reference
from graphs import Graph_Analysis
import json
import numpy as np
import pandas as pd
import seaborn as sns
from util import *

#work around for spyder bug in python 3
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)



#***************************************************
# ********* Load Data **********************
#************************************************** 
try:
    worker_lookup = json.load(open("../Data/worker_lookup.json",'r'))
    inverse_lookup = {v: k for k, v in worker_lookup.items()}
except IOError:
    print('no worker lookup found!')

try:
    worker_counts = json.load(open("../Data/worker_counts.json",'r'))
except IOError:
    print('no worker counts found!')
    
try:
    worker_pay = pd.read_json("../Data/worker_pay.json",'r')
except IOError:
    print('no worker pay found!')

#get pay
pay = worker_pay
workers = []
pay_list = [pay.total.get(inverse_lookup.get(w,'not found'),'not_found') if pay.base.get(inverse_lookup.get(w,'not found'),'not_found') != 60 else pay.bonuses.get(inverse_lookup.get(w,'not found'),'not_found') for w in workers]

#load Data
token = get_info('expfactory_token')
try:
    data_dir=get_info('data_directory')
except Exception:
    data_dir=get_info('base_directory') + 'Data/'


# read preprocessed data
data = pd.read_json(data_dir + 'mturk_discovery_data_post.json').reset_index(drop = True)

# get DV df
#DV_df = pd.read_json(data_dir + 'mturk_discovery_DV.json')
DV_df, valence_df = extract_DVs(data,use_group_fun = False)

#save data
save_task_data(data_dir, data)

    

# ************************************
# ********* Save Components of Data **
# ************************************
items_df = get_items(data)
items_pivot_df = items_df.pivot('worker','item_ID','coded_response')

# ************************************
# ********* DVs **********************
# ************************************
# get all DVs

#flip negative signed valence DVs
flip_df = valence_df.replace(to_replace ={'Pos': 1, 'NA': 1, 'Neg': -1}).iloc[0]
for c in DV_df.columns:
    print(c)
    try:
        DV_df.loc[:,c] = DV_df.loc[:,c] * flip_df.loc[c]
    except TypeError:
        continue
    
#drop na columns
DV_df.dropna(axis = 1, how = 'all', inplace = True)
# drop other columns of no interest
drop_vars = "missed_percent|acc|avg_rt_error|std_rt_error|avg_rt|std_rt|post_error_slowing|\
congruency_seq_rt|congruency_seq_acc|demographics|go_acc|stop_acc|go_rt_error|go_rt_std_error|\
go_rt|go_rt_std|stop_rt_error|stop_rt_error_std|SS_delay"
subset = DV_df.drop(DV_df.filter(regex=drop_vars).columns, axis = 1)

#make data subsets:
survey_df = subset.filter(regex = 'survey')

# plot graph
Graph_Analysis(subset.corr().dropna(axis=[0,1], how='all'), threshold = .2)
Graph_Analysis(abs(subset.corr().dropna(axis=[0,1], how='all')), threshold = .2)
Graph_Analysis(survey_df.corr().dropna(axis=[0,1], how='all'), threshold = .4)

# dendrogram heatmap
plot_df = subset.corr()
plot_df.columns = [' '.join(x.split('_')) for x in  plot_df.columns]
fig = dendroheatmap(plot_df.corr(), labels = True)


