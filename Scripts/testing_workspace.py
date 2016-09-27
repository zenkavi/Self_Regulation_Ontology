from expanalysis.experiments.processing import extract_row, post_process_data, post_process_exp, extract_experiment, calc_DVs, extract_DVs,flag_data,  get_DV, generate_reference
from graphs import Graph_Analysis, threshold_proportional_sign
import bct
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
failed_data = pd.read_json(data_dir + 'mturk_failed_data_post.json').reset_index(drop = True)
failed_data = failed_data[np.logical_not(failed_data.worker_id.str.contains('s5'))]

# get DV df
DV_df = pd.read_json(data_dir + 'mturk_discovery_DV.json')
valence_df = pd.read_json(data_dir + 'mturk_discovery_DV_valence.json')

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
subset = drop_vars(DV_df)
# make subset without EZ variables
noEZ_subset = drop_vars(subset, drop_vars = ['_EZ'])
# make subset without acc/rt vars
EZ_subset = drop_vars(subset, drop_vars = ['_acc', '_rt'])

#make data subsets:
survey_df = subset.filter(regex = 'survey')


graph_data = EZ_subset
# plot graph
t = .15
em = 'spearman'
t_f = bct.threshold_proportional
c_a = bct.community_louvain

G_w, connectivity_adj, threshold_visual = Graph_Analysis(graph_data, community_alg = c_a, thresh_func = t_f, edge_metric = em, 
                                                     reorder = False, threshold = t, plot_threshold = .04, weight = True, layout = 'kk',
                                                     print_options = {'lookup': {}}, 
                                                    plot_options = {'inline': False})
                                                    
t = .9
em = 'spearman'
t_f = threshold_proportional_sign
c_a = bct.modularity_louvain_und_sign                                               

# kk layout using from thresholded positive weights                                              
G_w, connectivity_mat, visual_style = Graph_Analysis(graph_data, community_alg = c_a, thresh_func = t_f, edge_metric = em, 
                                                     reorder = False, threshold = t, weight = True, layout = threshold_visual['layout'], 
                                                    plot_threshold = .02, print_options = {'lookup': {}}, 
                                                    plot_options = {'inline': False}) 
# circle layout                                                  
G_w, connectivity_mat, visual_style = Graph_Analysis(graph_data, community_alg = c_a, thresh_func = t_f, edge_metric = em, 
                                                     reorder = True, threshold = t, weight = True, layout = 'circle', 
                                                     plot_threshold = .03, print_options = {'lookup': {}}, 
                                                    plot_options = {'inline': False})
                                                    
# save graph
G_w, connectivity_mat, visual_style = Graph_Analysis(graph_data, community_alg = c_a, thresh_func = t_f, edge_metric = em, 
                                                     reorder = True, threshold = t, weight = True, layout = 'kk', 
                                                     print_options = {'lookup': {}, 'file': '../Plots/weighted_' + em + '.txt'}, 
                                                    plot_options = {'inline': False, 'target': '../Plots/weighted_' + em + '.pdf'})
                                                     
subgraph = community_reorder(get_subgraph(G_w,1))
print_community_members(subgraph)
subgraph_visual_style = get_visual_style(subgraph, vertex_size = 'eigen_centrality')
plot_graph(subgraph, visual_style = subgraph_visual_style, layout = 'circle', inline = False)

# dendrogram heatmap
plot_df = subset.corr()
plot_df.columns = [' '.join(x.split('_')) for x in  plot_df.columns]
fig = dendroheatmap(plot_df.corr(), labels = True)


