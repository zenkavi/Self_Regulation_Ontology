import bct
from collections import OrderedDict as odict
import fancyimpute
from graph_util import community_reorder, get_subgraph, get_visual_style, Graph_Analysis, plot_graph, \
                    print_community_members, threshold_proportional_sign
import numpy as np
from os import path
import pandas as pd
from util import dendroheatmap, get_info
import seaborn as sns

#load Data
data_dir=path.join(get_info('base_directory'),'Data/Discovery_9-26-16')

# get DV df
DV_df = pd.read_csv(path.join(data_dir,'meaningful_variables_EZ_contrasts.csv'), index_col = 0)
valence_df = pd.read_json(path.join(data_dir, 'mturk_discovery_DV_valence.json'))


#flip negative signed valence DVs
flip_df = valence_df.replace(to_replace ={'Pos': 1, 'NA': 1, 'Neg': -1}).mean()
for c in DV_df.columns:
    try:
        DV_df.loc[:,c] = DV_df.loc[:,c] * flip_df.loc[c]
    except TypeError:
        continue
    

# ************************************
# ************ Imputation *******************
# ************************************
DV_df_complete = fancyimpute.SoftImpute().complete(DV_df)
DV_df_complete = pd.DataFrame(DV_df_complete, index = DV_df.index, columns = DV_df.columns)

# ************************************
# ********* Heatmaps *******************
# ************************************
# dendrogram heatmap
plot_df = DV_df.corr()
plot_df.columns = [' '.join(x.split('_')) for x in  plot_df.columns]
fig = dendroheatmap(plot_df.corr(), labels = True)


# ************************************
# ********* Graphs *******************
# ************************************


                    
graph_data = DV_df

# threshold positive graph
t = .15
em = 'spearman'
t_f = bct.threshold_proportional
c_a = bct.community_louvain

G_w, connectivity_adj, threshold_visual_style = Graph_Analysis(graph_data, community_alg = c_a, thresh_func = t_f, edge_metric = em, 
                                                     reorder = False, threshold = t, plot_threshold = .075, weight = True, layout = 'kk',
                                                     print_options = {'lookup': {}}, 
                                                    plot_options = {'inline': False})

# signed graph
t = 1
em = 'spearman'
t_f = threshold_proportional_sign
c_a = bct.modularity_louvain_und_sign                                               

# circle layout                                                  
G_w, connectivity_mat, visual_style = Graph_Analysis(graph_data, community_alg = c_a, thresh_func = t_f, edge_metric = em, 
                                                     reorder = True, threshold = t, weight = True, layout = 'circle', 
                                                     plot_threshold = 1, print_options = {'lookup': {}}, 
                                                    plot_options = {'inline': False})

def get_top_community_tasks(G, community = 1):
    vs = G.vs.select(community = community)
    tasks = np.unique(list(map(lambda x: x.split('.')[0], vs['name'])))
    task_importance = {}
    for task in tasks:
        measures = {v['name'].split('.')[1]:v['subgraph_eigen_centrality'] for v in vs if v['name'].split('.')[0] == task}
        total_centrality = np.mean(list(measures.values()))
        task_importance[task] = {'importance': total_centrality, 'measures': measures}
    ordered_importance = odict(sorted(task_importance.items(), key = lambda t: t[1]['importance'], reverse = True))
    return ordered_importance
    
    
                                                  
# save graph
G_w, connectivity_mat, visual_style = Graph_Analysis(graph_data, community_alg = c_a, thresh_func = t_f, edge_metric = em, 
                                                     reorder = True, threshold = t, weight = True, layout = 'kk', 
                                                     print_options = {'lookup': {}, 'file': '../Plots/weighted_' + em + '.txt'}, 
                                                    plot_options = {'inline': False, 'target': '../Plots/weighted_' + em + '.pdf'})
                                                     
subgraph = community_reorder(get_subgraph(G_w,1))
print_community_members(subgraph)
subgraph_visual_style = get_visual_style(subgraph, vertex_size = 'eigen_centrality')
plot_graph(subgraph, visual_style = subgraph_visual_style, layout = 'circle', inline = False)












