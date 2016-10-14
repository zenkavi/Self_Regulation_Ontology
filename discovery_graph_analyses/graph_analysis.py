import bct
from collections import OrderedDict as odict
import fancyimpute
import igraph
import numpy as np
from os import path
import pandas as pd
import seaborn as sns
import sys

sys.path.append('../utils')
from graph_utils import calc_connectivity_mat, community_reorder, get_subgraph, get_visual_style, Graph_Analysis
from graph_utils import plot_graph, print_community_members, threshold_proportional_sign
from plot_utils import dendroheatmap
from utils import get_behav_data

# get dependent variables
DV_df = get_behav_data('Discovery_10-11-2016', use_EZ = True)
    
# ************************************
# ************ Imputation *******************
# ************************************
DV_df_complete = fancyimpute.SoftImpute().complete(DV_df)
DV_df_complete = pd.DataFrame(DV_df_complete, index = DV_df.index, columns = DV_df.columns)

# ************************************
# ************ Connectivity Matrix *******************
# ************************************

spearman_connectivity = calc_connectivity_mat(DV_df_complete, edge_metric = 'spearman')
distance_connectivity = calc_connectivity_mat(DV_df_complete, edge_metric = 'distance')


# ************************************
# ********* Heatmaps *******************
# ************************************
# dendrogram heatmap
plot_df = DV_df.corr(method = 'spearman')
plot_df.columns = [' '.join(x.split('_')) for x in  plot_df.columns]
fig = dendroheatmap(plot_df.corr(), labels = True)


# ************************************
# ********* Graphs *******************
# ************************************

def get_fully_connected_threshold(connectivity_matrix, initial_value = .1):
    '''Get a threshold above the initial value such that the graph is fully connected
    '''
    if type(connectivity_matrix) == pd.DataFrame:
        connectivity_matrix = connectivity_matrix.as_matrix()
    threshold = initial_value
    thresholded_mat = bct.threshold_proportional(connectivity_matrix,threshold)
    while np.any(np.max(thresholded_mat, axis = 1)==0):
        threshold += .01
        thresholded_mat = bct.threshold_proportional(connectivity_matrix,threshold)
    return threshold



# threshold positive graph
t = .5
plot_t = get_fully_connected_threshold(spearman_connectivity, .05)
t_f = bct.threshold_proportional
c_a = bct.community_louvain

G_w, connectivity_adj, threshold_visual_style = Graph_Analysis(spearman_connectivity, community_alg = c_a, 
                                                    thresh_func = t_f, threshold = t, plot_threshold = plot_t,
                                                     print_options = {'lookup': {}}, 
                                                    plot_options = {'inline': False})

# distance graph
t = 1
plot_t = get_fully_connected_threshold(distance_connectivity, .05)
t_f = bct.threshold_proportional
c_a = lambda x: bct.community_louvain(x, gamma = 1)

G_w, connectivity_adj, visual_style = Graph_Analysis(distance_connectivity, community_alg = c_a, thresh_func = t_f,
                                                      threshold = t, plot_threshold = plot_t,
                                                     print_options = {'lookup': {}}, 
                                                    plot_options = {'inline': False})


# signed graph
t = 1
t_f = threshold_proportional_sign
c_a = bct.modularity_louvain_und_sign                                               

# circle layout                                                  
G_w, connectivity_mat, visual_style = Graph_Analysis(spearman_connectivity, community_alg = c_a, thresh_func = t_f,
                                                     reorder = True, threshold = t,  layout = 'circle', 
                                                     plot_threshold = plot_t, print_options = {'lookup': {}}, 
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
# distance graph
t = 1
plot_t = .1
em = 'distance'
t_f = bct.threshold_proportional

layout = None
ref_community = None
for gamma in np.arange(.5,2,.05):
    c_a = lambda x: bct.community_louvain(x, gamma = gamma)
    layout = layout or 'kk'
    G_w, connectivity_adj, visual_style = Graph_Analysis(distance_connectivity, community_alg = c_a, ref_community = ref_community,
                                                         thresh_func = t_f, threshold = t, plot_threshold = plot_t, 
                                                         layout = layout,
                                                         print_options = {'lookup': {}, 'file': 'Plots/gamma=' + str(gamma) + '_weighted_distance.txt'}, 
                                                        plot_options = {'inline': False, 'target': 'Plots/gamma=' + str(gamma) + '_weighted_distance.pdf'})
    if type(layout) != igraph.layout.Layout:
        layout = visual_style['layout']
    ref_community = G_w.vs['community']
                                            
                                            
# spearman thresholded graph
t = .5
plot_t = .15
t_f = bct.threshold_proportional


layout = None
ref_community = None
for gamma in np.arange(.5,2,.05):
    layout = layout or 'kk'
    c_a = lambda x: bct.community_louvain(x, gamma = gamma)
    G_w, connectivity_adj, visual_style = Graph_Analysis(spearman_connectivity, community_alg = c_a, ref_community = ref_community,
                                                         thresh_func = t_f, threshold = t, plot_threshold = plot_t, 
                                                         layout = layout, 
                                                         print_options = {'lookup': {}, 'file': 'Plots/gamma=' + str(gamma) + '_weighted_spearman.txt'}, 
                                                        plot_options = {'inline': False, 'target': 'Plots/gamma=' + str(gamma) + '_weighted_spearman.pdf'})
    if type(layout) != igraph.layout.Layout:
        layout = visual_style['layout']
    ref_community = G_w.vs['community']

                                    
# spearman thresholded graph
t = 1
t_f = bct.threshold_proportional


layout = None
ref_community = None
reorder = True
graph_mat = spearman_connectivity
for gamma in np.arange(.5,2,.05):
    c_a = lambda x: bct.modularity_louvain_und_sign(x, gamma = gamma)
    layout = layout or 'circle'
    G_w, connectivity_adj, visual_style = Graph_Analysis(graph_mat, community_alg = c_a, ref_community = ref_community,
                                                         thresh_func = t_f, threshold = t, plot_threshold = plot_t, 
                                                         layout = layout, reorder = reorder,
                                                         print_options = {'lookup': {}, 'file': 'Plots/gamma=' + str(gamma) + '_weighted_spearman_sign.txt'}, 
                                                        plot_options = {'inline': False, 'target': 'Plots/gamma=' + str(gamma) + '_weighted_spearman_sign.pdf'})
    if type(layout) != igraph.layout.Layout:
        layout = visual_style['layout']
        reorder = False
        graph_mat = connectivity_adj
    ref_community = G_w.vs['community']
                                                    
                                                    
                                                    
                                                     
subgraph = community_reorder(get_subgraph(G_w,2))
print_community_members(subgraph)
subgraph_visual_style = get_visual_style(subgraph, vertex_size = 'eigen_centrality')
plot_graph(subgraph, visual_style = subgraph_visual_style, layout = 'circle', inline = False)












