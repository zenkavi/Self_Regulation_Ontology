import bct
import pickle
import igraph
import numpy as np
import pandas as pd
from pprint import pprint
import seaborn as sns
import matplotlib.pyplot as plt

# Utilities/Small World
def simulate(rep = 1000, fun = lambda: gen_random_graph(100,100)):
    output = []
    for _ in range(rep):
        output.append(fun())
    return output
        
def gen_random_graph(n = 10, m = 10, template_graph = None):
    if template_graph:
        G = template_graph.copy()
        G.rewire() # bct.randomizer_bin_und works on binary adjrices
    else:
        #  Generates a random binary graph with n vertices and m edges
        G = igraph.Graph.Erdos_Renyi(n = n, m = m)    
    # get cluster coeffcient. Transitivity is closed triangles/total triplets
    c = G.transitivity_undirected() 
    # get average (shortest) path length
    l = G.average_path_length()
    return (G,c,l,c/l)

def calc_small_world(G):
    # simulate random graphs with same number of nodes and edges
    sim_out = simulate(rep = 10000, fun = lambda: gen_random_graph(n = len(G.vs), m = len(G.es)))
    # get average C and L for random
    C_random = np.mean([i[1] for i in sim_out])
    L_random = np.mean([i[2] for i in sim_out])
    # calculate relative clustering and path length vs random networks
    Gamma = G.transitivity_undirected()/C_random
    Lambda = G.average_path_length()/L_random
    # small world coefficient
    Sigma = gamma/lam   
    return (Sigma, Gamma, Lambda)
    
# Visualization
def plot_graph(G, visual_style = None, inline = True):
    if not visual_style:
        visual_style = {}
        if 'weight' in G.es:
            visual_style['edge_width'] = [10 * weight for weight in G.es['weight']]
        visual_style['layout'] = G_bin.layout("kk")
    fig = igraph.plot(G, inline = inline, **visual_style)
    return fig

def plot_mat(mat, labels = []):
    fig = plt.figure(figsize = [16,12])
    ax = fig.add_axes([.25,.15,.7,.7]) 
    sns.heatmap(mat)
    if len(labels) > 0:
        ax.set_yticklabels(labels, rotation = 0, fontsize = 'large')
    return fig
    
def print_community_members(G, lookup = {}):
    assert set(['community','id','within_module_degree','name',  'eigen_centrality']) <=  set(G.vs.attribute_names()), \
        "Missing some required vertex attributes. Vertices must have id, name, part_coef, eigen_centrality, community and within_module_degree"
        
    print('Key: Node index, Within Module Degree, Measure, Eigenvector centrality')
    for community in np.unique(G.vs['community']):
        #find members
        members = [lookup.get(v['name'],v['name']) for v in G.vs if v['community'] == community]
        # ids and total degree
        ids = [v['id'] for v in G.vs if v['community'] == community]
        eigen = [round(v['eigen_centrality'],2) for v in G.vs if v['community'] == community]
        #sort by within degree
        within_degrees = [round(v['within_module_degree'],2) for v in G.vs if v['community'] == community]
        to_print = zip(ids, within_degrees,  members, eigen)
        to_print.sort(key = lambda x: -x[1])
        
        print('Members of community ' + str(community) + ':')
        pprint(to_print)
        print('')

# Main graph analysis function
def Graph_Analysis(data, threshold = .15, weight = True, layout = 'kk', reorder = True, display = True, filey = None):
    connectivity_matrix = data.corr(method = 'spearman').as_matrix()
    # remove diagnoal (required by bct) and uppder triangle
    np.fill_diagonal(connectivity_matrix,0)
    
    #threshold
    graph_mat = bct.threshold_proportional(connectivity_matrix,threshold)
    # make a binary version if not weighted
    if not weight:
        graph_mat = np.ceil(graph_mat)
        G = igraph.Graph.Adjacency(graph_mat.tolist(), mode = 'undirected')
    else:
        G = igraph.Graph.Weighted_Adjacency(graph_mat.tolist(), mode = 'undirected')

    # community detection
    # using louvain but also bct.modularity_und which is "Newman's spectral community detection"
    # bct.modularity_louvain_und_sign
    comm, mod = bct.community_louvain(graph_mat)
    
    #if reorder, reorder vertices by community membership
    if reorder:
        data = data.iloc[:,np.argsort(comm)]
        comm = np.sort(comm)
        connectivity_matrix = data.corr().as_matrix()
        # remove diagnoal (required by bct) and uppder triangle
        np.fill_diagonal(connectivity_matrix,0)
        
        #threshold
        graph_mat = bct.threshold_proportional(connectivity_matrix,threshold)
        # make a binary version if not weighted
        if not weight:
            graph_mat = np.ceil(graph_mat)
            G = igraph.Graph.Adjacency(graph_mat.tolist(), mode = 'undirected')
        else:
            G = igraph.Graph.Weighted_Adjacency(graph_mat.tolist(), mode = 'undirected')
    
    G.vs['community'] = comm
    G.vs['id'] = range(len(G.vs))
    G.vs['name'] = data.columns 
    G.vs['within_module_degree'] = bct.module_degree_zscore(graph_mat,comm)
    G.vs['part_coef'] = bct.participation_coef(graph_mat, comm)
    if weight:
        G.vs['eigen_centrality'] = G.eigenvector_centrality(directed = False, weights = G.es['weight'])
    else:
        G.vs['eigen_centrality'] = G.eigenvector_centrality(directed = False)
    
    if display:
        # plot community structure
        # make layout:
        graph_layout = G.layout(layout)
        
        
        
        # color by community and within-module-centrality
        # each community is a different color palette, darks colors are more central to the module
        if comm.max() <= 6:
            num_colors = 20.0
            palettes = ['Blues','Reds','Greens','Greys','Purples','Oranges']
            
            min_degree = np.min(G.vs['within_module_degree'])
            max_degree = np.max(G.vs['within_module_degree']-min_degree)
            within_degree = [(v-min_degree)/max_degree for v in G.vs['within_module_degree']]
            within_degree = np.digitize(within_degree, bins = np.arange(0,1,1/num_colors))
            
            vertex_color = [sns.color_palette(palettes[v['community']-1], int(num_colors)+1)[within_degree[i]] for i,v in enumerate(G.vs)]
        else:
            palette = sns.cubehelix_palette(comm.max())
            vertex_color = [palette[v['community']-1] for v in G.vs]
            
        visual_style = {'layout': graph_layout, 
                        'vertex_color': vertex_color, 
                        'vertex_size': [c*200+100 for c in G.vs['eigen_centrality']], 
                        'vertex_label': G.vs['id'],
                        'vertex_label_size': 75,
                        'bbox': (6000,6000),
                        'margin': 500}
        if weight:
            visual_style['edge_width'] = [w*10 for w in G.es['weight']]
        
        #visualize
        print_community_members(G, lookup = {})
        if filey:
            plot_graph(G, filey, visual_style = visual_style, inline = False)
        else:
            plot_graph(G, visual_style = visual_style, inline = False)
    return (G, graph_mat)