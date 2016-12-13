import bct
import igraph
from itertools import combinations 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.cluster import normalized_mutual_info_score

#work around for spyder bug in python 3
import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

# Utilities
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
    Sigma = Gamma/Lambda
    return (Sigma, Gamma, Lambda)

def distcorr(X, Y):
    """ Compute the distance correlation function
    
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    X,Y = zip(*[v for i,v in enumerate(zip(X,Y)) if not np.any(np.isnan(v))])
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

def distcorr_mat(M):
    n = M.shape[1]
    corr_mat = np.ones([n,n])
    for i in range(n):
        for j in range(i):
            distance_corr = distcorr(M[:,i], M[:,j])
            corr_mat[i,j] = corr_mat[j,i] =  distance_corr
    return corr_mat
    
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

def get_percentile_weight(W, percentile):
    return np.percentile(np.abs(W[np.tril_indices_from(W, k = -1)]),percentile)
    
def pairwise_MI(data):
    columns = data.columns
    MI_df = pd.DataFrame(index = columns, columns = columns)
    for c1,c2 in combinations(columns, 2):
        cleaned = data[[c1,c2]].dropna()
        MI = normalized_mutual_info_score(cleaned[c1], cleaned[c2])
        MI_df.loc[c1,c2] = MI
        MI_df.loc[c2,c1] = MI
    return MI_df.astype(float)
    
def simulate(rep = 1000, fun = lambda: gen_random_graph(100,100)):
    output = []
    for _ in range(rep):
        output.append(fun())
    return output
          
    
def threshold_proportional_sign(W, threshold):
    sign = np.sign(W)
    thresh_W = bct.threshold_proportional(np.abs(W), threshold)
    W = thresh_W * sign
    return W
        
# community functions      
def calc_connectivity_mat(data, edge_metric = 'pearson'):
    assert edge_metric in ['pearson','spearman','MI','abs_pearson','abs_spearman', 'distance'], \
        'Invalid edge metric passed. Must use "pearson", "spearman", "distance" or "MI"'
    if edge_metric == 'MI':
        connectivity_matrix = pd.DataFrame(pairwise_MI(data))
    elif edge_metric == 'distance':
        connectivity_matrix = pd.DataFrame(distcorr_mat(data.as_matrix()))
    else:
        *qualifier, edge_metric = edge_metric.split('_')
        if (qualifier or [None])[0] == 'abs':
            connectivity_matrix = abs(data.corr(method = edge_metric))
        else:
            connectivity_matrix = data.corr(method = edge_metric)
    connectivity_matrix.columns = data.columns
    connectivity_matrix.index = data.columns
    return connectivity_matrix
    
def get_fully_connected_threshold(connectivity_matrix):
    '''Get a threshold above the initial value such that the graph is fully connected
    '''
    threshold_mat = connectivity_matrix.values.copy()
    np.fill_diagonal(threshold_mat,0)
    abs_threshold = np.min(np.max(threshold_mat, axis = 1))
    proportional_threshold = np.mean(threshold_mat>=(abs_threshold-.001))
    return {'absolute': abs_threshold, 'proportional': proportional_threshold}            

def find_intersection(community, reference):
    ref_lists = [[i for i,c in enumerate(reference) if c==C] for C in np.unique(reference)]
    comm_lists = [[i for i,c in enumerate(community) if c==C] for C in np.unique(community)]
    # each element relates to a community
    intersection = [[len(set(ref).intersection(comm)) for ref in ref_lists] for comm in comm_lists]
    return np.array(intersection).T

def construct_community_tree(intersections, proportional=False):
    G = igraph.Graph()
    layer_start = 0
    colors = ['red','blue','green','violet']*4
    for intersection in intersections:
        if proportional:
            intersection = intersection/intersection.sum(axis=0)
        curr_color = colors.pop()
        origin_length = intersection.shape[0]
        target_length = intersection.shape[1]
        if len(G.vs)==0:
            G.add_vertices(origin_length)
        G.add_vertices(target_length)
        for i in range(origin_length):
            for j in range(target_length):
                G.add_edge(i+layer_start,j+origin_length+layer_start,weight=intersection[i,j],color = curr_color)
        layer_start+=intersection.shape[0]
    igraph.plot(G, layout = 'rt', **{'inline': False, 'vertex_label': range(len(G.vs)), 'edge_width':[w for w in G.es['weight']], 'edge_color': G.es['color'], 'bbox': (1000,1000)})
    #G.write_dot('test.dot')

    
# Graph Analysis Class Definition

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 22:40:47 2016

@author: ian
"""
import bct
import igraph
import numpy as np
import pandas as pd
from pprint import pprint
import seaborn as sns


class Graph_Analysis(object):
    def __init__(self):
        self.data = None
        self.graph_mat = None
        self.G = None
        self.node_order = []
        self.weight = True
        self.thresh_func = bct.threshold_proportional
        self.threshold = 1
        self.community_alg = bct.community_louvain
        self.ref_community = None
        self.visual_style = None
        self.print_options = {}
        self.plot_options = {}
        
    def setup(self, data=None, w=None, t=None, thresh_func=None, community_alg=None, 
              ref_community=None):
        """
        Creates and displays graphs of a data matrix.
        
        Parameters
        ----------
        data: pandas DataFrame
            data to use to create the graph
        thresh_func: function that takes in a connectivity matrix and thresholds
            any algorithm that returns a connectivity matrix of the same size as the original may be used.
            intended to be used with bct.threshold_proportional or bct.threshold_absolute
        community_alg: function that takes in a connectivity matrix and returns community assignment
            intended to use algorithms from brain connectivity toolbox like commnity_louvain or 
            modularity_und. Must return a list of community assignments followed by Q, the modularity
            index
        threshold: float 0 <= x <= 1, optional
            the proportion of weights to keep (to be passed to bct.threshold_proportion)
        weight: bool, optional
            if True, creates a weighted graph (vs. a binary)
            
        """
        if self.data is None:
            assert type(data) == pd.DataFrame, 'data must be a pandas dataframe'
        if t!=None:
            self.threshold = t
        if w!=None:
            assert(type(w)==bool), 'w must be a bool'
            self.weight = w
        if thresh_func!=None:
            self.thresh_func = thresh_func
        if community_alg!=None:
            self.community_alg = community_alg
        if ref_community!=None:
            self.ref_community = ref_community
        
        if data is not None:
            self.data = data
            # convert dataframe to matrix to be used for subsequent analyses
            graph_mat = self.thresh_func(data.values, self.threshold)
            # make a binary version if not weighted
            if not self.weight:
                graph_mat = np.ceil(graph_mat)
                G = igraph.Graph.Adjacency(graph_mat.tolist(), mode = 'undirected')
            else:
                G = igraph.Graph.Weighted_Adjacency(graph_mat.tolist(), mode = 'undirected')
            # label vertices of G
            G.vs['id'] = range(len(G.vs))
            G.vs['name'] = data.columns
            # set class variables
            self.graph_mat = graph_mat
            self.G = G
            self.node_order = list(range(graph_mat.shape[0]))
            
    def calculate_communities(self, reorder=False, **kwargs):
        G = self.G
        graph_mat = self.graph_mat
        # calculate community structure
        comm, mod = self.community_alg(graph_mat, **kwargs)
        # if there is a reference, relabel communities based on their closest association    
        if self.ref_community:
            comm = self._relabel_community(comm,self.ref_community)
        # label vertices of G
        G.vs['community'] = comm
        G.vs['within_module_degree'] = bct.module_degree_zscore(graph_mat,comm)
        if np.min(graph_mat) < 0:
            participation_pos, participation_neg = bct.participation_coef_sign(graph_mat, comm)
            G.vs['part_coef_pos'] = participation_pos
            G.vs['part_coef_neg'] = participation_neg
        else:
            G.vs['part_coef'] = bct.participation_coef(graph_mat, comm)
        
        if self.weight:
            G.vs['eigen_centrality'] = G.eigenvector_centrality(directed = False, weights = G.es['weight'])
        else:
            G.vs['eigen_centrality'] = G.eigenvector_centrality(directed = False)
        if reorder:
            self.reorder()
        # calculate subgraph (within-community) characteristics
        self._subgraph_analysis()
        return mod
    
    def create_visual_style(self, G, layout='kk', layout_graph = None, vertex_size = None, size = 1000, labels = None):
        """
        Creates an appropriate visual style for a graph. 
        
        Parameters
        ----------
        G: igraph object
            graph that the visual style will be based on
        layout: igraph.layout or str ('kk', 'circle', 'grid' or other igraph layouts), optional
            Determines how the graph is displayed. If a string is provided, assume it is
            a specification of a type of igraph layout using the graph provided. If an
            igraph.layout is provided it must conform to the number of vertices of the graph
        vertex_size: str, optional
            if defined, uses this graph metric to determine vertex size
        size: int, optinal
            determines overall size of graph display. Other sizes (vertex size, font size)
            are proportional to this value. Figures are always square
        labels: list the size of the number of vertices, optional
            Used to label vertices. Numbers are used if no labels are provided
        avg_num_edges: int > 1
            thresholds the edges on the graph so each node has, on average, avg_num_edges
            
        Returns
        ----------
        visual_style: dict
            the dictionary of attribute values to be used by igraph.plot
        """
        def make_layout():
            if type(layout) == igraph.layout.Layout:
                graph_layout = layout
                if layout_graph:
                    display_threshold = min([abs(i) for i in layout_graph.es['weight']])
                else:
                    display_threshold = 0
            elif type(layout) == str:
                if layout_graph:
                    graph_layout = layout_graph.layout(layout)
                    display_threshold = min([abs(i) for i in layout_graph.es['weight']])
                else:
                    graph_layout = G.layout(layout)
                    display_threshold = 0
            return graph_layout, display_threshold
        
        def get_vertex_colors():
            # color by community and within-module-centrality
            # each community is a different color palette, darks colors are more central to the module
            if 'community' in G.vs.attribute_names():
                community_count = np.max(G.vs['community'])
                if community_count <= 6 and 'subgraph_eigen_centrality' in G.vs.attribute_names():
                    num_colors = 20.0
                    palettes = ['Blues','Reds','Greens','Greys','Purples','Oranges']
                    
                    min_degree = np.min(G.vs['within_module_degree'])
                    max_degree = np.max(G.vs['within_module_degree']-min_degree)
                    within_degree = [(v-min_degree)/max_degree for v in G.vs['within_module_degree']]
                    within_degree = np.digitize(within_degree, bins = np.arange(0,1,1/num_colors))
                    
                    vertex_color = [sns.color_palette(palettes[v['community']-1], int(num_colors)+1)[within_degree[i]] for i,v in enumerate(G.vs)]
                else:
                    palette = sns.cubehelix_palette(max(community_count,10))
                    vertex_color = [palette[v['community']-1] for v in G.vs]
            else:
                vertex_color = 'red'
            return vertex_color
        
        def set_edges(visual_style):
            if 'weight' in G.es.attribute_names():
                thresholded_weights = [w if abs(w) > display_threshold else 0 for w in G.es['weight']]
                if layout_graph!=None:
                    if min(layout_graph.es['weight'])>0:
                        thresholded_weights = [w if w > display_threshold else 0 for w in G.es['weight']]
                visual_style['edge_width'] = [abs(w)**2.5*size/300.0 for w in thresholded_weights]
                if np.sum([e<0 for e in G.es['weight']]) > 0:
                    visual_style['edge_color'] = [['#3399FF','#696969','#FF6666'][int(np.sign(w)+1)] for w in G.es['weight']]
                else:
                    visual_style['edge_color'] = '#696969'
            
            
        G = self.G
        graph_layout, display_threshold = make_layout()
        vertex_color = get_vertex_colors()
        # set up visual style dictionary. Vertex label sizes are proportional to the total size
        visual_style = {'layout': graph_layout, 
                        'vertex_color': vertex_color, 
                        'vertex_label_size': size/130.0,
                        'bbox': (size,size),
                        'margin': size/20.0,
                        'inline': False}
        set_edges(visual_style)
        
        if vertex_size in G.vs.attribute_names():
            visual_style['vertex_size'] = [c*(size/60.0)+(size/100.0) for c in G.vs[vertex_size]]
        else:
            print('%s was not an attribute of G. Could not set vertex size!' % vertex_size)
        if labels:
            visual_style['vertex_label'] = labels
        
        return visual_style
        
    def display(self, plot=True, verbose=True,  print_options=None, plot_options=None):
        assert self.visual_style!=None, 'Must first call set_visual_style() !'
        if verbose:
            if print_options==None:
                print_options = {}
            try:
                self._print_community_members(**print_options)
            except KeyError:
                print('Communities not detected! Run calculate_communities() first!')
        if plot:
            if plot_options==None:
                plot_options = {}
                self._plot_graph(**plot_options)
                
    def get_subgraph(self, community = 1):
        G = self.G
        assert set(['community']) <=  set(G.vs.attribute_names()), \
            'No communities found! Call calculate_communities() first!'
        subgraph = G.induced_subgraph([v for v in G.vs if v['community'] == community])
        subgraph.vs['community'] = subgraph.vs['subgraph_community']
        subgraph.vs['eigen_centrality'] = subgraph.vs['subgraph_eigen_centrality']
        del subgraph.vs['subgraph_community']
        del subgraph.vs['subgraph_eigen_centrality']
        return subgraph
        
    def return_subgraph_analysis(self, community = 1):
        subgraph = self.graph_to_dataframe(self.get_subgraph(community))
        subgraph_GA=Graph_Analysis()
        subgraph_GA.setup(data=subgraph,
                          w=self.weight,
                          thresh_func = self.thresh_func,
                          community_alg=self.community_alg)
        return subgraph_GA
              
    
    def graph_to_dataframe(self, G=None):
        if G==None:
            G = self.G
        matrix = self._graph_to_matrix(G)
        graph_dataframe = pd.DataFrame(data = matrix, columns = G.vs['name'], index = G.vs['name'])
        return graph_dataframe
    
    def reorder(self, reorder_index=None):
        G = self.G
        if reorder_index==None:
            community = G.vs['community']
            reorder_index = np.argsort(community)
        
        # hold graph attributes:
        attribute_names = G.vs.attributes()
        attribute_values = [G.vs[a] for a in attribute_names]
        attribute_df = pd.DataFrame(attribute_values, index = attribute_names).T
        sorted_df = attribute_df.reindex(reorder_index).reset_index()
        
        # rearrange connectivity matrix
        graph_mat = self._graph_to_matrix(G)
        graph_mat = graph_mat[:,reorder_index][reorder_index]
    
        # make a binary version if not weighted
        if 'weight' in G.es.attribute_names():
            G = igraph.Graph.Weighted_Adjacency(graph_mat.tolist(), mode = 'undirected')
        else:
            G = igraph.Graph.Adjacency(graph_mat.tolist(), mode = 'undirected')
        for name in attribute_names:
            G.vs[name] = sorted_df[name]
        self.G = G
        self.graph_mat = self._graph_to_matrix(G)
        self.node_order = reorder_index
        
    def set_visual_style(self, layout ='kk', plot_threshold=None, labels='auto'):
        """
        layout: str: 'kk', 'circle', 'grid' or other igraph layouts, optional
        Determines how the graph is displayed
        """
        if layout=='circle':
            self.reorder()
        layout_graph = None
        if plot_threshold:
            layout_mat = self.thresh_func(self.data.values, plot_threshold)  
            layout_graph = igraph.Graph.Weighted_Adjacency(layout_mat.tolist(), mode = 'undirected')
        if labels=='auto':
            labels = self.G.vs['id']
        self.visual_style = self.create_visual_style(self.G, layout = layout, layout_graph = layout_graph, vertex_size = 'eigen_centrality', labels = labels,
                                        size = 6000)
        
  
    def _graph_to_matrix(self, G):
        if 'weight' in G.es.attribute_names():
            graph_mat = np.array(G.get_adjacency(attribute = 'weight').data)
        else:
            graph_mat = np.array(G.get_adjacency().data)
        return graph_mat
    
    def _plot_graph(self, G=None, visual_style=None, **kwargs):
        if G==None:
            G=self.G
        if visual_style==None:
            visual_style = self.visual_style
        visual_style.update(**kwargs)
        fig = igraph.plot(G, **visual_style)
        return fig
    
    def _print_community_members(self, G=None, lookup = {}, file = None):
        if G==None:
            G=self.G
        if file:
            f = open(file,'w')
        else:
            f = None
            
        print('Key: Node index, Subgraph Eigen Centrality, Measure, Eigenvector centrality', file = f)
        for community in np.unique(G.vs['community']):
            #find members
            members = [lookup.get(v['name'],v['name']) for v in G.vs if v['community'] == community]
            # ids and total degree
            ids = [v['id'] for v in G.vs if v['community'] == community]
            eigen = ["{0:.2f}".format(v['eigen_centrality']) for v in G.vs if v['community'] == community]
            #sort by within degree
            within_degrees = ["{0:.2f}".format(v['subgraph_eigen_centrality']) for v in G.vs if v['community'] == community]
            to_print = list(zip(ids, within_degrees,  members, eigen))
            to_print.sort(key = lambda x: -float(x[1]))
            
            print('Members of community ' + str(community) + ':', file = f)
            pprint(to_print, stream = f)
            print('', file = f)
        
    def _relabel_community(community, reference):
        ref_lists = [[i for i,c in enumerate(reference) if c==C] for C in np.unique(reference)]
        comm_lists = [[i for i,c in enumerate(community) if c==C] for C in np.unique(community)]
        relabel_dict = {}
        for ci,comm in enumerate(comm_lists):
            best_match = None
            best_count = 0
            for ri,ref in enumerate(ref_lists):
                count = len(set(ref).intersection(comm))
                if count > best_count:
                    best_count = count
                    best_match = ri + 1
            if best_match in relabel_dict.values():
                best_match = max(relabel_dict.values()) + 1
            relabel_dict[ci+1] = best_match
        return [relabel_dict[c] for c in community]
    
    def _subgraph_analysis(self):
        G = self.G
        community_alg = self.community_alg
        assert set(['community','id']) <=  set(G.vs.attribute_names()), \
            'Graph must have "community" and "id" as a vertex attributes'
        for c in np.unique(G.vs['community']):
            subgraph = G.induced_subgraph([v for v in G.vs if v['community'] == c])
            subgraph_mat = self._graph_to_matrix(subgraph)
            if 'weight' in G.es.attribute_names():
                subgraph.vs['eigen_centrality'] = subgraph.eigenvector_centrality(directed = False, weights = subgraph.es['weight'])
            else:
                subgraph.vs['eigen_centrality'] = subgraph.eigenvector_centrality(directed = False)
            G.vs.select(lambda v: v['id'] in subgraph.vs['id'])['subgraph_eigen_centrality'] = subgraph.vs['eigen_centrality']
            if community_alg:
                comm, Q = community_alg(subgraph_mat)
                subgraph.vs['community'] = comm
                G.vs.select(lambda v: v['id'] in subgraph.vs['id'])['subgraph_community'] = subgraph.vs['community']

    
        
