#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 22:40:47 2016

@author: ian
"""
import pandas as pd
import numpy as np
import bct
import igraph
from pprint import pprint


class Graph_Analysis(object):
    def __init__(self):
        self.data = None
        self.graph_mat = None
        self.G = None
        self.weight = True
        self.thresh_func = bct.threshold_proportional
        self.threshold = .15
        self.community_alg = bct.community_louvain
        self.ref_community = None
        self.reorder = False
        self.visual_style = None
        self.print_options = {}
        self.plot_options = {}
        
    def setup(self, data, w=None, t=None, thresh_func=None, community_alg=None, 
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
        assert type(data) == pd.DataFrame, 'data must be a pandas dataframe'
        self.data = data
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
            
    def calculate_communities(self):
        G = self.G
        graph_mat = self.graph_mat
        # bct.modularity_louvain_und_sign
        comm, mod = self.community_alg(graph_mat)
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
        # calculate subgraph (within-community) characteristics
        self._subgraph_analysis()
    
    def display(self, plot=True, verbose=True, print_options=None, plot_options=None):
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
        
    def graph_to_dataframe(self):
        matrix = self.graph_to_matrix()
        graph_dataframe = pd.DataFrame(data = matrix, columns = self.G.vs['name'], index = self.G.vs['name'])
        return graph_dataframe
    
    def set_visual_style(self, layout ='kk', plot_threshold=None, labels=None):
        """
        layout: str: 'kk', 'circle', 'grid' or other igraph layouts, optional
        Determines how the graph is displayed
        """
        layout_graph = None
        if plot_threshold:
            layout_mat = self.thresh_func(self.data.values, plot_threshold)  
            layout_graph = igraph.Graph.Weighted_Adjacency(layout_mat.tolist(), mode = 'undirected')
        if labels=='auto':
            labels = self.G.vs['id']
        self.visual_style = self._get_visual_style(layout = layout, layout_graph = layout_graph, vertex_size = 'eigen_centrality', labels = labels,
                                        size = 6000)
    
    def _get_visual_style(self, layout, layout_graph = None, vertex_size = None, size = 1000, labels = None):
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
        
    def _graph_to_matrix(self, G):
        if 'weight' in G.es.attribute_names():
            graph_mat = np.array(G.get_adjacency(attribute = 'weight').data)
        else:
            graph_mat = np.array(G.get_adjacency().data)
        return graph_mat
    
    def _plot_graph(self, **kwargs):
        visual_style = self.visual_style
        visual_style.update(**kwargs)
        fig = igraph.plot(self.G, **visual_style)
        return fig
    
        
    def _print_community_members(self, lookup = {}, file = None):
        G = self.G
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

    
        
