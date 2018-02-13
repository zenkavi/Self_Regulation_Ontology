# imports
from utils import abs_pdist, format_variable_names, set_seed
from plot_utils import save_figure, plot_loadings, plot_tree
from itertools import combinations
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
from os import makedirs, path
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import pdist, squareform
from selfregulation.utils.plot_utils import dendroheatmap, get_dendrogram_color_fun
from sklearn.manifold import MDS
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler, scale

def plot_clusterings(results, plot_dir=None, inp='data', figsize=(50,50),
                     titles=None, show_clusters=True, verbose=False, ext='png'):    
    HCA = results.HCA
    # get all clustering solutions
    clusterings = [(k ,v) for k,v in 
                    HCA.results.items() if inp in k]
    
    # plot dendrogram heatmaps
    for name, clustering in clusterings:
        if titles is None:
            title = name.split('-')[1] + '_metric-' + HCA.dist_metric
        else:
            title=titles.pop(0)
        filename = None
        if plot_dir is not None:
            filename = path.join(plot_dir, 'dendroheatmap_%s.%s' % (name, ext))
        if show_clusters == True:
            fig = dendroheatmap(clustering['linkage'], clustering['distance_df'], 
                                clustering['labels'],
                                figsize=figsize, title=title,
                                filename = filename)
        else:
            fig = dendroheatmap(clustering['linkage'], clustering['distance_df'], 
                                figsize=figsize, title=title,
                                filename = filename)
            
def plot_clustering_similarity(results, plot_dir=None, verbose=False, ext='png'):  
    HCA = results.HCA
    # get all clustering solutions
    clusterings = [(k ,v) for k,v in 
                    HCA.results.items() if 'clustering' in k]
    # plot cluster agreement across embedding spaces
    names = [k.split('-')[-1] for k,v in 
                    HCA.results.items() if 'clustering' in k]
    cluster_similarity = np.zeros((len(clusterings), len(clusterings)))
    cluster_similarity = pd.DataFrame(cluster_similarity, 
                                     index=names,
                                     columns=names)
    
    distance_similarity = np.zeros((len(clusterings), len(clusterings)))
    distance_similarity = pd.DataFrame(distance_similarity, 
                                     index=names,
                                     columns=names)
    for clustering1, clustering2 in combinations(clusterings, 2):
        name1 = clustering1[0].split('-')[-1]
        name2 = clustering2[0].split('-')[-1]
        # record similarity of distance_df
        dist_corr = np.corrcoef(squareform(clustering1[1]['distance_df']),
                                squareform(clustering2[1]['distance_df']))[1,0]
        distance_similarity.loc[name1, name2] = dist_corr
        distance_similarity.loc[name2, name1] = dist_corr
        # record similarity of clustering of dendrogram
        clusters1 = clustering1[1]['labels']
        clusters2 = clustering2[1]['labels']
        rand_score = adjusted_rand_score(clusters1, clusters2)
        MI_score = adjusted_mutual_info_score(clusters1, clusters2)
        cluster_similarity.loc[name1, name2] = rand_score
        cluster_similarity.loc[name2, name1] = MI_score
    
    with sns.plotting_context(context='notebook', font_scale=1.4):
        clust_fig = plt.figure(figsize = (12,12))
        sns.heatmap(cluster_similarity, square=True)
        plt.title('Cluster Similarity: TRIL: Adjusted MI, TRIU: Adjusted Rand',
                  y=1.02)
        
        dist_fig = plt.figure(figsize = (12,12))
        sns.heatmap(distance_similarity, square=True)
        plt.title('Distance Similarity, metric: %s' % HCA.dist_metric,
                  y=1.02)
        
    if plot_dir is not None:
        save_figure(clust_fig, path.join(plot_dir, 
                                   'cluster_similarity_across_measures.%s' % ext),
                    {'bbox_inches': 'tight'})
        save_figure(dist_fig, path.join(plot_dir, 
                                   'distance_similarity_across_measures.%s' % ext),
                    {'bbox_inches': 'tight'})
    
    if verbose:
        # assess relationship between two measurements
        rand_scores = cluster_similarity.values[np.triu_indices_from(cluster_similarity, k=1)]
        MI_scores = cluster_similarity.T.values[np.triu_indices_from(cluster_similarity, k=1)]
        score_consistency = np.corrcoef(rand_scores, MI_scores)[0,1]
        print('Correlation between measures of cluster consistency: %.2f' \
              % score_consistency)
        
    
def plot_subbranch(cluster_i, tree, loading, cluster_sizes, title=None,
                   figsize=(6,12), dpi=300, plot_loc=None):
    # get variables in subbranch based on coloring
    curr_index = 0
    curr_color = tree['color_list'][0]
    start = 0
    for i, color in enumerate(tree['color_list']):
        if color != curr_color:
            end = i
            if curr_index == cluster_i:
                break
            if color != "#808080":
                curr_index += 1
                start = i
            curr_color = color
    # plotting
    polar_size = [0,.6,1,.4]
    dendro_size = [0,.3,1,.2]
    heatmap_size = [0,.05,1,.25]
    fig = plt.figure(figsize=figsize)
    polar_ax = fig.add_axes(polar_size, projection='polar') 
    dendro_ax = fig.add_axes(dendro_size) 
    heatmap_ax = fig.add_axes(heatmap_size)
    # get subset of loading
    cumsizes = np.cumsum(cluster_sizes)
    if cluster_i==0:
        loading_start = 0
    else:
        loading_start = cumsizes[cluster_i-1]
    subset_loading = loading.T.iloc[:,loading_start:cumsizes[cluster_i]]
    plot_tree(tree, range(start, end), dendro_ax)
    dendro_ax.set_xticklabels('')
    sns.heatmap(subset_loading, ax=heatmap_ax, 
                cbar=False,
                yticklabels=True,
                vmin=np.min(loading.values), 
                vmax=np.max(loading.values),
                cmap=sns.diverging_palette(220,15,n=100,as_cmap=True))
    plot_loadings(polar_ax, abs(subset_loading).mean(1), kind='line', offset=.5,
                  plot_kws={'alpha': .8, 'c': tree['color_list'][start]})
    # tick properties of polar plot
    xtick_locs = np.arange(0.0, 2*np.pi, 2*np.pi/len(subset_loading))
    polar_ax.set_xticks(xtick_locs)
    polar_ax.set_xticks(xtick_locs+np.pi/len(subset_loading), minor=True)
    polar_ax.set_xticklabels(subset_loading.index, minor=True)
    polar_ax.tick_params(axis='x', which='minor', pad=20)
    
    # title and axes styling of dendrogram
    if title:
        dendro_ax.set_title(title, fontsize=20, y=1.05, fontweight='bold')
    dendro_ax.get_yaxis().set_visible(False)
    dendro_ax.spines['top'].set_visible(False)
    dendro_ax.spines['right'].set_visible(False)
    dendro_ax.spines['bottom'].set_visible(False)
    dendro_ax.spines['left'].set_visible(False)
    if plot_loc is not None:
        save_figure(fig, plot_loc, {'bbox_inches': 'tight', 'dpi': dpi})
    else:
        return fig
        
    
def plot_subbranches(results, c=None,  inp=None, cluster_range=None,
                     figsize=(6,10), dpi=300, ext='png', plot_dir=None):
    """ Plots HCA results as dendrogram with loadings underneath
    
    Args:
        results: results object
        c: number of components to use for loadings
        orientation: horizontal or vertical, which determines the direction
            the dendrogram leaves should be spread out on
        plot_dir: if set, where to save the plot
        inp: by default, plots all clusterings in results. Inp selects
            one. Clusterings are saved in the form "clustering_input-{inp}"
        titles: list of titles. Should correspond to number of clusters in
                results object if "inp" is not set. Otherwise should be a list of length 1.
    """
    HCA = results.HCA
    EFA = results.EFA
    loading = EFA.reorder_factors(EFA.get_loading(c))
    loading.index = format_variable_names(loading.index)
    # get all clustering solutions
    if inp is None:
        inp = ''
    clusterings = [(k ,v) for k,v in 
                    HCA.results.items() if inp in k]
    
    for name, clustering in clusterings:
        # extract cluster vars
        link = clustering['linkage']
        labels = clustering['clustered_df'].columns
        labels = format_variable_names(labels)
        ordered_loading = loading.loc[labels]
        # get cluster sizes
        cluster_labels = HCA.get_cluster_labels(inp=name.split('-')[1])
        cluster_sizes = [len(i) for i in cluster_labels]
        link_function, colors = get_dendrogram_color_fun(link, clustering['reorder_vec'],
                                                         clustering['labels'])
        tree = dendrogram(link,  link_color_func=link_function, no_plot=True,
                          no_labels=True)
        
        if plot_dir is not None:
            function_directory = 'subbranches_input-%s' % inp
            makedirs(path.join(plot_dir, function_directory), exist_ok=True)
            
        plot_loc = None
        if cluster_range is None:
            cluster_range = range(len(cluster_labels))
        figs = []
        for cluster_i in cluster_range:
            if plot_dir:
                filey = 'cluster_%s.%s' % (str(cluster_i).zfill(2), ext)
                plot_loc = path.join(plot_dir, function_directory, filey)
            fig = plot_subbranch(cluster_i, tree, ordered_loading, cluster_sizes,
                                 figsize=figsize, plot_loc=plot_loc)
            figs.append(fig)
        return figs
                           

def plot_dendrograms(results, c=None,  inp=None, titles=None, var_labels=False,
                     break_lines=True, orientation='horizontal',
                     figsize=(20,12),  dpi=300, ext='png', plot_dir=None):
    """ Plots HCA results as dendrogram with loadings underneath
    
    Args:
        results: results object
        c: number of components to use for loadings
        orientation: horizontal or vertical, which determines the direction
            the dendrogram leaves should be spread out on
        plot_dir: if set, where to save the plot
        inp: by default, plots all clusterings in results. Inp selects
            one. Clusterings are saved in the form "clustering_input-{inp}"
        titles: list of titles. Should correspond to number of clusters in
                results object if "inp" is not set. Otherwise should be a list of length 1.
    """
    subset = results.ID.split('_')[0]
    HCA = results.HCA
    EFA = results.EFA
    loading = EFA.reorder_factors(EFA.get_loading(c))
    # get all clustering solutions
    if inp is None:
        inp = ''
    clusterings = [(k ,v) for k,v in 
                    HCA.results.items() if inp in k]

        
    for name, clustering in clusterings:
        if titles is None:
            title = subset.title() + " " + name.split('-')[1] + ' metric-' + HCA.dist_metric
        else:
            title=titles.pop(0)
        # extract cluster vars
        link = clustering['linkage']
        labels = clustering['clustered_df'].columns
        ordered_loading = loading.loc[labels]
        # get cluster sizes
        cluster_labels = HCA.get_cluster_labels(inp=name.split('-')[1])
        cluster_sizes = [len(i) for i in cluster_labels]
        link_function, colors = get_dendrogram_color_fun(link, clustering['reorder_vec'],
                                                         clustering['labels'])
        # set up axes' size based on orientation
        if orientation == 'horizontal':
            dendro_size = [0,.4,.95,.45]
            heatmap_size = [0,.05,.95,.35]
            cbar_size = [.97,.05,.02,.25]
            cbar_orientation='vertical'
            dendro_orient='top'
            ordered_loading = ordered_loading.T
        elif orientation == 'vertical':
            dendro_size = [.4, 0, .45, .93]
            heatmap_size = [.05, 0, .35, .93]
            cbar_size = [.05,.97,.25,.02]
            cbar_orientation='horizontal'
            dendro_orient='right'
        with sns.axes_style('white'):
            fig = plt.figure(figsize=figsize)
            ax1 = fig.add_axes(dendro_size) 
            with plt.rc_context({'lines.linewidth': 2.5}):
                dendrogram(link, ax=ax1, link_color_func=link_function,
                           orientation=dendro_orient)
            # change axis properties
            ax1.tick_params(axis='x', which='major', labelsize=14,
                            labelbottom='off')
            # plot loadings as heatmap below
            ax2 = fig.add_axes(heatmap_size)
            ax3 = fig.add_axes(cbar_size)
            max_val = np.max(abs(loading.values))
            sns.heatmap(ordered_loading, ax=ax2, 
                        cbar=True, cbar_ax=ax3,
                        yticklabels=True,
                        vmax =  max_val, vmin = -max_val,
                        cbar_kws={'orientation': cbar_orientation},
                        cmap=sns.diverging_palette(220,15,n=100,as_cmap=True))
            # add lines to heatmap to distinguish clusters
            if break_lines == True:
                xlim = ax2.get_xlim(); 
                ylim = ax2.get_ylim()
                if orientation == 'horizontal':
                    step = xlim[1]/len(labels)
                    cluster_breaks = [i*step for i in np.cumsum(cluster_sizes)]
                    ax2.vlines(cluster_breaks[:-1], ylim[0], ylim[1], linestyles='dashed',
                               linewidth=3, colors=[.5,.5,.5])
                elif orientation == 'vertical':
                    step = max(ylim)/len(labels)
                    cluster_breaks = [ylim[1]-i*step for i in np.cumsum(cluster_sizes)]
                    ax2.hlines(cluster_breaks[:-1], xlim[0], xlim[1], linestyles='dashed',
                               linewidth=2, colors=[.5,.5,.5])
            # change axis properties based on orientation
            if orientation == 'horizontal':
                ax2.tick_params(labelsize=figsize[0]*.8)
            elif orientation == 'vertical':
                ax1.invert_yaxis()
                ax2.tick_params(labelsize=figsize[1]*12/10)
            # add title
            ax1.set_title(title, fontsize=40, y=1.05)
            ax1.get_yaxis().set_visible(False)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
            ax1.spines['left'].set_visible(False)
            # set label visibility
            if not var_labels:
                if orientation == 'horizontal':
                    ax2.tick_params(labelbottom='off')  
                else:
                    ax2.tick_params(labelleft='off') 
        
        if plot_dir is not None:
            save_figure(fig, path.join(plot_dir, 
                                             'dendrogram_%s.%s' % (name, ext)),
                        {'bbox_inches': 'tight', 'dpi': dpi})
        
def plot_graphs(HCA_graphs, plot_dir=None, ext='png'):
    if plot_dir is not None:
        makedirs(path.join(plot_dir, 'graphs'))
    plot_options = {'inline': False,  'target': None}
    for i, GA in enumerate(HCA_graphs):
        if plot_dir is not None:
            plot_options['target'] = path.join(plot_dir, 
                                                'graphs', 
                                                'graph%s.%s' % (i, ext))
        GA.set_visual_style()
        GA.display(plot_options)
    
    
    
@set_seed(seed=15)
def MDS_visualization(results, c, plot_dir=None, 
                      dist_metric='abs_correlation', ext='png', **plot_kws):
    """ visualize EFA loadings and compares to raw space """
    def scale_plot(input_data, data_colors=None, cluster_colors=None,
                   cluster_sizes=None, dissimilarity='euclidean', filey=None):
        """ Plot MDS of data and clusters """
        if data_colors is None:
            data_colors = 'r'
        if cluster_colors is None:
            cluster_colors='b'
        if cluster_sizes is None:
            cluster_sizes = 2200
            
        # scale
        mds = MDS(dissimilarity=dissimilarity)
        mds_out = mds.fit_transform(input_data)
        
        with sns.axes_style('white'):
            f=plt.figure(figsize=(14,14))
            plt.scatter(mds_out[n_clusters:,0], mds_out[n_clusters:,1], 
                        s=75, color=data_colors)
            plt.scatter(mds_out[:n_clusters,0], mds_out[:n_clusters,1], 
                        marker='*', s=cluster_sizes, color=cluster_colors,
                        edgecolor='black', linewidth=2)
            # plot cluster number
            offset = .011
            font_dict = {'fontsize': 17, 'color':'white'}
            for i,(x,y) in enumerate(mds_out[:n_clusters]):
                if i<9:
                    plt.text(x-offset,y-offset,i+1, font_dict)
                else:
                    plt.text(x-offset*2,y-offset,i+1, font_dict)
            plt.title(path.basename(filey)[:-4], fontsize=20)
        if filey is not None:
            save_figure(f, filey)
            
    # set up variables
    data = results.data
    HCA = results.HCA
    EFA = results.EFA
    
    cluster_loadings = HCA.get_cluster_loading(EFA, 'data', c)
    cluster_loadings_mat = np.vstack([i[1] for i in cluster_loadings])
    EFA_loading = abs(EFA.get_loading(c))
    EFA_loading_mat = EFA_loading.values
    EFA_space = np.vstack([cluster_loadings_mat, EFA_loading_mat])
    
    # set up colors
    n_clusters = cluster_loadings_mat.shape[0]
    color_palette = sns.color_palette(palette='hls', n_colors=n_clusters)
    colors = []
    for var in EFA_loading.index:
        # find which cluster this variable is in
        index = [i for i,cluster in enumerate(cluster_loadings) \
                 if var in cluster[0]][0]
        colors.append(color_palette[index])
    # set up cluster sizes proportional to number of members
    n_members = np.reshape([len(i) for i,j in cluster_loadings], [-1,1])
    scaler = MinMaxScaler()
    relative_members = scaler.fit_transform(n_members).flatten()
    sizes = 1500+2000*relative_members
    
    if dist_metric == 'abs_correlation':
        EFA_space_distances = squareform(abs_pdist(EFA_space))
    else: 
        EFA_space_distances = squareform(pdist(EFA_space, dist_metric))
    
    # repeat the same thing as above but with raw distances
    scaled_data = pd.DataFrame(scale(data).T,
                               index=data.columns,
                               columns=data.index)
    clusters_raw = []
    for labels, EFA_vec in cluster_loadings:
        subset = scaled_data.loc[labels,:]
        cluster_vec = subset.mean(0)
        clusters_raw.append(cluster_vec)
    raw_space = np.vstack([clusters_raw, scaled_data])
    # turn raw space into distances
    if dist_metric == 'abs_correlation':
        raw_space_distances = squareform(abs_pdist(raw_space))
    else:
        raw_space_distances = squareform(pdist(raw_space, dist_metric))
    
    # plot distances
    distances = {'EFA%s' % c: EFA_space_distances,
                 'subj': raw_space_distances}
    filey=None
    for label, space in distances.items():
        if plot_dir is not None:
            filey = path.join(plot_dir, 
                              'MDS_%s_metric-%s.%s' % (label, dist_metric, ext))
        scale_plot(space, data_colors=colors,
                   cluster_colors=color_palette,
                   cluster_sizes=sizes,
                   dissimilarity='precomputed',
                   filey=filey)

def visualize_importance(importance, ax, xticklabels=True, 
                           yticklabels=True, pad=0, ymax=None, legend=True):
    """Plot task loadings on one axis"""
    importance_vars = importance[0]
    importance_vals = [abs(i)+pad for i in importance[1].T]
    plot_loadings(ax, importance_vals, kind='line', offset=.5,
                  plot_kws={'alpha': 1})
    # set up x ticks
    xtick_locs = np.arange(0.0, 2*np.pi, 2*np.pi/len(importance_vars))
    ax.set_xticks(xtick_locs)
    ax.set_xticks(xtick_locs+np.pi/len(importance_vars), minor=True)
    if xticklabels:
        if type(importance_vars[0]) == str:
            ax.set_xticklabels(importance_vars, 
                               y=.08, minor=True)
        else:
            ax.set_xticklabels(['Fac %s' % str(i+1) for i in importance_vars], 
                               y=.08, minor=True)
    # set up yticks
    if ymax:
        ax.set_ylim(top=ymax)
    ytick_locs = ax.yaxis.get_ticklocs()
    new_yticks = np.linspace(0, ytick_locs[-1], 7)
    ax.set_yticks(new_yticks)
    if yticklabels:
        labels = np.round(new_yticks,2)
        replace_dict = {i:'' for i in labels[::2]}
        labels = [replace_dict.get(i, i) for i in labels]
        ax.set_yticklabels(labels)
    if legend:
        ax.legend(loc='upper center', bbox_to_anchor=(.5,-.15))
        
def plot_cluster_factors(results, c, inp='data', ext='png', plot_dir=None):
    """
    Args:
        EFA: EFA_Analysis object
        c: number of components for EFA
        task_sublists: a dictionary whose values are sets of tasks, and 
                        whose keywords are labels for those lists
    """
    # set up variables
    HCA = results.HCA
    EFA = results.EFA
    
    cluster_loadings = HCA.get_cluster_loading(EFA, inp, c)
    max_loading = max([max(abs(i[1])) for i in cluster_loadings])
    # plot
    colors = sns.hls_palette(len(cluster_loadings))
    ncols = min(5, len(cluster_loadings))
    nrows = ceil(len(cluster_loadings)/ncols)
    f, axes = plt.subplots(nrows, ncols, 
                               figsize=(ncols*10,nrows*(8+nrows)),
                               subplot_kw={'projection': 'polar'})
    axes = f.get_axes()
    for i, (measures, loading) in enumerate(cluster_loadings):
        plot_loadings(axes[i], loading, kind='line', offset=.5,
              plot_kws={'alpha': .8, 'c': colors[i]})
        axes[i].set_title('Cluster %s' % i, y=1.14, fontsize=25)
        # set tick labels
        xtick_locs = np.arange(0.0, 2*np.pi, 2*np.pi/len(loading))
        axes[i].set_xticks(xtick_locs)
        axes[i].set_xticks(xtick_locs+np.pi/len(loading), minor=True)
        if i%(ncols*2)==0 or i%(ncols*2)==(ncols-1):
            axes[i].set_xticklabels(loading.index,  y=.08, minor=True)
            # set ylim
            axes[i].set_ylim(top=max_loading)
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    plt.subplots_adjust(hspace=.5, wspace=.5)
    
    filename = 'polar_factors_EFA%s_inp-%s.%s' % (c, inp, ext)
    if plot_dir is not None:
        save_figure(f, path.join(plot_dir, filename),
                    {'bbox_inches': 'tight'})
        
            
def plot_HCA(results, plot_dir=None, verbose=False, ext='png'):
    c = results.EFA.num_factors
    # plots, woo
    if verbose: print("Plotting dendrogram heatmaps")
    plot_clusterings(results, inp='data', plot_dir=plot_dir, verbose=verbose, ext=ext)
    plot_clusterings(results, inp='EFA%s' % c, plot_dir=plot_dir, verbose=verbose, ext=ext)
    if verbose: print("Plotting dendrograms")
    plot_dendrograms(results, c, inp='data', plot_dir=plot_dir, ext=ext)
    plot_dendrograms(results, c, inp='EFA%s' % c, plot_dir=plot_dir, ext=ext)
    if verbose: print("Plotting dendrogram subbranches")
    plot_subbranches(results, c,  inp='data', plot_dir=plot_dir, ext=ext)
    plot_subbranches(results, c,  inp='EFA%s' % c, plot_dir=plot_dir, ext=ext)
    if verbose: print("Plotting clustering similarity")
    plot_clustering_similarity(results, plot_dir=plot_dir, verbose=verbose, ext=ext)
    if verbose: print("Plotting cluster polar plots")
    plot_cluster_factors(results, c, inp='data', plot_dir=plot_dir, ext=ext)
    plot_cluster_factors(results, c, inp='EFA%s' % c, plot_dir=plot_dir, ext=ext)
    if verbose: print("Plotting MDS space")
    for metric in ['abs_correlation']:
        MDS_visualization(results, c, plot_dir=plot_dir,
                          dist_metric=metric, ext=ext)








