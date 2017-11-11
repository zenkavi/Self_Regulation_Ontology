from collections import OrderedDict as odict
from dynamicTreeCut import cutreeHybrid
import fancyimpute
import functools
import hdbscan
from itertools import combinations
from glob import glob
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage, cut_tree
from scipy.spatial.distance import pdist, squareform
from selfregulation.utils.plot_utils import dendroheatmap
from selfregulation.utils.r_to_py_utils import psychFA
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, scale
# imports for behavior prediction
import sys
import importlib
import traceback
import selfregulation.prediction.behavpredict as behavpredict
importlib.reload(behavpredict)

def set_seed(seed):
    def seeded_fun_decorator(fun):
        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            np.random.seed(seed)
            out = fun(*args, **kwargs)
            np.random.seed()
            return out
        return wrapper
    return seeded_fun_decorator
    
    
class Imputer(object):
    """ Imputation class so that fancyimpute can be used with scikit pipeline"""
    def __init__(self, imputer=None):
        if imputer is None:
            self.imputer = fancyimpute.SimpleFill()
        else:
            self.imputer = imputer(verbose=False)
        
    def transform(self, X):
        transformed = self.imputer.complete(X)
        return transformed
    
    def fit(self, X, y=None):
        return self

def distcorr(X, Y, flip=True):
    """ Compute the distance correlation function
    
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    
    Taken from: https://gist.github.com/satra/aa3d19a12b74e9ab7941
    """
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
    if flip == True:
        dcor = 1-dcor
    return dcor

def abs_pdist(mat, square=False):
    correlation_dist = pdist(mat, metric='correlation')
    correlations = 1-correlation_dist
    absolute_distance = 1-abs(correlations)
    if square == True:
        absolute_distance = squareform(absolute_distance)
    return absolute_distance

def load_results(datafile):
    results = {}
    result_files = glob('Output/%s/*/results*.pkl' % (datafile))
    for filey in result_files:
        name = os.path.basename(os.path.dirname(filey))
        results[name] = pickle.load(open(filey,'rb'))
    return results

def not_regex(txt):
    return '^((?!%s).)*$' % txt

def save_figure(fig, loc, save_kws=None):
    """ Saves figure in location and creates directory tree if needed """
    if save_kws is None:
        save_kws = {}
    directory = os.path.dirname(loc)
    if directory != "":
        os.makedirs(directory, exist_ok=True)
    fig.savefig(loc, **save_kws)

# ****************************************************************************
# helper functions for hierarchical clustering
# ****************************************************************************
def hierarchical_cluster(df, compute_dist=True,  pdist_kws=None, 
                         plot=False, cluster_kws=None, plot_kws=None):
    """
    plot hierarchical clustering and heatmap
    :df: a correlation matrix
    parse_heatmap: int (optional). If defined, devides the columns of the 
                    heatmap based on cutting the dendrogram
    """
    
    # if compute_dist = False, assume df is a distance matrix. Otherwise
    # compute distance on df rows
    if compute_dist == True:
        if pdist_kws is None:
            pdist_kws= {'metric': 'correlation'}
        if pdist_kws['metric'] == 'abscorrelation':
            # convert to absolute correlations
            dist_vec = abs_pdist(df)
        else:
            dist_vec = pdist(df, **pdist_kws)
        dist_df = pd.DataFrame(squareform(dist_vec), 
                               index=df.index, 
                               columns=df.index)
    else:
        assert df.shape[0] == df.shape[1]
        dist_df = df
        dist_vec = squareform(df.values)
    #clustering
    link = linkage(dist_vec, method='ward')    
    #dendrogram
    reorder_vec = leaves_list(link)
    clustered_df = dist_df.iloc[reorder_vec, reorder_vec]
    # clustering
    if cluster_kws is None:
        cluster_kws = {'minClusterSize': 1}
    clustering = cutreeHybrid(link, dist_vec, **cluster_kws)
    if plot == True:
        if plot_kws is None:
            plot_kws = {}
        dendroheatmap(link, dist_df, clustering['labels'], **plot_kws)
        
    return {'linkage': link, 
            'distance_df': dist_df, 
            'clustered_df': clustered_df,
            'reorder_vec': reorder_vec,
            'clustering': clustering,
            'labels': clustering['labels']}


def hdbscan_cluster(df, compute_dist=True,  pdist_kws=None, 
                    plot=False, cluster_kws=None, plot_kws=None):
    """
    plot hierarchical clustering and heatmap
    :df: a correlation matrix
    parse_heatmap: int (optional). If defined, devides the columns of the 
                    heatmap based on cutting the dendrogram
    """
    
    # if compute_dist = False, assume df is a distance matrix. Otherwise
    # compute distance on df rows
    if compute_dist == True:
        if pdist_kws is None:
            pdist_kws= {'metric': 'correlation'}
        if pdist_kws['metric'] == 'abscorrelation':
            # convert to absolute correlations
            dist_vec = abs_pdist(df)
        else:
            dist_vec = pdist(df, **pdist_kws)
        dist_df = pd.DataFrame(squareform(dist_vec), 
                               index=df.index, 
                               columns=df.index)
    else:
        assert df.shape[0] == df.shape[1]
        dist_df = df
        dist_vec = squareform(df.values)
    #clustering
    if cluster_kws is None:
        cluster_kws = {'min_cluster_size': 4,
                       'min_samples': 4}
    clusterer = hdbscan.HDBSCAN(metric='precomputed',
                                cluster_selection_method='leaf',
                                **cluster_kws)
    clusterer.fit(dist_df)  
    link = clusterer.single_linkage_tree_.to_pandas().iloc[:,1:]   
    labels = clusterer.labels_
    probs = clusterer.probabilities_
    #dendrogram
    reorder_vec = leaves_list(link)
    clustered_df = dist_df.iloc[reorder_vec, reorder_vec]
    
    # clustering
    if plot == True:
        if plot_kws is None:
            plot_kws = {}
        dendroheatmap(link, dist_df, labels, **plot_kws)
        
    return {'clusterer': clusterer,
            'distance_df': dist_df,
            'clustered_df': clustered_df,
            'labels': labels,
            'probs': probs,
            'link': link}

        


# ****************************************************************************
# helper functions for dealing with factor analytic results
# ****************************************************************************
def corr_lower_higher(higher_dim, lower_dim, cross_only=True):
    """
    Returns a correlation matrix between factors at different dimensionalities
    cross_only: bool, if True only display the correlations between dimensions
    """
    # higher dim is the factor solution with fewer factors
    higher_dim = higher_dim.copy()
    lower_dim = lower_dim.copy()
    higher_n = higher_dim.shape[1]
    
    lower_dim.columns = ['l%s' % i  for i in lower_dim.columns]
    higher_dim.columns = ['h%s' % i for i in higher_dim.columns]
    corr = pd.concat([higher_dim, lower_dim], axis=1).corr()
    if cross_only:
        corr = corr.iloc[:higher_n, higher_n:]
    return corr

# functions to fit and extract factor analysis solutions
def find_optimal_components(data, minc=1, maxc=50, metric='BIC'):
    """
    Fit EFA over a range of components and returns the best c. If metric = CV
    uses sklearn. Otherwise uses psych
    metric: str, method to use for optimal components. Options 'BIC', 'SABIC',
            and 'CV'
    """
    steps_since_best = 0 # count steps since last best metric.
    metrics = {}
    n_components = range(minc,maxc)
    scaler = StandardScaler()
    if metric != 'CV':
        scaled_data = scaler.fit_transform(data)
        for c in n_components:
            fa, output = psychFA(scaled_data, c, method='ml')
            last_metric = output[metric]
            # iterate counter if new metric isn't better than previous metric
            if len(metrics) > 0:
                if last_metric > metrics[c-1]:
                    steps_since_best += 1
                else:
                    steps_since_best = 0
            metrics[c] = last_metric
            if steps_since_best > 2:
                break
        best_c = min(metrics, key=metrics.get)
    else:
        for c in n_components:
            fa = FactorAnalysis(c)
            scaler = StandardScaler()
            imputer = Imputer()
            pipe = Pipeline(steps = [('impute', imputer),
                                     ('scale', scaler),
                                     ('fa', fa)])
            cv = cross_val_score(pipe, data, cv=10)
            # iterate counter if new metric isn't better than previous metric
            if len(metrics) > 0:
                if cv < metrics[c-1]:
                    steps_since_best += 1
                else:
                    steps_since_best = 0
            metrics[c] = np.mean(cv)
            if steps_since_best > 2:
                break
        best_c = max(metrics, key=metrics.get)
    return best_c, metrics

def get_loadings(fa_output, labels, sort=False):
    """
    Takes output of psychFA, and a list of labels and returns a loadings dataframe
    """
    loading_df = pd.DataFrame(fa_output['loadings'], index=labels)
    if sort == True:
        # sort by maximum loading on surveys
        sorting_index = np.argsort(loading_df.filter(regex='survey',axis=0).abs().mean()).tolist()[::-1]
        loading_df = loading_df.loc[:,sorting_index]
        loading_df.columns = range(loading_df.shape[1])
    return loading_df

def get_top_factors(loading_df, n=4, verbose=False):
    """
    Takes output of get_loadings and prints the absolute top variables per factor
    """
    # number of variables to display
    factor_top_vars = {}
    for i,column in loading_df.iteritems():
        sort_index = np.argsort(abs(column))[::-1] # descending order
        top_vars = column[sort_index]
        factor_top_vars[i] = top_vars
        if verbose:
            print('\nFACTOR %s' % i)
            print(top_vars[0:n])
    return factor_top_vars

def reorder_data(data, groups, axis=1):
    ordered_cols = []
    for i in groups:
        ordered_cols += i[1]
    new_data = data.reindex_axis(ordered_cols, axis)
    return new_data

def create_factor_tree(data, component_range=(1,13), component_list=None):
    """
    Runs "visualize_factors" at multiple dimensionalities and saves them
    to a pdf
    data: dataframe to run EFA on at multiple dimensionalities
    groups: group list to be passed to visualize factors
    filename: filename to save pdf
    component_range: limits of EFA dimensionalities. e.g. (1,5) will run
                     EFA with 1 component, 2 components... 5 components.
    component_list: list of specific components to calculate. Overrides
                    component_range if set
    """
    def get_similarity_order(lower_dim, higher_dim):
        "Helper function to reorder factors into correspondance between two dimensionalities"
        subset = corr_lower_higher(higher_dim, lower_dim)
        max_factors = np.argmax(abs(subset.values), axis=0)
        return np.argsort(max_factors)

    EFA_results = {}
    full_fa_results = {}
    # plot
    if component_list is None:
        components = range(component_range[0],component_range[1]+1)
    else:
        components = component_list
    for c in components:
        fa, output = psychFA(data, c)
        tmp_loading_df = get_loadings(output, labels=data.columns)
        if (c-1) in EFA_results.keys():
            reorder_index = get_similarity_order(tmp_loading_df, EFA_results[c-1])
            tmp_loading_df = tmp_loading_df.iloc[:, reorder_index]
            tmp_loading_df.columns = sorted(tmp_loading_df.columns)
        EFA_results[c] = tmp_loading_df
        full_fa_results[c] = fa
    return EFA_results, full_fa_results

def get_factor_groups(loading_df):
    index_assignments = np.argmax(abs(loading_df).values,axis=1)
    names = loading_df.columns
    factor_groups = []
    for assignment in np.unique(index_assignments):
        name = names[assignment]
        assignment_vars = [var for i,var in enumerate(loading_df.index) if index_assignments[i] == assignment]
        # sort assignment_vars by maximum loading on their assigned factor
        assignment_vars = abs(loading_df.loc[assignment_vars, name]).sort_values()
        assignment_vars = list(assignment_vars.index)[::-1] # get names
        factor_groups.append([name, assignment_vars])
    return factor_groups

def get_hierarchical_groups(loading_df, n_groups=8):
    # helper function
    def remove_adjacent(nums):
        result = []
        for num in nums:
            if len(result) == 0 or num != result[-1]:
                result.append(num)
        return result
    # distvec
    dist_vec = pdist(loading_df, metric=distcorr)
    # create linkage matrix for variables projected into a component loading
    row_clusters = linkage(dist_vec, method='ward')   
    # use the dendorgram function to order the leaves appropriately
    row_dendr = dendrogram(row_clusters, labels=loading_df.T.columns, no_plot = True)
    cluster_reorder_index = row_dendr['leaves']
    # cut the linkage graph such that there are only n groups
    n_groups = n_groups
    index_assignments = [i[0] for i in cut_tree(row_clusters, n_groups)]
    # relabel groups such that 0 is the 'left' most in the dendrogram
    group_order = remove_adjacent([index_assignments[i] for i in cluster_reorder_index])
    index_assignments = [group_order.index(i) for i in index_assignments]
    # using the groups and the dendrogram ordering, create a number of groups
    hierarchical_groups = []
    for assignment in np.unique(index_assignments):
        # get variables that are in the correct group
        assignment_vars = [var for i,var in enumerate(loading_df.index) if index_assignments[i] == assignment]
        hierarchical_groups.append([assignment,assignment_vars])
    return cluster_reorder_index, hierarchical_groups

def get_scores_from_subset(data, fa_output, task_subset):
    match_cols = []
    for i, c in enumerate(data.columns):
        if np.any([task in c for task in task_subset]):
            match_cols.append(i)

    weights_subset = fa_output['weights'][match_cols,:]
    data_subset = scale(data.iloc[:, match_cols])
    subset_scores = data_subset.dot(weights_subset)

    # concat subset and full scores into one dataframe
    labels = ['%s_full' % i for i in list(range(fa_output['scores'].shape[1]))]
    labels+=[i.replace('full','subset') for i in labels]
    concat_df = pd.DataFrame(np.hstack([fa_output['scores'], subset_scores]),
                             columns = labels)
    
    # calculate variance explained by subset
    lr = LinearRegression()
    lr.fit(concat_df.filter(regex='subset'), 
           concat_df.filter(regex='full'))
    scores = r2_score(lr.predict(concat_df.filter(regex='subset')), 
                      concat_df.filter(regex='full'), 
                      multioutput='raw_values')
    return concat_df, scores


def quantify_higher_nesting(higher_dim, lower_dim):
    """
    Quantifies how well higher levels of the tree can be reconstructed from 
    lower levels
    """
    lr = LinearRegression()
    best_score = -1
    relationship = []
    # quantify how well the higher dimensional solution can reconstruct
    # the lower dimensional solution using a linear combination of two factors
    for higher_name, higher_c in higher_dim.iteritems():
        for lower_c1, lower_c2 in combinations(lower_dim.columns, 2):
            # combined prediction
            predict_mat = higher_dim.loc[:,[lower_c1, lower_c2]]
            lr.fit(predict_mat, higher_c)
            score = lr.score(predict_mat, higher_c)
            # individual correlation
            lower_subset = lower_dim.drop(higher_name, axis=1)
            higher_subset = higher_dim.drop([lower_c1, lower_c2], axis=1)
            corr = corr_lower_higher(higher_subset, lower_subset)
            if len(corr)==1:
                other_cols = [corr.iloc[0,0]]
            else:
                other_cols = corr.apply(lambda x: max(x**2)-sorted(x**2)[-2],
                                        axis=1)
            total_score = np.mean(np.append(other_cols, score))
            if total_score>best_score:
                best_score = total_score
                relationship = {'score': score,
                                'lower_factor': higher_c.name, 
                                'higher_factors': (lower_c1, lower_c2), 
                                'coefficients': lr.coef_}
    return relationship

def quantify_lower_nesting(factor_tree):
    """
    Quantifies how well lower levels of the tree can be reconstruted from
    higher levels
    """
    lr = LinearRegression()
    relationships = odict()
    for higher_c, lower_c in combinations(factor_tree.keys(), 2):
        higher_dim = factor_tree[higher_c]
        lower_dim = factor_tree[lower_c]
        lr.fit(higher_dim, lower_dim)
        scores = r2_score(lr.predict(higher_dim), 
                                 lower_dim, 
                                 multioutput='raw_values')
        relationship = {'scores': scores,
                        'coefs': lr.coef_}
        relationships[(higher_c,lower_c)] = relationship
    return relationships

# ****************************************************************************
# Helper functions for visualization of component loadings
# ****************************************************************************

def plot_loadings(ax, component_loadings, groups=None, colors=None, 
                  width_scale=1, offset=0, kind='bar', plot_kws=None):
    """Plot component loadings
    
    Args:
        ax: axis to plot on. If a polar axist, a polar bar plot will be created.
            Otherwise, a histogram will be plotted
        component_loadings (array or pandas Series): loadings to plot
        groups (list, optional): ordered list of tuples of the form: 
            [(group_name, list of group members), ...]. If not supplied, all
            elements will be treated as one group
        colors (list, optional): if supplied, specifies the colors for the groups
        width_scale (float): scale of bars. Default is 1, which fills the entire
            plot
        offset (float): offset as a proportion of width. Used to plot multiple
            columns side by side under one factor
        bar_kws (dict): keywords to pass to ax.bar
    """
    if plot_kws is None:
        plot_kws = {}
    N = len(component_loadings)
    if groups is None:
        groups = [('all', [0]*N)]
    if colors is not None:
        assert(len(colors) == len(groups))
    else:
        colors = sns.hls_palette(len(groups), l=.5, s=.8)
    ax.set_xticklabels([''])
    ax.set_yticklabels([''])
    
    width = np.pi/(N/2)*width_scale*np.ones(N)
    theta = np.arange(0.0, 2*np.pi, 2*np.pi/N) + width[0]*offset
    radii = component_loadings
    if kind == 'bar':
        bars = ax.bar(theta, radii, width=width, bottom=0.0, **plot_kws)
        for i,r,bar in zip(range(N),radii, bars):
            color_index = sum((np.cumsum([len(g[1]) for g in groups])<i+1))
            bar.set_facecolor(colors[color_index])
    elif kind == 'line':
        theta = np.append(theta, theta[0])
        radii = np.append(radii, radii[0])
        bars = ax.plot(theta, radii, linewidth=5, **plot_kws)
    return colors
        
def create_categorical_legend(labels,colors, ax):
    """Take a list of labels and colors and creates a legend"""
    import matplotlib
    def create_proxy(color):
        line = matplotlib.lines.Line2D([0], [0], linestyle='none',
                    mec='none', marker='o', color=color)
        return line
    proxies = [create_proxy(item) for item in colors]
    ncol = max(len(proxies)//6, 1)
    ax.legend(proxies, labels, numpoints=1, markerscale=2.5, ncol=ncol,
              bbox_to_anchor=(1, .95), prop={'size':20})

def visualize_factors(loading_df, groups=None, n_rows=2, 
                      legend=True, input_axes=None, colors=None):
    """
    Takes in a dataset to run EFA on, and a list of groups in the form
    of a list of tuples. Each element of this list should be of the form
    ("group name", [list of variables]). Each element of the list should
    be mututally exclusive. These groups are used for coloring the plots
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    loading_df = loading_df.select_dtypes(include=numerics)
    if groups:
        loading_df = reorder_data(loading_df, groups, axis=0)
            
    n_components = loading_df.shape[1]
    n_cols = int(np.ceil(n_components/n_rows))
    sns.set_style("white")
    if input_axes is None:
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*5,n_rows*5),
                           subplot_kw={'projection': 'polar'})
        axes = fig.get_axes()
        fig.tight_layout()
    else:
        axes = input_axes
    for i in range(n_components):
        component_loadings = loading_df.iloc[:,i]
        colors = plot_loadings(axes[i], abs(component_loadings), groups,
                               colors=colors)
    for j in range(n_components, len(axes)):
        axes[j].set_visible(False)
    if legend and groups is not None:
        create_categorical_legend([g[0] for g in groups], 
                                  colors, axes[n_components-1])
    if input_axes is None:
        return fig

def visualize_task_factors(task_loadings, ax, xticklabels=True, 
                           yticklabels=False, pad=0, ymax=None, legend=True):
    """Plot task loadings on one axis"""
    n_measures = len(task_loadings)
    colors = sns.hls_palette(len(task_loadings), l=.5, s=.8)
    for i, (name, DV) in enumerate(task_loadings.iterrows()):
        plot_loadings(ax, abs(DV)+pad, width_scale=1/(n_measures), 
                      colors = [colors.pop()], offset=i+.5,
                      kind='line',
                      plot_kws={'label': name, 'alpha': .8})
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
    # set up x ticks
    xtick_locs = np.arange(0.0, 2*np.pi, 2*np.pi/len(DV))
    ax.set_xticks(xtick_locs)
    ax.set_xticks(xtick_locs+np.pi/len(DV), minor=True)
    if xticklabels:
        ax.set_xticklabels(task_loadings.columns,  y=.08, minor=True)
    if legend:
        ax.legend(loc='upper center', bbox_to_anchor=(.5,-.15))
        
def plot_factor_tree(factor_tree, groups=None, filename=None):
    """
    Runs "visualize_factors" at multiple dimensionalities and saves them
    to a pdf
    data: dataframe to run EFA on at multiple dimensionalities
    groups: group list to be passed to visualize factors
    filename: filename to save pdf
    component_range: limits of EFA dimensionalities. e.g. (1,5) will run
                     EFA with 1 component, 2 components... 5 components.
    reorder_list: optional. List of index values in an order that will be used
                  to rearrange data
    """
    max_c = np.max(list(factor_tree.keys()))
    min_c = np.min(list(factor_tree.keys()))
    n_cols = max_c
    n_rows = max_c-min_c+1
    f,axes = plt.subplots(n_rows, n_cols, subplot_kw=dict(projection='polar'),
               figsize=(n_cols*5,n_rows*5))
    f.tight_layout()
    # move axes:
    for i,row in enumerate(axes):
        for j,ax in enumerate(row):
            pos1 = ax.get_position() # get the original position
            pos2 = [pos1.x0 + (n_rows-i-1)*pos1.width/1.76, pos1.y0,  pos1.width, pos1.height] 
            ax.set_position(pos2) # set a new position

    # plot
    for rowi, c in enumerate(range(min_c,max_c+1)):
        tmp_loading_df = factor_tree[c]
        if rowi == 0:
            visualize_factors(tmp_loading_df, groups, 
                              n_rows=1, input_axes=axes[rowi,0:c], legend=True)
        else:
            visualize_factors(tmp_loading_df, groups, 
                              n_rows=1, input_axes=axes[rowi,0:c], legend=False)
        for ax in axes[rowi,c:]:
            ax.set_axis_off()
    if filename:
        save_figure(f, filename, {'bbox_inches': 'tight'})
    else:
        return f



# ****************************************************************************
# Helper functions for prediction
# ****************************************************************************
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from selfregulation.utils.utils import get_behav_data

def survey_task_prediction(datafile):
    # assess survey task independence
    data = get_behav_data(datafile, file='meaningful_variables_imputed.csv')
    task_data = data.filter(regex=not_regex('survey')+'|cognitive_reflection')
    survey_data = data.filter(regex= not_regex(not_regex('survey')+'|cognitive_reflection'))
    
    pipe = make_pipeline(StandardScaler(),RidgeCV())
    #fit survey to task and task to survey
    st_score = np.mean(cross_val_score(pipe, survey_data, scale(task_data), cv=10))
    ts_score = np.mean(cross_val_score(pipe, task_data, scale(survey_data), cv=10))
    # without cross validation
    pipe.fit(survey_data, scale(task_data))
    st_score_within = pipe.score(survey_data, scale(task_data))
    pipe.fit(task_data, scale(survey_data))
    ts_score_within = pipe.score(task_data, scale(survey_data))
    return {'st_scores_within': st_score_within,
            'ts_scores_within': ts_score_within,
            'st_scores_cv': st_score,
            'ts_scores_cv': ts_score}
    

def run_EFA_prediction(dataset, factor_scores, output_base, save=True,
                       verbose=False, classifier='lasso',
                       shuffle=False, n_jobs=2, imputer="SimpleFill",
                       smote_threshold=.05, freq_threshold=.1, icc_threshold=.25,
                       no_baseline_vars=True, singlevar=None):
    
    output_dir=os.path.join(output_base,'prediction_outputs')
    if dataset is 'baseline' or no_baseline_vars:
        baselinevars=False
        if verbose:
            print("turning off inclusion of baseline vars")
    else:
        baselinevars=True
        if verbose:
            print("including baseline vars in prediction models")
            
    # skip several variables because they crash the estimation tool
    bp=behavpredict.BehavPredict(verbose=verbose,
                                 dataset=dataset,
         drop_na_thresh=100,n_jobs=n_jobs,
         skip_vars=['RetirementPercentStocks',
         'HowOftenFailedActivitiesDrinking',
         'HowOftenGuiltRemorseDrinking',
         'AlcoholHowOften6Drinks'],
         output_dir=output_dir,shuffle=shuffle,
         classifier=classifier,
         add_baseline_vars=baselinevars,
         smote_cutoff=smote_threshold,
         freq_threshold=freq_threshold,
         imputer=imputer)
    
    bp.load_demog_data()
    bp.get_demogdata_vartypes()
    bp.remove_lowfreq_vars()
    bp.binarize_ZI_demog_vars()
    bp.behavdata = factor_scores
    #bp.filter_by_icc(icc_threshold)
    bp.get_joint_datasets()
    
    if not singlevar:
        vars_to_test=[v for v in bp.demogdata.columns if not v in bp.skip_vars]
    else:
        vars_to_test=singlevar
    
    vars_to_test = ['BMI', 'AlcoholHowManyDrinksDay', 'SmokeEveryDay', 'CannabisHowOften', 'DaysLostLastMonth']
    for v in vars_to_test:
        bp.lambda_optim=None
        print('RUNNING:',v,bp.data_models[v],dataset)
        try:
            bp.scores[v],bp.importances[v]=bp.run_crossvalidation(v,nlambda=100)
            bp.scores_insample[v],_=bp.run_lm(v,nlambda=100)
            # fit model with no regularization
            if bp.data_models[v]=='binary':
                bp.lambda_optim=[0]
            else:
                bp.lambda_optim=[0,0]
            bp.scores_insample_unbiased[v],_=bp.run_lm(v,nlambda=100)
        except:
            e = sys.exc_info()
            print('error on',v,':',e)
            bp.errors[v]=traceback.format_tb(e[2])
    if save == True:
        if singlevar:
            bp.write_data(vars_to_test,listvar=True)
        else:
            bp.write_data(vars_to_test)
    return bp

