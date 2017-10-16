from collections import OrderedDict as odict
from itertools import combinations
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from scipy.spatial.distance import pdist, squareform
from selfregulation.utils.r_to_py_utils import psychFA
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, scale

def distcorr(X, Y, flip=True):
    """ Compute the distance correlation function
    
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
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

# functions to fit and extract factor analysis solutions
def find_optimal_components(data, minc=1, maxc=30, metric='BIC'):
    """
    Fit EFA over a range of components and returns the best c. If metric = CV
    uses sklearn. Otherwise uses psych
    metric: str, method to use for optimal components. Options 'BIC', 'SABIC',
            and 'CV'
    """
    metrics = {}
    n_components = range(minc,maxc)
    scaler = StandardScaler()
    if metric != 'CV':
        scaled_data = scaler.fit_transform(data)
        for c in n_components:
            fa, output = psychFA(scaled_data, c, method='ml')
            metrics[c] = output[metric]
        best_c = min(metrics, key=metrics.get)
    else:
        for c in n_components:
            fa = FactorAnalysis(c)
            scaler = StandardScaler()
            pipe = Pipeline(steps = [('scale', scaler),
                                     ('fa', fa)])
            cv = cross_val_score(pipe, data, cv=10)
            metrics[c] = np.mean(cv)
        best_c = max(metrics, key=metrics.get)
    print('Best Component: ', best_c)
    return best_c, metrics

def get_loadings(fa_output, labels):
    """
    Takes output of psychFA, and a list of labels and returns a loadings dataframe
    """
    loading_df = pd.DataFrame(fa_output['loadings'], index=labels)
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
        
# ****************************************************************************
# Other helper functions for dealing with factor analytic results
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

def get_factor_groups(loading_df):
    index_assignments = np.argmax(abs(loading_df).values,axis=1)
    factor_groups = []
    for assignment in np.unique(index_assignments):
        assignment_vars = [var for i,var in enumerate(loading_df.index) if index_assignments[i] == assignment]
        factor_groups.append([assignment,assignment_vars])
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


  
# ****************************************************************************
# Helper functions for visualization of component loadings
# ****************************************************************************

def plot_loadings(ax, component_loadings, groups=None):
    """
    Takes in an axis, a vector and a list of groups and 
    plots the loadings. If a polar axis is passed this will
    result in a polar bar plot, otherwise a normal bar plot will
    be made. 
    """
    N = len(component_loadings)
    if groups is None:
        groups = [('all', [0]*N)]
    colors = sns.hls_palette(len(groups), l=.5, s=.8)
    ax.set_xticklabels([''])
    ax.set_yticklabels([''])
    
    theta = np.arange(0.0, 2*np.pi, 2*np.pi/N)
    radii = component_loadings
    width = np.pi/(N/2)*np.ones(N)
    bars = ax.bar(theta, radii, width=width, bottom=0.0)
    for i,r,bar in zip(range(N),radii, bars):
        color_index = sum((np.cumsum([len(g[1]) for g in groups])<i))
        bar.set_facecolor(colors[color_index])
        bar.set_alpha(1)
    plt.legend()
    return colors
        
def create_categorical_legend(labels,colors, ax):
    """
    Takes in a list of labels and colors and creates a legebd
    for an axis object which assigns each label to the corresponding
    color
    """
    import matplotlib
    def create_proxy(color):
        line = matplotlib.lines.Line2D([0], [0], linestyle='none',
                    mec='none', marker='o', color=color)
        return line
    proxies = [create_proxy(item) for item in colors]
    ax.legend(proxies, labels, numpoints=1, markerscale=2.5, bbox_to_anchor=(1, .95), prop={'size':20})
    
def visualize_factors(loading_df, groups=None, n_rows=2, 
                      legend=True, input_axes=None):
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
                           subplot_kw=dict(projection='polar'))
        axes = fig.get_axes()
        fig.tight_layout()
    else:
        axes = input_axes
    for i in range(n_components):
        component_loadings = loading_df.iloc[:,i]
        colors = plot_loadings(axes[i], abs(component_loadings), groups)
    for j in range(n_components, len(axes)):
        axes[j].set_visible(False)
    if legend and groups is not None:
        create_categorical_legend([g[0] for g in groups], 
                                  colors, axes[n_components-1])
    if input_axes is None:
        return fig

# helper functions
def reorder_data(data, groups, axis=1):
    ordered_cols = []
    for i in groups:
        ordered_cols += i[1]
    new_data = data.reindex_axis(ordered_cols, axis)
    return new_data

def create_factor_tree(data, component_range=(1,13)):
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
    def get_similarity_order(lower_dim, higher_dim):
        "Helper function to reorder factors into correspondance between two dimensionalities"
        subset = corr_lower_higher(higher_dim, lower_dim)
        max_factors = np.argmax(abs(subset.values), axis=0)
        return np.argsort(max_factors)

    EFA_results = {}
    full_fa_results = {}
    # plot
    for c in range(component_range[0],component_range[1]+1):
        fa, output = psychFA(data, c)
        tmp_loading_df = get_loadings(output, labels=data.columns)
        if (c-1) in EFA_results.keys():
            reorder_index = get_similarity_order(tmp_loading_df, EFA_results[c-1])
            tmp_loading_df = tmp_loading_df.iloc[:, reorder_index]
            tmp_loading_df.columns = sorted(tmp_loading_df.columns)
        EFA_results[c] = tmp_loading_df
        full_fa_results[c] = fa
    return EFA_results, full_fa_results

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
        f.savefig(filename)
    else:
        return f

