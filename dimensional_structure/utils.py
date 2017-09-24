from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from selfregulation.utils.r_to_py_utils import psychFA
from sklearn.preprocessing import StandardScaler

# functions to fit and extract factor analysis solutions
def find_optimal_components(data, minc=1, maxc=20):
    """
    Fit psychFA over a range of components and returns the best c 
    """
    BICs = {}
    outputs = []
    n_components = range(minc,maxc)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    for c in n_components:
        fa, output = psychFA(scaled_data, c, method='ml')
        BICs[c] = output['BIC']
        outputs.append(output)
    best_c = np.argmin(BICs)+1
    print('Best Component: ', best_c)
    return best_c, BICs

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

def print_top_factors(loading_df, n=4):
    """
    Takes output of get_loadings and prints the absolute top variables per factor
    """
    # number of variables to display
    for i,column in loading_df.iteritems():
        sort_index = np.argsort(abs(column))[::-1] # descending order
        top_vars = column[sort_index][0:n]
        print('\nFACTOR %s' % i)
        print(top_vars)
        
        
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
    if legend and groups is not None:
        create_categorical_legend([g[0] for g in groups], colors, axes[-1])
    if input_axes is None:
        return fig

# helper functions
def reorder_data(data, groups):
    ordered_cols = [j for i in groups for j in i[1]]
    new_data = data.reindex_axis(ordered_cols, axis=1)
    return new_data

def create_factor_tree(data, groups=None, component_range=(1,13)):
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
    def get_similarity_order(higher_dim, lower_dim):
        "Helper function to reorder factors into correspondance between two dimensionalities"
        corr = abs(pd.concat([higher_dim,lower_dim], axis=1).corr())
        subset = corr.iloc[c:,:c] # rows are former EFA result, cols are current
        max_factors = np.argmax(subset.values, axis=1)
        remaining = np.sum(range(c))-np.sum(max_factors)
        return np.append(max_factors, remaining)

    EFA_results = {}
    if groups != None:
        data = reorder_data(data, groups)
    
    # plot
    for c in range(component_range[0],component_range[1]+1):
        fa, output = psychFA(data, c)
        tmp_loading_df = get_loadings(output, labels=data.columns)
        if (c-1) in EFA_results.keys():
            reorder_index = get_similarity_order(tmp_loading_df, EFA_results[c-1])
            tmp_loading_df = tmp_loading_df.iloc[:, reorder_index]
        EFA_results[c] = tmp_loading_df
    return EFA_results

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
    return f

# helper function
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from scipy.spatial.distance import pdist, squareform

def get_hierarchical_groups(loading_df, n_groups=8):
    # helper function
    def remove_adjacent(nums):
        result = []
        for num in nums:
            if len(result) == 0 or num != result[-1]:
                result.append(num)
        return result

    # create linkage matrix for variables projected into a component loading
    row_clusters = linkage(pdist(loading_df), method='ward')   
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