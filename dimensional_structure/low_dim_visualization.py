# imports

import matplotlib.pyplot as plt
from os import makedirs, path
import pandas as pd
import pickle
import seaborn as sns
from sklearn.manifold import t_sne, MDS
from sklearn.preprocessing import scale


# ****************************************************************************
# Laad Data
# ****************************************************************************
datafile = 'Complete_10-08-2017'
plot_file = path.join('Plots', datafile)
output_file = path.join('Output', datafile)
makedirs(plot_file, exist_ok = True)
makedirs(output_file, exist_ok = True)

results = pickle.load(open(path.join(output_file, 'EFA_results.pkl'),'rb'))


inputs = {'data': scale(results['data']).T, 
          'EFA22': results['EFA']['factor_tree'][22]}



def visualize_loading(results, c)
    HCA = results.HCA
    EFA = results.EFA
    c = 9
    
    cluster_loadings = HCA.get_cluster_loading(EFA, 'data', c)
    cluster_loadings_mat = np.vstack([i[1] for i in cluster_loadings])
    EFA_loading = abs(EFA.get_loading(c))
    EFA_loading_mat = EFA_loading.values
    input_data = np.vstack([cluster_loadings_mat, EFA_loading_mat])
    
    
    n_clusters = cluster_loadings_mat.shape[0]
    color_palette = sns.color_palette(palette='hls', n_colors=n_clusters)
    colors = []
    for var in EFA_loading.index:
        # find which cluster this variable is in
        index = [i for i,cluster in enumerate(cluster_loadings) \
                 if var in cluster[0]][0]
        colors.append(color_palette[index])
        
    
    mds = MDS()
    mds_out = mds.fit_transform(input_data)
    
    
    plt.figure(figsize=(20,20))
    plt.scatter(mds_out[:n_clusters,0], mds_out[:n_clusters,1], 
                marker='*', s=400, color=color_palette)
    plt.scatter(mds_out[n_clusters:,0], mds_out[n_clusters:,1], 
                s=50, color=colors)

