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



for name, input_data in inputs.items():
    HCA = results['HCA']['clustering_metric-distcorr_input-%s' % name]
    clusters = HCA['dynamic_clusters']['labels']
    color_palette = sns.color_palette(palette='hls', n_colors=max(clusters))
    colors = [color_palette[i-1] for i in clusters]
    
    # apply two different dimensionality reductions
    tsne = t_sne.TSNE(perplexity=5)
    tsne_out = tsne.fit_transform(input_data)
    mds = MDS()
    mds_out = mds.fit_transform(input_data)
    
    f = plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.scatter(tsne_out[:,0], tsne_out[:,1], c=colors)
    plt.subplot(1,2,2)
    plt.scatter(mds_out[:,0], mds_out[:,1], c=colors)
