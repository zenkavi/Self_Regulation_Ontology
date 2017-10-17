# imports
from dimensional_structure.utils import distcorr
import matplotlib.pyplot as plt
from os import makedirs, path
import pandas as pd
import pickle
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
from selfregulation.utils.plot_utils import dendroheatmap

# ****************************************************************************
# Laad Data
# ****************************************************************************
datafile = 'Complete_10-08-2017'
plot_file = path.join('Plots', datafile)
output_file = path.join('Output', datafile)
makedirs(plot_file, exist_ok = True)
makedirs(output_file, exist_ok = True)

results = pickle.load(open(path.join(output_file, 'EFA_results.pkl'),'rb'))
data = results['data']


# ****************************************************************************
# Analysis
# ****************************************************************************

fig, output = dendroheatmap(data.T, figsize=(50,50), parse=10,
                            pdist_kws={'metric': distcorr})
fig.savefig(path.join(plot_file,'dendrogramheatmap_metric-distcorr.png'))
results['distcorr_clustering'] = output

# plot distance correlation for factor solutions in the same order as the
# clustered solution
clustered_df = output['clustered_df']
cluster_order = clustered_df.index

fig = plt.figure(figsize=(40,40))
sns.heatmap(clustered_df, square=True)
fig.savefig(path.join(plot_file,
                          'heatmap_metric-distcorr.png'), bbox_inches='tight')

factor_distances = {}
for c, loadings in results['factor_tree'].items():
    if c>2:
        loadings = loadings.copy().loc[cluster_order, :]
        distances = squareform(pdist(loadings, metric=distcorr))
        distances = pd.DataFrame(distances, 
                                 index=loadings.index, 
                                 columns=loadings.index)
        factor_distances[c] = squareform(distances)
        # plot
        fig = plt.figure(figsize=(40,40))
        sns.heatmap(distances, square=True)
        fig.savefig(path.join(plot_file,
                              'heatmap_metric-distcorr_c-%02d.png' % c), 
                    bbox_inches='tight')

factor_distances = pd.DataFrame(factor_distances)
factor_distances.loc[:, 'raw'] = squareform(clustered_df)

with sns.plotting_context('notebook', font_scale=1.8):
    f = plt.figure(figsize=(12,8))
    factor_distances.corr()['raw'][:-1].plot()
    plt.xlabel('Factors in EFA')
    plt.ylabel('Correlation with Raw Values')
    plt.title('Distance Matrix Correlations')
    f.savefig(path.join(plot_file, 
                        'distance_correlations_across_factors_metric-distcorr.png'))
    
