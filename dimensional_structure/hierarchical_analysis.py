# imports
from dimensional_structure.utils import distcorr
from os import makedirs, path
import pickle
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

metric = 'SABIC_c'
data = results['data']
best_c = results[metric]
loading = results['factor_tree'][best_c]



# ****************************************************************************
# Plotting
# ****************************************************************************
fig, output = dendroheatmap(data.T, figsize=(50,50), parse=10,
                            pdist_kws={'metric': distcorr})

fig.savefig(path.join(plot_file,'data_distcorr_dendorgramheatmap.png'))
results['distcorr_clustering'] = output


fig, output = dendroheatmap(loading, figsize=(50,50), parse=10)
fig.savefig(path.join(plot_file,'EFA%s_pearson_dendorgramheatmap.png' % best_c))

fig, output = dendroheatmap(data.T, figsize=(50,50), parse=10)
fig.savefig(path.join(plot_file,'data__pearson_dendorgramheatmap.png'))

