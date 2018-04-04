import matplotlib.pyplot as plt
import numpy as np
from os import path
import pandas as pd
from scipy.spatial.distance import  squareform
from sklearn.manifold import MDS
from sklearn.preprocessing import scale
import seaborn as sns
from dimensional_structure.HCA_plots import abs_pdist
from dimensional_structure.plot_utils import  save_figure
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.plot_utils import format_num

# load data
results = load_results('Complete_03-29-2018')
data = results['task'].data
out = results['task'].EFA.get_loading()
subset = pd.concat([
    out.filter(regex='\.hddm_drift$', axis=0)[1:4],
    out.filter(regex='\.hddm_thresh$', axis=0)[1:4],
    out.filter(regex='\.hddm_non_decision$', axis=0)[1:4],
    out.filter(regex='\.SSRT', axis=0)[0:5]], axis=0)

subset_data = data.loc[:, subset.index]
subset_data = pd.DataFrame(scale(subset_data), 
                                 index=subset_data.index,
                                 columns=subset_data.columns)

    

colors = ['r', 'g', 'k', 'm']
color_lookup = {'drift': colors[0],
                'thresh': colors[1], 
                'non-decision': colors[2],
                'SSRT': colors[3]}
f = plt.figure(figsize=(12,14))
ax = f.add_axes([0,.35,.4,.4]) 
ax2 = f.add_axes([.45,.36,.3,.39]) 
ax3 = f.add_axes([.05,0,.3,.3]) 
ax4 = f.add_axes([.45,0,.3,.3]) 
cbar_ax = f.add_axes([.77,.4,.03,.3]) 

# plot participants
for i, (label, vals) in enumerate(subset_data.iteritems()):
    if 'drift' in label:
        name = 'drift'
    elif 'thresh' in label:
        name = 'thresh'
    elif 'non_decision' in label:
        name = 'non-decision'
    else:
        name = 'SSRT'
    vals = pd.rolling_mean(vals, 20)
    ax.plot(vals[::20]+i, linewidth=3, color = color_lookup[name])

ax.set_yticks([1,4,7,11])
ax.set_yticklabels(['Drift', 'Threshold', 'Non-Decision', 'SSRT'], fontsize=24)
ax.set_xlabel('Participant', fontsize=24, labelpad=5)
# make x ticks invisible
ax.set_xticklabels('')
ax.tick_params(axis='both', length=0)
# change tick color
[t.set_color(i) for (i,t) in
 zip(colors,ax.yaxis.get_ticklabels())]
# remove splines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

max_val = abs(subset).max().max()
sns.heatmap(subset.iloc[::-1,:], ax=ax2, yticklabels=False, xticklabels=False, 
            cbar_ax=cbar_ax, vmax =  max_val, vmin = -max_val,
            cbar_kws={'ticks': [-max_val, 0, max_val]},
            cmap=sns.diverging_palette(220,15,n=100, as_cmap=True))
ax2.hlines(4, 0, 5, linestyles='dashed', linewidth=3, colors=[1,1,1])
ax2.hlines(7, 0, 5, linestyles='dashed', linewidth=3, colors=[1,1,1])
ax2.hlines(10, 0, 5, linestyles='dashed',  linewidth=3, colors=[1,1,1])
ax2.set_xlabel('Factor Loading', fontsize=24, labelpad=15)
cbar_ax.set_yticklabels([format_num(-max_val), 0, format_num(max_val)])
cbar_ax.tick_params(labelsize=15)


# plot raw MDS
np.random.seed(1000)
space_distances = squareform(abs_pdist(subset_data.T))
mds = MDS(dissimilarity='precomputed')
mds_out = mds.fit_transform(space_distances)
ax3.scatter(mds_out[:,0], mds_out[:,1], 
            s=250,
            facecolors=[colors[0]]*3+[colors[1]]*3+[colors[2]]*3+[colors[3]]*4,
            edgecolors='white')
ax3.set_ylabel('MDS Projection', fontsize=24)
ax3.set_xticklabels(''); ax3.set_yticklabels('')
ax3.tick_params(axis='both', length=0)
# plot loading MDS
space_distances = squareform(abs_pdist(subset))
mds = MDS(dissimilarity='precomputed')
mds_out = mds.fit_transform(space_distances)
ax4.scatter(mds_out[:,0], mds_out[:,1], 
            s=250,
            facecolors=[colors[0]]*3+[colors[1]]*3+[colors[2]]*3+[colors[3]]*4,
            edgecolors='white')
ax4.set_xticklabels(''); ax4.set_yticklabels('')
ax4.tick_params(axis='both', length=0)
    
plot_file = path.dirname(results['task'].plot_dir)
f.savefig(path.join(plot_file, 'toyplot.pdf'), 
                bbox_inches='tight', 
                dpi=300)

