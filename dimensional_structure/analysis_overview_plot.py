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
task_subset = pd.concat([
    out.filter(regex='choice_reaction_time', axis=0),
    out.filter(regex='^stop_signal', axis=0)[1:5]])

task_subset_data = data.loc[:, task_subset.index]
task_subset_data = pd.DataFrame(scale(task_subset_data), 
                                 index=task_subset_data.index,
                                 columns=task_subset_data.columns)
var_subset =  pd.concat([
    out.filter(regex='\.hddm_drift$', axis=0),
    out.filter(regex='\.hddm_thresh$', axis=0),
    out.filter(regex='\.hddm_non_decision$', axis=0),
    out.filter(regex='\.SSRT', axis=0)], axis=0)
var_subset_data = data.loc[:, var_subset.index]
var_subset_data = pd.DataFrame(scale(var_subset_data), 
                                 index=var_subset_data.index,
                                 columns=var_subset_data.columns)   

colors = ['r', 'g', 'k', 'm']
color_lookup = {'drift': colors[0],
                'thresh': colors[1], 
                'non-decision': colors[2],
                'SSRT': colors[3]}
f = plt.figure(figsize=(12,14))
back = f.add_axes([0,0,1,1])
back.axis('off')
task1_ax = f.add_axes([0, .375, .2, .15])
task2_ax = f.add_axes([0, .575, .2, .15])

participant_ax1 = f.add_axes([.25,.355,.25,.15]) 
participant_ax2 = f.add_axes([.25,.55,.25,.2]) 

loading_ax1 = f.add_axes([.625,.355,.25,.15]) 
loading_ax2 = f.add_axes([.625,.55,.25,.2]) 

participant_mds = f.add_axes([.25,0,.25,.25]) 
loading_mds = f.add_axes([.625,0,.25,.25]) 
cbar_ax = f.add_axes([.92,.4,.03,.3]) 

# label 
back.text(0, .77, 'Measure', horizontalalignment='center', fontsize=17,
          fontweight='bold')
back.text(.22, .77, 'Sub-Metric', horizontalalignment='center', fontsize=17,
          fontweight='bold')
task1_ax.text(0,.5, 'Stop Signal', fontsize=15,
              horizontalalignment='center', verticalalignment='center',
              rotation=90)
task2_ax.text(0,.5, 'Choice Reaction Time', fontsize=15,
              horizontalalignment='center', verticalalignment='center',
              rotation=90)

task1_ax.axis('off'); task2_ax.axis('off')

tasks = sorted(np.unique([i.split('.')[0] for i in task_subset.index]))
task_axes = [task1_ax, task2_ax]
participant_axes = [participant_ax1, participant_ax2]
loading_axes = [loading_ax1, loading_ax2]
for task_i in range(len(tasks)):
    tick_names = []

    # ***** plot participants on two tasks ***** 
    ax = participant_axes[task_i]
    plot_data = task_subset_data.filter(regex=tasks[task_i], axis=1)
    for i, (label, vals) in enumerate(plot_data.iteritems()):
        if 'drift' in label:
            name = 'drift'
        elif 'thresh' in label:
            name = 'thresh'
        elif 'non_decision' in label:
            name = 'non-decision'
        else:
            name = 'SSRT'
        tick_names.append(name)
        vals = pd.rolling_mean(vals, 20)
        plot_vals = vals[::40]*.75+i
        ax.hlines(i, 0, len(plot_vals), linestyle='--', color=color_lookup[name])
        ax.plot(range(len(plot_vals)), plot_vals,
                'o', linewidth=3, color=color_lookup[name])
    # make x ticks invisible
    ax.set_xticklabels('')
    ax.tick_params(axis='both', length=0)
    # remove splines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # add tick labels
    ax.set_yticks(range(len(tick_names)))
    ax.set_yticklabels(tick_names, fontsize=15)
    # change tick color
    tick_colors = [color_lookup[name] for name in tick_names]
    [t.set_color(i) for (i,t) in
         zip(tick_colors,ax.yaxis.get_ticklabels())]
    
    # ***** plot loading ***** 
    max_val = abs(task_subset).max().max()
    loading_data = task_subset.filter(regex=tasks[task_i], axis=0)
    sns.heatmap(loading_data.iloc[::-1,:], ax=loading_axes[task_i], yticklabels=False, xticklabels=False, 
                cbar_ax=cbar_ax, vmax =  max_val, vmin = -max_val,
                cbar_kws={'ticks': [-max_val, 0, max_val]},
                cmap=sns.diverging_palette(220,15,n=100, as_cmap=True))
    cbar_ax.set_yticklabels([format_num(-max_val), 0, format_num(max_val)])
    cbar_ax.tick_params(labelsize=15)
    for i in range(loading_data.shape[0]):
        loading_axes[task_i].hlines(i, -.1, 5.1, color='white', linewidth=3)
    
# add labels and arrows
participant_ax1.spines['bottom'].set_visible(True)
participant_ax1.set_xlabel('Participant (n=522)', fontsize=15)
loading_ax1.set_xlabel('Factor Loading', fontsize=15, labelpad=10)
# arrows
back.arrow(.52, .525, .05, 0, width=.005, facecolor='k')
back.arrow(.375, .31, 0, -.02, width=.005, facecolor='k')
back.arrow(.75, .31, 0, -.02, width=.005, facecolor='k')
back.text(.5625, .27, 'MDS Projection', fontsize=24, 
          horizontalalignment='center')


# plot raw MDS
num_drift = len(var_subset.filter(regex='drift', axis=0))
num_thresh = len(var_subset.filter(regex='thresh', axis=0))
num_non = len(var_subset.filter(regex='non_decision', axis=0))
num_SSRT = len(var_subset.filter(regex='SSRT', axis=0))
colors = [colors[0]]*num_drift+[colors[1]]*num_thresh+[colors[2]]*num_non+[colors[3]]*num_SSRT
        
np.random.seed(1000)
space_distances = squareform(abs_pdist(var_subset_data.T))
mds = MDS(dissimilarity='precomputed')
mds_out = mds.fit_transform(space_distances)
participant_mds.scatter(mds_out[:,0], mds_out[:,1], 
            s=200,
            facecolors=colors,
            edgecolors='white')
participant_mds.set_xticklabels(''); participant_mds.set_yticklabels('')
participant_mds.tick_params(axis='both', length=0)
participant_mds.axis('off')
# plot loading MDS
space_distances = squareform(abs_pdist(var_subset))
mds = MDS(dissimilarity='precomputed')
mds_out = mds.fit_transform(space_distances)
loading_mds.scatter(mds_out[:,0], mds_out[:,1], 
            s=200,
            facecolors=colors,
            edgecolors='white')
loading_mds.set_xticklabels(''); loading_mds.set_yticklabels('')
loading_mds.tick_params(axis='both', length=0)
loading_mds.axis('off'); 

#plot_file = path.dirname(results['task'].plot_dir)
#f.savefig(path.join(plot_file, 'toyplot.pdf'), 
#                bbox_inches='tight', 
#                dpi=300)

