import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from os import path
import pandas as pd
from scipy.spatial.distance import  squareform
from sklearn.manifold import MDS
from sklearn.preprocessing import scale
import seaborn as sns
from dimensional_structure.HCA_plots import abs_pdist
from dimensional_structure.plot_utils import  save_figure
from selfregulation.utils.utils import get_behav_data
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.plot_utils import format_num, DDM_plot

# load data
results = load_results('Complete_03-29-2018')
data = results['task'].data
out = results['task'].EFA.get_loading()
nfactors = out.shape[1]
task_subset = pd.concat([
    out.filter(regex='choice_reaction_time', axis=0),
    out.filter(regex='^stop_signal\.(hddm|SSRT)', axis=0)[1:5]])

task_subset_data = data.loc[:, task_subset.index]

var_subset =  pd.concat([
    out.filter(regex='\.hddm_drift$', axis=0),
    out.filter(regex='\.hddm_thresh', axis=0),
    out.filter(regex='\.hddm_non_decision$', axis=0),
    out.filter(regex='\.SSRT', axis=0)], axis=0)
var_subset_data = data.loc[:, var_subset.index]


# Ridiculous analysis overview plot
colors = [(1.0, 0.0, 0.0), 
          (0.0, 0.5, 0.0), 
          (0.0, 0.75, 0.75), 
          (0.75, 0.0, 0.75)]
color_lookup = {'drift': colors[0],
                'thresh': colors[1], 
                'non-decision': colors[2],
                'SSRT': colors[3]}
f = plt.figure(figsize=(12,12))
task1_ax = f.add_axes([0, .555, .2, .15])
task2_ax = f.add_axes([0, .755, .2, .2])
task1_ax.axis('off'); task2_ax.axis('off')

participant_ax1 = f.add_axes([.25,.555,.28,.15]) 
participant_ax2 = f.add_axes([.25,.75,.28,.2]) 

loading_ax1 = f.add_axes([.625,.555,.25,.15]) 
loading_ax2 = f.add_axes([.625,.75,.25,.2]) 

participant_distance = f.add_axes([.3,.32,.15,.15]) 
loading_distance = f.add_axes([.675,.32,.15,.15]) 
#participant_distance.axis('off'); loading_distance.axis('off')

participant_mds = f.add_axes([.25,0,.25,.25]) 
loading_mds = f.add_axes([.625,0,.25,.25]) 
cbar_ax = f.add_axes([.92,.6,.03,.3]) 
back = f.add_axes([0,0,1,1])
back.axis('off')
back.patch.set_alpha(0)
# label 
back.text(.05, .75, 'Measure', horizontalalignment='center', 
          verticalalignment='center',
          fontsize=20,
          fontweight='bold', rotation=90)
back.text(.22, .97, 'Sub-Metric', horizontalalignment='center', fontsize=20,
          fontweight='bold')
task1_ax.text(.5,.7, 'Choice Reaction Time', fontsize=15,
              horizontalalignment='center', verticalalignment='center',
              rotation=90)
task2_ax.text(.5,.7, 'Stop Signal', fontsize=15,
              horizontalalignment='center', verticalalignment='center',
              rotation=90)


tasks = sorted(np.unique([i.split('.')[0] for i in task_subset.index]))
task_axes = [task1_ax, task2_ax]
participant_axes = [participant_ax1, participant_ax2]
loading_axes = [loading_ax1, loading_ax2]
for task_i in range(len(tasks)):
    tick_names = []
    # *************************************************************************
    # ***** plot participants on two tasks ***** 
    # *************************************************************************
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
        plot_vals = scale(vals[20:40])*.25+i*1.5
        ax.hlines(i*1.5, 0, len(plot_vals)*.8, alpha=.6,
                  linestyle='--', color=color_lookup[name])
        scatter_colors = [list(color_lookup[name])+[alpha] for alpha in np.linspace(1,0, len(plot_vals))]
        ax.scatter(range(len(plot_vals)), plot_vals, color=scatter_colors)
    # make x ticks invisible
    ax.set_xticklabels('')
    ax.tick_params(axis='both', length=0)
    # remove splines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # add tick labels
    ax.set_yticks([x*1.5 for x in range(len(tick_names))])
    ax.set_yticklabels(tick_names, fontsize=15)
    # change tick color
    tick_colors = [color_lookup[name] for name in tick_names]
    [t.set_color(i) for (i,t) in
         zip(tick_colors,ax.yaxis.get_ticklabels())]
    
    # *************************************************************************
    # ***** plot loading ***** 
    # *************************************************************************
    max_val = round(abs(task_subset).max().max(),1)
    loading_data = task_subset.filter(regex=tasks[task_i], axis=0)
    sns.heatmap(loading_data.iloc[::-1,:], ax=loading_axes[task_i], 
                yticklabels=False, xticklabels=False,
                cbar_ax=cbar_ax, vmax =  max_val, vmin = -max_val,
                cbar_kws={'ticks': [-max_val, 0, max_val]},
                cmap=sns.diverging_palette(220,15,n=100, as_cmap=True))
    # format cbar
    cbar_ax.set_yticklabels([format_num(-max_val, 1), 0, format_num(max_val, 1)])
    cbar_ax.tick_params(axis='y', length=0)
    cbar_ax.tick_params(labelsize=15)
    for i in range(loading_data.shape[0]):
        loading_axes[task_i].hlines(i, -.1, 5.1, color='white', linewidth=4)
    
# add labels 
loading_ax1.set_xlabel('Factor Loading', fontsize=15, labelpad=10)
# loading ticks
loading_ax2.xaxis.set_ticks_position('top')
loading_ax2.set_xticks(np.arange(.5,5.5,1))
loading_ax2.set_xticklabels(['Factor %s' % i for i in range(1,nfactors+1)],
                            rotation=45, ha='left', fontsize=14)

# participant box
back.add_patch(Rectangle((.3385,.55), width=.011, height=.39, 
                         facecolor="none", edgecolor='grey', linewidth=1.5))
back.text(.3385, .53, 'One Participant', fontsize=12, 
          horizontalalignment='center', color='grey')


# arrows
back.arrow(.52, .725, .05, 0, width=.005, facecolor='k')
# from data to heatmap
back.arrow(.375, .51, 0, -.01, width=.004, facecolor='k')
back.arrow(.75, .51, 0, -.01, width=.004, facecolor='k')
# from heatmap to MDS
back.arrow(.375, .31, 0, -.01, width=.004, facecolor='k')
back.arrow(.75, .31, 0, -.01, width=.004, facecolor='k')
back.text(.567, .26, 'MDS Projection', fontsize=24, 
          horizontalalignment='center')

# ****************************************************************************
# Distance Matrices
# ****************************************************************************
participant_distances = squareform(abs_pdist(data.T))
participant_distances = results['task'].HCA.results['clustering_input-data']['clustered_df']
loading_distances = results['task'].HCA.results['clustering_input-EFA5']['clustered_df']
sns.heatmap(participant_distances, ax=participant_distance,
            xticklabels=False, yticklabels=False, square=True, cbar=False)
sns.heatmap(loading_distances, ax=loading_distance,
            xticklabels=False, yticklabels=False, square=True, cbar=False)
# ****************************************************************************
# MDS Plots
# ****************************************************************************

mds_colors = [[.5, .5, .5]]*loading_distances.shape[0]
for i, label in enumerate(loading_distances.index):
    if 'drift' in label:
        name = 'drift'
    elif 'thresh' in label:
        name = 'thresh'
    elif 'non_decision' in label:
        name = 'non-decision'
    elif 'SSRT' in label:
        name = 'SSRT'
    else:
        continue
    mds_colors[i] = color_lookup[name]
# plot raw MDS
np.random.seed(700)
f, axes = plt.subplots(1,2, figsize=(10,5))
participant_mds, loading_mds = axes
mds = MDS(dissimilarity='precomputed')
mds_out = mds.fit_transform(participant_distances)
participant_mds.scatter(mds_out[:,0], mds_out[:,1], 
            s=220,
            marker='h',
            facecolors=mds_colors,
            edgecolors='white')
participant_mds.set_xticklabels(''); participant_mds.set_yticklabels('')
participant_mds.tick_params(axis='both', length=0)
participant_mds.axis('off')
# plot loading MDS
mds = MDS(dissimilarity='precomputed')
mds_out = mds.fit_transform(loading_distances)
loading_mds.scatter(mds_out[:,0], mds_out[:,1], 
            s=220,
            marker='h',
            facecolors=mds_colors,
            edgecolors='white')
loading_mds.set_xticklabels(''); loading_mds.set_yticklabels('')
loading_mds.tick_params(axis='both', length=0)
loading_mds.axis('off'); 

"""
# save
plot_file = path.dirname(results['task'].plot_dir)
f.savefig(path.join(plot_file, 'analysis_overview.pdf'), 
                bbox_inches='tight', 
                dpi=300)


# example DDM plot
np.random.seed(1000)
ddm_plot, trajectories = DDM_plot(2, .2, 3, n=100, plot_n=7, 
                                  file=path.join(plot_file, 'DDM.pdf'))
"""