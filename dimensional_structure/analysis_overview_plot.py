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

# Ridiculous analysis overview plot
colors = [[.7, 0.0, 0.0], 
          [0.0, 0.5, 0.0], 
          [0.0, 0.75, 0.75], 
          [0.75, 0.0, 0.75]]
color_lookup = {'drift rate': colors[0],
                'threshold': colors[1], 
                'non-decision': colors[2],
                'SSRT': colors[3]}
f = plt.figure(figsize=(4.6, 4.6))
basefont = 6
basemarker = 40
basewidth = .6

#f = plt.figure(figsize=(12, 12))
#basefont = 16
#basemarker = 220
#basewidth = 2


participant_ax1 = f.add_axes([.25,.555,.28,.16]) 
participant_ax2 = f.add_axes([.25,.75,.28,.2]) 

loading_ax1 = f.add_axes([.625,.555,.25,.146]) 
loading_ax2 = f.add_axes([.625,.75,.25,.189]) 

participant_distance = f.add_axes([.3,.32,.16,.16]) 
loading_distance = f.add_axes([.675,.32,.16,.16]) 
#participant_distance.axis('off'); loading_distance.axis('off')
participant_mds = f.add_axes([.25,0,.25,.25]) 
loading_mds = f.add_axes([.625,0,.25,.25]) 
# color bars for heatmaps
cbar_ax = f.add_axes([.92,.6,.03,.3]) 
cbar_ax2 = f.add_axes([.86,.34,.02,.12]) 
# set background
back = f.add_axes([0,0,1,1])
back.axis('off')
back.patch.set_alpha(0)
back.set_xlim([0,1]); back.set_ylim([0,1])

tasks = sorted(np.unique([i.split('.')[0] for i in task_subset.index]))
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
            name = 'drift rate'
        elif 'thresh' in label:
            name = 'threshold'
        elif 'non_decision' in label:
            name = 'non-decision'
        else:
            name = 'SSRT'
        tick_names.append(name)
        plot_vals = scale(vals[20:40])*.25+i*1.5
        # add mean line
        ax.hlines(i*1.5, 0, len(plot_vals)*.8, alpha=.6,
                  linestyle='--', color=color_lookup[name],
                  linewidth=basewidth)
        # plot values
        scatter_colors = [list(color_lookup[name])+[alpha] for alpha in np.linspace(1,0, len(plot_vals))]
        ax.scatter(range(len(plot_vals)), plot_vals, color=scatter_colors,
                   s=basemarker*.23)
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
    ax.set_yticklabels(tick_names, fontsize=basefont)
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
                cmap=sns.diverging_palette(220,16,n=100, as_cmap=True))
    # format cbar
    cbar_ax.set_yticklabels([format_num(-max_val, 1), 0, format_num(max_val, 1)])
    cbar_ax.tick_params(axis='y', length=0)
    cbar_ax.tick_params(labelsize=basefont)
    for i in range(1,loading_data.shape[0]):
        loading_axes[task_i].hlines(i, -.2, 6.1, color='white', linewidth=basewidth*3)
    
# ****************************************************************************
# Distance Matrices
# ****************************************************************************
participant_distances = squareform(abs_pdist(data.T))
participant_distances = results['task'].HCA.results['clustering_input-data']['clustered_df']
loading_distances = results['task'].HCA.results['clustering_input-EFA5']['clustered_df']
sns.heatmap(participant_distances, ax=participant_distance,
            xticklabels=False, yticklabels=False, square=True, cbar=False)
sns.heatmap(loading_distances, ax=loading_distance,
            xticklabels=False, yticklabels=False, square=True, 
            cbar_kws={'ticks': [0, .99]}, cbar_ax=cbar_ax2)
participant_distance.set_ylabel('DV', fontsize=basefont*.875)
loading_distance.set_ylabel('DV', fontsize=basefont*.875)
# format cbar
cbar_ax2.set_yticklabels([0, 1])
cbar_ax2.tick_params(axis='y', length=0)
cbar_ax2.tick_params(labelsize=basefont*.75)
# ****************************************************************************
# MDS Plots
# ****************************************************************************

mds_colors = np.array([[.5, .5, .5, .4]]*loading_distances.shape[0])
interest_index = []
misc_index = []
for i, label in enumerate(loading_distances.index):
    if '.hddm_drift' in label:
        name = 'drift rate'
    elif '.hddm_thresh' in label:
        name = 'threshold'
    elif '.hddm_non_decision' in label:
        name = 'non-decision'
    elif 'SSRT' in label:
        name = 'SSRT'
    else:
        misc_index.append(i)
        continue
    interest_index.append(i)
    mds_colors[i] = color_lookup[name]+[1]
mds_index = misc_index + interest_index

# plot raw MDS
np.random.seed(700)
mds = MDS(dissimilarity='precomputed')
mds_out = mds.fit_transform(participant_distances)
participant_mds.scatter(mds_out[mds_index,0], mds_out[mds_index,1], 
            s=basemarker,
            marker='h',
            facecolors=mds_colors[mds_index],
            edgecolors='white')
participant_mds.set_xticklabels(''); participant_mds.set_yticklabels('')
participant_mds.tick_params(axis='both', length=0)
participant_mds.axis('off')
# plot loading MDS
mds = MDS(dissimilarity='precomputed')
mds_out = mds.fit_transform(loading_distances)
loading_mds.scatter(mds_out[mds_index,0], mds_out[mds_index,1], 
            s=basemarker,
            marker='h',
            facecolors=mds_colors[mds_index],
            edgecolors='white')
loading_mds.set_xticklabels(''); loading_mds.set_yticklabels('')
loading_mds.tick_params(axis='both', length=0)
loading_mds.axis('off'); 

# get example points
var_locs = []
subplot_colors=[]
for label in task_subset.index:
    if 'drift' in label:
        var_color = color_lookup['drift rate']
    elif 'thresh' in label:
        var_color = color_lookup['threshold']
    elif 'non_decision' in label:
        var_color = color_lookup['non-decision']
    else:
        var_color = color_lookup['SSRT']
    index = np.where(loading_distances.index==label)[0][0]
    var_loc = mds_out[index]
    var_locs.append((label, var_loc))
    subplot_colors.append(var_color)

width = sum(np.abs(list(loading_mds.get_xlim())))
height = sum(np.abs(list(loading_mds.get_ylim())))
loading_mds.scatter([v[1][0] for v in var_locs],
                    [v[1][1] for v in var_locs],
                    edgecolors='white',
                    facecolors=subplot_colors,
                    marker='h',
                    s=basemarker)
loading_mds.scatter([v[1][0] for v in var_locs],
                    [v[1][1] for v in var_locs],
                    edgecolors='white',
                    facecolors='yellow',
                    marker='.',
                    s=basemarker*.7)
# ****************************************************************************
# Text and additional pretty lines
# ****************************************************************************
# label 
back.text(-.03, .96, 'Measure', horizontalalignment='center', fontsize=basefont,
          fontweight='bold')
back.text(.21, .94, 'DV', horizontalalignment='center', fontsize=basefont,
          fontweight='bold')
# task labels
#back.text(.05, .75, 'Measure', horizontalalignment='center', 
#          verticalalignment='center',
#          fontsize=basefont*1.56250,
 #         fontweight='bold', rotation=90)
back.text(-.03,.62, 'Choice RT', fontsize=basefont, rotation=0, 
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.87, 'Stop Signal', fontsize=basefont, rotation=0, 
              horizontalalignment='center', verticalalignment='center')
# other tasks
back.text(-.03,.6, 'Bickel Titrator', fontsize=basefont, rotation=0, alpha=.5,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.58, 'ART', fontsize=basefont, rotation=0, alpha = .4,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.56, 'ANT', fontsize=basefont, rotation=0, alpha=.3,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.54, 'Adaptive N-Back', fontsize=basefont, rotation=0, alpha=.2,
              horizontalalignment='center', verticalalignment='center')

back.text(-.03,.64, 'CCT-Cold', fontsize=basefont, rotation=0, alpha=.5,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.66, 'CCT-Hot', fontsize=basefont, rotation=0, alpha = .4,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.68, 'Dietary Decision', fontsize=basefont, rotation=0, alpha=.3,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.7, 'Digit Span', fontsize=basefont, rotation=0, alpha=.2,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.72, 'Directed Forgetting', fontsize=basefont, rotation=0, alpha=.1,
              horizontalalignment='center', verticalalignment='center')

back.text(-.03,.89, 'Stroop', fontsize=basefont, rotation=0, alpha=.5,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.91, 'Three-By-Two', fontsize=basefont, rotation=0, alpha = .4,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.93, 'Tower of London', fontsize=basefont, rotation=0, alpha=.3,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.85, 'Stim-Selective SS', fontsize=basefont, rotation=0, alpha=.5,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.83, 'Spatial Span', fontsize=basefont, rotation=0, alpha = .4,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.81, 'Simple RT', fontsize=basefont, rotation=0, alpha=.3,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.79, 'Simon', fontsize=basefont, rotation=0, alpha=.2,
              horizontalalignment='center', verticalalignment='center')
back.text(-.03,.77, 'Shift Task', fontsize=basefont, rotation=0, alpha=.1,
              horizontalalignment='center', verticalalignment='center')
# add labels 
cbar_ax.tick_params('y', which='major', pad=basefont*.5)
cbar_ax.set_ylabel('Factor Loading', rotation=-90, fontsize=basefont, labelpad=basefont)
cbar_ax2.tick_params('y', which='major', pad=basefont*.5)
cbar_ax2.set_ylabel('Distance', rotation=-90, fontsize=basefont*.875, labelpad=basefont)
back.text(.375, .535, 'Participants (n=522)', fontsize=basefont, horizontalalignment='center')

# loading ticks
loading_ax2.tick_params('x', length=basewidth*2, which='major', pad=basefont*.5)
loading_ax2.xaxis.set_ticks_position('top')
loading_ax2.set_xticks(np.arange(.5,5.5,1))
loading_ax2.set_xticklabels(['Factor %s' % i for i in range(1,nfactors+1)],
                            rotation=45, ha='left', fontsize=basefont*.9)
# participant box
back.add_patch(Rectangle((.3385,.56), width=.0115, height=.39, 
                         facecolor="none", edgecolor='grey', linewidth=basewidth*.75))
back.text(.3385, .96, 'One Participant', fontsize=basefont*.75, 
          horizontalalignment='center', color='grey')
# legend for mds
back.text(.15, .18, 'DVs (130)', fontsize=basefont, 
          horizontalalignment='center')
back.text(.15, .15, 'Threshold', fontsize=basefont, 
          horizontalalignment='center', color=color_lookup['threshold'])
back.text(.15, .125, 'Non-Decision', fontsize=basefont, 
          horizontalalignment='center', color=color_lookup['non-decision'])
back.text(.15, .1, 'Drift Rate', fontsize=basefont, 
          horizontalalignment='center', color=color_lookup['drift rate'])
back.text(.15, .075, 'SSRT', fontsize=basefont, 
          horizontalalignment='center', color=color_lookup['SSRT'])
back.text(.15, .05, 'Other', fontsize=basefont, 
          horizontalalignment='center', color='grey')

# add connecting lines between participants and loading
# top task
back.hlines(.773, .53, .6, alpha=.4,linestyle=':', linewidth=basewidth)
back.hlines(.82, .53, .6, alpha=.4,linestyle=':', linewidth=basewidth)
back.hlines(.869, .53, .6, alpha=.4,linestyle=':', linewidth=basewidth)
back.hlines(.918, .53, .6, alpha=.4,linestyle=':', linewidth=basewidth)
# bottom task
back.hlines(.677, .53, .6, alpha=.4,linestyle=':', linewidth=basewidth)
back.hlines(.626, .53, .6, alpha=.4,linestyle=':', linewidth=basewidth)
back.hlines(.575, .53, .6, alpha=.4,linestyle=':', linewidth=basewidth)

back.vlines(.565, .3, .42, alpha=.4, linestyle='-', linewidth=basewidth)
back.vlines(.565, .05, .2, alpha=.4, linestyle='-', linewidth=basewidth)

# arrows
# from tasks to DVs
arrowcolor = [.5,.5,.5]
back.arrow(.03,.62,.1,.06, width=basewidth/1000, color=arrowcolor)
back.arrow(.03,.62,.07,.011, width=basewidth/1000, color=arrowcolor)
back.arrow(.03,.62,.1,-.04, width=basewidth/1000, color=arrowcolor)

back.arrow(.05,.87,.08,.045, width=basewidth/1000, color=arrowcolor)
back.arrow(.05,.87,.07,-.005, width=basewidth/1000, color=arrowcolor)
back.arrow(.05,.87,.08,-.045, width=basewidth/1000, color=arrowcolor)
back.arrow(.05,.87,.1,-.075, width=basewidth/1000, color=arrowcolor)

# from participant to EFA
back.arrow(.53, .725, .05, 0, width=basewidth/200, facecolor='k')
back.text(.55, .735, 'EFA', fontsize=basefont, 
          horizontalalignment='center')
# from data to heatmap
back.arrow(.375, .515, 0, -.01, width=basewidth/250, facecolor='k')
back.arrow(.75, .515, 0, -.01, width=basewidth/250, facecolor='k')
back.text(.567, .48, 'Pairwise Distance', fontsize=basefont, 
          horizontalalignment='center')
back.text(.567, .455, '1-abs(correlation)', fontsize=basefont*.75, 
          horizontalalignment='center')
# from heatmap to MDS
back.arrow(.375, .31, 0, -.01, width=basewidth/250, facecolor='k')
back.arrow(.75, .31, 0, -.01, width=basewidth/250, facecolor='k')
back.text(.567, .24, 'MDS Projection', fontsize=basefont, 
          horizontalalignment='center')

# figure labels
back.text(-.13, 1, 'a', fontsize=basefont*1.56255, fontweight='bold')
back.text(.12, 1, 'b', fontsize=basefont*1.56255, fontweight='bold')
back.text(.62, 1, 'c', fontsize=basefont*1.56255, fontweight='bold')
back.text(.25, .49, 'd', fontsize=basefont*1.56255, fontweight='bold')
back.text(.85, .49, 'e', fontsize=basefont*1.56255, fontweight='bold')
back.text(.25, .24, 'f', fontsize=basefont*1.56255, fontweight='bold')
back.text(.85, .24, 'g', fontsize=basefont*1.56255, fontweight='bold')



# save
plot_file = path.dirname(results['task'].plot_dir)
f.savefig(path.join(plot_file, 'analysis_overview.pdf'), 
                bbox_inches='tight', 
                dpi=300)

"""
# example DDM plot
np.random.seed(1000)
ddm_plot, trajectories = DDM_plot(2, .2, 3, n=100, plot_n=7, 
                                  file=path.join(plot_file, 'DDM.pdf'))
"""