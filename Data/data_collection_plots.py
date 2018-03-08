import json
import matplotlib.pyplot as plt
import numpy as np
from os import path
import pandas as pd
import seaborn as sns
from selfregulation.utils.utils import get_behav_data, get_info, get_var_category
from selfregulation.utils.plot_utils import format_num
sns.set_palette("Set1", 8, .75)

base_dir = get_info('base_directory')
ext = 'png'
"""
# Load data if plots need to be regenerated

post_process_data_loc = ''
data = pd.load_pickle(post_process_data_loc)
"""

# plt total time on tasks
(data.groupby('worker_id').ontask_time.sum()/3600).hist(bins=40, 
                                                        grid=False, 
                                                        density=True,
                                                        figsize=(12,8))
plt.xlabel('Time (Hours)')
plt.title('Total Time on Tasks', weight='bold')



# plot distribution of times per task
tasks = data.experiment_exp_id.unique()
N = len(tasks)

f, axes = plt.subplots(3,1,figsize=(16,20))
for i in range(3):
    for exp in tasks[i*N//3: (i+1)*N//3]:
        task_time = data.query('experiment_exp_id == "%s"' % exp).ontask_time/3600
        task_time.name = ' '.join(exp.split('_'))
        if not pd.isnull(task_time.sum()):
            sns.kdeplot(task_time, linewidth=3, ax=axes[i])
    axes[i].set_xlim(0,1)
    axes[i].legend(ncol=3)
plt.xlabel('Time (Hours)')


"""
# Load worker completions if plot needs to be regenerated
worker_completion_loc = '/mnt/OAK/behavioral_data/admin/worker_counts.json'
worker_completions = json.load(open(worker_completion_loc, 'r'))
"""
with sns.plotting_context('poster'):
    save_dir = path.join(base_dir, 'Data', 'Plots', 'worker_completions.%s' % ext)
    completion_rate = np.mean(np.array(list(worker_completions.values())) ==63)
    completion_rate = format_num(completion_rate*100, 1)
    analyzed_rate = 522/len(worker_completions)
    analyzed_rate = format_num(analyzed_rate*100, 1)
    plt.figure(figsize=(12,8))
    plt.hist(worker_completions.values(), bins=40, width=5)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.text(5, 400, 'Completion Rate: %s' % completion_rate, size=20)
    ax.text(5, 350, 'Passed QC: %s' % analyzed_rate, size=20)
    plt.xlabel('Number of Tasks Completed', fontsize=20)
plt.savefig(save_dir, dpi=300, bbox_inches='tight')

# ****************************************************************************
# plot psychometric reliability
# ****************************************************************************

sns.set_context('poster')
meaningful_vars = get_behav_data(file='meaningful_variables_imputed.csv').columns
meaningful_vars = [i.replace('.logTr','') for i in meaningful_vars]
meaningful_vars = [i.replace('.ReflogTr','') for i in meaningful_vars]

retest_data = get_behav_data(dataset='Retest_02-03-2018', file='bootstrap_merged.csv.gz')
retest_data = retest_data.groupby('dv').mean()
retest_data.rename({'dot_pattern_expectancy.BX.BY_hddm_drift': 'dot_pattern_expectancy.BX-BY_hddm_drift',
                    'dot_pattern_expectancy.AY.BY_hddm_drift': 'dot_pattern_expectancy.AY-BY_hddm_drift'},
                    axis='index',
                    inplace=True)
# onyl select meaningful variables
retest_data = retest_data.query('dv in %s' % list(meaningful_vars))

# create reliability dataframe
measure_cat = [get_var_category(v).title() for v in retest_data.index]
retest_data.loc[:,'Measure Category'] = measure_cat
Survey_N = np.sum(retest_data.loc[:, 'Measure Category']=='Survey')
Task_N = len(retest_data)-Survey_N

# plot
save_dir = path.join(base_dir, 'Data', 'Plots', 'ICC_stripplot.%s' % ext)
plt.figure(figsize=(12,8))
ax = sns.pointplot(y='icc', x='Measure Category', 
                   color='black',
                   data=retest_data, 
                   join=False)
plt.setp(ax.collections, sizes=[200], zorder=20)
ax = sns.stripplot(y='icc', x='Measure Category', 
                    data=retest_data, 
                    jitter=True, alpha=.5, size=10)
plt.savefig(save_dir, dpi=300, bbox_inches='tight')

# box plot
colors = sns.color_palette('Blues_d',3) 
save_dir = path.join(base_dir, 'Data', 'Plots', 'ICC_distplot.%s' % ext)
f = plt.figure(figsize=(12,8))
# plot boxes
box_ax = f.add_axes([0,0,1,.6]) 
sns.boxplot(x='icc', y='Measure Category', ax=box_ax, data=retest_data,
            palette={'Survey': colors[0], 'Task': colors[1]}, saturation=1,
            width=.5)
box_ax.text(0, 1.2, '%s Task Measures' % Task_N, color=colors[1], fontsize=24)
box_ax.text(0, 1, '%s Survey Measures' % Survey_N, color=colors[0], fontsize=24)
box_ax.set_ylabel('Measure Category', fontsize=24, labelpad=10)
box_ax.set_xlabel('ICC', fontsize=24, labelpad=10)
box_ax.tick_params(labelsize=20)
# plot distributions
dist_ax = f.add_axes([0,.6,1,.4]) 
dist_ax.set_xlim(*box_ax.get_xlim())
dist_ax.set_xticklabels('')
dist_ax.tick_params(length=0)
for i, (name, g) in enumerate(retest_data.groupby('Measure Category')):
    sns.kdeplot(g['icc'], color=colors[i], ax=dist_ax, linewidth=4, 
                shade=True, legend=False)
dist_ax.axis('off')
plt.savefig(save_dir, dpi=300, bbox_inches='tight')


# violin plot
colors = sns.color_palette('Blues_d',3) 
save_dir = path.join(base_dir, 'Data', 'Plots', 'ICC_violinplot.%s' % ext)
plt.figure(figsize=(12,8))
ax = sns.violinplot(y='icc', x='Measure Category', 
                 data=retest_data,
                 palette = colors, saturation=1)
ax.text(.7, .3, '%s Task Measures' % Task_N, color=colors[0], fontsize=24)
ax.text(.7, .2, '%s Survey Measures' % Survey_N, color=colors[1], fontsize=24)
plt.ylabel('ICC', fontsize=24, labelpad=10)
plt.xlabel('Measure Category', fontsize=24, labelpad=10)
plt.tick_params(labelsize=20)

plt.savefig(save_dir, dpi=300, bbox_inches='tight')
