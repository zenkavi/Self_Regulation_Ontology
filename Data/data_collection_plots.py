import json
import matplotlib.pyplot as plt
import numpy as np
from os import path
import pandas as pd
import seaborn as sns
from selfregulation.utils.utils import get_behav_data, get_info, get_var_category
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
worker_completion_loc = ''
worker_completions = json.load(open(worker_completion_loc, 'r'))
"""
save_dir = path.join(base_dir, 'Data', 'Plots', 'worker_completions.%s' % ext)
completion_rate = np.mean(np.array(list(worker_completions.values())) ==63)
completion_rate = "{0:0.1f}%".format(completion_rate*100)
plt.figure(figsize=(12,8))
plt.hist(worker_completions.values(), bins=40, width=5)
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.text(5, 400, 'Completion Rate: %s' % completion_rate, size=20)
plt.xlabel('Number of Tasks Completed')
plt.savefig(save_dir, dpi=300, bbox_inches='tight')


# plot psychometric reliability
sns.set_context('poster')
meaningful_vars = get_behav_data().columns
retest_data = get_behav_data(dataset='Retest_02-03-2018', file='bootstrap_merged.csv.gz')
retest_data = retest_data.groupby('dv').mean()
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

# boxplot
colors = sns.color_palette(n_colors=2, desat=.75)
save_dir = path.join(base_dir, 'Data', 'Plots', 'ICC_boxplot.%s' % ext)
plt.figure(figsize=(12,8))
ax = sns.boxplot(y='icc', x='Measure Category', 
                 data=retest_data,
                 palette = colors, saturation=1)
ax.text(.7, .3, '%s Task Measures' % Task_N, color=colors[0])
ax.text(.7, .2, '%s Survey Measures' % Survey_N, color=colors[1])

plt.savefig(save_dir, dpi=300, bbox_inches='tight')
