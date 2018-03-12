import math
import matplotlib.pyplot as plt
import numpy as np
from os import path
import pandas as pd
import seaborn as sns
from selfregulation.utils.plot_utils import beautify_legend, format_num, format_variable_names
from selfregulation.utils.utils import get_behav_data, get_demographics, get_info

base_dir = get_info('base_directory')
ext= 'png'
data = get_behav_data()

# two_stage
two_stage_df = get_behav_data(file='Individual_Measures/two_stage_decision.csv.gz')
# subset two subjects who passed quality control
successful_two_stage = data.filter(regex='two_stage').dropna(how='any').index
two_stage_df = two_stage_df.query('worker_id in %s' % list(successful_two_stage))
two_stage_df = two_stage_df.query('rt_first != -1 and feedback_last in [0,1]')
colors = sns.hls_palette(2)
plot_df = (1-two_stage_df.groupby(['worker_id','stage_transition_last','feedback_last']).switch.mean()).reset_index()
plot_df.feedback_last = plot_df.feedback_last.replace({0:'Unrewarded', 1:'Rewarded'})
plot_df.stage_transition_last = \
    plot_df.stage_transition_last.replace({'infrequent':'Rare', 'frequent':'Common'})
f = plt.figure(figsize=(12,8))
sns.barplot(x='feedback_last', y='switch', hue='stage_transition_last', 
            data=plot_df, 
            order=['Rewarded', 'Unrewarded'],
            hue_order=['Common', 'Rare'],
            palette=colors)
plt.xlabel('')
plt.ylabel('Stay Probability', fontsize=24)
plt.title('Two Step Task', y=1.04, fontsize=30)
plt.tick_params(labelsize=20)
plt.ylim([.5,1])
ax = plt.gca()
leg = ax.get_legend()
leg.set_title('')
beautify_legend(leg, colors=colors, fontsize=20)
save_dir = path.join(base_dir, 'Results', 'replication', 'Plots', 'two_stage_replication.%s' % ext)
f.savefig(save_dir, dpi=300, bbox_inches='tight')
plt.close()


# shift
shift_df = get_behav_data(file='Individual_Measures/shift_task.csv.gz')
# subset two subjects who passed quality control
successful_shift = data.filter(regex='shift').dropna(how='any').index
shift_df = shift_df.query('worker_id in %s' % list(successful_shift))
shift_df = shift_df.query('rt != -1')
shift_df = shift_df.groupby(['worker_id','trials_since_switch']).correct.mean().reset_index()
# plot
f = plt.figure(figsize=(14,8))
sns.pointplot('trials_since_switch', 'correct', data=shift_df)
plt.xticks(range(0,25,5), range(0,25,5))
plt.xlabel('Trials After Change-Point', fontsize=24)
plt.ylabel('Percent Correct', fontsize= 24)
plt.title('Shift Task', y=1.04, fontsize=30)
plt.tick_params(labelsize=20)
save_dir = path.join(base_dir, 'Results', 'replication', 'Plots', 'shift_task_replication.%s' % ext)
f.savefig(save_dir, dpi=300, bbox_inches='tight')
plt.close()
# plot