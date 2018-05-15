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

#replicate "The neural circuitry supporting goal maintenance during cognitive control: a comparison of expectancy AX-CPT and dot probe expectancy paradigms"
dpx = get_behav_data(file='Individual_Measures/dot_pattern_expectancy.csv.gz')
N = len(dpx.worker_id.unique())
acc = 1-dpx.query('rt!=-1').groupby(['worker_id', 'condition']).correct.mean()
acc_stats = acc.groupby('condition').agg(["mean","std"])

rt = dpx.groupby(['worker_id', 'condition']).rt.median()
rt_stats = rt.groupby('condition').agg(["mean","std"])
# plot
sns.set_context('poster')
f, axes = plt.subplots(2,1, figsize=(8, 12))
axes[0].errorbar(range(rt_stats.shape[0]), 
                 acc_stats.loc[:,'mean'], 
                 yerr=acc_stats.loc[:,'std']*2/(N**.5),
                 color='#D3244F', linewidth=5)
axes[0].set_ylim([0, .22])
axes[0].set_ylabel(r'$Mean \pm SEM error rate$', fontsize=20)
axes[0].set_yticks(np.arange(0,.24,.02))
axes[0].set_xticks(range(rt_stats.shape[0]))
axes[0].set_xticklabels([])
axes[0].grid(axis='x')
# plot reaction time
axes[1].errorbar(range(rt_stats.shape[0]), 
                 rt_stats.loc[:,'mean'], 
                 yerr=rt_stats.loc[:,'std']*2/(N**.5),
                 color='#D3244F', linewidth=5)
axes[1].set_ylim([400, 700])
axes[1].set_ylabel(r'$Median \pm SEM reaction time$', fontsize=20)
axes[1].set_xticks(range(rt_stats.shape[0]))
axes[1].set_xticklabels(['AX', 'AY', 'BX', 'BY'])
axes[1].set_xlabel('Trial Type', fontsize=20)
axes[1].grid(axis='x')