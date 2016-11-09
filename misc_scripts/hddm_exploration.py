from expanalysis.experiments.processing import  extract_experiment, extract_DVs, get_DV
from os import path
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ttest_1samp
import sys
sys.path.append('../utils')
from utils import get_behav_data
from data_preparation_utils import drop_vars


DV_df = get_behav_data('Discovery_11-09-2016', file = 'variables_exhaustive.csv')
subset = drop_vars(DV_df, saved_vars = ['\.std_rt$','\.avg_rt$','\.acc$'])

ddm_tasks = np.unique([c.split('.')[0] for c in DV_df.filter(regex = 'EZ').columns])
ddm_regex = '(' + '|'.join(ddm_tasks) + ')'

drift = subset.filter(regex = ddm_regex + '.*(drift|rt$|\.acc)')
thresh = subset.filter(regex = 'thresh')
non_decision = subset.filter(regex = 'non_decision')


df = drift
tasks = np.unique([c.split('.')[0] for c in df.columns])
task_correlations = []
for task in tasks:
    task_subset = df.filter(regex = '^%s' % task)
    # sort by variable
    column_order = np.argsort([c.replace('EZ','hddm') for c in task_subset.columns])
    task_subset = task_subset.iloc[:,column_order]
    # move ddm parameters to beginning
    columns = task_subset.columns
    column_order = np.append(np.where(['drift' in c for c in columns])[0],np.where(['drift' not in c for c in columns])[0])
    task_subset = task_subset.iloc[:,column_order]
    corr_mat = task_subset.corr()
    correlations = []
    for c in task_subset.filter(regex = 'hddm').columns:
        c_ez = c.replace('hddm', 'EZ')
        try:
            correlations.append(corr_mat.loc[c,c_ez])
        except KeyError:
            print('Problem with %s' % c)
    task_correlations.append(correlations)
    
mean_reliability = np.mean([np.mean(i) for i in task_correlations])
print('Correlation across measures: %s' % mean_reliability)

# non decision correlations
sns.heatmap(DV_df.filter(regex = '\.(EZ|hddm)_non_decision$').corr())



sns.heatmap(DV_df.filter(regex = '\.hddm_non_decision$|\.avg_rt$').corr())

#variability
ddm_std = DV_df.filter(regex = '\.(hddm|EZ)').std()
ddm_std=ddm_std.groupby([lambda x: 'EZ' in x,lambda x: ['non_decision','drift','thresh'][('drift' in x) + ('thresh' in x)*2]]).mean().reset_index()
ddm_std.columns = ['Routine','Param','STD']
ddm_std.replace({False: 'hddm', True: 'EZ'}, inplace = True)

# ddm parameters contrasts
EZnon_decision_contrasts = {x[0]: ttest_1samp(x[1].dropna(),0).statistic for x in DV_df.filter(regex = '\.[a-z]+.*EZ_non_decision$').iteritems()}
EZdrift_contrasts = {x[0]: ttest_1samp(x[1].dropna(),0).statistic for x in DV_df.filter(regex = '\.[a-z]+.*EZ_drift$').iteritems()}
EZthresh_contrasts = {x[0]: ttest_1samp(x[1].dropna(),0).statistic for x in DV_df.filter(regex = '\.[a-z]+.*EZ_thresh$').iteritems()}
print('Mean EZ Non Decision Contrast: %s' % np.mean(list(EZnon_decision_contrasts.values())))
print('Mean EZ Drift Contrast: %s' % np.mean(list(EZdrift_contrasts.values())))
print('Mean EZ Thresh Contrast: %s' % np.mean(list(EZthresh_contrasts.values())))


Hdrift_contrasts = {x[0]: ttest_1samp(x[1].dropna(),0).statistic for x in DV_df.filter(regex = '\.[a-z]+.*hddm_drift$').iteritems()}
Hthresh_contrasts = {x[0]: ttest_1samp(x[1].dropna(),0).statistic for x in DV_df.filter(regex = '\.[a-z]+.*hddm_thresh$').iteritems()}
print('Mean hddm Drift Contrast: %s' % np.mean(list(Hdrift_contrasts.values())))
print('Mean hddm Thresh Contrast: %s' % np.mean(list(Hthresh_contrasts.values())))




