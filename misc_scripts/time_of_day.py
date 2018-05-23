from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import scale
from selfregulation.utils.utils import get_behav_data

def convert_to_time(date_str):
    dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    time = dt.time()
    return time.hour
    
behav_data = get_behav_data(file='meaningful_variables_imputed.csv')

task_name ='stroop'
# look at task
task = get_behav_data(file='Individual_Measures/%s.csv.gz' % task_name)
task_DVs = behav_data.filter(regex=task_name)
task_DVs.columns = [i.split('.')[1] for i in task_DVs.columns]
# scale
task_DVs = pd.DataFrame(scale(task_DVs), index=task_DVs.index, columns=task_DVs.columns)

finishtimes = task.groupby('worker_id').finishtime.apply(lambda x: np.unique(x)[0])
daytime = finishtimes.apply(convert_to_time)
daytime.name='hour'
task_DVs = pd.concat([task_DVs, daytime], axis=1)
# add on time split in half and melt
split_time = task_DVs.hour>task_DVs.hour.median()
task_DVs = task_DVs.assign(split_time = split_time)
melted = task_DVs.melt(value_vars=task_DVs.columns[:-2],
                         id_vars='split_time')

f, (ax1,ax2) = plt.subplots(1, 2, figsize=(16,8))
for name in task_DVs.columns[:-2]:
    sns.regplot('hour', name, data=task_DVs, lowess=True, label=name,
                ax=ax1)
ax1.legend()
sns.boxplot('split_time', 'value', hue='variable', data=melted, ax=ax2)

import statsmodels.formula.api as smf
for name in task_DVs.columns[:-2]:
    rs = smf.ols(formula = '%s ~ hour' % name, data=task_DVs).fit()
    print(name, '\nBeta: %s' % rs.params.hour, '\nPvalue: %s\n\n' % rs.pvalues.hour)
