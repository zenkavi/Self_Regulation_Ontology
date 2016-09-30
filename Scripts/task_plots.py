from os import path
import numpy as np
import pandas as pd
from util import get_info
import seaborn as sns

#load Data
data_dir=path.join(get_info('base_directory'),'Data/Discovery_9-26-16')

# get DV df
DV_df = pd.read_csv(path.join(data_dir,'meaningful_variables.csv'))

tasks = np.unique(DV_df.columns.map(lambda x: x.split('.')[0]))
for task in 
p = sns.pairplot(subset, diag_kws = {'bins': 50})
p.savefig('Plots/pair_plot_all_variables.pdf', dpi = 300)




plt.figure(figsize=(12,8))
data.groupby('worker_id').passed_QC.sum().hist(bins = 50)

exp = 'simon'
a,b = get_DV(data, exp, use_group_fun = False)
x = [i['acc']['value'] for i in a.values()]
y = [i['simon_rt']['value'] for i in a.values()]
plot_df = pd.DataFrame({'acc': x, 'measure': y})
plt.figure(figsize = (12,10))
sns.regplot('acc','measure', data = plot_df)

exp = 'stroop'
a,b = get_DV(data, exp, use_group_fun = False)
x = [i['acc']['value'] for i in a.values()]
y = [i['stroop_rt']['value'] for i in a.values()]
plot_df = pd.DataFrame({'acc': x, 'measure': y})
plt.figure(figsize = (12,10))
sns.regplot('acc','measure', data = plot_df)

exp = 'threebytwo'
a,b = get_DV(data, exp, use_group_fun = False)
x = [i['EZ_drift']['value'] for i in a.values()]
y = [i['cue_switch_cost']['value'] for i in a.values()]
plot_df = pd.DataFrame({'acc': x, 'measure': y})
plt.figure(figsize = (12,10))
sns.regplot('acc','measure', data = plot_df)