# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 08:58:43 2016

@author: ian
"""
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