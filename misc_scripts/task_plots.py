from os import path
import numpy as np
import pandas as pd
from util import get_info
import seaborn as sns

#load Data
data_dir=path.join(get_info('base_directory'),'Data/Discovery_9-26-16')

# get DV df
DV_df = pd.read_csv(path.join(data_dir,'meaningful_variables_noEZ_contrasts.csv'), index_col = 0)

tasks = np.unique(DV_df.columns.map(lambda x: x.split('.')[0]))
for task in tasks:
    subset = DV_df.filter(regex = '^%s' % task)
    subset = subset.dropna(how = 'all').dropna(axis = 1)
    sns.set(font_scale = 1.5)
    p = sns.pairplot(subset, kind = 'reg', size = 5, diag_kws = {'bins': 50})
    p.savefig('Plots/%s_pair_plot.pdf' % task, dpi = 300)


# look across stop signals
stop1 = DV_df.filter(regex = '^%s' % 'stop_signal')
stop2 = DV_df.filter(regex = '^%s' % 'motor_selective')
stop3 = DV_df.filter(regex = '^%s' % 'stim_selective')
stop1 = stop1.dropna(how = 'all').dropna(axis = 1)
stop2 = stop2.dropna(how = 'all').dropna(axis = 1)
stop3 = stop3.dropna(how = 'all').dropna(axis = 1)
stop = stop1.join(stop2).join(stop3).dropna()
sns.set(font_scale = 1.5)
p = sns.pairplot(stop, kind = 'reg', size = 5, diag_kws = {'bins': 50})
p.savefig('Plots/%s_pair_plot.pdf' % 'all_stops', dpi = 300)