import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_1samp
from selfregulation.utils.utils import get_behav_data


# get dependent variables
DV_df = get_behav_data('Discovery_10-14-2016', use_EZ = True)

EZ_df = DV_df.filter(regex = 'EZ')
EZ_df.apply([np.mean, np.std])


ttest = EZ_df.apply(lambda x: ttest_1samp(x,0, nan_policy = 'omit'))
basic_df = pd.DataFrame(list(zip(*ttest)), index = ['T-score','pval'], columns = ttest.index).T
basic_df.loc[:,'mean'] = EZ_df.mean()
basic_df.loc[:,'std'] = EZ_df.std()
basic_df.to_csv('~/basic_stats.csv')