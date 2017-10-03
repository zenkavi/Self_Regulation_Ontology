# imports
from collections import OrderedDict as odict
import numpy as np
from os import makedirs, path
import pandas as pd
import pickle
from selfregulation.utils.utils import get_behav_data
from selfregulation.utils.r_to_py_utils import get_Rpsych
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from utils import find_optimal_components, get_loadings, psychFA, reorder_data
from utils import create_factor_tree, plot_factor_tree, quantify_nesting
from utils import get_hierarchical_groups
# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# load the psych R package
psych = get_Rpsych()

# ****************************************************************************
# Laad Data
# ****************************************************************************
datafile = 'Complete_07-08-2017'
plot_file = path.join('Plots', datafile)
output_file = path.join('Output', datafile)
makedirs(plot_file, exist_ok = True)
makedirs(output_file, exist_ok = True)

# load data
raw_data = get_behav_data(dataset=datafile, file='meaningful_variables_imputed.csv')
results = {'data': raw_data}

# ****************************************************************************
# Peform factor analysis
# ****************************************************************************

# test if sample is suitable for factor analysis
def adequacy_test(data):
    # KMO test should be > .6
    KMO_MSA = psych.KMO(data.corr())[0][0]
    # barlett test should be significant
    Barlett_p = psych.cortest_bartlett(data.corr(), data.shape[0])[1][0]
    adequate = KMO_MSA>.6 and Barlett_p < .05
    return adequate, {'Barlett_p': Barlett_p, 'KMO': KMO_MSA}

adequate, adequacy_stats = adequacy_test(results['data'])
print('Is the data adequate for factor analysis? %s' % ['No', 'Yes'][adequate])
results['adequacy'] = {'adequate': adequate, 'adequacy_stats': adequacy_stats}

# ************************* calculate optimal FA ******************************

# using BIC
BIC_c, BICs = find_optimal_components(results['data'], metric='BIC')
results['BIC_c'] = BIC_c
results['BICs'] = BICs
# using SABIC
SABIC_c, SABICs = find_optimal_components(results['data'], metric='SABIC')
results['SABIC_c'] = SABIC_c
results['SABICs'] = SABICs
# parallel analysis
parallel_out = psych.fa_parallel(results['data'], fa='fa', fm='ml')
results['parallel_c'] = parallel_out[parallel_out.names.index('nfact')][0]

# perform factor analysis with optimal number of components
best_metric = 'BIC_c'
fa, output = psychFA(results['data'], results[best_metric])
results['optimal_fa'] = (fa,output)
optimal_loading = get_loadings(output, labels=raw_data.columns)

# *********************** create groups ************************************
# create putative groups
sorted_columns = []
survey_cols = ('survey', results['data'].filter(regex='survey').columns.tolist())
drift_cols = ('drift', results['data'].filter(regex='\.hddm_drift').columns.tolist())
drift_contrast_cols = ('drift con', results['data'].filter(regex='\..*_hddm_drift').columns.tolist())
thresh_cols = ('thresh', results['data'].filter(regex='\.hddm_thresh').columns.tolist())
thresh_contrast_cols = ('thresh con', results['data'].filter(regex='\..*_hddm_thresh').columns.tolist())
non_decision_cols = ('non-decision', results['data'].filter(regex='\.hddm_non_decision').columns.tolist())
non_decision_contrast_cols = ('non-decision con', results['data'].filter(regex='\..*_hddm_non_decision').columns.tolist())
stop_cols = ('stop', results['data'].filter(regex='stop').columns.tolist())
discount_cols = ('discount', results['data'].filter(regex='discount').columns.tolist())
leftover_cols = ('misc', results['data'].columns)

tmp_groups = [survey_cols,
              drift_cols, drift_contrast_cols, 
              thresh_cols, thresh_contrast_cols,
              non_decision_cols, non_decision_contrast_cols,
              stop_cols, discount_cols,
              leftover_cols]
putative_groups = []
for name, group in tmp_groups:
    if len(group)>0:
        group = sorted(list(set(group)-set(sorted_columns)))
        sorted_columns+=group
        putative_groups.append((name,group))
results['putative_groups'] = putative_groups

# create hierarchical groups
cluster_reorder_index, groups = get_hierarchical_groups(optimal_loading)
results['hierarchical_groups'] = groups



# ************************* create factor trees ******************************

for group_name in ['putative', 'hierarchical']:
    
    # Use Putative groups
    factor_tree = create_factor_tree(results['data'], 
                                     results['%s_groups' % group_name], 
                                     (1,results['BIC_c']+10))
    results['%s_factor_tree' % group_name] = factor_tree

# quantify nesting of factor tree:
nesting_results = odict()
for c in range(3,len(factor_tree)+1):
    out = quantify_nesting(factor_tree[c], factor_tree[c-1])
    nesting_results[(c,c-1)] = out
    results['nesting_tree'] = nesting_results

# saving

pickle.dump(results, open(path.join(output_file, 'EFA_results.pkl'),'wb'))

# ****************************************************************************
# Plotting
# ****************************************************************************
sns.set_context('notebook', font_scale=1.4)

with sns.axes_style('dark'):
    x = list(results['BICs'].keys())
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x, list(results['BICs'].values()), c='c', lw=3)
    ax1.set_ylabel('BIC', fontsize=20)
    BIC_c = results['BIC_c']
    ax1.plot(BIC_c,results['BICs'][BIC_c],'k.', markersize=30)
    ax2.plot(x, list(results['SABICs'].values()), c='m', lw=3)
    ax2.set_ylabel('SABIC', fontsize=20)
    SABIC_c = results['SABIC_c']
    ax2.plot(SABIC_c,results['SABICs'][SABIC_c],'k.', markersize=30)

# plot nesting tree across components
f = plt.figure(figsize=(16,12))
plt.subplot(3,1,1)
y = [i[0]['score'] for i in results['nesting_tree'].values()]    
plt.plot(range(2, len(y)+2), y, 'mo-', lw = 3)
plt.ylabel('R^2 for Upper Recovery')

plt.subplot(3,1,2)
f = lambda x: abs(abs(x[0])-abs(x[1]))
y = [f(i[0]['coefficients']) for i in results['nesting_tree'].values()]    
plt.plot(range(2, len(y)+2), y, 'mo-', lw = 3)
plt.ylabel('Difference between contributions')

plt.subplot(3,1,3)
lower_limit = -2
df = pd.DataFrame([i[1] for i in results['nesting_tree'].values()],
                  index=results['nesting_tree'].keys())
df.insert(0, 'name', df.index)
df = pd.melt(df, 'name', value_name='Lower Recovery')
df.loc[df.loc[:,'Lower Recovery']<-2, 'Lower Recovery']=lower_limit
sns.stripplot('name','Lower Recovery',data=df, size=8, jitter=True)
plt.hlines(0, -1, len(df.name.unique())+1, linestyles='dashed')
plt.xlabel('# Higher-Order Factors', fontsize=20)
plt.xticks(range(len(df.name.unique())), range(2, len(y)+2))
plt.ylim([lower_limit,1])
plt.tight_layout()

# plot mini factor tree
plot_factor_tree({i: results['hierarchical_factor_tree'][i] for i in [1,2]},
                 groups=results['hierarchical_groups'],
                 filename = 'hierarchical_mini_tree')

# plot full factor tree
plot_factor_tree(results['hierarchical_factor_tree'],
                 groups=results['hierarchical_groups'],
                 filename = 'hierarchical_full_tree')




# threshold
def threshold_fa(loadings, thresh):
    loadings = loadings.copy()
    loadings[abs(loadings)<thresh]=0
    return loadings

cut_thresh = .2 # threshold for factor loadings
thresh_optimal_loading = threshold_fa(optimal_loading, cut_thresh)




pca = PCA(2)
reduced_loadings = pca.fit_transform(optimal_loading)
plt.plot(reduced_loadings[:,0], reduced_loadings[:,1], 'o')
plot_factor_tree(factor_tree, )