# imports
from os import makedirs, path
from selfregulation.utils.utils import get_behav_data
from selfregulation.utils.r_to_py_utils import get_Rpsych
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

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

# ****************************************************************************
# Peform factor analysis
# ****************************************************************************
from utils import find_optimal_components, get_loadings, psychFA, reorder_data
from utils import create_factor_tree, plot_factor_tree

# test if sample is suitable for factor analysis
def adequacy_test(data):
    # KMO test should be > .6
    KMO_MSA = psych.KMO(data.corr())[0][0]
    # barlett test should be significant
    Barlett_p = psych.cortest_bartlett(data.corr(), data.shape[0])[1][0]
    adequate = KMO_MSA>.6 and Barlett_p < .05
    return adequate

adequate = adequacy_test(results['data'])
print('Is the data adequate for factor analysis? %s' % ['No', 'Yes'][adequate])

# calculate optimal number of components
best_c, BICs = find_optimal_components(results['data'])
results['best_c'] = best_c
results['BICs'] = BICs

# perform factor analysis
groups = results['putative_groups']
ordered_data = reorder_data(results['data'], groups)
fa, output = psychFA(ordered_data, results['best_c'])
results['optimal_fa'] = (fa,output)

optimal_loading = get_loadings(output, labels=raw_data.columns)
putative_factor_tree = create_factor_tree(results['data'], groups, 
                                          (1,results['best_c']+5))
results['putative_factor_tree'] = putative_factor_tree

# threshold
def threshold_fa(loadings, thresh):
    loadings = loadings.copy()
    loadings[abs(loadings)<thresh]=0
    return loadings

cut_thresh = .2 # threshold for factor loadings
thresh_optimal_loading = threshold_fa(optimal_loading, cut_thresh)




pca = PCA(2)
reduced_loadings = pca.fit_transform(optimal_loadings)
plt.plot(reduced_loadings[:,0], reduced_loadings[:,1], 'o')
plot_factor_tree(putative_factor_tree, groups)