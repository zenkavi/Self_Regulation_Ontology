# imports
import numpy as np
from os import makedirs, path
import pandas as pd
import pickle
from selfregulation.utils.utils import get_behav_data
from selfregulation.utils.r_to_py_utils import get_Rpsych
from sklearn.preprocessing import scale
from dimensional_structure.utils import (
        create_factor_tree, find_optimal_components, get_factor_groups,
        get_hierarchical_groups, get_scores_from_subset,
        get_loadings, plot_factor_tree, get_top_factors, psychFA,
        quantify_lower_nesting, visualize_factors
        )
# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# load the psych R package
psych = get_Rpsych()

# ****************************************************************************
# Laad Data
# ****************************************************************************
datafile = 'Complete_10-08-2017'
plot_file = path.join('Plots', datafile)
output_file = path.join('Output', datafile)
makedirs(plot_file, exist_ok = True)
makedirs(output_file, exist_ok = True)
try:
    results = pickle.load(open(path.join(output_file, 'EFA_results.pkl'),'rb'))
except FileNotFoundError:
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
results['EFA_adequacy'] = {'adequate': adequate, 'adequacy_stats': adequacy_stats}

# ************************* calculate optimal FA ******************************
if 'parallel_c' not in results.keys():
    # using BIC
    BIC_c, BICs = find_optimal_components(results['data'], metric='BIC')
    results['BIC_c'] = BIC_c
    results['BICs'] = BICs
    # using SABIC
    SABIC_c, SABICs = find_optimal_components(results['data'], metric='SABIC')
    results['SABIC_c'] = SABIC_c
    results['SABICs'] = SABICs
     # using CV
    CV_c, CVs = find_optimal_components(results['data'], maxc=50, metric='CV')
    results['CV_c'] = CV_c
    results['CVs'] = CVs
    # parallel analysis
    parallel_out = psych.fa_parallel(results['data'], fa='fa', fm='ml',
                                     **{'n.iter': 100})
    results['parallel_c'] = parallel_out[parallel_out.names.index('nfact')][0]


# *********************** create groups ************************************
if 'factor_groups' not in results.keys():
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
    
    # create hierarchical groups on data
    # perform factor analysis for hierarchical grouping
    for metric in ['SABIC', 'BIC']:
        grouping_metric = '%s_c' % metric
        fa, output = psychFA(results['data'], results[grouping_metric])
        grouping_loading = get_loadings(output, labels=results['data'].columns)
        
        cluster_reorder_index, groups = get_hierarchical_groups(grouping_loading,
                                                                n_groups=8)
        # label groups
        if metric == 'SABIC':
            groups[0][0] = 'Self Awareness'
            groups[1][0] = 'Risk Attitude (ART/CCT)'
            groups[2][0] = 'Temporal Discounting'
            groups[3][0] = 'Intelligence'
            groups[4][0] = 'Risk Attitude (Dospert)'
            groups[5][0] = 'Information Processing'
            groups[6][0] = 'Context Setting'
            groups[7][0] = 'Stimulus Processing'
        elif metric == 'BIC':
            groups[0][0] = 'Self Awareness'
            groups[1][0] = 'No Idea...'
            groups[2][0] = 'Self Awareness'
            groups[3][0] = 'Temporal Discounting'
            groups[4][0] = 'Intelligence/Information Use'
            groups[5][0] = 'Stimulus Processing'
            groups[6][0] = 'Context Setting (Thresh)'
            groups[7][0] = 'Information Processing'
            
        results['%s_hierarchical_groups' % grouping_metric] = groups
        # create factor groups
        factor_groups = get_factor_groups(grouping_loading)
        for i in factor_groups:
            i[0] = 'Factor %s' % i[0]
        results['%s_factor_groups' % metric] = factor_groups

# ************************* create factor trees ******************************
run_FA = results.get('factor_tree', [])
max_factors = max([results['SABIC_c'], results['BIC_c']])+5
if len(run_FA) < max_factors:
    # Use Putative groups
    factor_tree, factor_tree_rout = create_factor_tree(results['data'],
                                     (1,max_factors))
    results['factor_tree'] = factor_tree
    results['factor_tree_Rout'] = factor_tree_rout

# quantify nesting of factor tree:
results['lower_nesting'] = quantify_lower_nesting(results['factor_tree'])



# ***************************** saving ****************************************
pickle.dump(results, open(path.join(output_file, 'EFA_results.pkl'),'wb'))


    
# ********************** Verify factor solution and analysis******************
# recreate psych calculation to verify scores
fa, output = psychFA(results['data'], 10)
loadings = get_loadings(output, labels=results['data'].columns)

scores = output['scores'] # factor scores per subjects derived from psychFA
scaled_data = scale(results['data'])
redone_scores = scaled_data.dot(output['weights'])
redone_score_diff = np.mean(scores-redone_scores)
assert(redone_score_diff < 1e-5)

# analyze nesting
factor_tree = results['factor_tree']
explained_threshold = .5
explained_scores = -np.ones((len(factor_tree), len(factor_tree)-1))
sum_explained = np.zeros((len(factor_tree), len(factor_tree)-1))
for key in results['lower_nesting'].keys():
    r = results['lower_nesting'][key]
    adequately_explained = r['scores'] > explained_threshold
    explained_score = np.mean(r['scores'][adequately_explained])
    if np.isnan(explained_score): explained_score = 0
    
    explained_scores[key[1]-1, key[0]-1] = explained_score
    sum_explained[key[1]-1, key[0]-1] = (np.sum(adequately_explained/key[0]))
    
    
# ********************* Pull out representation of specific tasks *******

tasks_of_interest = ['stroop', 'adaptive_n_back', 'threebytwo', 
                     'stop_signal', 'motor_selective_stop_signal',]
subset_scores, r2_scores = get_scores_from_subset(results['data'],
                                                  output,
                                                  tasks_of_interest)

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

# plot optimal factor breakdown in polar format
for group in ['hierarchical', 'factor']:
    # plot mini factor tree
    plot_factor_tree({i: results['factor_tree'][i] for i in [1,2]},
                     groups=results['%s_groups' % group],
                     filename = path.join(plot_file, '%s_mini_factor_tree.png' % group))

    for metric in ['BIC', 'SABIC']:
        c = results['%s_c' % metric]
        loadings = results['factor_tree'][c]
        
        # plot polar plot factor visualization for metric loadings
        fig = visualize_factors(loadings, n_rows=4, 
                                groups=results['%s_groups' % group])
        fig.savefig(path.join(plot_file, 
                              '%s_%s_factor_polar.png' % (group, metric)))
        
        # plot factor tree around optimal metric
        plot_factor_tree({i: results['factor_tree'][i] for i in [c-1,c,c+1]},
                          groups=results['%s_groups' % group],
                          filename = path.join(plot_file, 
                                               '%s_%s_factor_tree.png' % (group, metric)))
        
        # plot bar of each factor
        sorted_vars = get_top_factors(loadings) # sort by loading
        
        grouping = results['%s_factor_groups' % metric]
        flattened_factor_order = []
        for sublist in [i[1] for i in grouping]:
            flattened_factor_order += sublist
            
        n_factors = len(sorted_vars)
        f = plt.figure(figsize=(30, n_factors*3))
        axes = []
        for i in range(n_factors):
            axes.append(plt.subplot2grid((n_factors, 4), (i,0), colspan=3))
            axes.append(plt.subplot2grid((n_factors, 4), (i,3), colspan=1))
        with sns.plotting_context(font_scale=1.3) and sns.axes_style('white'):
            # plot optimal factor breakdown in bar format to better see labels
            for i, (k,v) in list(enumerate(sorted_vars.items())):
                ax1 = axes[2*i]
                ax2 = axes[2*i+1]
                # plot distribution of factors
                colors = [['r','b'][int(i)] for i in (np.sign(v)+1)/2]
                abs(v).plot(kind='bar', ax=ax2, color=colors)
                # plot actual values
                ordered_v = v[flattened_factor_order]
                ordered_colors = [['r','b'][int(i)] for i in (np.sign(ordered_v)+1)/2]
                abs(ordered_v).plot(kind='bar', ax=ax1, color=ordered_colors)
                # draw lines separating groups
                for x_val in np.cumsum([len(i[1]) for i in grouping]):
                    ax1.vlines(x_val, 0, 1.1, lw=2, color='grey')
                # set axes properties
                ax1.set_ylim(0,1.1); ax2.set_ylim(0,1.1)
                ax1.set_yticklabels(''); ax2.set_yticklabels('')
                ax2.set_xticklabels('')
                labels = ax1.get_xticklabels()
                locs = ax1.xaxis.get_ticklocs()
                ax1.set_ylabel('Factor %s' % (i+1))
                if i == 0:
                    ax_copy = ax1.twiny()
                    ax_copy.set_xticks(locs[::2])
                    ax_copy.set_xticklabels(labels[::2], rotation=90)
                    ax2.set_title('Factor Loading Distribution')
                if i == len(sorted_vars)-1:
                    # and other half on bottom
                    ax1.set_xticks(locs[1::2])
                    ax1.set_xticklabels(labels[1::2], rotation=90)
                else:
                    ax1.set_xticklabels('')
        f.savefig(path.join(plot_file, 
                            '%s_%s_factor_bars.png' % (group, metric)))
        
# plot lower nesting
fig, ax = plt.subplots(1, 1, figsize=(30,30))
cbar_ax = fig.add_axes([.905, .3, .05, .3])
sns.heatmap(sum_explained, annot=explained_scores,
            fmt='.2f', mask=(explained_scores==-1), square=True,
            ax = ax, vmin=.2, cbar_ax=cbar_ax,
            xticklabels = range(1,sum_explained.shape[1]+1),
            yticklabels = range(1,sum_explained.shape[0]+1))
plt.xlabel('Higher Factors (Explainer)', fontsize=25)
plt.ylabel('Lower Factors (Explainee)', fontsize=25)
plt.title('Nesting of Lower Level Factors based on R2', fontsize=30)
fig.savefig(path.join(plot_file, 'lower_nesting_heatmap.png'))

