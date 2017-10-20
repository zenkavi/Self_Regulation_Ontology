# imports
import argparse
from math import ceil
from dimensional_structure.utils import (
        create_factor_tree, find_optimal_components, get_factor_groups,
        get_hierarchical_groups, get_scores_from_subset,
        get_loadings, plot_factor_tree, get_top_factors, 
        quantify_lower_nesting, save_figure,
        visualize_factors, visualize_task_factors
        )
import matplotlib.pyplot as plt
import numpy as np
from os import makedirs, path
import pandas as pd
import pickle
from scipy.stats import entropy
import seaborn as sns
from selfregulation.utils.utils import get_behav_data
from selfregulation.utils.r_to_py_utils import get_Rpsych, psychFA
from sklearn.preprocessing import scale

# load the psych R package
psych = get_Rpsych()

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-rerun', action='store_true')
parser.add_argument('-no_plot', action='store_true')
args = parser.parse_args()

rerun = args.rerun
plot_on = not args.no_plot

# ****************************************************************************
# Laad Data
# ****************************************************************************
datafile = 'Complete_10-08-2017'
plot_file = path.join('Plots', datafile, 'EFA')
output_file = path.join('Output', datafile)
makedirs(plot_file, exist_ok = True)
makedirs(output_file, exist_ok = True)
try:
    results = pickle.load(open(path.join(output_file, 'EFA_results.pkl'),'rb'))
except FileNotFoundError:
    # load data
    imputed_data = get_behav_data(dataset=datafile, file='meaningful_variables_imputed.csv')
    cleaned_data = get_behav_data(dataset=datafile, file='meaningful_variables_clean.csv')
    results = {'data': imputed_data, 'data_no_impute': cleaned_data,
               'EFA': {}}

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
results['EFA']['EFA_adequacy'] = {'adequate': adequate, 'adequacy_stats': adequacy_stats}

# ********************** Verify factor solution and analysis******************
# recreate psych calculation to verify scores
fa, output = psychFA(results['data'], 10)
loadings = get_loadings(output, labels=results['data'].columns)

scores = output['scores'] # factor scores per subjects derived from psychFA
scaled_data = scale(results['data'])
redone_scores = scaled_data.dot(output['weights'])
redone_score_diff = np.mean(scores-redone_scores)
assert(redone_score_diff < 1e-5)

# ************************* calculate optimal FA ******************************
if 'c_metric-parallel' not in results['EFA'].keys() or rerun == True:
    print('Calculating optimal number of factors')
    # using BIC
    BIC_c, BICs = find_optimal_components(results['data'], metric='BIC')
    results['EFA']['c_metric-BIC'] = BIC_c
    results['EFA']['cscores_metric-BIC'] = BICs
    # using SABIC
    SABIC_c, SABICs = find_optimal_components(results['data'], metric='SABIC')
    results['EFA']['c_metric-SABIC'] = SABIC_c
    results['EFA']['cscores_metric-SABIC'] = SABICs
     # using CV
    CV_c, CVs = find_optimal_components(results['data_no_impute'], 
                                        maxc=50, metric='CV')
    results['EFA']['c_metric-CV'] = CV_c
    results['EFA']['cscores_metric-CV'] = CVs
    # parallel analysis
    parallel_out = psych.fa_parallel(results['data'], fa='fa', fm='ml',
                                     **{'n.iter': 100})
    results['EFA']['c_metric-parallel'] = parallel_out[parallel_out.names.index('nfact')][0]


# *********************** create groups ************************************
if 'factor_groups_metric-SABIC' not in results['EFA'].keys() or rerun == True:
    print('Creating factor groups')
    # create hierarchical groups on data
    # perform factor analysis for hierarchical grouping
    for metric in ['SABIC', 'BIC']:
        grouping_metric = 'c_metric-%s' % metric
        fa, output = psychFA(results['data'], results['EFA'][grouping_metric])
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
            
        results['EFA']['hierarchical_groups_metric-%s' % metric] = groups
        # create factor groups
        factor_groups = get_factor_groups(grouping_loading)
        for i in factor_groups:
            i[0] = 'Factor %s' % i[0]
        results['EFA']['factor_groups_metric-%s' % metric] = factor_groups

# ************************* create factor trees ******************************
run_FA = results['EFA'].get('factor_tree', [])
max_factors = max([v for k,v in results['EFA'].items() if 'c_metric-' in k])
if len(run_FA) < max_factors or rerun == True:
    print('Creating factor tree')
    # Use Putative groups
    factor_tree, factor_tree_rout = create_factor_tree(results['data'],
                                     (1,max_factors))
    results['EFA']['factor_tree'] = factor_tree
    results['EFA']['factor_tree_Rout'] = factor_tree_rout

# quantify nesting of factor tree:
results['EFA']['lower_nesting'] = quantify_lower_nesting(results['EFA']['factor_tree'])

# quantify entropy of each measure
for metric in ['BIC', 'SABIC']:
    c = results['EFA']['c_metric-%s' % metric]
    loadings = results['EFA']['factor_tree'][c]
    # calculate entropy of each variable
    loading_entropy = abs(loadings).apply(entropy, 1)
    max_entropy = entropy([1/loadings.shape[1]]*loadings.shape[1])
    results['EFA']['entropy_metric-%s' % metric] = loading_entropy/max_entropy


# *************Additional analyses***********************************
# analyze nesting
factor_tree = results['EFA']['factor_tree']
explained_threshold = .5
explained_scores = -np.ones((len(factor_tree), len(factor_tree)-1))
sum_explained = np.zeros((len(factor_tree), len(factor_tree)-1))
for key in results['EFA']['lower_nesting'].keys():
    r = results['EFA']['lower_nesting'][key]
    adequately_explained = r['scores'] > explained_threshold
    explained_score = np.mean(r['scores'][adequately_explained])
    if np.isnan(explained_score): explained_score = 0
    
    explained_scores[key[1]-1, key[0]-1] = explained_score
    sum_explained[key[1]-1, key[0]-1] = (np.sum(adequately_explained/key[0]))

# analyze entropy across factor solutions
entropies = {}
for c, loadings in results['EFA']['factor_tree'].items():
    if c > 1:
        # calculate entropy of each variable
        loading_entropy = abs(loadings).apply(entropy, 1)
        max_entropy = entropy([1/loadings.shape[1]]*loadings.shape[1])
        entropies[c] = loading_entropy/max_entropy
entropies = pd.DataFrame(entropies)
results['EFA']['entropies'] = entropies

# create null entropy distributions
null_entropies = {}
num_shuffles = 50
for c, loadings in results['EFA']['factor_tree'].items():
    if c > 1:
        max_entropy = entropy([1/loadings.shape[1]]*loadings.shape[1])
        permuted_entropies = np.array([])
        abs_loadings = abs(loadings)
        for _ in range(num_shuffles):
            # shuffle matrix
            for i, col in enumerate(abs_loadings.values.T):
                shuffle_vec = np.random.permutation(col)
                abs_loadings.iloc[:, i] = shuffle_vec
            # calculate entropy of each variable
            loading_entropy = abs_loadings.apply(entropy, 1)
            permuted_entropies = np.append(permuted_entropies,
                                           (loading_entropy/max_entropy).values)
        null_entropies[c] = permuted_entropies
        
        
null_entropies = pd.DataFrame(null_entropies)


# ********************* Pull out representation of specific tasks *******
tasks_of_interest = ['stroop', 'adaptive_n_back', 'threebytwo', 
                     'stop_signal', 'motor_selective_stop_signal',]
subset_scores, r2_scores = get_scores_from_subset(results['data'],
                                                  output,
                                                  tasks_of_interest)

# ***************************** saving ****************************************
pickle.dump(results, open(path.join(output_file, 'EFA_results.pkl'),'wb'))

# ****************************************************************************
# Plotting
# ****************************************************************************
if plot_on:
    print('Plotting')
    sns.set_context('notebook', font_scale=1.4)
    
    # Plot BIC and SABIC curves
    with sns.axes_style('white'):
        x = list(results['EFA']['cscores_metric-BIC'].keys())
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        # BIC
        BIC_scores = list(results['EFA']['cscores_metric-BIC'].values())
        BIC_c = results['EFA']['c_metric-BIC']
        ax1.plot(x, BIC_scores, c='c', lw=3, label='BIC')
        ax1.set_ylabel('BIC', fontsize=20)
        ax1.plot(BIC_c, BIC_scores[BIC_c],'k.', markersize=30)
        # SABIC
        SABIC_scores = list(results['EFA']['cscores_metric-SABIC'].values())
        SABIC_c = results['EFA']['c_metric-SABIC']
        ax2.plot(x, SABIC_scores, c='m', lw=3, label='SABIC')
        ax2.set_ylabel('SABIC', fontsize=20)
        ax2.plot(SABIC_c, SABIC_scores[SABIC_c],'k.', markersize=30)
        # set up legend
        ax1.plot(np.nan, c='m', lw=3, label='SABIC')
        ax1.legend(loc='upper center')
        save_figure(fig, path.join(plot_file, 'BIC_SABIC_curves.png'),
                    {'bbox_inches': 'tight'})
        
    # plot lower nesting
    fig, ax = plt.subplots(1, 1, figsize=(30,30))
    cbar_ax = fig.add_axes([.905, .3, .05, .3])
    sns.heatmap(sum_explained, annot=explained_scores,
                fmt='.2f', mask=(explained_scores==-1), square=True,
                ax = ax, vmin=.2, cbar_ax=cbar_ax,
                xticklabels = range(1,sum_explained.shape[1]+1),
                yticklabels = range(1,sum_explained.shape[0]+1))
    ax.set_xlabel('Higher Factors (Explainer)', fontsize=25)
    ax.set_ylabel('Lower Factors (Explainee)', fontsize=25)
    ax.set_title('Nesting of Lower Level Factors based on R2', fontsize=30)
    save_figure(fig, path.join(plot_file, 'lower_nesting_heatmap.png'),
                {'bbox_inches': 'tight'})
    
    # Plot factor loadings
    for group in ['hierarchical', 'factor']:
        for metric in ['BIC', 'SABIC']:
            c = results['EFA']['c_metric-%s' % metric]
            loadings = results['EFA']['factor_tree'][c]
            
            # plot polar plot factor visualization for metric loadings
            filename =  'factor_polar_metric-%s_group-%s.png' % (metric, group)
            fig = visualize_factors(loadings, n_rows=4, 
                                    groups=results['EFA']['%s_groups_metric-%s' % (group, metric)])
            save_figure( fig, path.join(plot_file, metric, filename),  
                        {'bbox_inches': 'tight'})
            
            # plot mini factor tree
            filename = 'mini_factor_tree_metric-%s_group-%s.png' % (metric,group)
            plot_factor_tree({i: results['EFA']['factor_tree'][i] for i in [1,2]},
                              groups=results['EFA']['%s_groups_metric-%s' % (group, metric)],
                              filename = path.join(plot_file, metric, filename))
        
            # plot factor tree around optimal metric
            filename = 'factor_tree_metric-%s_group-%s.png' % (metric, group)
            plot_factor_tree({i: results['EFA']['factor_tree'][i] for i in [c-1,c,c+1]},
                              groups=results['EFA']['%s_groups_metric-%s' % (group, metric)],
                              filename = path.join(plot_file, metric, filename))
            
            # plot bar of each factor
            sorted_vars = get_top_factors(loadings) # sort by loading
            
            grouping = results['EFA']['%s_groups_metric-%s' % (group, metric)]
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
            filename = 'factor_bars_metric-%s_group-%s.png' % (metric, group)
            save_figure(f, path.join(plot_file, metric, filename),
                        {'bbox_inches': 'tight'})
    
    # plot task factor loading
    factor_tree = results['EFA']['factor_tree']
    tasks = np.unique([i.split('.')[0] for i in factor_tree[1].index])
    ncols = 6
    for metric in ['BIC', 'SABIC']:
        c = results['EFA']['c_metric-%s' % metric]
        loadings = factor_tree[c]
        max_loading = abs(loadings).max().max()
        
        task_sublists = {'surveys': [t for t in tasks if 'survey' in t],
                        'tasks': [t for t in tasks if 'survey' not in t]}
        for sublist_name, task_sublist in task_sublists.items():
            nrows = ceil(len(task_sublist)/ncols)
            adjusted_cols = min(ncols, len(task_sublist))
            # plot loading distributions. Each measure is scaled so absolute
            # comparisons are impossible. Only the distributions can be compared
            f, axes = plt.subplots(nrows, adjusted_cols, 
                                   figsize=(adjusted_cols*10,nrows*(8+nrows)),
                                   subplot_kw={'projection': 'polar'})
            axes = f.get_axes()
            for i, task in enumerate(task_sublist):
                task_loadings = loadings.filter(regex=task, axis=0)
                # add entropy to index
                task_entropies = entropies[c][task_loadings.index]
                task_loadings.index = [i+'(%.2f)' % task_entropies.loc[i] for i in task_loadings.index]
                # plot
                visualize_task_factors(task_loadings, axes[i])
                axes[i].set_title(' '.join(task.split('_')), 
                                  y=1.14, fontsize=25)
                
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
            plt.subplots_adjust(hspace=.5, wspace=.5)
            filename = 'factor_DVdistributions_metric-%s_subset-%s.png' % (metric, sublist_name)
            save_figure(f, path.join(plot_file, metric, filename), 
                        {'bbox_inches': 'tight'})
            
    
    # plot entropies
    entropies.loc[:, 'group'] = 'real'
    null_entropies.loc[:, 'group'] = 'null'
    plot_entropies = pd.concat([entropies, null_entropies], 0)
    plot_entropies = plot_entropies.melt(id_vars= 'group',
                                         var_name = 'EFA',
                                         value_name = 'entropy')
    with sns.plotting_context('notebook', font_scale=1.8):
        f = plt.figure(figsize=(20,8))
        sns.boxplot(x='EFA', y='entropy', data=plot_entropies, hue='group')
        plt.xlabel('# Factors')
        plt.ylabel('Entropy')
        plt.title('Distribution of Measure Specificity across Factor Solutions')
        f.savefig(path.join(plot_file, 'entropies_across_factors.png'), 
                  bbox_inches='tight')