# Running script to create results objects for subsets of data and plot

# imports
import argparse
# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-dataset', default=None)
parser.add_argument('-no_analysis', action='store_false')
parser.add_argument('-no_prediction', action='store_false')
parser.add_argument('-no_plot', action='store_false')
parser.add_argument('-no_group_analysis', action='store_false')
parser.add_argument('-no_group_plot', action='store_false')
parser.add_argument('-bootstrap', action='store_true')
parser.add_argument('-boot_iter', type=int, default=1000)
parser.add_argument('-shuffle_repeats', type=int, default=1)
parser.add_argument('-subsets', nargs='+', default=['task', 'survey'])
parser.add_argument('-classifiers', nargs='+', default=['lasso', 'ridge',  'svm', 'rf'])
parser.add_argument('-plot_backend', default=None)
parser.add_argument('-dpi', type=int, default=300)
parser.add_argument('-size', type=int, default=4.6)
parser.add_argument('-ext', default='pdf')
parser.add_argument('-quiet', action='store_false')
args = parser.parse_args()

dataset = args.dataset
run_analysis = args.no_analysis
run_prediction = args.no_prediction
run_plot = args.no_plot
group_analysis = args.no_group_analysis
group_plot = args.no_group_plot
bootstrap = args.bootstrap
boot_iter = args.boot_iter
shuffle_repeats = args.shuffle_repeats
classifiers = args.classifiers
selected_subsets = args.subsets
verbose = args.quiet

# import matplotlib and set backend
import matplotlib
if args.plot_backend:
    matplotlib.use('Agg')
    
# imports
from glob import glob
import numpy as np
from os import makedirs, path
import random
from shutil import copyfile, copytree, rmtree
import subprocess
import time

from dimensional_structure.results import Results
from dimensional_structure.cross_results_plots import (plot_corr_heatmap, 
                                                       plot_BIC,
                                                       plot_cross_silhouette,
                                                       plot_cross_communality)
from dimensional_structure.cross_results_utils import run_group_prediction
from dimensional_structure.DA_plots import plot_DA
from dimensional_structure.EFA_plots import plot_EFA
from dimensional_structure.HCA_plots import plot_HCA
from dimensional_structure.prediction_plots import (plot_prediction, 
                                                    plot_prediction_scatter,
                                                    plot_prediction_comparison,
                                                    plot_factor_fingerprint)

from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_info, get_recent_dataset


if verbose:
    print('Running Analysis? %s, Prediction? %s, Plotting? %s, Bootstrap? %s, Selected Subsets: %s' 
        % (['No', 'Yes'][run_analysis],  
            ['No', 'Yes'][run_prediction], 
            ['No', 'Yes'][run_plot], 
            ['No', 'Yes'][bootstrap],
            ', '.join(selected_subsets)))

# get dataset of interest
basedir=get_info('base_directory')
if dataset == None:
    dataset = get_recent_dataset()
dataset = path.join(basedir,'Data',dataset)
datafile = dataset.split(path.sep)[-1]

# label subsets
demographic_factor_names = ['Drug Use', 
                            'Mental Health',
                            'Problem Drinking',
                            'Daily Smoking',
                            'Binge Drinking',
                            'Obesity',
                            'Lifetime Smoking',
                            'Unsafe Drinking',
                            'Income / Life Milestones']
subsets = [{'name': 'task', 
            'regex': 'task',
            'factor_names': ['Speeded IP', 'Strategic IP', 'Discounting',
                             'Perc / Resp', 'Caution'],
            'cluster_names': [],
            'predict': True},
            {'name': 'survey',
             'regex': 'survey',
             'factor_names':  ['Sensation Seeking', 'Mindfulness', 'Impulsivity', 
                               'Emotional Control', 'Reward Sensitivity', 'Goal-Directedness', 
                               'Risk Perception', 'Eating Control', 'Ethical Risk-Taking', 
                               'Social Risk-Taking', 'Financial Risk-Taking', 'Agreeableness'],
             'predict': True},
             {'name': 'main_subset', 
            'regex': 'main',
            'factor_names': [],
            'cluster_names': [],
            'predict': False},
             {'name': 'all', 
              'regex': '.',
              'factor_names': [],
              'predict': False}]
results = None
ID = str(random.getrandbits(16)) 
# create/run results for each subset
for subset in subsets:
    name = subset['name']
    if verbose:
        print('*'*79)
        print('SUBSET: %s' % name.upper())
        print('*'*79)
    if selected_subsets is not None and name not in selected_subsets:
        continue
    if run_analysis == True:
        print('*'*79)
        print('Analyzing Subset: %s' % name)
        # ****************************************************************************
        # Laad Data
        # ****************************************************************************
        # run dimensional analysis
        start = time.time()
        results = Results(datafile=datafile, 
                          dist_metric='abscorrelation',
                          name=subset['name'],
                          filter_regex=subset['regex'],
                          boot_iter=boot_iter,
                          ID=ID,
                          residualize_vars=['Age', 'Sex'])
        results.run_demographic_analysis(verbose=verbose, bootstrap=bootstrap)
        results.run_EFA_analysis(verbose=verbose, bootstrap=bootstrap)
        results.run_clustering_analysis(verbose=verbose, run_graphs=False)
        ID = results.ID.split('_')[1]
        # name factors and clusters
        factor_names = subset.get('factor_names', None)
        cluster_names = subset.get('cluster_names', None)
        if factor_names:
            results.EFA.name_factors(factor_names)
        if cluster_names:
            results.HCA.name_clusters(cluster_names)
        results.DA.name_factors(demographic_factor_names)
        if verbose: print('Saving Subset: %s' % name)
        id_file = results.save_results()
        # ***************************** saving ****************************************
        # copy latest results and prediction to higher directory
        copyfile(id_file, path.join(path.dirname(results.get_output_dir()), 
                                    '%s_results.pkl' % name))

    if run_prediction == True:   
        if verbose:
            print('*'*79)
            print('Running prediction: %s' % name)
        if results is None or name not in results.ID:
            results = load_results(datafile, name=name)[name]
        # run behavioral prediction using the factor results determined by BIC
        for classifier in classifiers:
            results.run_prediction(classifier=classifier, verbose=verbose)
            results.run_prediction(classifier=classifier, shuffle=shuffle_repeats, verbose=verbose) # shuffled
            # predict demographic changes
            results.run_change_prediction(classifier=classifier, verbose=verbose)
            results.run_change_prediction(classifier=classifier, shuffle=shuffle_repeats, verbose=verbose) # shuffled
        # ***************************** saving ****************************************
        prediction_dir = path.join(results.get_output_dir(), 'prediction_outputs')
        for classifier in classifiers:
            for change_flag in [False, True]:
                prediction_files = glob(path.join(prediction_dir, '*%s*' % classifier))
                # filter by change
                prediction_files = filter(lambda x: ('change' in x) == change_flag, prediction_files)
                # sort by creation time and get last two files
                prediction_files = sorted(prediction_files, key = path.getmtime)[-4:]
                for filey in prediction_files:
                    filename = '_'.join(path.basename(filey).split('_')[:-1])
                    copyfile(filey, path.join(path.dirname(results.get_output_dir()), 
                                              '%s_%s.pkl' % (name, filename)))

    # ****************************************************************************
    # Plotting
    # ****************************************************************************
    dpi = args.dpi
    ext = args.ext
    size = args.size
    if run_plot==True:
        if verbose:
            print('*'*79)
            print('Plotting Subset: %s' % name)
        if results is None or name not in results.ID:
            results = load_results(datafile, name=name)[name]
        plot_dir = results.get_plot_dir()
        DA_plot_dir = path.join(plot_dir, 'DA')
        EFA_plot_dir = path.join(plot_dir, 'EFA')
        HCA_plot_dir = path.join(plot_dir, 'HCA')
        prediction_plot_dir = path.join(plot_dir, 'prediction')
        makedirs(DA_plot_dir, exist_ok = True)
        makedirs(EFA_plot_dir, exist_ok = True)
        makedirs(HCA_plot_dir, exist_ok = True)
        
        # set up kws for plotting functions
        tasks = np.unique([i.split('.')[0] for i in results.data.columns])
        if name == 'task':
            plot_task_kws= {'task_sublists': {'tasks': [t for t in tasks if 'survey' not in t]}}
        elif name == 'survey':
            plot_task_kws= {'task_sublists': {'surveys': [t for t in tasks if 'survey' in t]}}
        else:
            plot_task_kws={}
         
            # Plot EFA
        if verbose: print("Plotting EFA")
        plot_DA(results, DA_plot_dir, verbose=verbose, size=size, dpi=dpi, ext=ext)
        
        # Plot EFA
        if verbose: print("Plotting EFA")
        plot_EFA(results, EFA_plot_dir, verbose=verbose, size=size, dpi=dpi, 
                 ext=ext, plot_task_kws=plot_task_kws)
            
        # Plot HCA
        if verbose: print("Plotting HCA")
        plot_HCA(results, HCA_plot_dir, size=size, dpi=dpi, ext=ext)
        
        # Plot prediction
        if results.load_prediction_object() is not None:
            target_order = results.DA.reorder_factors(results.DA.get_loading()).columns
            change_target_order = [i + ' Change' for i in target_order]
            for classifier in classifiers:
                for EFA in [True, False]:
                    print("Plotting Prediction, classifier: %s, EFA: %s" % (classifier, EFA))
                    plot_prediction(results, target_order=target_order, EFA=EFA, 
                                    classifier=classifier, plot_dir=prediction_plot_dir,
                                    dpi=dpi,
                                    ext=ext,
                                    size=size)
                    plot_prediction_scatter(results, target_order=target_order, EFA=EFA, 
                                    classifier=classifier, plot_dir=prediction_plot_dir,
                                    dpi=dpi,
                                    ext=ext,
                                    size=size)
                    print("Plotting Change Prediction, classifier: %s, EFA: %s" % (classifier, EFA))
                    try:
                        plot_prediction(results, target_order=change_target_order, 
                                        EFA=EFA, change=True,
                                        classifier=classifier, plot_dir=prediction_plot_dir,
                                        dpi=dpi,
                                        ext=ext,
                                        size=size)
                        plot_prediction_scatter(results, target_order=change_target_order, 
                                        EFA=EFA, change=True,
                                        classifier=classifier, plot_dir=prediction_plot_dir,
                                        dpi=dpi,
                                        ext=ext,
                                        size=size)
                    except AssertionError:
                        print('No shuffled data was found for %s change predictions, EFA: %s' % (name, EFA))
                        
            plot_prediction_comparison(results, change=False, size=size, ext=ext,
                                       dpi=dpi, plot_dir=prediction_plot_dir)
            plot_prediction_comparison(results, change=True, size=size, ext=ext, 
                                       dpi=dpi, plot_dir=prediction_plot_dir)
            plot_factor_fingerprint(results, change=False, size=size, ext=ext,
                                    dpi=dpi, plot_dir=prediction_plot_dir)
            plot_factor_fingerprint(results, change=True, size=size, ext=ext,
                                  dpi=dpi, plot_dir=prediction_plot_dir)
        
        # copy latest results and prediction to higher directory
        generic_dir = '_'.join(plot_dir.split('_')[0:-1])
        if path.exists(generic_dir):
            rmtree(generic_dir)
        copytree(plot_dir, generic_dir)

# ****************************************************************************
# group analysis (across subsets)
# ****************************************************************************
        
if group_analysis == True:
    all_results = load_results(datafile)
    run_group_prediction(all_results, 
                         shuffle=False, classifier='ridge',
                         include_raw_demographics=False, rotate='oblimin',
                         verbose=False)

if group_plot == True:
    if verbose:
        print('*'*79)
        print('*'*79)
        print("Group Plots")
    all_results = load_results(datafile)
    plot_file = path.dirname(all_results['task'].get_plot_dir())
    plot_corr_heatmap(all_results, size=size, ext=ext, dpi=dpi, plot_dir=plot_file)
    plot_BIC(all_results, size=size, ext=ext, dpi=dpi, plot_dir=plot_file)
    plot_cross_silhouette(all_results, size=size, ext=ext, dpi=dpi, plot_dir=plot_file)
    plot_cross_communality(all_results, size=size, ext=ext, dpi=dpi, plot_dir=plot_file)
# ****************************************************************************
# move plots to paper directory
# ****************************************************************************
if all_results is not None:
    plot_file = path.dirname(all_results['task'].get_plot_dir())
else:
    plot_file = results.get_plot_dir()
paper_dir = path.join(basedir, 'Results', 'Psych_Ontology_Paper')
figure_lookup = {
        'analysis_overview': 'Fig01_Analysis_Overview',
        'data_correlations': 'Fig02_Task-Survey_Correlation',
        'survey/HCA/dendrogram_EFA12_oblimin': 'Fig03_Survey_Dendrogram',
        'task/HCA/dendrogram_EFA5_oblimin': 'Fig04_Task_Dendrogram',
        'survey/prediction/EFA_ridge_prediction_bar': 'Fig05_Survey_prediction',
        'task/prediction/EFA_ridge_prediction_bar': 'Fig06_Task_prediction',
        'survey/EFA/factor_heatmap_EFA12': 'FigS02_Survey_EFA',
        'task/EFA/factor_heatmap_EFA5': 'FigS03_Task_EFA',
        'task/DA/factor_heatmap_DA9': 'FigS04_Outcome_EFA',
        'BIC_curves': 'FigS05_BIC_curves',
        'survey/EFA/factor_correlations_EFA12': 'FigS06_Survey_2nd-order',
        'task/EFA/factor_correlations_EFA5': 'FigS07_Task_2nd-order',
        'task/DA/factor_correlations_DA9': 'FigS08_Outcome_2nd-order',
        'communality_adjustment': 'FigS09_communality',
        'survey/HCA/dendrogram_data': 'FigS10_Survey_Raw_Dendrogram',
        'task/HCA/dendrogram_data': 'FigS11_Task_Raw_Dendrogram',
        'silhouette_analysis': 'FigS12_Survey_Silhouette',
        # survey clusters
        # task clusters
        # combined EFA prediction
        'survey/prediction/IDM_ridge_prediction_bar': 'FigS16_Survey_IDM_prediction',
        'task/prediction/IDM_ridge_prediction_bar': 'FigS17_Task_IDM_prediction',
        'survey/prediction/EFA_ridge_factor_fingerprint': 'FigS18_Survey_Factor_Fingerprints'
        }
    

for filey in figure_lookup.keys():
    orig_file = path.join(plot_file, filey+'.'+ext)
    new_file = path.join(paper_dir, 'Plots', figure_lookup[filey]+'.'+ext)
    a=subprocess.Popen('cpdf -scale-to-fit "4.6in PH mul 4.6in div PW" %s -o %s' % (orig_file, new_file),
                     shell=True, 
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
    out, err = a.communicate()
    if 'cpdf: not found' in str(err):
        try:
            copyfile(orig_file, 
                     new_file)
        except FileNotFoundError:
            print('%s not found' % filey)
    elif 'No such file or directory' in str(err):
        print('%s not found' % filey)