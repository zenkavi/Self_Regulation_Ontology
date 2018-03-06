# Running script to create results objects for subsets of data and plot

# imports
import argparse
from glob import glob
import numpy as np
from os import makedirs, path
import pickle
from shutil import copyfile, copytree, rmtree
import time

from dimensional_structure.results import Results
from dimensional_structure.DA_plots import plot_DA
from dimensional_structure.EFA_plots import plot_EFA
from dimensional_structure.HCA_plots import plot_HCA
from dimensional_structure.prediction_plots import plot_prediction, plot_prediction_comparison
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_info, get_recent_dataset

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-dataset', default=None)
parser.add_argument('-no_analysis', action='store_false')
parser.add_argument('-no_plot', action='store_false')
parser.add_argument('-bootstrap', action='store_true')
parser.add_argument('-boot_iter', type=int, default=1000)
parser.add_argument('-dpi', type=int, default=300)
parser.add_argument('-subset', default=None)
args = parser.parse_args()

dataset = args.dataset
run_analysis = args.no_analysis
run_plot = args.no_plot
bootstrap = args.bootstrap
boot_iter = args.boot_iter
dpi = args.dpi
selected_subset = args.subset
print('Running Analysis? %s, Plotting? %s, Bootstrap? %s' % (['No', 'Yes'][run_analysis], 
                                                             ['No', 'Yes'][run_plot],
                                                             ['No', 'Yes'][bootstrap]))
# get dataset of interest
basedir=get_info('base_directory')
if dataset == None:
    dataset = get_recent_dataset()
dataset = path.join(basedir,'Data',dataset)

datafile = dataset.split(path.sep)[-1]

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
            'cluster_names': []},
            {'name': 'survey',
             'regex': 'survey',
             'factor_names':  ['Sensation Seeking', 'Mindfulness', 'Emotional Control', 
                               'Impulsivity', 'Goal-Directedness', 'Reward Sensitivity',
                               'Risk Perception', 'Eating Control', 'Ethical Risk-Taking', 
                               'Social Risk-Taking', 'Financial Risk-Taking', 'Agreeableness']},
             {'name': 'all', 
            'regex': '.',
            'factor_names': []}]
classifiers = ['lasso', 'ridge', 'tikhonov']
ID = None # ID will be created
results = None
# create/run results for each subset
for subset in subsets[0:2]:
    name = subset['name']
    if selected_subset is not None and name != selected_subset:
        continue
    print('*'*79)
    print('*'*79)
    print('Running Subset: %s' % name)
    if run_analysis == True:
        # ****************************************************************************
        # Laad Data
        # ****************************************************************************
        # run dimensional analysis
        start = time.time()
        results = Results(datafile, dist_metric='abscorrelation',
                          name=subset['name'],
                          filter_regex=subset['regex'],
                          boot_iter=boot_iter,
                          ID=ID)
        results.run_demographic_analysis(verbose=True, bootstrap=bootstrap)
        results.run_EFA_analysis(verbose=True, bootstrap=bootstrap)
        results.run_clustering_analysis(verbose=True, run_graphs=False)
        ID = results.ID.split('_')[1]
        # name factors and clusters
        factor_names = subset.get('factor_names', None)
        cluster_names = subset.get('cluster_names', None)
        if factor_names:
            results.EFA.name_factors(factor_names)
        if cluster_names:
            results.HCA.name_clusters(cluster_names)
        results.DA.name_factors(demographic_factor_names)
        # run behavioral prediction using the factor results determined by BIC
        for classifier in classifiers:
            results.run_prediction(classifier=classifier, verbose=True)
            results.run_prediction(classifier=classifier, shuffle=True, verbose=True) # shuffled
        run_time = time.time()-start
        
        # ***************************** saving ****************************************
        print('Saving Subset: %s' % name)
        id_file = path.join(results.output_file,  'results_ID-%s.pkl' % results.ID)
        pickle.dump(results, open(id_file,'wb'))
        # copy latest results and prediction to higher directory
        copyfile(id_file, path.join(path.dirname(results.output_file), 
                                    '%s_results.pkl' % name))
        prediction_dir = path.join(results.output_file, 'prediction_outputs')
        for classifier in classifiers:
            prediction_files = glob(path.join(prediction_dir, '*%s*' % classifier))
            # sort by creation time and get last two files
            prediction_files = sorted(prediction_files, key = path.getmtime)[-4:]
            for filey in prediction_files:
                filename = '_'.join(path.basename(filey).split('_')[:-1])
                copyfile(filey, path.join(path.dirname(results.output_file), 
                                          '%s_%s.pkl' % (name, filename)))

    # ****************************************************************************
    # Plotting
    # ****************************************************************************
    if run_plot==True:
        if results is None or name not in results.ID:
            results = load_results(datafile, name=name)[name]
        DA_plot_dir = path.join(results.plot_file, 'DA')
        EFA_plot_dir = path.join(results.plot_file, 'EFA')
        HCA_plot_dir = path.join(results.plot_file, 'HCA')
        prediction_plot_dir = path.join(results.plot_file, 'prediction')
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
        print("Plotting EFA")
        plot_DA(results, DA_plot_dir, verbose=True, dpi=dpi)
        
        # Plot EFA
        print("Plotting EFA")
        plot_EFA(results, EFA_plot_dir, verbose=True,  dpi=dpi, plot_task_kws=plot_task_kws)
            
        # Plot HCA
        print("Plotting HCA")
        plot_HCA(results, HCA_plot_dir)
        
        # Plot prediction
        target_order = results.DA.reorder_factors(results.DA.get_loading()).columns
        for classifier in classifiers:
            print("Plotting Prediction, classifier: %s" % classifier)
            plot_prediction(results, target_order=target_order, EFA=True, 
                            classifier=classifier, dpi=dpi, 
                            plot_dir=prediction_plot_dir)
            plot_prediction(results, target_order=target_order, EFA=False, 
                            classifier=classifier, dpi=dpi, 
                            plot_dir=prediction_plot_dir)
        plot_prediction_comparison(results, dpi=dpi, plot_dir=prediction_plot_dir)
        
        # copy latest results and prediction to higher directory
        plot_dir = results.plot_file
        generic_dir = '_'.join(plot_dir.split('_')[0:-1])
        if path.exists(generic_dir):
            rmtree(generic_dir)
        copytree(plot_dir, generic_dir)
        