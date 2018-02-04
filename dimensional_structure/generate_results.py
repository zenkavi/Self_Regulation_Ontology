# Running script to create results objects for subsets of data and plot

# imports
import argparse
from EFA_plots import plot_EFA
from HCA_plots import plot_HCA
from prediction_plots import plot_prediction
from results import Results
from glob import glob
import numpy as np
from os import makedirs, path
import pickle
from selfregulation.utils.utils import get_info, sorting
from shutil import copyfile, copytree, rmtree
import time
from utils import load_results

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-dataset', default=None)
parser.add_argument('-no_analysis', action='store_false')
parser.add_argument('-no_plot', action='store_false')
parser.add_argument('-bootstrap', action='store_true')
args = parser.parse_args()

dataset = args.dataset
run_analysis = args.no_analysis
run_plot = args.no_plot
bootstrap = args.bootstrap
print('Running Analysis? %s, Plotting? %s, Bootstrap? %s' % (['No', 'Yes'][run_analysis], 
                                                             ['No', 'Yes'][run_plot],
                                                             ['No', 'Yes'][bootstrap]))
# get dataset of interest
basedir=get_info('base_directory')
if dataset == None:
    files = glob(path.join(basedir,'Data/Complete*'))
    files.sort(key=sorting)
    dataset = files[-1]
else:
    dataset = path.join(basedir,'Data',dataset)
    
subsets = [{'name': 'task', 
            'regex': 'task',
            'factor_names': ['Speeded IP', 'Discounting', 'Perc/Resp',
                             'Strategic IP', 'Caution']},
            {'name': 'survey',
             'regex': 'survey',
             'factor_names':  ['Sensation Seeking', 'Mindfulness', 'Emotional Control', 'Impulsivity', 'Goal-Directedness', 'Reward Sensitivity', 'Risk Perception', 'Eating Control', 'Ethical Risk Taking', 'Social Risk Taking', 'Financial Risk Taking', 'Agreeableness']},
             {'name': 'all', 
            'regex': '.',
            'factor_names': []}]

ID = None # ID will be created
results = None
# create/run results for each subset
for subset in subsets[0:2]:
    name = subset['name']
    print('Running Subset: %s' % name)
    if run_analysis == True:
        # ****************************************************************************
        # Laad Data
        # ****************************************************************************
        # run dimensional analysis
        start = time.time()
        results = Results(dataset, dist_metric='abscorrelation',
                          name=subset['name'],
                          filter_regex=subset['regex'],
                          ID=ID)
        results.run_EFA_analysis(verbose=True, bootstrap=bootstrap)
        results.run_clustering_analysis(verbose=True, run_graphs=False)
        ID = results.ID.split('_')[1]
        # name factors
        factor_names = subset.get('factor_names', None)
        if factor_names:
            results.EFA.name_factors(factor_names)
        # run behavioral prediction using the factor results determined by BIC
        c = results.EFA.num_factors
        results.run_prediction(c=c)
        results.run_prediction(c=c, shuffle=True) # shuffled
        run_time = time.time()-start
        
        # ***************************** saving ****************************************
        id_file = path.join(results.output_file,  'results_ID-%s.pkl' % results.ID)
        pickle.dump(results, open(id_file,'wb'))
        # copy latest results and prediction to higher directory
        copyfile(id_file, path.join(path.dirname(results.output_file), 
                                    '%s_results.pkl' % name))
        prediction_dir = path.join(results.output_file, 'prediction_outputs')
        prediction_files = glob(path.join(prediction_dir, '*'))
        # sort by creation time and get last two files
        prediction_files = sorted(prediction_files, key = path.getmtime)[-2:]
        for filey in prediction_files:
            if 'shuffle' in filey:
                copyfile(filey, path.join(path.dirname(results.output_file), 
                                          '%s_prediction_shuffle.pkl' % name))
            else:
                copyfile(filey, path.join(path.dirname(results.output_file), 
                                          '%s_prediction.pkl' % name))
    
       
    # ****************************************************************************
    # Plotting
    # ****************************************************************************
    if run_plot==True:
        if results is None or name not in results.ID:
            results = load_results(datafile, name=name)[name]
        EFA_plot_dir = path.join(results.plot_file, 'EFA')
        HCA_plot_dir = path.join(results.plot_file, 'HCA')
        prediction_plot_dir = path.join(results.plot_file, 'prediction')
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
        plot_EFA(results, EFA_plot_dir, verbose=True,  plot_task_kws=plot_task_kws)
            
        # Plot HCA
        print("Plotting HCA")
        plot_HCA(results, HCA_plot_dir)
        
        # Plot prediction
        print("Plotting Prediction")
        plot_prediction(results, prediction_plot_dir)
        
        # copy latest results and prediction to higher directory
        plot_dir = results.plot_file
        generic_dir = '_'.join(plot_dir.split('_')[0:-1])
        if path.exists(generic_dir):
            rmtree(generic_dir)
        copytree(plot_dir, generic_dir)
        