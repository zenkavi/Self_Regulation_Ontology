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
from shutil import copyfile, copytree, rmtree
import time
from utils import load_results

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-no_analysis', action='store_true')
parser.add_argument('-no_plot', action='store_true')
args = parser.parse_args()

run_analysis = not args.no_analysis
run_plot = not args.no_plot
print('Running Analysis? %s, Plotting? %s' % (['No', 'Yes'][run_analysis], 
                                              ['No', 'Yes'][run_plot]))

datafile = 'Complete_01-17-2018'
subsets = [{'name': 'all', 
            'regex': '.',
            'factor_names': ['Pros Plan', 'Sensation Seeking', 'Mind Over Matter', 'Info Processing', 'Discounting', 'Stim Processing', 'Caution', 'Planning/WM', 'Env Resp']},
           {'name': 'task', 
            'regex': 'task',
            'factor_names': ['Decision Speed', 'DPX', 'WM/IQ', 'ART', 'Stim Processing', 'Strategic Flexibility', 'Discounting']},
            {'name': 'survey',
             'regex': 'survey',
             'factor_names': ['Immediacy', 'Future', 'Sensation Seeking', 'DOSPERT', 'DOSPERT_fin', 'Agreeableness', 'DOSPERT_RP', 'Hedonism', 'Social', 'Emotional Control', 'Eating', 'Mindfulness']}]

           

ID = None # ID will be created
results = None
# create/run results for each subset
for subset in subsets:
    name = subset['name']
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
                          ID=ID)
        results.run_EFA_analysis(verbose=True)
        results.run_clustering_analysis(verbose=True, run_graphs=False)
        ID = results.ID.split('_')[1]
        # name factors
        factor_names = subset.get('factor_names', None)
        if factor_names is not None:
            results.EFA.name_factors(factor_names)
        # run behavioral prediction using the factor results determined by BIC
        c = results.EFA.get_metric_cs()['c_metric-BIC']
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
    # Bootstrap run
    # ****************************************************************************
    #start = time.time()
    #for _ in range(10):
    #    results.run_bootstrap(verbose=True, save=True)
    #boot_time = time.time()-start
    
    #def eval_data_clusters(results, boot_results):
    #    orig_data_clusters = results.HCA.results['clustering_metric-abscorrelation_input-data']['clustering']['labels']
    #    boot_clusters = []
    #    for r in boot_results:
    #        HCA_results = r['HCA_solutions']
    #        data_HCA = HCA_results['clustering_metric-abscorrelation_input-data']
    #        data_clusters = data_HCA['clustering']['labels']
    #        boot_clusters.append(data_clusters)
    #    boot_consistency = [adjusted_rand_score(a,b) for a,b 
    #                        in  combinations(boot_clusters,2)]
    #    data_consistency = [adjusted_rand_score(orig_data_clusters,a) for a
    #                        in  boot_clusters]
    #    return {'boot_consistency': np.mean(boot_consistency),  
    #            'data_consistency': np.mean(data_consistency)}
    #    
    #def get_dimensionality_estimates(boot_results):
    #    return [i['metric_cs'] for i in boot_results]
    
       
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
        