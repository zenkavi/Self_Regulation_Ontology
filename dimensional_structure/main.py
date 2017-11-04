# imports
import argparse
from EFA_plots import plot_EFA
from HCA_plots import plot_HCA
from results import Results
from glob import glob
import numpy as np
from os import makedirs, path, remove
import pickle
from shutil import copyfile
import time
"""
# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-rerun', action='store_true')
parser.add_argument('-no_plot', action='store_true')
args = parser.parse_args()

rerun = args.rerun
plot_on = not args.no_plot
"""
def not_regex(txt):
    return '^((?!%s).)*$' % txt
datafile = 'Complete_10-08-2017'
subsets = [{'name': 'task', 'regex': not_regex('survey')},
           {'name': 'all', 'regex': '.'},
            {'name': 'survey', 'regex': 'survey'}]

for subset in subsets:
    name = subset['name']
    print('Running Subset: %s' % name)
    # ****************************************************************************
    # Laad Data
    # ****************************************************************************
    # run dimensional analysis
    start = time.time()
    results = Results(datafile, dist_metric='abscorrelation',
                      name=subset['name'],
                      filter_regex=subset['regex'])
    results.run_EFA_analysis(verbose=True)
    results.run_HCA_analysis(verbose=True)
    run_time = time.time()-start
    
    # ***************************** saving ****************************************
    id_file = path.join(results.output_file,  'results_ID-%s.pkl' % results.ID)
    pickle.dump(results, open(id_file,'wb'))
    copyfile(id_file, path.join(results.output_file, 'results_%s.pkl' % name))
    
    
    # ***************************** loading ****************************************
    result_file = glob('Output/%s/%s/results_ID*.pkl' % (datafile, name))[-1]
    results = pickle.load(open(result_file, 'rb'))
    
    # add function to existing class
    # results.fun = fun.__get__(results)
    
    # *************************Aim 2 Task Choice**************************************
    EFA = results.EFA
    loadings = EFA.get_loading(EFA.get_metric_cs()['c_metric-BIC'])
    
    for task in ['stop_signal', 'kirby', 'stroop', 'threebytwo']:
        print(task)
    
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
    EFA_plot_dir = path.join(results.plot_file, 'EFA')
    HCA_plot_dir = path.join(results.plot_file, 'HCA')
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
    for i, c in enumerate(results.EFA.get_metric_cs().values()):
        if i==0:
            plot_EFA(results.EFA, c, EFA_plot_dir, verbose=True,
                     plot_task_kws=plot_task_kws)
        else:
            plot_EFA(results.EFA, c, EFA_plot_dir, 
                     verbose=True, plot_generic=False,
                     plot_task_kws=plot_task_kws)
        
    # Plot HCA
    print("Plotting HCA")
    plot_HCA(results, HCA_plot_dir)