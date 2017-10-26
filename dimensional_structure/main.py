# imports
import argparse
from dimensional_structure.EFA_plots import plot_EFA
from dimensional_structure.HCA_plots import plot_clusterings
from dimensional_structure.results import Results
from os import makedirs, path
import pickle

"""
# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-rerun', action='store_true')
parser.add_argument('-no_plot', action='store_true')
args = parser.parse_args()

rerun = args.rerun
plot_on = not args.no_plot
"""

# ****************************************************************************
# Laad Data
# ****************************************************************************
datafile = 'Complete_10-08-2017'
results = Results(datafile, dist_metric='abscorrelation')
results.run_EFA_analysis(verbose=True)
results.run_HCA_analysis(verbose=True)

# ***************************** saving ****************************************
distmetric = results.HCA.metric_name
pickle.dump(results, open(path.join(results.output_file, 
                                    'results_distmetric-%s.pkl' % distmetric),'wb'))


# ***************************** loading ****************************************
# results = pickle.load(open('Output/%s/results.pkl' % datafile, 'rb'))

# add function to existing class
# results.fun = fun.__get__(results)
# ****************************************************************************
# Bootstrap run
# ****************************************************************************

results.run_bootstrap(verbose=True, save_dir='/home/ian/tmp')

import time
start = time.time()
results.run_parallel_boot(2, save_dir='/home/ian/tmp')
end = time.time()-start

def eval_data_clusters(results, boot_results):
    orig_data_clusters = results.HCA.results['clustering_metric-abscorrelation_input-data']['clustering']['labels']
    boot_clusters = []
    for r in boot_results:
        HCA_results = r['HCA_solutions']
        data_HCA = HCA_results['clustering_metric-abscorrelation_input-data']
        data_clusters = data_HCA['clustering']['labels']
        boot_clusters.append(data_clusters)
    boot_consistency = [adjusted_rand_score(a,b) for a,b 
                        in  combinations(boot_clusters,2)]
    data_consistency = [adjusted_rand_score(orig_data_clusters,a) for a
                        in  boot_clusters]
    avg_data_consistency = np.mean(data_consistency)
        
        
# ****************************************************************************
# Plotting
# ****************************************************************************
EFA_plot_dir = path.join(results.plot_file, 'EFA')
HCA_plot_dir = path.join(results.plot_file, 'HCA')
makedirs(EFA_plot_dir, exist_ok = True)
makedirs(HCA_plot_dir, exist_ok = True)

# Plot EFA
for i, c in enumerate(results.EFA.get_metric_cs().values()):
    if i==0:
        plot_EFA(results.EFA, c, EFA_plot_dir, verbose=True)
    else:
        plot_EFA(results.EFA, c, EFA_plot_dir, 
                 verbose=True, plot_generic=False)
    
# Plot HCA
plot_clusterings(results.HCA, HCA_plot_dir, verbose=True)
