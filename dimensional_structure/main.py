# imports
import argparse
from dimensionality_structure.EFA_plots import plot_EFA
from dimensionality_structure.results import Results
from os import makedirs, path
import pickle


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
results = Results(datafile)
results.run_EFA_analysis(verbose=True)
results.run_HCA_analysis(verbose=True)

# ***************************** saving ****************************************
pickle.dump(results, open(path.join(results.output_file, 'results.pkl'),'wb'))

# ****************************************************************************
# Plotting
# ****************************************************************************
EFA_plot_dir = path.join(results.plot_file, 'EFA')
HCA_plot_dir = path.join(results.plot_file, 'HCA')
makedirs(EFA_plot_dir, exist_ok = True)
makedirs(HCA_plot_dir, exist_ok = True)

for c in results.EFA.get_metric_cs().values():
    plot_EFA(results.EFA, c, EFA_plot_dir)