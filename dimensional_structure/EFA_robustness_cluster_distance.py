import numpy as np
from os import path
import pandas as pd
import pickle

from dimensional_structure.utils import get_loadings
from ontology_mapping.reconstruction_utils import reorder_FA
from selfregulation.utils.r_to_py_utils import psychFA
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_recent_dataset


# EFA robustness
# Check to see how sensitive the EFA solution is to any single measure
def drop_EFA(data, measures, c):
    
    to_drop = data.filter(regex='|'.join(measures)).columns
    subset = data.drop(to_drop, axis=1)
    fa, output = psychFA(subset, c, method='ml', rotate='oblimin')
    loadings = get_loadings(output, labels=subset.columns)
    return loadings

results = load_results(get_recent_dataset())
for result in results.values():
    output_dir = result.get_output_dir()
    c = result.EFA.get_c()
    orig_loadings = result.EFA.get_loading(c=c)
    measures = np.unique([c.split('.')[0] for c in result.data.columns])
    # drop a single measure
    factor_correlations = {}
    for measure in measures:
        data = result.data
        new_loadings = drop_EFA(data, [measure], c)
        new_loadings = reorder_FA(orig_loadings.loc[new_loadings.index], 
                                  new_loadings, thresh=-1)
        
        corr = pd.concat([new_loadings, orig_loadings], axis=1, sort=False) \
                .corr().iloc[:c, c:]
        diag = {c:i for c,i in zip(new_loadings.columns, np.diag(corr))}
        factor_correlations[measure] = diag

    
    # save pair factor correlations
    save_file = path.join(output_dir, 'EFAdrop_robustness.pkl')
    to_save = factor_correlations
    pickle.dump(to_save, open(save_file, 'wb'))

# cluster distance
# compare the absolute correlation distance within cluster to between clusters
task = results['task']
corrs = {}
for name, group in task.HCA.get_cluster_DVs(inp='EFA5_oblimin').items():
    corr = abs(orig_loadings.loc[group].T.corr())
    corrs[name] = 1-corr.values[np.tril_indices_from(corr, -1)]
    
cluster_loadings = task.HCA.get_cluster_loading(EFA=task.EFA)

cluster_dists = 1-abs(pd.DataFrame(task.HCA.get_cluster_loading(EFA=task.EFA)).corr())
cluster_dists = cluster_dists.values[np.tril_indices_from(cluster_dists, -1)]
