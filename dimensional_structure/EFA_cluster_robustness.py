from collections import OrderedDict as odict
import numpy as np
from os import path
import pandas as pd
import pickle
from sklearn.metrics import adjusted_mutual_info_score

from dimensional_structure.utils import get_loadings, hierarchical_cluster
from ontology_mapping.reconstruction_utils import reorder_FA
from selfregulation.utils.r_to_py_utils import psychFA
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_recent_dataset

# helper functions
def drop_EFA(data, measures, c):
    
    to_drop = data.filter(regex='|'.join(measures)).columns
    subset = data.drop(to_drop, axis=1)
    fa, output = psychFA(subset, c, method='ml', rotate='oblimin')
    loadings = get_loadings(output, labels=subset.columns)
    return loadings

def convert_cooccurence(labels):
    mat = np.zeros((len(labels), len(labels)))
    for i, val in enumerate(labels):
        mat[i] = (labels==val)
    return mat


def tril(square):
    indices = np.tril_indices_from(square, -1)
    return square[indices]

def get_nearest_clusters(HCA, inp, cluster):
    names, DVs = zip(*HCA.get_cluster_DVs(inp).items())
    cluster_i = names.index(cluster)
    nearest_clusters = []
    nearest_DVs = []
    i_1 = cluster_i-1 if cluster_i>0 else cluster_i+2
    i_2 = cluster_i+1 if cluster_i+1 < len(names) else cluster_i-2
    nearest_clusters.append(names[i_1])
    nearest_DVs.extend(DVs[i_1])
    nearest_clusters.append(names[i_2])
    nearest_DVs.extend(DVs[i_2])
    return nearest_clusters, nearest_DVs
    
# EFA robustness
# Check to see how sensitive the EFA solution is to any single measure
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


# cluster robustness
# simulate based on EFA bootstraps
output_dir = path.dirname(results['task'].get_output_dir())
save_file = path.join(output_dir, 'cluster_robustness.pkl')
sim_reps = 5000
cluster_robustness = {}
cooccurences = {}
relative_cooccurences = {}
for name, result in results.items():
    scores = []
    cooccurence = np.zeros((result.data.shape[1], result.data.shape[1]))
    c = result.EFA.get_c()
    inp = 'EFA%s_oblimin' % c
    orig_clustering = result.HCA.results[inp]['labels']
    for _ in range(sim_reps):
        stats = result.EFA.get_boot_stats(c=c)
        loadings = np.random.normal(size=stats['means'].shape)*stats['sds']+stats['means']
        clustering = hierarchical_cluster(loadings,
                                          method='average',
                                          min_cluster_size=3,
                                          pdist_kws={'metric': 'abscorrelation'})
        labels = clustering['labels']
        score = adjusted_mutual_info_score(orig_clustering,labels)
        scores.append(score)
        cooccurence += convert_cooccurence(labels)
    cluster_robustness[name] = scores
    cooccurences[name] = pd.DataFrame(cooccurence/sim_reps, index=loadings.index,
                                       columns=loadings.index)
    # calculate inter/intra cooccurence
    
    relative_cooccurence = odict({})
    for cluster, DVs in result.HCA.get_cluster_DVs(inp).items():
        nearest_clusters, nearest_DVs = get_nearest_clusters(result.HCA, inp, cluster)
        intra_subset = tril(cooccurences[name].loc[DVs, DVs].values)
        inter_subset = cooccurences[name].drop(DVs, axis=1).loc[DVs]
        nearest_subset = cooccurences[name].loc[DVs, nearest_DVs]
        relative_cooccurence[cluster] = (np.mean(intra_subset),
                                         np.mean(inter_subset.values.flatten()),
                                         np.mean(nearest_subset.values.flatten()))
    relative_cooccurences[name] = relative_cooccurence


    
# saving
pickle.dump({'cluster_robustness': cluster_robustness,
             'cluster_cooccurence': cooccurences,
             'relative_cooccurence': relative_cooccurences}, 
    open(save_file, 'wb'))