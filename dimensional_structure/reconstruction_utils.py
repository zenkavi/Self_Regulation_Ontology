import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor, Lasso, Ridge
from selfregulation.utils.r_to_py_utils import psychFA

# utils for deriving and evaluating ontological factors for out-of-model tasks
def linear_reconstruction(results, var, pseudo_pop_size=60,
                              n_reps=100, clf=None, robust=False, verbose=True):
    def get_coefs(clf):
        try:
            return clf.coef_
        except AttributeError:
            return clf.estimator_.coef_
    if verbose: 
        print('Starting Linear reconstruction, var', var)
        print('*'*79)
    data = results.data
    c = results.EFA.results['num_factors']
    full_scores = results.EFA.get_scores(c)
    loadings = results.EFA.get_loading(c)
    # refit an EFA model without variable    
    subset = data.drop(var, axis=1)
    fa, out = psychFA(subset, c)
    scores = pd.DataFrame(out['scores'], 
                          columns=full_scores.columns,
                          index=full_scores.index)
    
    orig_estimate = loadings.loc[var]
    if clf is None:
        clf = LinearRegression(fit_intercept=False)
    if robust:
        clf = RANSACRegressor(base_estimator=clf)
    if verbose: print('Starting full reconstruction')
    clf.fit(scores, data.loc[:, var])
    full_reconstruction = pd.Series(get_coefs(clf), index=orig_estimate.index)
    estimated_loadings = []
    if verbose: print('Starting partial reconstruction, pop size:', pseudo_pop_size)
    for rep in range(n_reps):
        if verbose and rep%100==0: 
            print('Rep', rep)
        random_subset = np.random.choice(scores.index,
                                         pseudo_pop_size, 
                                         replace=False)
        X = scores.loc[random_subset]
        y = data.loc[random_subset, var]
        clf.fit(X,y)
        estimated_loadings.append(get_coefs(clf))
    estimated_loadings = pd.DataFrame(estimated_loadings, columns=orig_estimate.index).T
    # calculate average distance from coefficient estimat eacross runs
    return orig_estimate, estimated_loadings, full_reconstruction

def get_ontology_blend(weights, ref_loadings, weighted=True):
    """ Take a set of distances and reference loadings and return a reconstructed loading
    Args:
        distances: pandas Series of distances from the to-be-reconstructed variable in descending order
        ref_loading: the lookup loading matrix to index with the closest variables taken from distances
        k_list: list of k_values to use

    """
    weights /= weights.sum() # normalize
    if weighted:
        reconstruction = ref_loadings.loc[weights.index].multiply(weights,axis=0).sum(0)
        reconstruction['reconstruction_weights'] = tuple(weights)
    else:
        reconstruction = ref_loadings.loc[weights.index].mean(0)
        reconstruction['reconstruction_weights'] = tuple([1]*len(weights))
    reconstruction['reconstruction_vars'] = tuple(weights.index)
    return reconstruction

def reorder_FA(ref_FA, new_FA):
    c = len(ref_FA.columns)
    corr = pd.concat([ref_FA, new_FA], axis=1, sort=False).corr().iloc[c:, :c]
    new_FA = new_FA.loc[:,corr.idxmax()]
    new_FA.columns = ref_FA.columns
    return new_FA

def k_nearest_reconstruction(results, drop_regex, reconstruct_vars=None, 
                             pseudo_pop_size=60, n_reps=100, 
                             k_list=None, EFA_rotation='oblimin', verbose=True):

    if k_list is None:
        k_list = [3]
    data = results.data
    c = results.EFA.results['num_factors']
    orig_loadings = results.EFA.get_loading(c, rotate=EFA_rotation)
    # refit an EFA model without variable    
    drop_vars = list(data.filter(regex=drop_regex).columns)
    subset = data.drop(drop_vars, axis=1)
    fa, out = psychFA(subset, c, rotate=EFA_rotation)
    loadings = pd.DataFrame(out['loadings'], index=subset.columns)
    loadings = reorder_FA(orig_loadings, loadings)
    if reconstruct_vars is None:
        reconstruct_vars = drop_vars
    if verbose: 
        print('*'*79)
        print('Starting K Nearest reconstruction, measures:', reconstruct_vars)
        print('*'*79)

    # full reconstruction
    if verbose: print('Starting full reconstruction')
    full_reconstruction = []
    corr = data.corr().drop(drop_vars, axis=1)
    for var in drop_vars:
        distances = corr.loc[var].sort_values(ascending=False)
        for k in k_list:
            if k>len(distances): k=len(distances)
            for weighted in [True, False]:
                reconstruction = get_ontology_blend(distances[:k], loadings, weighted)
                reconstruction['weighted'] = weighted
                reconstruction['k'] = k
                reconstruction['var'] = var
                full_reconstruction.append(reconstruction)
    full_reconstruction = pd.DataFrame(full_reconstruction)

    if verbose: print('Starting partial reconstruction, pop size:', pseudo_pop_size)
    estimated_loadings = []
    for rep in range(n_reps):
        if verbose and rep%100==0: 
            print('Rep', rep)
        random_subset = np.random.choice(data.index,
                                         pseudo_pop_size, 
                                         replace=False)
        corr = data.loc[random_subset,:].corr().drop(drop_vars, axis=1)
        for var in drop_vars:
            distances = corr.loc[var].sort_values(ascending=False)
            for k in k_list:
                if k>len(distances): k=len(distances)
                for weighted in [True, False]:
                    reconstruction = get_ontology_blend(distances[:k], loadings, weighted)
                    reconstruction['weighted'] = weighted
                    reconstruction['sample'] = rep
                    reconstruction['k'] = k
                    reconstruction['var'] = var
                    estimated_loadings.append(reconstruction)
    estimated_loadings = pd.DataFrame(estimated_loadings)
    return estimated_loadings, full_reconstruction

def linear_weighted_reconstruction(results, clf, drop_regex, reconstruct_vars=None,
                                   pseudo_pop_size=60, n_reps=100,
                                   EFA_rotation='oblimin', verbose=True):
    data = results.data
    c = results.EFA.results['num_factors']
    orig_loadings = results.EFA.get_loading(c, rotate=EFA_rotation)
    # refit an EFA model without variable    
    drop_vars = list(data.filter(regex=drop_regex).columns)
    subset = data.drop(drop_vars, axis=1)
    fa, out = psychFA(subset, c, rotate=EFA_rotation)
    loadings = pd.DataFrame(out['loadings'], index=subset.columns)
    loadings = reorder_FA(orig_loadings, loadings)
    if reconstruct_vars is None:
        reconstruct_vars = drop_vars
    if verbose: 
        print('*'*79)
        print('Starting linear weighted reconstruction, measures:', reconstruct_vars)
        print('*'*79)
    # full reconstruction
    if verbose: print('Starting full reconstruction')
    full_reconstruction = []
    for var in drop_vars:
        # get weights
        X = data.drop(drop_vars, axis=1)
        clf.fit(X.values, data.loc[:,var])
        weights = pd.Series(clf.coef_, index=X.columns)
        weights = weights[weights!=0] # remove zero
        reconstruction = get_ontology_blend(weights, loadings, True)
        reconstruction['var'] = var
        full_reconstruction.append(reconstruction)
    full_reconstruction = pd.DataFrame(full_reconstruction)

    if verbose: print('Starting partial reconstruction, pop size:', pseudo_pop_size)
    estimated_loadings = []
    for rep in range(n_reps):
        if verbose and rep%100==0: 
            print('Rep', rep)
        random_subset = np.random.choice(data.index,
                                         pseudo_pop_size, 
                                         replace=False)
        for var in drop_vars:
            X = data.drop(drop_vars, axis=1).loc[random_subset]
            clf.fit(X, data.loc[random_subset,var])
            weights = pd.Series(clf.coef_, index=X.columns)
            weights = weights[weights!=0] # remove zero
            reconstruction = get_ontology_blend(weights, loadings, True)
            reconstruction['sample'] = rep
            reconstruction['var'] = var
            estimated_loadings.append(reconstruction)
    estimated_loadings = pd.DataFrame(estimated_loadings)
    return estimated_loadings, full_reconstruction


def organize_reconstruction(reconstruction_results, scoring_funs=None):
    # organize the output from the simulations
    reconstruction_df = reconstruction_results.pop('true')
    reconstruction_df.loc[:,'label'] = 'true'
    for pop_size, out in reconstruction_results.items():
        for k, v in out.items():
            c = len(v[0])
            combined = pd.concat([v[1], v[0]], sort=False)
            combined.reset_index(drop=True, inplace=True)
            labels = ['full_reconstruct']
            if len(v[1].shape) == 2:
                labels += ['full_reconstruct']*(v[1].shape[0]-1)
            labels += ['partial_reconstruct']*v[0].shape[0]
            combined.loc[:, 'label'] = labels
            combined.loc[:, 'pop_size'] = pop_size
            reconstruction_df = pd.concat([reconstruction_df, combined], sort=False)
    reconstruction_df = reconstruction_df.infer_objects().reset_index(drop=True)
    # drop redundant reconstructions
    pop_sizes = reconstruction_df.pop_size.dropna().unique()[1:]
    drop_indices = reconstruction_df[(reconstruction_df.label=="full_reconstruct") & 
                                     (reconstruction_df.pop_size.isin(pop_sizes))].index
    reconstruction_df.drop(drop_indices, inplace=True)
    reconstruction_df.loc[(reconstruction_df.label=="full_reconstruct"), 'pop_size'] = np.nan
    if scoring_funs:
        for fun in scoring_funs:
            fun(reconstruction_df)
    return reconstruction_df

def corr_scoring(organized_results):
    for v, group in organized_results.groupby('var'):
        corr_scores = np.corrcoef(x=group.iloc[:,:5].astype(float))[:,0]
        organized_results.loc[group.index, 'corr_score'] = corr_scores
        
def get_reconstruction_results(results, measure_list, pop_sizes=(100,200), 
                               recon_fun=linear_reconstruction, 
                               scoring_funs=(corr_scoring,), 
                               **kwargs):
    loadings = results.EFA.get_loading(c=results.EFA.results['num_factors'])
    reconstructed_DVs = set()
    reconstruction_results = {}
    # convert list of measures to a regex lookup
    for pop_size in pop_sizes:     
        out = {}
        for measure in measure_list:
            estimated, full = recon_fun(results, drop_regex=measure, reconstruct_vars=None,
                                        pseudo_pop_size=pop_size, **kwargs)
            out[measure] = [estimated, full]
            reconstructed_DVs = reconstructed_DVs | set(full['var'])
        reconstruction_results[pop_size] = out
        
    true = loadings.loc[reconstructed_DVs]
    true.loc[:,'var'] = true.index
    reconstruction_results['true'] = true
    return organize_reconstruction(reconstruction_results, scoring_funs=scoring_funs)
    
