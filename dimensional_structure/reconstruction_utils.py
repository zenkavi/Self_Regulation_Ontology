import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression, RANSACRegressor, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale
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


def reorder_FA(ref_FA, new_FA):
    c = len(ref_FA.columns)
    corr = pd.concat([ref_FA, new_FA], axis=1, sort=False).corr().iloc[c:, :c]
    new_FA = new_FA.loc[:,corr.idxmax()]
    new_FA.columns = ref_FA.columns
    return new_FA

def run_linear(scores, test_vars, clf=LinearRegression(fit_intercept=False)):
    """
    Run Knearest neighbor using precomputed distances to create an ontological mapping
    
    Args:
        scores: ontological scores
        test_vars: variable to reconstruct
        clf: linear model that returns coefs
    """
    clf.fit(scores, test_vars)
    out = clf.coef_
    if len(out.shape)==1:
        out = out.reshape(1,-1)
    out = pd.DataFrame(out, columns=scores.columns)
    out['var'] = test_vars.columns
    return out

def linear_reconstruction(results, drop_regex, 
                          pseudo_pop_size=60, n_reps=100, 
                          clf=LinearRegression(fit_intercept=False),
                          EFA_rotation='oblimin', verbose=True):
    data = results.data
    c = results.EFA.results['num_factors']
    orig_scores = results.EFA.get_scores(c, rotate=EFA_rotation)
    # refit an EFA model without variable    
    drop_vars = list(data.filter(regex=drop_regex).columns)
    subset = data.drop(drop_vars, axis=1)
    fa, out = psychFA(subset, c, rotate=EFA_rotation)
    scores = pd.DataFrame(out['scores'], index=subset.index)
    scores = reorder_FA(orig_scores, scores)
    if verbose:
        print('*'*79)
        print('Reconstructing', drop_vars)
        print('*'*79)
        
    if verbose: print('Starting full reconstruction')
    full_reconstruction = run_linear(scores, scale(data.loc[:, drop_vars]), clf)
    full_reconstruction.reset_index(drop=True)

    if verbose: print('Starting partial reconstruction, pop size:', pseudo_pop_size)
    estimated_loadings = pd.DataFrame()
    for rep in range(n_reps):
        if verbose and rep%100==0: 
            print('Rep', rep)
        random_subset = np.random.choice(data.index,pseudo_pop_size, replace=False)
        out = run_linear(scores.loc[random_subset], scale(data.loc[random_subset, drop_vars]), clf)
        out['rep'] = rep+1
        estimated_loadings = pd.concat([estimated_loadings, out], sort=False)
    estimated_loadings.reset_index(drop=True)
    return estimated_loadings, full_reconstruction

    
def run_kNeighbors(distances, loadings, test_vars, 
                   weightings=('uniform',), k_list=(3)):
    """
    Run Knearest neighbor using precomputed distances to create an ontological mapping
    
    Args:
        distances: square distance matrix to pass to KNeighborsRegressors
        loadings: loading matrix for training
        test_vars: variable to reconstruct
        weightings: (optional) list of weightings to pass to KNeighbors
        k_list: list of k values to pass to KNeighbors as n_neighbors
    """
    train_distances = distances.loc[loadings.index, :]
    test_distances = distances.loc[test_vars, :]
    to_return = pd.DataFrame()
    for weighting in weightings:
        for k in k_list:
            clf = KNeighborsRegressor(metric='precomputed', n_neighbors=k, weights=weighting)
            clf.fit(train_distances, loadings)
            out = clf.predict(test_distances)
            out = pd.DataFrame(out, columns=loadings.columns)
            out['var'] = test_vars
            out['k'] = k
            out['weighting'] = weighting
            # add neighbors and distances
            neighbors = clf.kneighbors(test_distances)
            out['distances'] = tuple(neighbors[0])
            out['neighbors'] = tuple(test_distances.columns[neighbors[1]])
            to_return = pd.concat([to_return, out], sort=False)
    return to_return
    
def k_nearest_reconstruction(results, drop_regex, available_vars=None,
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
    weightings = ['uniform', 'distance']
    if available_vars is not None:
        data = data.loc[:, set(available_vars) | set(drop_vars)]
        loadings = loadings.loc[available_vars,:]
    if verbose:
        print('*'*79)
        print('Reconstructing', drop_vars)
        print('*'*79)
    if verbose: print('Starting full reconstruction')
    distances = pd.DataFrame(squareform(pdist(data.T, metric='correlation')), 
                             index=data.columns, 
                             columns=data.columns).drop(drop_vars, axis=1)

    full_reconstruction = run_kNeighbors(distances, loadings, drop_vars, weightings, k_list)
    full_reconstruction.reset_index(drop=True)

    if verbose: print('Starting partial reconstruction, pop size:', pseudo_pop_size)
    estimated_loadings = pd.DataFrame()
    for rep in range(n_reps):
        if verbose and rep%100==0: 
            print('Rep', rep)
        random_subset = data.loc[np.random.choice(data.index, 
                                                  pseudo_pop_size, 
                                                  replace=False)]
        distances = pd.DataFrame(squareform(pdist(random_subset.T, metric='correlation')), 
                                 index=random_subset.columns, 
                                 columns=random_subset.columns).drop(drop_vars, axis=1)
        out = run_kNeighbors(distances, loadings, drop_vars, weightings, k_list)
        out['rep'] = rep+1
        estimated_loadings = pd.concat([estimated_loadings, out], sort=False)
    estimated_loadings.reset_index(drop=True)
    return estimated_loadings, full_reconstruction

def corr_scoring(organized_results):
    for v, group in organized_results.groupby('var'):
        corr_scores = np.corrcoef(x=group.iloc[:,:5].astype(float))[:,0]
        organized_results.loc[group.index, 'corr_score'] = corr_scores
        
def organize_reconstruction(reconstruction_results, scoring_funs=None):
    # organize the output from the simulations
    reconstruction_df = reconstruction_results.pop('true')
    reconstruction_df.loc[:,'label'] = 'true'
    for pop_size, (estimated, full) in reconstruction_results.items():
        c = len(estimated)
        combined = pd.concat([full, estimated], sort=False)
        combined.reset_index(drop=True, inplace=True)
        labels = ['full_reconstruct']
        if len(full.shape) == 2:
            labels += ['full_reconstruct']*(full.shape[0]-1)
        labels += ['partial_reconstruct']*estimated.shape[0]
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

def get_reconstruction_results(results, measure_list, pop_sizes=(100,200), 
                               EFA_rotation='oblimin',
                               recon_fun=linear_reconstruction, 
                               scoring_funs=(corr_scoring,), 
                               **kwargs):
    loadings = results.EFA.get_loading(rotate=EFA_rotation, c=results.EFA.results['num_factors'])
    out = {}
    # convert list of measures to a regex lookup
    for measure in measure_list:
        reconstruction_results = {}
        for pop_size in pop_sizes:  
            estimated, full = recon_fun(results, drop_regex=measure, 
                                        pseudo_pop_size=pop_size, 
                                        EFA_rotation=EFA_rotation, **kwargs)

            reconstruction_results[pop_size] = [estimated, full]
        true = loadings.loc[set(full['var'])]
        true.loc[:,'var'] = true.index
        reconstruction_results['true'] = true
        organized = organize_reconstruction(reconstruction_results, scoring_funs=scoring_funs)
        out[measure.lstrip('^')] = organized  
    return out
    
