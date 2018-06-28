from os import path
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr
from dimensional_structure.prediction_utils import run_prediction

def run_group_prediction(all_results, shuffle=False, classifier='lasso',
                       include_raw_demographics=False, rotate='oblimin',
                       verbose=False, save=True):
    if verbose:
        print('*'*79)
        print('Running Prediction, shuffle: %s, classifier: %s' % (shuffle, classifier))
        print('*'*79)
    
    names = [r.ID.split('_')[0] for r in all_results.values()]
    name = '_'.join(names)
    factor_scores = pd.concat([r.EFA.get_scores(rotate=rotate) 
                                for r in all_results.values()], axis=1)
    tmp_results = list(all_results.values())[0]
    output_dir = path.dirname(tmp_results.get_output_dir())
    demographics = tmp_results.DA
    demographic_factors = demographics.reorder_factors(demographics.get_scores())

    
    targets = [('demo_factors', demographic_factors)]
    if include_raw_demographics:
        targets.append(('demo_raw', tmp_results.demographics))
    out = {}
    for target_name, target in targets:
        predictors = ('EFA_%s_%s' % (name, rotate), factor_scores)
        # predicting using best EFA
        if verbose: print('**Predicting using %s**' % predictors[0])
        prediction = run_prediction(predictors[1], 
                        target, 
                        output_dir,
                        outfile='%s_%s_prediction' % (predictors[0], target_name), 
                        shuffle=shuffle,
                        classifier=classifier, 
                        verbose=verbose, 
                        save=save)
        out[target_name] = prediction
    return out
        



def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = pearsonr(df[r], df[c])[1]
    return pvalues

def FDR_correction(pval_df, alpha=.01):
    pvals = squareform(pval_df)
    sort_indices = np.argsort(pvals)
    sorted_pvals = pvals[sort_indices]
    thresholds = [i/len(sorted_pvals)*alpha for i in range(len(sorted_pvals))]
    highest_k = np.where(sorted_pvals<=thresholds)[0][-1]
    significant_values = np.zeros(len(pvals), dtype=np.int8)
    significant_values[sort_indices[:highest_k]]=1
    return pd.DataFrame(squareform(significant_values), 
                        columns=pval_df.columns[:],
                        index=pval_df.index[:])
    
    
def calc_survey_task_relationship(all_results, EFA=False, alpha=.01):
    def get_EFA_HCA(results, EFA):
        if EFA == False:
            return results.HCA.results['data']
        else:
            c = results.EFA.results['num_factors']
            return results.HCA.results['EFA%s' % c]
    survey_order = get_EFA_HCA(all_results['survey'], EFA)['reorder_vec']
    task_order = get_EFA_HCA(all_results['task'], EFA)['reorder_vec']
    
    if EFA == False:
        all_data = pd.concat([all_results['task'].data.iloc[:, task_order], 
                              all_results['survey'].data.iloc[:, survey_order]], 
                            axis=1)
    else:
        all_data = pd.concat([all_results['task'].EFA.get_loading().T.iloc[:, task_order], 
                              all_results['survey'].EFA.get_loading().T.iloc[:, survey_order]], 
                            axis=1)
        
    pvals = calculate_pvalues(all_data)
    corrected_results = {}
    for alpha in [.05, .01]:
        corrected_results=FDR_correction(pvals, alpha)
    return corrected_results
