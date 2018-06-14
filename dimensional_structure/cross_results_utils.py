from os import path
import pandas as pd
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
        
