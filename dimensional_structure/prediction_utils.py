# ****************************************************************************
# Helper functions for prediction
# ****************************************************************************
import numpy as np
import os
import pandas as pd
from selfregulation.prediction.behavpredict_V2 import BehavPredict

def run_prediction(factor_scores, demographics, output_base, 
                   outfile='prediction', save=True,
                   verbose=False, classifier='lasso',
                   shuffle=False, n_jobs=2, imputer="SimpleFill",
                   smote_threshold=.05, freq_threshold=.1):
    
    output_dir=os.path.join(output_base,'prediction_outputs')
    
    bp = BehavPredict(behavdata=factor_scores,
                      demogdata=demographics,
                      classifier=classifier,
                      output_dir=output_dir,
                      outfile=outfile,
                      shuffle=shuffle)
    bp.binarize_ZI_demog_vars()
    vars_to_test=[v for v in bp.demogdata.columns if not v in bp.skip_vars]
    for v in vars_to_test:
        # run regression into non-null number is found. Should only be run once!
        # but occasionally a nan is returned for some reason
        cv_scores = insample_scores = [np.nan, np.nan]
        cv_repeats = insample_repeats = 0
        while pd.isnull(cv_scores[0]) and cv_repeats <= 2:
            cv_repeats += 1
            cv_scores, cv_importances = bp.run_crossvalidation(v,nlambda=100)
        while pd.isnull(insample_scores[0]) and insample_repeats <= 2:
            insample_repeats += 1
            insample_scores,_ = bp.run_lm(v)
        if verbose:
            if insample_repeats == 2 or cv_repeats == 2:
                print('Regressions was repeated var: %s.\n\
                CV run %s times. Insample run %s times' % (v, 
                                                           cv_repeats, 
                                                           insample_repeats))
            elif insample_repeats > 2 or cv_repeats > 2:
                print('Regressions FAILED for var: %s.\n\
                CV run %s times. Insample run %s times' % (v, 
                                                           cv_repeats, 
                                                           insample_repeats))
                
        bp.scores[v],bp.importances[v] = cv_scores, cv_importances
        bp.scores_insample[v] = insample_scores
        
    if save == True:
        bp.write_data(vars_to_test)
    return bp