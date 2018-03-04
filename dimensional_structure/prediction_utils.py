# ****************************************************************************
# Helper functions for prediction
# ****************************************************************************
import os
from selfregulation.prediction.behavpredict_V2 import BehavPredict

def run_prediction(factor_scores, demographics, output_base, 
                   outfile='prediction', save=True,
                   verbose=False, classifier='lasso',
                   shuffle=False, n_jobs=2, imputer="SimpleFill",
                   smote_threshold=.05, freq_threshold=.1):
    
    output_dir=os.path.join(output_base,'prediction_outputs')
    
    bp = BehavPredict(behavdata=factor_scores,
                      demogdata=demographics,
                      classifier='tikhonov',
                      output_dir=output_dir,
                      outfile=outfile,
                      shuffle=shuffle)
    bp.binarize_ZI_demog_vars()
    vars_to_test=[v for v in bp.demogdata.columns if not v in bp.skip_vars]
    for v in vars_to_test:
        bp.scores[v],bp.importances[v]=bp.run_crossvalidation(v,nlambda=100)
        bp.scores_insample[v],_=bp.run_lm(v,nlambda=100)
    if save == True:
        bp.write_data(vars_to_test)
    return bp