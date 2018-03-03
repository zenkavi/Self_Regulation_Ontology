# ****************************************************************************
# Helper functions for prediction
# ****************************************************************************
import os
import sys
import traceback
import selfregulation.prediction.behavpredict_V1 as behavpredict
from selfregulation.prediction.prediction_utils import get_demographic_model_type

def run_EFA_prediction(dataset, factor_scores, demographics, output_base, save=True,
                       verbose=False, classifier='lasso',
                       shuffle=False, n_jobs=2, imputer="SimpleFill",
                       smote_threshold=.05, freq_threshold=.1, icc_threshold=.25,
                       no_baseline_vars=True, singlevar=None):
    
    output_dir=os.path.join(output_base,'prediction_outputs')
    if dataset is 'baseline' or no_baseline_vars:
        baselinevars=False
        if verbose:
            print("turning off inclusion of baseline vars")
    else:
        baselinevars=True
        if verbose:
            print("including baseline vars in prediction models")
            
    # skip several variables because they crash the estimation tool
    bp=behavpredict.BehavPredict(verbose=verbose,
                                 dataset=dataset,
         drop_na_thresh=100,n_jobs=n_jobs,
         skip_vars=['RetirementPercentStocks',
         'HowOftenFailedActivitiesDrinking',
         'HowOftenGuiltRemorseDrinking',
         'AlcoholHowOften6Drinks'],
         output_dir=output_dir,shuffle=shuffle,
         classifier=classifier,
         add_baseline_vars=baselinevars,
         smote_cutoff=smote_threshold,
         freq_threshold=freq_threshold,
         imputer=imputer)
    
    bp.behavdata = factor_scores
    bp.demogdata = demographics
    model_types = get_demographic_model_type(demographics)
    bp.data_models = {k:v for i, (k,v) in model_types.iterrows()}
    bp.binarize_ZI_demog_vars()
    bp.behavdata = factor_scores
    #bp.filter_by_icc(icc_threshold)
    bp.get_joint_datasets()
    
    vars_to_test=[v for v in bp.demogdata.columns if not v in bp.skip_vars]
    for v in vars_to_test:
        bp.lambda_optim=None
        print('RUNNING:',v,bp.data_models[v],dataset)
        try:
            bp.scores[v],bp.importances[v]=bp.run_crossvalidation(v,nlambda=100)
            bp.scores_insample[v],_=bp.run_lm(v,nlambda=100)
            # fit model with no regularization
            if bp.data_models[v]=='binary':
                bp.lambda_optim=[0]
            else:
                bp.lambda_optim=[0,0]
            bp.scores_insample_unbiased[v],_=bp.run_lm(v,nlambda=100)
        except:
            e = sys.exc_info()
            print('error on',v,':',e)
            bp.errors[v]=traceback.format_tb(e[2])
    if save == True:
        bp.write_data(vars_to_test)
    return bp

