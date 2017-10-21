"""
perform prediction on demographic data

use different strategy depending on the nature of the variable:
- lasso classification (logistic regression) for binary variables
- lasso regression for normally distributed variables
- lasso-regularized zero-inflated poisson regression for zero-inflated variables
-- via R mpath library using rpy2

compare each model to a baseline with age and sex as regressors

TODO:
- add metadata including dataset ID into results output
- break icc thresholding into separate method
- use a better imputation method than SimpleFill
"""

import sys,os
import random
import pickle
import importlib

import numpy

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LassoCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from selfregulation.utils.utils import get_info
import selfregulation.prediction.behavpredict as behavpredict
importlib.reload(behavpredict)

import argparse
import fancyimpute

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v',"--verbose", help="increase output verbosity",
                        default=0, action='count')
    parser.add_argument('-c',"--classifier", help="classifier",
                            default='lasso')

    parser.add_argument("--report_features", help="print features",
                        action='store_true')
    parser.add_argument("--print_report", help="print report at the end",
                        action='store_true')
    parser.add_argument('-s',"--shuffle", help="shuffle target variable",
                        action='store_true')
    parser.add_argument('-i',"--icc_threshold", help="threshold for ICC filtering",
                        type=float,default=0.25)
    parser.add_argument("--freq_threshold", help="threshold for binary variable frequency",
                        type=float,default=0.04)
    parser.add_argument("--no_baseline_vars",
                        help="don't include baseline vars in task/survey model",
                        action='store_true')
    parser.add_argument('-d',"--dataset", help="dataset for prediction",
                            required=True)
    parser.add_argument('-j',"--n_jobs", help="number of processors",type=int,
                            default=2)
    parser.add_argument('-w',"--workdir", help="working directory")
    parser.add_argument('-r',"--resultsdir", help="results directory")
    parser.add_argument("--singlevar", nargs='*',help="run with single variables")

    parser.add_argument("--smote_threshold", help="threshold for applying smote (distance from 0.5)",
                        type=float,default=0.05)
    args=parser.parse_args()
    print(args)
    print(args.dataset)

    # parameters to set

    if args.resultsdir is None:
        try:
            output_base=get_info('results_directory')
        except:
            output_base='.'
    else:
        output_base=args.resultsdir
    output_dir=os.path.join(output_base,'prediction_outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #assert args.dataset in ['survey','mirt','task','all','baseline']
    assert args.classifier in ['lasso','rf']
    # don't regress out baseline vars for baseline model
    if args.dataset=='baseline' or args.no_baseline_vars:
        baselinevars=False
        if args.verbose:
            print("turning off inclusion of baseline vars")
    else:
        baselinevars=True
        if args.verbose:
            print("including baseline vars in survey/task models")


    # skip several variables because they crash the estimation tool
    bp=behavpredict.BehavPredict(verbose=args.verbose,
         drop_na_thresh=100,n_jobs=args.n_jobs,
         skip_vars=['RetirementPercentStocks',
         'HowOftenFailedActivitiesDrinking',
         'HowOftenGuiltRemorseDrinking'],
         output_dir=output_dir,shuffle=args.shuffle,
         classifier=args.classifier,
         add_baseline_vars=baselinevars,
         smote_cutoff=args.smote_threshold,
         freq_threshold=args.freq_threshold)
    bp.load_demog_data()
    bp.get_demogdata_vartypes()
    bp.remove_lowfreq_vars()
    bp.binarize_ZI_demog_vars()

    bp.load_behav_data('task')
    bp.add_varset('discounting',[v for v in list(bp.behavdata.columns) if v.find('discount')>-1])
    bp.add_varset('stopping',[v for v in list(bp.behavdata.columns) if v.find('stop_signal')>-1 or v.find('nogo')>-1])
    bp.add_varset('intelligence',[v for v in list(bp.behavdata.columns) if v.find('raven')>-1 or v.find('cognitive_reflection')>-1])
    bp.load_behav_data(args.dataset)
    bp.filter_by_icc(args.icc_threshold)

    # if args.shuffle:
    #     tmp=bp.demogdata.values.copy()
    #     numpy.random.shuffle(tmp)
    #     bp.demogdata.iloc[:,:]=tmp
    #     print('WARNING: shuffling target data')
    bp.get_joint_datasets()

    if not args.singlevar:
        vars_to_test=[v for v in bp.demogdata.columns if not v in bp.skip_vars]
    else:
        vars_to_test=args.singlevar

    for v in vars_to_test:
        if args.verbose:
            print(v,bp.data_models[v])
        try:
            bp.scores[v],bp.importances[v]=bp.run_crossvalidation(v,
                                    imputer=fancyimpute.SimpleFill,nlambda=100)
            bp.scores_insample[v],_=bp.run_lm(v,imputer=fancyimpute.SimpleFill,
                                    nlambda=150)
            # fit model with no regularization
            bp.lambda_optim=[0,0]
            bp.scores_insample_unbiased[v],_=bp.run_lm(v,imputer=fancyimpute.SimpleFill,
                                    nlambda=150)
        except:
            e = sys.exc_info()
            print('error on',v,':',e)
            bp.errors[v]=e

    bp.write_data(v)
