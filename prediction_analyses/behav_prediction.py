"""
perform prediction on demographic data

use different strategy depending on the nature of the variable:
- lasso classification (logistic regression) for binary variables
- lasso regression for normally distributed variables
- lasso-regularized zero-inflated poisson regression for zero-inflated variables
-- via R mpath library using rpy2

compare each model to a baseline with age and sex as regressors

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
    parser.add_argument("--no_baseline_regress", help="don't regress out baseline",
                        action='store_true')
    parser.add_argument('-d',"--dataset", help="dataset for prediction",
                            required=True)
    parser.add_argument('-j',"--n_jobs", help="number of processors",type=int,
                            default=2)
    parser.add_argument('-w',"--workdir", help="working directory")
    parser.add_argument('-r',"--resultsdir", help="results directory")
    parser.add_argument("--singlevar", nargs='*',help="run with single variables")

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

    assert args.dataset in ['survey','mirt','task','all','baseline']
    assert args.classifier in ['lasso','rf']
    # don't regress out baseline vars for baseline model
    if args.dataset=='baseline' or args.no_baseline_regress:
        preregress=False
        if args.verbose:
            print("turning off pre-regression by baseline vars")
    else:
        preregress=True
        if args.verbose:
            print("turning on pre-regression by baseline vars")
    

    # skip RetirementPercentStocks because it crashes the estimation tool
    bp=behavpredict.BehavPredict(verbose=args.verbose,
         drop_na_thresh=100,n_jobs=args.n_jobs,
         skip_vars=['RetirementPercentStocks'])
    bp.load_demog_data()
    bp.get_demogdata_vartypes()
    bp.load_behav_data(args.dataset)

    if args.shuffle:
        tmp=bp.demogdata.values.copy()
        numpy.random.shuffle(tmp)
        bp.demogdata.iloc[:,:]=tmp
        print('WARNING: shuffling target data')
    bp.get_joint_datasets()

    if not args.singlevar:
        vars_to_test=bp.demogdata.columns
    else:
        vars_to_test=args.singlevar

    print(vars_to_test)

    for v in vars_to_test:
        if v in bp.skip_vars:
            print('skipping',v)
            continue
        if numpy.mean(bp.demogdata[v]>0)<0.04:
            print('skipping due to low freq:',v,numpy.mean(bp.demogdata[v]>0))
            continue
        try:
            bp.scores[v],bp.importances[v]=bp.run_crossvalidation(v,
                     classifier=args.classifier,
                     preregress_baseline_vars=preregress)
            if args.report_features and numpy.mean(bp.scores[v])>0.65:
                print('')
                meanimp=numpy.mean(bp.importances[v],0)
                meanimp_sortidx=numpy.argsort(meanimp)
                for i in meanimp_sortidx[-1:-4:-1]:
                    print(bp.behavdata.columns[i],meanimp[i])
                for i in meanimp_sortidx[:3][::-1]:
                    print(bp.behavdata.columns[i],meanimp[i])
        except:
            e = sys.exc_info()
            print('error on',v,':',e)

        h='%08x'%random.getrandbits(32)
        shuffle_flag='shuffle_' if args.shuffle else ''
        varflag='%s_'%v
        outfile='prediction_%s_%s_%s%s%s.pkl'%(args.dataset,args.classifier,shuffle_flag,varflag,h)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        if args.verbose:
            print('saving to',os.path.join(output_dir,outfile))
        pickle.dump((bp.scores[v],bp.importances[v]),open(os.path.join(output_dir,outfile),'wb'))

    # print a report
    if args.print_report:
        for v in vars_to_test:
            if not v in bp.scores:
                continue
            t=bp.data_models[v]
            if t=='binary':
                cutoff=0.65
            else:
                cutoff=0.15

            if bp.scores[v]<cutoff:
                continue

            print('%s\t%s\t%f\t(%s)'%(v,t,bp.scores[v],'\t'.join(['%f'%i for i in bp.importances[v].tolist()[0]])))
