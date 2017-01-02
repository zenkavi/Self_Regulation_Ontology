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

if __name__=='__main__':

    # parameters to set
    report_features=False
    if len(sys.argv)>1:
        shuffle=int(sys.argv[1])
    else:
        shuffle=False
    if len(sys.argv)>2:
        datasubset=sys.argv[2]
    else:
        datasubset='baseline'
    if len(sys.argv)>2:
        vars=[sys.argv[3]]
    else:
        vars=None

    try:
        output_base=get_info('results_directory')
    except:
        output_base='.'
    output_dir=os.path.join(output_base,'prediction_outputs')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    assert datasubset in ['survey','mirt','task','all','baseline']

    # set up classifier
    clf=LassoCV()

    bp=behavpredict.BehavPredict(verbose=2,
         drop_na_thresh=100,n_jobs=2)
    bp.load_demog_data()
    bp.get_demogdata_vartypes()
    bp.load_behav_data(datasubset)
    if shuffle:
        tmp=bp.demogdata.values.copy()
        numpy.random.shuffle(tmp)
        bp.demogdata.iloc[:,:]=tmp
        print('WARNING: shuffling target data')
    bp.get_joint_datasets()

    if not vars:
        vars_to_test=bp.demogdata.columns
    else:
        vars_to_test=vars
    for v in vars_to_test:
        if numpy.mean(bp.demogdata[v]>0)<0.04:
            print('skipping due to low freq:',v,numpy.mean(bp.demogdata[v]>0))
            continue
        try:
            bp.scores[v],bp.importances[v]=bp.run_crossvalidation(v)
            if report_features and numpy.mean(bp.scores[v])>0.65:
                print('')
                meanimp=numpy.mean(bp.importances[v],0)
                meanimp_sortidx=numpy.argsort(meanimp)
                for i in meanimp_sortidx[-1:-4:-1]:
                    print(bp.behavdata.columns[i],meanimp[i])
                for i in meanimp_sortidx[:3][::-1]:
                    print(bp.behavdata.columns[i],meanimp[i])
        except:
            e = sys.exc_info()[0]
            print('error on',v,':',e)

    h='%08x'%random.getrandbits(32)
    shuffle_flag='shuffle_' if shuffle else ''
    outfile='prediction_%s_%s%s.pkl'%(datasubset,shuffle_flag,h)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    pickle.dump((bp.scores,bp.importances),open(os.path.join(output_dir,outfile),'wb'))

    # print a report

    for v in vars_to_test:
        t=bp.data_models[v]
        if t=='binary':
            cutoff=0.65
        else:
            cutoff=0.15

        if bp.scores[v]<cutoff:
            continue

        print('%s\t%s\t%f\t(%s)'%(v,t,bp.scores[v],'\t'.join(['%f'%i for i in bp.importances[v].tolist()[0]])))
