"""
perform classification on demographic data
after binarizing continuous variables
"""

import sys,os
import random
import pickle
import importlib

import numpy

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LassoCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,ShuffleSplit

from get_balanced_folds import BalancedKFold
import behavpredict
importlib.reload(behavpredict)

if __name__=='__main__':
    # variables to be binarized for classification - dictionary
    # with threshold for each variable

    # parameters to set
    report_features=True
    shuffle=False
    output_dir='regression_outputs'
    clfname='forest'
    datasubset='all'


    assert datasubset in ['survey','task','all']

    bp=behavpredict.BehavPredict(verbose=2,use_smote=False)

    # set up classifier
    if clfname=='lasso':
        clf=LassoCV()
    elif clfname=='forest':
        clf=ExtraTreesRegressor(n_estimators=250,n_jobs=bp.n_jobs)

    else:
        raise Exception('clfname %s is not defined'%clfname)

    bp.load_demog_data(binarize=False)
    bp.load_behav_data(datasubset)
    bp.get_joint_datasets()
    #bp.binarize_demog_vars()
    for v in bp.demogdata.columns:
        print('')
        bp.rocscores[v],bp.importances[v]=bp.run_crossvalidation(v,clf=clf,
                                    shuffle=shuffle,scoring='r2',
                                    outer_cv=BalancedKFold(bp.n_outer_splits))
        if report_features and numpy.mean(bp.rocscores[v])>0.2:
            meanimp=numpy.mean(bp.importances[v],0)
            meanimp_sortidx=numpy.argsort(meanimp)
            for i in meanimp_sortidx[-1:-4:-1]:
                print(bp.behavdata.columns[i],meanimp[i])
            for i in meanimp_sortidx[:3][::-1]:
                print(bp.behavdata.columns[i],meanimp[i])

    h='%08x'%random.getrandbits(32)
    shuffle_flag='shuffle_' if shuffle else ''
    outfile='prediction_%s_%s_%s%s.pkl'%(datasubset,clfname,shuffle_flag,h)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    pickle.dump((bp.rocscores,bp.importances),open(os.path.join(output_dir,outfile),'wb'))
