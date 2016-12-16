"""
perform classification on demographic data
after binarizing continuous variables
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

import behavpredict
importlib.reload(behavpredict)


if __name__=='__main__':
    # variables to be binarized for classification - dictionary
    # with threshold for each variable

    # parameters to set
    report_features=True
    shuffle=False
    output_dir='prediction_outputs'
    clfname='lasso'
    datasubset='task'


    assert datasubset in ['survey','task','all']

    # set up classifier
    if clfname=='lasso':
        clf=LassoCV()
    elif clfname=='forest':
        clf=ExtraTreesClassifier(n_estimators=250,n_jobs=self.n_jobs,
                                        class_weight='balanced')
    elif clfname=='rbfsvm':
        # RBF SVM was tested but found to have even worse performance
        # than RF in terms of zero-variance predictions
        tuned_parameters={'gamma': 10.**numpy.arange(-5,5),
                     'C': 10.**numpy.arange(-2,3)}
        clf = GridSearchCV(SVC(), tuned_parameters, cv=4,
                       scoring='roc_auc')

    else:
        raise Exception('clfname %s is not defined'%clfname)

    bp=behavpredict.BehavPredict(verbose=2)
    bp.load_demog_data(binarize=True)
    bp.load_behav_data(datasubset)
    bp.get_joint_datasets()
    bp.binarize_demog_vars()
    for v in bp.demogdata.columns:
        print('')
        bp.rocscores[v],bp.importances[v]=bp.run_crossvalidation(v,clf=clf,
                                    shuffle=shuffle)
        if report_features and numpy.mean(bp.rocscores[v])>0.65:
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
