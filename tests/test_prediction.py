
import sys,os,glob
import random
import pickle
import importlib
import shutil
import numpy

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LassoCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from selfregulation.utils.utils import get_info
import selfregulation.prediction.behavpredict as behavpredict

# test using variables that should be perfectly predictable
def run_prediction(dataset,shuffle=False,classifier='lasso',
                    vars_to_test=['Age','Sex'],cleanup=True):
        bp=behavpredict.BehavPredict(verbose=True,
             drop_na_thresh=100,n_jobs=1,
             skip_vars=['RetirementPercentStocks'],
             output_dir='/tmp/bptest',shuffle=shuffle,
             classifier=classifier)
        bp.load_demog_data()
        bp.get_demogdata_vartypes()
        bp.load_behav_data(dataset)
        bp.filter_by_icc()
        bp.get_joint_datasets()
        # test one binary and one continuous variable
        for v in vars_to_test:
            bp.scores[v],bp.importances[v]=bp.run_crossvalidation(v)
        bp.write_data(vars_to_test)
        if cleanup:
            print('cleaning up')
            shutil.rmtree('/tmp/bptest')
        return bp.scores

def test_prediction_task():
    scores=run_prediction('task')
    assert scores['Age'][0]>0.95
    assert scores['Sex'][0]>0.95

def test_prediction_task_shuffle():
    scores=run_prediction('task',True)
    print(scores)
    assert numpy.abs(scores['Age'][0])<0.1
    assert numpy.abs(scores['Sex'][0] - 0.5) < 0.1
def test_prediction_survey():
    scores=run_prediction('survey')
    assert scores['Age'][0]>0.95
    assert scores['Sex'][0]>0.95
def test_prediction_survey_shuffle():
    scores=run_prediction('survey',True)
    assert numpy.abs(scores['Age'][0])<0.1
    assert numpy.abs(scores['Sex'][0] - 0.5) < 0.1
def test_prediction_baseline():
    scores=run_prediction('baseline')
    assert scores['Age'][0]>0.95
    assert scores['Sex'][0]>0.95
def test_prediction_baseline_shuffle():
    scores=run_prediction('baseline',True)
    assert numpy.abs(scores['Age'][0])<0.1
    assert numpy.abs(scores['Sex'][0] - 0.5) < 0.1
def test_prediction_mean():
    scores=run_prediction('mean')
    assert scores['Age'][0]>0.95
    assert scores['Sex'][0]>0.95
