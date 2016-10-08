
# coding: utf-8

# This notebook assesses the ability to predict demographic outcomes from survey data.

# In[1]:

import os,glob,sys,pickle
import importlib

import warnings
import numpy,pandas
from sklearn.svm import LinearSVC,SVC,OneClassSVM
from sklearn.linear_model import LinearRegression,LogisticRegressionCV,RandomizedLogisticRegression,ElasticNet,ElasticNetCV,Ridge,RidgeCV
from sklearn.preprocessing import scale,StandardScaler,FunctionTransformer
from sklearn.decomposition import FactorAnalysis,PCA
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV,StratifiedShuffleSplit,cross_val_score
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import fancyimpute
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import scipy.stats

def print_confusion_matrix(y_true,y_pred,labels=[0,1]):
    cm=confusion_matrix(y_true,y_pred)
    print('Confusion matrix')
    print('\t\tPredicted')
    print('\t\t0\t1')
    print('Actual\t0\t%d\t%d'%(cm[0,0],cm[0,1]))
    print('\t1\t%d\t%d'%(cm[1,0],cm[1,1]))

# this is kludgey but it works
sys.path.append('../utils')
import utils
import crossvalidation
importlib.reload(utils)
importlib.reload(crossvalidation)

#warnings.filterwarnings("ignore") # only turn this on in production mode
                                  # to keep log files from overflowing

dataset='Discovery_9-26-16'
basedir=utils.get_info('base_directory')
derived_dir=os.path.join(basedir,'Data/Derived_Data/%s'%dataset)

if not os.path.exists('surveypred'):
    os.mkdir('surveypred')

binary_vars=["Sex","ArrestedChargedLifeCount","DivorceCount","GamblingProblem","ChildrenNumber",
            "CreditCardDebt","RentOwn","RetirementAccount","TrafficTicketsLastYearCount","Obese",
             "TrafficAccidentsLifeCount","CaffienatedSodaCansPerDay","Nervous",
             'Hopeless', 'RestlessFidgety', 'Depressed',
             'EverythingIsEffort', 'Worthless','CigsPerDay','LifetimeSmoke100Cigs',
             'CannabisPast6Months']
# try:
#     binary_vars=[sys.argv[1]]
# except:
#     print('specify variable as command line argument')
#     binary_vars=['Nervous'] #hsys.exit(1)

if len(sys.argv)>2:
    shuffle=True
    shufflenum=int(sys.argv[2])
else:
    shuffle=False

# for testing
#shuffle=True
#shufflenum=1

if shuffle:
    shuffletag='_shuffle%04d'%shufflenum
else:
    shuffletag=''

def softimpute(X, y=None):
    return(fancyimpute.SoftImpute(verbose=False).complete(X))

imputer = FunctionTransformer(softimpute, validate=False)

# for some items, we want to use somethign other than the minimum as the
# cutoff:
item_thresholds={'Nervous':1,
                'Hopeless':1,
                'RestlessFidgety':1,
                'Depressed':1,
                'EverythingIsEffort':1,
                'Worthless':1}

def get_demog_data(binary_vars=binary_vars,ordinal_vars=[],item_thresholds={},binarize=True):
    demogdata=pandas.read_csv(os.path.join(derived_dir,
        'surveydata/demographics.tsv'),index_col=0,delimiter='\t')
    healthdata=pandas.read_csv(os.path.join(derived_dir,
        'surveydata/health_ordinal.tsv'),index_col=0,delimiter='\t')
    alcdrugs=pandas.read_csv(os.path.join(derived_dir,
        'surveydata/alcohol_drugs_ordinal.tsv'),index_col=0,delimiter='\t')
    assert all(demogdata.index==healthdata.index)
    assert all(demogdata.index==alcdrugs.index)
    demogdata=demogdata.merge(healthdata,left_index=True,right_index=True)
    demogdata=demogdata.merge(alcdrugs,left_index=True,right_index=True)
    # remove a couple of outliers - this is only for cases when we include BMI/obesity
    if 'BMI' in ordinal_vars or 'Obese' in binary_vars:
        demogdata=demogdata.query('WeightPounds>50')
        demogdata=demogdata.query('HeightInches>36')
        demogdata=demogdata.query('CaffienatedSodaCansPerDay>-1')
        demogdata=demogdata.assign(BMI=demogdata['WeightPounds']*0.45 / (demogdata['HeightInches']*0.025)**2)
        demogdata=demogdata.assign(Obese=(demogdata['BMI']>30).astype('int'))

    if binarize:
        demogdata=demogdata[binary_vars]
        demogdata=demogdata.loc[demogdata.isnull().sum(1)==0]

        for i in range(len(binary_vars)):
            v=binary_vars[i]
            if v in item_thresholds:
                threshold=item_thresholds[v]
            else:
                threshold=demogdata[v].min()
            demogdata.loc[demogdata[v]>threshold,v]=1
            assert demogdata[v].isnull().sum()==0
    return demogdata


def get_joint_dataset(d1,d2):
    d1_index=set(d1.index)
    d2_index=set(d2.index)
    inter=list(d1_index.intersection(d2_index))
    return d1.ix[inter],d2.ix[inter]
    return inter

behavdata_orig=utils.get_behav_data('Discovery_9-26-16',use_EZ=True)

demogdata,behavdata=get_joint_dataset(get_demog_data(),behavdata_orig)
assert list(demogdata.index)==list(behavdata.index)
print('%d joint subjects found'%demogdata.shape[0])
behavvars=list(behavdata.columns)
print('%d task variables found'%len(behavvars))
print('Demographic variables to test:')
print(list(demogdata.columns))


# First get binary variables and test classification based on survey data.
# Only include variables that have at least 10% of the infrequent category.
# Some of these were not collected as binary variables, but we binarize by
# calling anything above the minimum value a positive outcome.


bvardata=numpy.array(demogdata)
sdata_orig=numpy.array(behavdata).copy()

# just impute randomly here, since we don't have a way to work our
# imputation into the CV loop

#nanvals=numpy.where(numpy.isnan(sdata_orig))
sdata=sdata_orig
#sdata[nanvals]=numpy.random.randn(nanvals[0].shape[0])*0.1
results=pandas.DataFrame(columns=['variable','fa_ctr','trainf1','testf1'])

clf_params={}

ctr=0

accuracy={}
imp={}
null_accuracy={}
cutoff={}
importances={}

for varname in binary_vars:
    print(varname)

    y=numpy.array(demogdata[varname].copy())
    if numpy.var(y)==0:
        print('skipping %s: no variance'%varname)
        print('')
        continue

    if shuffle:
        numpy.random.shuffle(y)
        print('y shuffled')

    forest = ExtraTreesClassifier(n_estimators=250)
    outer_cv = StratifiedShuffleSplit(n_splits=5)
    estimators = [('imputer',imputer),('reduce_dim', PCA()), ('clf',forest)]
    pipeline=Pipeline(steps=estimators)
    params=dict(reduce_dim=[None, FactorAnalysis(10), FactorAnalysis(25)])

    grid_search = GridSearchCV(pipeline, param_grid=params)
    cvscores=[]
    nruns=10
    nruns_shuf=250
    for i in range(nruns):
        nested_score = cross_val_score(grid_search, X=sdata, y=y,
                            cv=outer_cv,scoring='roc_auc')
        cvscores.append(nested_score.mean())
    accuracy[varname]=cvscores
    print(numpy.mean(accuracy[varname]),
            numpy.min(accuracy[varname]),numpy.max(accuracy[varname]))

    cvscores=[]
    # do shuffling to get null dist
    for i in range(nruns_shuf):
        y_shuf=y.copy()
        numpy.random.shuffle(y_shuf)
        assert not all(y_shuf==y)
        nested_score = cross_val_score(pipeline, X=sdata, y=y_shuf,
                            cv=outer_cv,scoring='roc_auc')
        cvscores.append(nested_score.mean())
    null_accuracy[varname]=cvscores
    cutoff[varname]=scipy.stats.scoreatpercentile(cvscores,95)
    print('95% cutoff:',cutoff)
    assert all(y==numpy.array(demogdata[varname]))

    # fit on all data and compute importance
    forest.fit(fancyimpute.SoftImpute(verbose=False).complete(sdata), y)
    importances[varname] = forest.feature_importances_
    std = numpy.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = numpy.argsort(importances[varname])[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    for f in range(5):
        print("%s. feature %d (%f)" % (behavvars[indices[f]], indices[f],
                            importances[varname][indices[f]]))
    print('')

import pickle
pickle.dump((accuracy,importances,null_accuracy,cutoff),
        open('behavpred_pipeline_performance.pkl','wb'))

(accuracy,importances,null_accuracy,cutoff)=pickle.load(
        open('behavpred_pipeline_performance.pkl','rb'))
for varname in binary_vars:
    if not varname in importances:
        continue
    indices = numpy.argsort(importances[varname])[::-1]
    print('')
    print('Variable:',varname)
    print('Prediction accuracy (ROC AUC):',numpy.mean(accuracy[varname][:10]))
    pval=1-scipy.stats.percentileofscore(null_accuracy[varname],
                numpy.mean(accuracy[varname][:10]))/100.
    print('null mean %f, pval: %f'%(numpy.mean(null_accuracy[varname]),pval))
    if pval<0.1:
        # Print the feature ranking
        print("Feature ranking:")
        for f in range(5):
            print("%s. feature %d (%f)" % (behavvars[indices[f]], indices[f],
                        importances[varname][indices[f]]))
        print('')
