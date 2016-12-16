"""
This code performs predictive anlaysis on the UH2 data
as specified in the pre-registration at https://osf.io/7t677/

This is a refactor of demographic_feature_importance_behav.py
that reimplements it as a class

#NOTE: differences from the methods proposed in the pre-registration:

We had proposed to use a RandomForest classifier, under the assumption
that it would perform well with the larger dataset.  However, initial
testing found that it performed poorly on variables with minimum class
frequency of 0.25 or less, often returning predictions with no variance.
Because of this, we decided to try a LassoCV classifier, and found that it
was much less likely to return constant predictions.

Russ Poldrack
12/13/2016
"""

import sys,os
import random
import pickle

import numpy

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score,StratifiedKFold,ShuffleSplit,GridSearchCV
from sklearn.metrics import roc_auc_score,r2_score
from sklearn.linear_model import LassoCV
from sklearn.svm import SVC

import fancyimpute
from imblearn.combine import SMOTETomek

sys.path.append('../utils')
from utils import get_info,get_behav_data,get_demographics



class BehavPredict:
    def __init__(self,verbose=False,dataset=None,
                    use_full_dataset=True,
                    drop_na_thresh=50,
                    binary_min_proportion=0.05,
                    categorical_vars=None,n_jobs=1,
                    n_outer_splits=4,
                    use_smote=True,smote_cutoff=0.1):
        # set up arguments
        self.verbose=verbose
        if not dataset:
            self.dataset=get_info('dataset')
        else:
            self.dataset=dataset
        self.use_full_dataset=use_full_dataset
        self.drop_na_thresh=drop_na_thresh
        if not categorical_vars is None:
            self.categorical_vars=categorical_vars
        else:
            self.categorical_vars=['HispanicLatino','Race',
                                'DiseaseDiagnoses', 'DiseaseDiagnosesOther',
                                'MotivationForParticipation', 'MotivationOther',
                                'NeurologicalDiagnoses',
                                'NeurologicalDiagnosesDescribe',
                                'OtherDebtSources',
                                'OtherDrugs', 'OtherRace', 'OtherTobaccoProducts',
                                'PsychDiagnoses',
                                'PsychDiagnosesOther']
        self.binary_min_proportion=binary_min_proportion
        self.n_jobs=n_jobs
        self.n_outer_splits=n_outer_splits
        # define internal variables
        self.demogdata=None
        self.behavdata=None
        self.dropped_na_columns=None
        self.binary_cutoffs={}
        self.rocscores={}
        self.importances={}
        self.use_smote=use_smote
        self.smote_cutoff=smote_cutoff

    def load_demog_data(self,cleanup=True,binarize=False,
                        drop_categorical=True):
        self.demogdata=get_behav_data(self.dataset,'demographic_health.csv',
                        full_dataset=self.use_full_dataset)
        if cleanup:
            q=self.demogdata.query('WeightPounds<50')
            for i in q.index:
                self.demogdata.loc[i,'WeightPounds']=numpy.nan
            if self.verbose:
                print('replacing bad WeightPounds value for',q.index)
            q=self.demogdata.query('HeightInches<36')
            for i in q.index:
                self.demogdata.loc[i,'HeightInches']=numpy.nan
            if self.verbose:
                print('replacing bad HeightInches value for',q.index)
            q=self.demogdata.query('CaffienatedSodaCansPerDay<0')
            for i in q.index:
                self.demogdata.loc[i,'CaffienatedSodaCansPerDay']=numpy.nan
            if self.verbose:
                print('replacing bad CaffienatedSodaCansPerDay value for',q.index)

        self.demogdata=self.demogdata.assign(BMI=self.demogdata['WeightPounds']*0.45 / (self.demogdata['HeightInches']*0.025)**2)
        self.demogdata=self.demogdata.assign(Obese=(self.demogdata['BMI']>30).astype('int'))
        if drop_categorical:
            for v in self.categorical_vars:
                del self.demogdata[v]
                if self.verbose:
                    print('dropping categorical variable:',v)

    def load_behav_data(self,datasubset='all'):
        self.behavdata=get_behav_data(self.dataset,
                                'meaningful_variables_clean.csv',
                                full_dataset=self.use_full_dataset)
        if datasubset=='survey':
            for v in self.behavdata.columns:
                if not v.find('survey')>-1:
                    del self.behavdata[v]
                    if self.verbose>1:
                        print('dropping non-survey var:',v)
        if datasubset=='task':
            for v in self.behavdata.columns:
                if v.find('survey')>-1:
                    del self.behavdata[v]
                    if self.verbose>1:
                        print('dropping non-survey var:',v)

        if self.drop_na_thresh>0:
            na_count=numpy.sum(numpy.isnan(self.behavdata),0)

            self.dropped_na_columns=self.behavdata.columns[na_count>self.drop_na_thresh]
            for c in self.dropped_na_columns:
                if self.verbose>1:
                    print('dropping',c,numpy.sum(numpy.isnan(self.behavdata[c])))
                del self.behavdata[c]
            if self.verbose:
                print('dropping %d vars due to excessive NAs'%len(self.dropped_na_columns))
                print('%d behavioral variables remaining'%self.behavdata.shape[1])

    def get_joint_datasets(self):
        demog_index=set(self.demogdata.index)
        behav_index=set(self.behavdata.index)
        inter=list(demog_index.intersection(behav_index))
        self.demogdata=self.demogdata.loc[inter,:]
        self.behavdata=self.behavdata.loc[inter,:]
        assert all(self.demogdata.index==self.behavdata.index)
        if self.verbose:
            print('datasets successfully aligned')
            print('%d subjects in joint dataset'%self.demogdata.shape[0])

    def binarize_demog_vars(self):
        print('binarizing demographic data...')
        for v in self.demogdata.columns:
            # first check to see if it's a binary variable already:
            if len(self.demogdata[v].unique())==2:
                if self.verbose>1:
                    print('already binary:',v)
            else:
                c=numpy.percentile(self.demogdata[v].dropna(),50)
                self.binary_cutoffs[v]=[c,numpy.sum(self.demogdata[v]<=c),
                        numpy.sum(self.demogdata[v]>c)]
                if self.binary_cutoffs[v][2]/(self.binary_cutoffs[v][1]+self.binary_cutoffs[v][2])<self.binary_min_proportion:
                    if self.verbose:
                        print('dropping binary var due to low frequency:',v,
                            self.binary_cutoffs[v][2]/(self.binary_cutoffs[v][1]+self.binary_cutoffs[v][2]))
                    del self.demogdata[v]
                else:
                    self.demogdata[v]=(self.demogdata[v]>c).astype('int')

    def run_crossvalidation(self,v,clf=None,outer_cv=None,
                            imputer=fancyimpute.SimpleFill,
                            shuffle=False,scoring='roc_auc'):
        """
        v is the variable on which to run crosvalidation
        """
        if self.verbose:
            print('classifying',v,numpy.mean(self.demogdata[v]))
        if not clf:
            clf = ExtraTreesClassifier(n_estimators=250,n_jobs=self.n_jobs,
                                            class_weight='balanced')
        # set up crossvalidation
        if not outer_cv:
            outer_cv=StratifiedKFold(n_splits=self.n_outer_splits,shuffle=True)
        Ydata=self.demogdata[v].dropna().copy()
        Xdata=self.behavdata.loc[Ydata.index,:].copy()
        Ydata=Ydata.values
        if shuffle:
            if self.verbose:
                print('shuffling Y variable')
                numpy.random.shuffle(Ydata)
        Xdata=Xdata.values
        scores=[]
        importances=[]
        for train,test in outer_cv.split(Xdata,Ydata):
            Xtrain=imputer().complete(Xdata[train,:])
            Xtest=imputer().complete(Xdata[test,:])
            Ytrain=Ydata[train]
            if numpy.abs(numpy.mean(Ytrain)-0.5)>self.smote_cutoff and self.use_smote:
                if self.verbose>1:
                    print("using SMOTE to oversample")
                smt = SMOTETomek()
                Xtrain,Ytrain=smt.fit_sample(Xtrain.copy(),Ydata[train])
            clf.fit(Xtrain,Ytrain)
            pred=clf.predict(Xtest)

        if numpy.var(pred)>0:
            if scoring=='r2':
                scores=r2_score(Ydata[test],pred)
            elif scoring=='roc_auc':
                scores=roc_auc_score(Ydata[test],pred)
        else:
           if self.verbose:
               print(v,'zero variance in predictions')
           scores=numpy.nan
        if hasattr(clf,'feature_importances_'):  # for random forest
            importances.append(clf.feature_importances_)
        elif hasattr(clf,'coef_'):  # for lasso
            importances.append(clf.coef_)
        if self.verbose:
            print('mean accuracy = %0.3f'%scores)
        try:
            imp=numpy.vstack(importances)
        except:
            imp=None
        return scores,imp
