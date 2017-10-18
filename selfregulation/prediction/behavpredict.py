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


TBD:
- back out demog cleaning into main package
- do better job of catergorization of task/survey (holt-laury)
- clean up filtering for SMOTE
"""

import sys,os,socket,getpass
import random
import pickle
import warnings
import datetime

import numpy,pandas
import importlib

from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.model_selection import cross_val_score,StratifiedKFold,ShuffleSplit,GridSearchCV
from sklearn.metrics import roc_auc_score,r2_score,explained_variance_score,mean_absolute_error
from sklearn.linear_model import LassoCV,LinearRegression,LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

import fancyimpute
from imblearn.combine import SMOTETomek

from selfregulation.utils.utils import get_info,get_behav_data,get_demographics

from selfregulation.utils.get_balanced_folds import BalancedKFold
import selfregulation.prediction.prediction_utils as prediction_utils
importlib.reload(prediction_utils)

from marshmallow import Schema, fields


class UserSchema(Schema):
    hostname = fields.Str()
    dataset = fields.Str()
    drop_na_thresh = fields.Integer()
    n_jobs = fields.Integer()
    baseline_vars = fields.List(fields.Str())
    username = fields.Str()
    created_at = fields.DateTime()
    shuffle=fields.Boolean()
    use_smote=fields.Boolean()
    smote_cutoff=fields.Float()



class BehavPredict:
    def __init__(self,verbose=False,dataset=None,
                    use_full_dataset=True,
                    drop_na_thresh=50,
                    n_jobs=1,
                    categorical_vars=None,
                    n_outer_splits=8,
                    use_smote=True,smote_cutoff=0.3,
                    baseline_vars=['Age','Sex'],
                    add_baseline_vars=True,
                    skip_vars=[],
                    shuffle=False,
                    classifier='rf',
                    output_dir='prediction_outputs'):
        # set up arguments
        self.created_at = datetime.datetime.now()
        self.hostname= socket.gethostname()
        self.username = getpass.getuser()
        self.verbose=verbose
        self.shuffle=shuffle
        self.classifier=classifier
        self.output_dir=output_dir
        if not dataset:
            self.dataset=get_info('dataset')
        else:
            self.dataset=dataset
        self.skip_vars=skip_vars
        self.use_full_dataset=use_full_dataset
        self.drop_na_thresh=drop_na_thresh
        self.add_baseline_vars=add_baseline_vars
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
        self.n_jobs=n_jobs
        self.n_outer_splits=n_outer_splits
        self.baseline_vars=baseline_vars

        # define internal variables
        self.predictor_set=None
        self.demogdata=None
        self.behavdata=None
        self.dropped_na_columns=None
        self.binary_cutoffs={}
        self.scores={}
        self.importances={}
        self.use_smote=use_smote
        self.smote_cutoff=smote_cutoff
        self.data_models={}
        self.pred=None
        self.reliabilities=None

    def dump(self):
        schema = UserSchema()
        result = schema.dump(self)
        return result.data

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


    def get_demogdata_vartypes(self,zinf_cutoff=0.5):
        # for each variable, get info on how it is distributed
        basedir=get_info('base_directory')
        with open(os.path.join(basedir,'prediction_analyses/demographic_model_type.txt')) as f:
            lines=[i.strip().split() for i in f.readlines()]
            for l in lines:
                self.data_models[l[0]]=l[1]

    def load_reliabilities(self,infile='boot_df.csv'):
        basedir=get_info('base_directory')
        icc_boot=pandas.DataFrame.from_csv(os.path.join(basedir,
                'retest_analyses',infile))
        self.reliabilities=icc_boot.groupby('dv').mean().icc

    def load_behav_data(self,datasubset='all',
                        add_baseline_vars=False,
                        cognitive_vars=['cognitive_reflection',
                        'holt_laury']):
        self.behavdata=get_behav_data(self.dataset,
                                'meaningful_variables_clean.csv',
                                full_dataset=self.use_full_dataset)
        self.predictor_set=datasubset
        if datasubset=='survey':
            for v in self.behavdata.columns:
                dropvar=True
                if v.find('survey')>-1:
                    dropvar=False
                for cv in cognitive_vars:
                     if v.find(cv)>-1:
                         dropvar=True
                if dropvar:
                    del self.behavdata[v]
                    if self.verbose>1:
                        print('dropping non-survey var:',v)

        if datasubset=='task':
            for v in self.behavdata.columns:
                dropvar=False
                if v.find('survey')>-1:
                    dropvar=True
                for cv in cognitive_vars:
                     if v.find(cv)>-1:
                         dropvar=False
                if dropvar:
                    del self.behavdata[v]
                    if self.verbose>1:
                        print('dropping non-task var:',v)

        if datasubset=='baseline':
            self.behavdata=self.demogdata[self.baseline_vars].copy()
        elif add_baseline_vars:
            for v in self.baseline_vars:
                self.behavdata[v]=self.demogdata[v].copy()

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

    def filter_by_icc(self,icc_threshold=0.25,verbose=False):
        if verbose:
            print('filtering X variables by ICC > ',icc_threshold)
        self.load_reliabilities()
        orig_shape=len(self.behavdata.columns)
        for v in self.behavdata.columns:
            if v in ['Age','Sex']:
                continue
            try:
                icc=self.reliabilities.loc[v]
            except KeyError:
                print('key', v,'not in ICC data frame - leaving in the list for now')
                continue
            if icc<icc_threshold:
                del self.behavdata[v]
                if verbose:
                    print('removing',v,icc)
        new_shape=len(self.behavdata.columns)
        if verbose:
            print('removed %d columns'%int(orig_shape - new_shape))


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


    def run_crossvalidation(self,v,outer_cv=None,
                            imputer=fancyimpute.SimpleFill):
        """
        v is the variable on which to run crosvalidation
        """

        if self.data_models[v]=='binary':
            return self.run_crossvalidation_binary(v,outer_cv,
                                                imputer)
        else:
            return self.run_crossvalidation_regression(v,outer_cv,
                                                imputer)


    def run_crossvalidation_binary(self,v,outer_cv=None,
                            imputer=fancyimpute.SoftImpute):
        """
        run CV for binary data
        """

        if self.classifier=='rf':
            clf=ExtraTreesClassifier()
        elif self.classifier=='lasso':
            clf=LogisticRegressionCV(Cs=100)
        else:
            raise ValueError('classifier not in approved list')

        if self.verbose:
            print('classifying',v,numpy.mean(self.demogdata[v]))
            print('using classifier:',self.classifier)
        # set up crossvalidation
        if not outer_cv:
            outer_cv=StratifiedKFold(n_splits=self.n_outer_splits,shuffle=True)
        Ydata=self.demogdata[v].dropna().copy()
        Xdata=self.behavdata.loc[Ydata.index,:].copy()
        if self.add_baseline_vars:
            for v in self.baseline_vars:
                Xdata[v]=self.demogdata[v].dropna().copy()

        Ydata=Ydata.values
        if self.shuffle:
            if self.verbose:
                print('shuffling Y variable')
            numpy.random.shuffle(Ydata)
        Xdata=Xdata.values
        scores=[]
        importances=[]
        self.pred=numpy.zeros(Ydata.shape[0])
        scale=StandardScaler()
        for train,test in outer_cv.split(Xdata,Ydata):
            Xtrain=Xdata[train,:]
            Xtest=Xdata[test,:]
            if numpy.sum(numpy.isnan(Xtrain))>0:
                Xtrain=imputer().complete(Xdata[train,:])
            if numpy.sum(numpy.isnan(Xtest))>0:
                Xtest=imputer().complete(Xdata[test,:])

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Data with input dtype int64 was converted to float64 by StandardScaler.")
                Xtrain=scale.fit_transform(Xtrain)
                Xtest=scale.transform(Xtest)
            Ytrain=Ydata[train]

            if numpy.abs(numpy.mean(Ytrain)-0.5)>self.smote_cutoff and self.use_smote:
                if self.verbose>1:
                    print("using SMOTE to oversample")
                smt = SMOTETomek()
                Xtrain,Ytrain=smt.fit_sample(Xtrain.copy(),Ydata[train])
            clf.fit(Xtrain,Ytrain)
            self.pred[test]=clf.predict(Xtest)

        if numpy.var(self.pred)>0:
            scores=[roc_auc_score(Ydata,self.pred)]
        else:
           if self.verbose:
               print(v,'zero variance in predictions')
           scores=[numpy.nan]
        if hasattr(clf,'feature_importances_'):  # for random forest
            importances.append(clf.feature_importances_)
        elif hasattr(clf,'coef_'):  # for lasso
            importances.append(clf.coef_)
        if self.verbose:
            print('mean accuracy = %0.3f'%scores[0])
        try:
            imp=numpy.vstack(importances)
        except:
            imp=None
        return scores,imp

    def run_crossvalidation_regression(self,v,outer_cv=None,
                            imputer=fancyimpute.SoftImpute):
        """
        run CV for binary data
        """

        if self.verbose:
            print('%s regression on'%self.data_models[v],v,numpy.mean(self.demogdata[v]>0))
            print('using classifier:',self.classifier)
        if self.classifier=='rf':
            clf=ExtraTreesRegressor()
        elif self.classifier=='lasso':
            if not self.data_models[v]=='gaussian':
                print('using R to fit model...')
                clf=prediction_utils.RModel(self.data_models[v],self.n_jobs)
            else:
                clf=LassoCV()
        else:
            raise ValueError('classfier not in approved list!')
        # set up crossvalidation
        if not outer_cv:
            outer_cv=BalancedKFold()
        Ydata=self.demogdata[v].dropna().copy()
        Xdata=self.behavdata.loc[Ydata.index,:].copy()

        if self.add_baseline_vars:
            for v in self.baseline_vars:
                Xdata[v]=self.demogdata[v].dropna().copy()
        Ydata=Ydata.values
        if self.shuffle:
            if self.verbose:
                print('shuffling Y variable')
            numpy.random.shuffle(Ydata)
        Xdata=Xdata.values
        scores=[]
        importances=[]
        self.pred=numpy.zeros(Xdata.shape[0])
        scale=StandardScaler()
        for train,test in outer_cv.split(Xdata,Ydata):
            Xtrain=Xdata[train,:]
            Xtest=Xdata[test,:]
            if numpy.sum(numpy.isnan(Xtrain))>0:
                Xtrain=imputer().complete(Xdata[train,:])
            if numpy.sum(numpy.isnan(Xtest))>0:
                Xtest=imputer().complete(Xdata[test,:])

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Data with input dtype int64 was converted to float64 by StandardScaler.")
                Xtrain=scale.fit_transform(Xtrain)
                Xtest=scale.transform(Xtest)
            Ytrain=Ydata[train]
            clf.fit(Xtrain,Ytrain)
            p=clf.predict(Xtest)
            if len(p.shape)>1:
                p=p[:,0]
            self.pred[test]=p

        if numpy.var(self.pred)>0:
            scores=[numpy.corrcoef(Ydata,self.pred)[0,1]**2,
                    mean_absolute_error(Ydata,self.pred)]
        else:
           if self.verbose:
               print(v,'zero variance in predictions')
           scores=[numpy.nan,numpy.nan]
        if hasattr(clf,'feature_importances_'):  # for random forest
            importances.append(clf.feature_importances_)
        elif hasattr(clf,'coef_'):  # for lasso
            importances.append(clf.coef_)

        if self.verbose:
            print('scores:',scores)
        try:
            imp=numpy.vstack(importances)
        except:
            imp=None
        return scores,imp
    def write_data(self,v):
        h='%08x'%random.getrandbits(32)
        shuffle_flag='shuffle_' if self.shuffle else ''
        varflag='%s_'%v
        outfile='prediction_%s_%s_%s%s%s.pkl'%(self.predictor_set,
            self.classifier,shuffle_flag,varflag,h)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        if self.verbose:
            print('saving to',os.path.join(self.output_dir,outfile))
        info=self.dump()
        info['variable']=v
        info['predvars']=list(self.behavdata.columns)
        pickle.dump((self.scores[v],self.importances[v],info),
            open(os.path.join(self.output_dir,outfile),'wb'))
        return info
    def print_importances(self,v,nfeatures=3):
            print('Most important predictors for',v)
            meanimp=numpy.mean(self.importances[v],0)
            meanimp_sortidx=numpy.argsort(meanimp)
            for i in meanimp_sortidx[-1:-1*(nfeatures+1):-1]:
                print(self.behavdata.columns[i],meanimp[i])
