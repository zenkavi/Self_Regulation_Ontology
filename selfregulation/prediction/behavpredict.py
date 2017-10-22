"""
This code performs predictive anlaysis on the UH2 data
as specified in the pre-registration at https://osf.io/7t677/

TBD:
- back out demog cleaning into main package
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
from sklearn.linear_model import LassoCV,LinearRegression,LogisticRegressionCV,Lasso,LogisticRegression
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
    errors = fields.List(fields.Str())
    username = fields.Str()
    created_at = fields.DateTime()
    finished_at = fields.DateTime()
    shuffle=fields.Boolean()
    use_smote=fields.Boolean()
    smote_cutoff=fields.Float()
    shuffle=fields.Boolean()
    classifier=fields.Str()
    predictior_set=fields.Str()
    freq_threshold=fields.Integer()
    drop_threshold=fields.Integer()

class BehavPredict:
    def __init__(self,verbose=False,dataset=None,
                    use_full_dataset=True,
                    drop_na_thresh=50,
                    n_jobs=1,
                    categorical_vars=None,
                    n_outer_splits=8,
                    use_smote=True,
                    smote_cutoff=0.3,
                    baseline_vars=['Age','Sex'],
                    add_baseline_vars=True,
                    skip_vars=[],
                    shuffle=False,
                    classifier='rf',
                    output_dir='prediction_outputs',
                    freq_threshold=0.04,
                    drop_threshold=0.2):
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
        self.freq_threshold=freq_threshold
        self.skip_vars=skip_vars
        self.use_full_dataset=use_full_dataset
        self.drop_na_thresh=drop_na_thresh
        self.drop_threshold=drop_threshold
        if self.dataset=='mean':
            if self.verbose:
                print("modeling only the mean: excluding baseline vars")
            self.add_baseline_vars=False
        else:
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
        self.finished_at=None
        self.predictor_set=None
        self.demogdata=None
        self.behavdata=None
        self.dropped_na_columns=None
        self.binary_cutoffs={}
        self.scores={}
        self.importances={}
        self.scores_insample={}
        self.scores_insample_unbiased={}
        self.use_smote=use_smote
        self.smote_cutoff=smote_cutoff
        self.data_models=None
        self.pred=None
        self.reliabilities=None
        self.varsets={}
        # for debugging purposes
        self.Xdata=None
        self.Ydata=None
        self.lambda_optim=None
        self.clf=None
        self.errors={}

    def dump(self):
        schema = UserSchema()
        result = schema.dump(self)
        return result.data

    def add_varset(self,name,vars):
        self.varsets[name]=vars

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


    def get_demogdata_vartypes(self):
        # for each variable, get info on how it is distributed
        self.data_models={}
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

        elif datasubset=='task':
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

        elif datasubset=='baseline':
            self.behavdata=self.demogdata[self.baseline_vars].copy()

        elif datasubset=='mean': # model with just mean
            self.behavdata=pandas.DataFrame({'mean':numpy.ones(self.demogdata.shape[0])})
            self.behavdata.index=self.demogdata.index
        elif datasubset in self.varsets.keys():
            for v in self.behavdata.columns:
                if not v in self.varsets[datasubset]:
                    del self.behavdata[v]
                    if self.verbose>1:
                        print('dropping off-list var:',v)
            assert self.behavdata.shape[1]==len(self.varsets[datasubset])

        else:
            raise ValueError('datasubset %s is not defined'%datasubset)

        if add_baseline_vars:
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

    def remove_lowfreq_vars(self):
        if self.data_models is None:
            self.get_demogdata_vartypes()
        for v in self.demogdata.columns:
            if not self.data_models[v]=='binary':
                data=(self.demogdata[v]>0).astype('int')
            else:
                data=self.demogdata[v]
            varmean=data.dropna().mean()
            if varmean<=self.freq_threshold:
                if self.verbose:
                    print('dropping %s: freq too small (%f)'%(v,varmean))
                del self.demogdata[v]

    def binarize_ZI_demog_vars(self,vars=None,replace=False):
        """
        for zero-inflated demographic vars, create a binary version
        replace: should we replace or create a new version?
        zithresh: proportion of zeros necessary to include variable
        """
        if vars is None:
            if self.verbose:
                print('binarizing all appropriate variables')
            vars=list(self.demogdata.columns)
        elif not isinstance(vars,list):
            vars=[vars]
        print('binarizing demographic data...')
        if self.data_models is None:
            self.get_demogdata_vartypes()

        vars=[v for v in vars if not self.data_models[v]=='binary']

        newvars=[]
        for v in vars:
            m=self.demogdata[v].dropna().min()
            pct_min=numpy.mean(self.demogdata[v].dropna()==m)
            if pct_min<=self.freq_threshold or pct_min>=(1-self.freq_threshold):
                if self.verbose:
                    print('not binarizing %s: pct min too small (%f)'%(v,pct_min))
                continue

            newv=v+'.binarized'
            newvars.append(newv)

            self.binary_cutoffs[v]=[m,numpy.sum(self.demogdata[v]<=m),
                    numpy.sum(self.demogdata[v]>m)]
            self.demogdata[newv]=(self.demogdata[v]>m).astype('int')
            self.data_models[newv]='binary'
            if replace:
                del self.demogdata[v]
            if (1-pct_min)<self.drop_threshold:
                if self.verbose:
                    print('dropping %s due to too few nonzero vals'%v)
                    del self.demogdata[v]

        return newvars

    def run_lm(self,v,imputer=fancyimpute.SoftImpute,nlambda=100):
        """
        compute in-sample r^2/auroc
        """
        if self.data_models[v]=='binary':
            return self.run_lm_binary(v,imputer)
        else:
            return self.run_lm_regression(v,imputer,nlambda)

    def run_lm_binary(self,v,imputer=fancyimpute.SoftImpute):
        if self.classifier=='rf':
            clf=ExtraTreesClassifier()
        elif self.classifier=='lasso':
            if self.lambda_optim is not None:
                if self.verbose:
                    if self.lambda_optim[0]==0:
                        # sklearn uses different coding - 0 will break it
                        self.lambda_optim[0]=1
                clf=LogisticRegression(C=self.lambda_optim[0],penalty='l1',solver='liblinear')
            else:
                clf=LogisticRegressionCV(Cs=100,penalty='l1',solver='liblinear')
        else:
            raise ValueError('classifier not in approved list')

        Ydata=self.demogdata[v].dropna().copy()
        Xdata=self.behavdata.loc[Ydata.index,:].copy()
        Ydata=Ydata.values
        scale=StandardScaler()
        if self.add_baseline_vars:
            for bv in self.baseline_vars:
                Xdata[bv]=self.demogdata[bv].dropna().copy()
        if self.shuffle:
            if self.verbose:
                print('shuffling Y variable')
            numpy.random.shuffle(Ydata)
        Xdata=Xdata.values
        if numpy.sum(numpy.isnan(Xdata))>0:
            Xdata=imputer().complete(Xdata)
        scores=[]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Data with input dtype int64 was converted to float64 by StandardScaler.")
            Xdata=scale.fit_transform(Xdata)
        clf.fit(Xdata,Ydata)
        self.pred=clf.predict(Xdata)
        scores=[roc_auc_score(Ydata,self.pred)]

        if hasattr(clf,'feature_importances_'):  # for random forest
            importances=clf.feature_importances_
        elif hasattr(clf,'coef_'):  # for lasso
            importances=clf.coef_
        if self.verbose:
            print('overfit mean accuracy = %0.3f'%scores[0])
        return scores,importances

    def run_lm_regression(self,v,imputer=fancyimpute.SoftImpute,nlambda=100):
        if self.classifier=='rf':
            self.clf=ExtraTreesRegressor()
        elif self.classifier=='lasso':
            if not self.data_models[v]=='gaussian':
                print('using R to fit model...')
                if self.lambda_optim is not None:
                    if len(self.lambda_optim)>2:
                        lambda_optim=numpy.hstack(self.lambda_optim).T.mean(0)
                    else:
                        lambda_optim=self.lambda_optim
                    print('using optimal lambdas from CV:')
                    print(lambda_optim)
                else:
                    lambda_optim=None
                self.clf=prediction_utils.RModel(self.data_models[v],self.verbose,
                                            self.n_jobs,
                                            lambda_preset=lambda_optim)

            else:
                if self.lambda_optim is not None:
                    if self.lambda_optim[0]==0:
                        self.clf=LinearRegression()
                    else:
                        self.clf=Lasso(alpha=self.lambda_optim)
                else:
                    self.clf=LassoCV()
        else:
            raise ValueError('classfier not in approved list!')
        # set up crossvalidation
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
        if numpy.sum(numpy.isnan(Xdata))>0:
            Xdata=imputer().complete(Xdata)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Data with input dtype int64 was converted to float64 by StandardScaler.")
            Xdata=scale.fit_transform(Xdata)

        self.Xdata=Xdata
        self.Ydata=Ydata

        self.clf.fit(Xdata,Ydata)
        self.pred=self.clf.predict(Xdata)

        if numpy.var(self.pred)>0:
            scores=[numpy.corrcoef(Ydata,self.pred)[0,1]**2,
                    mean_absolute_error(Ydata,self.pred)]
        else:
           if self.verbose:
               print(v,'zero variance in predictions')
           scores=[numpy.nan,numpy.nan]
        if hasattr(self.clf,'feature_importances_'):  # for random forest
            importances.append(self.clf.feature_importances_)
        elif hasattr(self.clf,'coef_'):  # for lasso
            importances.append(self.clf.coef_)

        if self.verbose:
            print('overfit scores:',scores)
        try:
            imp=numpy.vstack(importances)
        except:
            imp=None
        return scores,imp

    def run_crossvalidation(self,v,outer_cv=None,
                            imputer=fancyimpute.SoftImpute,
                            nlambda=100):
        """
        v is the variable on which to run crosvalidation
        """

        if self.data_models[v]=='binary':
            return self.run_crossvalidation_binary(v,outer_cv,
                                                imputer)
        else:
            return self.run_crossvalidation_regression(v,outer_cv,
                                                imputer,nlambda)


    def run_crossvalidation_binary(self,v,outer_cv=None,
                            imputer=fancyimpute.SoftImpute):
        """
        run CV for binary data
        """


        if self.verbose:
            print('classifying',v,numpy.mean(self.demogdata[v]))
            print('using classifier:',self.classifier)
        if self.classifier=='rf':
            clf=ExtraTreesClassifier()
        elif self.classifier=='lasso':
            clf=LogisticRegressionCV(Cs=100,penalty='l1',solver='liblinear')
        else:
            raise ValueError('classifier not in approved list')
        # set up crossvalidation
        if not outer_cv:
            outer_cv=StratifiedKFold(n_splits=self.n_outer_splits,shuffle=True)
        Ydata=self.demogdata[v].dropna().copy()
        Xdata=self.behavdata.loc[Ydata.index,:].copy()
        if self.add_baseline_vars:
            for bv in self.baseline_vars:
                Xdata[bv]=self.demogdata[bv].dropna().copy()

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
        if hasattr(clf,'C_'):
            self.lambda_optim=[clf.C_[0]]
            if self.verbose:
                print('optimal lambdas:',self.lambda_optim)
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
                            imputer=fancyimpute.SoftImpute,
                            nlambda=100):
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
                clf=prediction_utils.RModel(self.data_models[v],
                                            self.verbose,
                                            self.n_jobs,
                                            nlambda=nlambda)

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
        if hasattr(clf,'lambda_optim'):
            self.lambda_optim=clf.lambda_optim
            if self.verbose:
                print('optimal lambdas:',self.lambda_optim)
        if self.verbose:
            print('scores:',scores)
        try:
            imp=numpy.vstack(importances)
        except:
            imp=None
        return scores,imp
    def write_data(self,vars):
        h='%08x'%random.getrandbits(32)
        shuffle_flag='shuffle_' if self.shuffle else ''
        outfile='prediction_%s_%s_%s%s.pkl'%(self.predictor_set,
            self.classifier,shuffle_flag,h)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        if self.verbose:
            print('saving to',os.path.join(self.output_dir,outfile))
        if not isinstance(vars,list):
            vars=[vars]

        info={}
        self.finished_at=datetime.datetime.now()
        info['info']=self.dump()
        info['info']['hash']=h
        info['data']={}
        for v in vars:
            info['data'][v]={}
            info['data'][v]['predvars']=list(self.behavdata.columns)
            info['data'][v]['scores_cv']=self.scores[v]
            info['data'][v]['importances']=self.importances[v]
            info['data'][v]['scores_insample']=self.scores_insample[v]
            info['data'][v]['scores_insample_unbiased']=self.scores_insample_unbiased[v]
        pickle.dump(info,
            open(os.path.join(self.output_dir,outfile),'wb'))
        if len(self.errors)>0:
            pickle.dump(self.errors,
                open(os.path.join(self.output_dir,'error_'+outfile),'wb'))

            return info
    def print_importances(self,v,nfeatures=3):
            print('Most important predictors for',v)
            meanimp=numpy.mean(self.importances[v],0)
            meanimp_sortidx=numpy.argsort(meanimp)
            for i in meanimp_sortidx[-1:-1*(nfeatures+1):-1]:
                print(self.behavdata.columns[i],meanimp[i])
