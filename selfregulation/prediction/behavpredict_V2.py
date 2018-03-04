"""
This code performs predictive anlaysis on the UH2 data
as specified in the pre-registration at https://osf.io/7t677/

V2 simplifies the class to make a prediction "module", rather than a self-contained
prediction analysis
"""
import datetime
from marshmallow import Schema, fields
import numpy
import os
import pickle
import random
import socket
import subprocess
import warnings

from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.model_selection import cross_val_score,StratifiedKFold,ShuffleSplit,GridSearchCV
from sklearn.metrics import roc_auc_score,r2_score,explained_variance_score,mean_absolute_error
from sklearn.linear_model import LassoCV,LinearRegression,LogisticRegressionCV,Lasso,LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import type_of_target

import fancyimpute
from imblearn.combine import SMOTETomek

from selfregulation.utils.logreg import LogReg
from selfregulation.utils.get_balanced_folds import BalancedKFold
from selfregulation.prediction.prediction_utils import get_demographic_model_type, RModel
from tikhonov.TikhonovRegression import find_tikhonov_from_covariance, TikhonovCV

def get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])

class UserSchema(Schema):
    hostname = fields.Str()
    dataset = fields.Str()
    drop_na_thresh = fields.Integer()
    n_jobs = fields.Integer()
    errors = fields.List(fields.Str())
    finished_at = fields.DateTime()
    shuffle=fields.Boolean()
    use_smote=fields.Boolean()
    smote_cutoff=fields.Float()
    shuffle=fields.Boolean()
    classifier=fields.Str()
    predictior_set=fields.Str()
    freq_threshold=fields.Integer()
    drop_threshold=fields.Integer()
    imputer=fields.Str()
    git_commit=fields.Str()

class BehavPredict:
    def __init__(self,
                 behavdata,
                 demogdata,
                 reliabilities=None,
                 output_dir='prediction_outputs',
                 outfile=None,
                 binary_classifier='rf',
                 classifier='rf',
                 verbose=False,
                 n_jobs=1,
                 categorical_vars=None,
                 n_outer_splits=8,
                 use_smote=True,
                 smote_cutoff=0.3,
                 skip_vars=[],
                 shuffle=False,
                 freq_threshold=0.04,
                 drop_threshold=0.2,
                 imputer='SoftImpute'):
        # set up arguments
        self.behavdata = behavdata
        self.demogdata = demogdata
        self.reliabilities = reliabilities
        self.git_commit = get_git_revision_short_hash().strip()
        self.hostname = socket.gethostname()
        self.verbose = verbose
        self.shuffle = shuffle
        self.classifier = classifier
        self.output_dir = output_dir
        self.outfile = outfile
        self.freq_threshold=freq_threshold
        self.skip_vars=skip_vars
        self.drop_threshold=drop_threshold
        self.n_jobs=n_jobs
        self.n_outer_splits=n_outer_splits
        self.imputer=imputer

        # define internal variables
        self.finished_at=None
        self.predictor_set=None
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
        self.varsets={}
        
        # initialize
        self.get_joint_datasets()
        self.get_demogdata_vartypes()

    def dump(self):
        schema = UserSchema()
        result = schema.dump(self)
        return result.data
    
    # data manipulation functions
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
        if self.verbose:
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
    
    def filter_by_reliability(self, threshold=0.25, verbose=False):
        if verbose:
            print('filtering X variables by reliability > ',threshold)
        if self.reliabilities is None:
            print('No reliabilities found - must pass a reliabilities dataframe!')
        orig_shape=len(self.behavdata.columns)
        for v in self.behavdata.columns:
            if v in ['Age','Sex']:
                continue
            try:
                reliability=self.reliabilities.loc[v]
            except KeyError:
                print('key', v,'not in ICC data frame - leaving in the list for now')
                continue
            if reliability < threshold:
                del self.behavdata[v]
                if verbose:
                    print('removing', v, reliability)
        new_shape=len(self.behavdata.columns)
        if verbose:
            print('removed %d columns'%int(orig_shape - new_shape))
            
    def get_demogdata_vartypes(self):
        # for each variable, get info on how it is distributed
        model_types = get_demographic_model_type(self.demogdata)
        self.data_models = {k:v for i, (k,v) in model_types.iterrows()}

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

    
    # linear models without cross validation for testing
    def run_lm(self,v,nlambda=100):
        """
        compute in-sample r^2/auroc
        """
        imputer=eval('fancyimpute.%s'%self.imputer)

        if self.data_models[v]=='binary':
            return self.run_lm_binary(v,imputer)
        else:
            return self.run_lm_regression(v,imputer,nlambda)

    def run_lm_binary(self,v,imputer):
        if self.binary_classifier=='rf':
            self.binary_clf=ExtraTreesClassifier()
        elif self.binary_classifier=='lasso':
            if self.lambda_optim is not None:
                if self.lambda_optim[0]==0:
                    # sklearn uses different coding - 0 will break it
                    self.binary_clf=LogReg()

                else:
                    if self.verbose:
                        print('using lambda_optim:',self.lambda_optim[0])
                    self.binary_clf=LogisticRegression(C=self.lambda_optim[0],penalty='l1',solver='liblinear')
            else:
                self.binary_clf=LogisticRegressionCV(Cs=100,penalty='l1',solver='liblinear')
        else:
            self.binary_clf = self.binary_classifier

        Ydata=self.demogdata[v].dropna().copy()
        idx=Ydata.index
        Xdata=self.behavdata.loc[idx,:].copy()
        Ydata=Ydata.values
        scale=StandardScaler()
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
        self.binary_clf.fit(Xdata,Ydata)
        self.pred=self.binary_clf.predict(Xdata)
        if numpy.var(self.pred)==0:
            if self.verbose:
                print('zero variance in predictions')
            scores=[numpy.nan]
        else:
            scores=[roc_auc_score(Ydata,self.pred)]

        if hasattr(self.binary_clf,'feature_importances_'):  # for random forest
            importances=self.binary_clf.feature_importances_
        elif hasattr(self.binary_clf,'coef_'):  # for lasso
            importances=self.binary_clf.coef_
        if self.verbose:
            print('overfit mean accuracy = %0.3f'%scores[0])
        return scores,importances

    def run_lm_regression(self,v,imputer,nlambda=100):
        if self.classifier=='rf':
            clf=ExtraTreesRegressor()
        elif self.classifier=='lasso':
            if not self.data_models[v]=='gaussian':
                if self.verbose:
                    print('using R to fit model...')
                if self.lambda_optim is not None:
                    if len(self.lambda_optim)>2:
                        lambda_optim=numpy.hstack(self.lambda_optim).T.mean(0)
                    else:
                        lambda_optim=self.lambda_optim
                    if self.verbose:
                        print('using optimal lambdas from CV:')
                        print(lambda_optim)
                else:
                    lambda_optim=None
                clf=RModel(self.data_models[v],self.verbose,
                                self.n_jobs,
                                lambda_preset=lambda_optim)

            else:
                if self.lambda_optim is not None:
                    if self.lambda_optim[0]==0:
                        clf=LinearRegression()
                    else:
                        clf=Lasso(alpha=self.lambda_optim)
                else:
                    clf=LassoCV(max_iter=5000)
        elif self.classifier=='tikhonov':
            clf = TikhonovCV()
        else:
            clf = self.classifier
        # run regression
        Ydata=self.demogdata[v].dropna().copy()
        Xdata=self.behavdata.loc[Ydata.index,:].copy()
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
        # impute and scale data
        if numpy.sum(numpy.isnan(Xdata))>0:
            Xdata=imputer().complete(Xdata)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Data with input dtype int64 was converted to float64 by StandardScaler.")
            Xdata=scale.fit_transform(Xdata)

        # run regression
        if self.classifier == 'tikhonov':
            L = find_tikhonov_from_covariance(numpy.corrcoef(Xdata.T))
            clf.fit(Xdata,Ydata,L=L)
        else:
            clf.fit(Xdata,Ydata)
        self.pred=clf.predict(Xdata)

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
            print('overfit scores:',scores)
        try:
            imp=numpy.vstack(importances)
        except:
            imp=None
        return scores,imp
    
    # linear models with cross validation
    def run_crossvalidation(self,v,outer_cv=None,
                            nlambda=100):
        """
        v is the variable on which to run crosvalidation
        """
        imputer=eval('fancyimpute.%s'%self.imputer)

        if self.data_models[v]=='binary':
            return self.run_crossvalidation_binary(v,imputer,outer_cv)
        else:
            return self.run_crossvalidation_regression(v,imputer,outer_cv,
                                                nlambda)


    def run_crossvalidation_binary(self,v,imputer,outer_cv=None):
        """
        run CV for binary data
        """


        if self.verbose:
            print('classifying',v,numpy.mean(self.demogdata[v]))
            print('using classifier:',self.classifier)
        if self.binary_classifier=='rf':
            clf=ExtraTreesClassifier()
        elif self.binary_classifier=='lasso':
            clf=LogisticRegressionCV(Cs=100,penalty='l1',solver='liblinear')
        else:
            clf = self.binary_classifier
        # set up crossvalidation
        if not outer_cv:
            outer_cv=StratifiedKFold(n_splits=self.n_outer_splits,shuffle=True)
        # set up data
        Ydata=self.demogdata[v].dropna().copy()
        Xdata=self.behavdata.loc[Ydata.index,:].copy()

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
        # run cross validation loops
        for train,test in outer_cv.split(Xdata,Ydata):
            Xtrain=Xdata[train,:]
            Xtest=Xdata[test,:]
            if numpy.sum(numpy.isnan(Xtrain))>0:
                Xtrain=imputer().complete(Xdata[train,:])
            if numpy.sum(numpy.isnan(Xtest))>0:
                Xtest=imputer().complete(Xdata[test,:])
            # scale data
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Data with input dtype int64 was converted to float64 by StandardScaler.")
                Xtrain=scale.fit_transform(Xtrain)
                Xtest=scale.transform(Xtest)
            Ytrain=Ydata[train]
            # correct for imbalanced Y
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
        # determine feature importance by fitting on the whole dataset
        clf.fit(Xdata, Ydata)
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

    def run_crossvalidation_regression(self,v,imputer,outer_cv=None,
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
                if self.verbose:
                    print('using R to fit model...')
                clf=RModel(self.data_models[v],
                           self.verbose,
                           self.n_jobs,
                           nlambda=nlambda)

            else:
                clf=LassoCV(max_iter=5000)
        elif self.classifier=='tikhonov':
            clf = TikhonovCV()
        else:
            clf = self.classifier
        # set up crossvalidation
        # set up data
        Ydata=self.demogdata[v].dropna().copy()
        Xdata=self.behavdata.loc[Ydata.index,:].copy()
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
        # set up CV object
        if not outer_cv:
            if type_of_target(Ydata) == 'continuous':
                outer_cv=BalancedKFold(nfolds=self.n_outer_splits)
            else:
                outer_cv=StratifiedKFold(n_splits=self.n_outer_splits,shuffle=True)
        # run cross validation loops
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
            if self.classifier == 'tikhonov':
                L = find_tikhonov_from_covariance(numpy.corrcoef(Xtrain.T))
                clf.fit(Xtrain,Ytrain,L=L)
            else:
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
        # determine feature importance by fitting on the whole dataset
        clf.fit(Xdata, Ydata)
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
    
    def write_data(self, vars):
        shuffle_flag='shuffle_' if self.shuffle else ''
        h='%08x'%random.getrandbits(32)
        if self.outfile is None:
            outfile='prediction_%s_%s_%s%s%s.pkl' % (self.predictor_set,
                                                     self.classifier,
                                                     shuffle_flag,
                                                     h)
        else:
            outfile = self.outfile.rstrip('.pkl')
            outfile = outfile + shuffle_flag + '_%s.pkl' % h
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
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
            try:
                info['data'][v]['scores_cv']=self.scores[v]
                info['data'][v]['importances']=self.importances[v]
            except KeyError:
                if self.verbose:
                    print("no cv scores for",v)
                continue
            try:
                info['data'][v]['scores_insample']=self.scores_insample[v]
                info['data'][v]['scores_insample_unbiased']=self.scores_insample_unbiased[v]
            except KeyError:
                if self.verbose:
                    print('no insample scores for',v)
                pass
        pickle.dump(info,
                    open(os.path.join(self.output_dir,outfile),'wb'))
        
    def print_importances(self,v,nfeatures=3):
            print('Most important predictors for',v)
            meanimp=numpy.mean(self.importances[v],0)
            meanimp_sortidx=numpy.argsort(meanimp)
            for i in meanimp_sortidx[-1:-1*(nfeatures+1):-1]:
                print(self.behavdata.columns[i],meanimp[i])
