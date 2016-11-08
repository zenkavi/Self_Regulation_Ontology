"""
objective functions for search
"""

import pandas,numpy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import scale
from sklearn.decomposition import FactorAnalysis

def get_reconstruction_error_vars(chosen_vars,data,nsplits=4,clf='kridge',n_jobs=1):
    """
    get reconstruction error for chosen vars (not tasks)
    """

    kf = KFold(n_splits=nsplits,shuffle=True)
    fulldata=data.values
    if clf=='kridge':
        linreg=KernelRidge(alpha=1)
    elif clf=='rf':
        linreg=RandomForestRegressor()
    else:
       linreg=LinearRegression(n_jobs=n_jobs)
    scaler=StandardScaler()
    pred=numpy.zeros(fulldata.shape)
    for train, test in kf.split(fulldata):
        # fit scaler to train data and apply to test
        fulldata_train=scaler.fit_transform(fulldata[train,:])
        fulldata_test=scaler.transform(fulldata[test,:])
        subdata_train=fulldata_train[:,chosen_vars]
        subdata_test=fulldata_test[:,chosen_vars]
        linreg.fit(subdata_train,fulldata_train)
        pred[test,:]=linreg.predict(subdata_test)
    cc=numpy.corrcoef(scaler.transform(fulldata).ravel(),pred.ravel())[0,1]
    return cc

def get_subset_corr_vars(chosen_vars,data):
    """
    get subset correlation for chosen vars (not tasks)
    """

    subcorr=numpy.corrcoef(data)[numpy.triu_indices(data.shape[0],1)]
    chosen_data=data.ix[:,chosen_vars].values
    chosen_data=scale(chosen_data)
    subcorr_subset=numpy.corrcoef(chosen_data)[numpy.triu_indices(data.shape[0],1)]
    return(numpy.corrcoef(subcorr,subcorr_subset)[0,1])

def get_subset_corr(ct,data):
    subcorr=numpy.corrcoef(data)[numpy.triu_indices(data.shape[0],1)]
    tasknames=[i.split('.')[0] for i in data.columns]
    tasks=list(set(tasknames))
    tasks.sort()
    chosen_vars=[]
    for i in ct:
        vars=[j for j in range(len(tasknames)) if tasknames[j].split('.')[0]==tasks[i]]
        chosen_vars+=vars

    chosen_data=data.ix[:,chosen_vars].values
    chosen_data=scale(chosen_data)
    subcorr_subset=numpy.corrcoef(chosen_data)[numpy.triu_indices(data.shape[0],1)]
    return(numpy.corrcoef(subcorr,subcorr_subset)[0,1])

def get_reconstruction_error(ct,data,nsplits=4,clf='kridge',n_jobs=1):
    tasknames=[i.split('.')[0] for i in data.columns]
    tasks=list(set(tasknames))
    tasks.sort()
    chosen_vars=[]
    #print(ct,tasks,tasknames)
    for i in ct:
        vars=[j for j in range(len(tasknames)) if tasknames[j].split('.')[0]==tasks[i]]
        chosen_vars+=vars
    kf = KFold(n_splits=nsplits,shuffle=True)
    fulldata=data.values
    #subdata=data.ix[:,chosen_vars].values
    if clf=='kridge':
        linreg=KernelRidge(alpha=1)
    elif clf=='rf':
        linreg=RandomForestRegressor()
    else:
       linreg=LinearRegression(n_jobs=n_jobs)
    scaler=StandardScaler()
    pred=numpy.zeros(fulldata.shape)
    for train, test in kf.split(fulldata):
        #fulldata_train=fulldata[train,:]
        #fulldata_test=fulldata[test,:]
        # fit scaler to train data and apply to test
        fulldata_train=scaler.fit_transform(fulldata[train,:])
        fulldata_test=scaler.transform(fulldata[test,:])
        subdata_train=fulldata_train[:,chosen_vars]
        subdata_test=fulldata_test[:,chosen_vars]
        linreg.fit(subdata_train,fulldata_train)
        pred[test,:]=linreg.predict(subdata_test)
    cc=numpy.corrcoef(scaler.transform(fulldata).ravel(),pred.ravel())[0,1]
    return cc
