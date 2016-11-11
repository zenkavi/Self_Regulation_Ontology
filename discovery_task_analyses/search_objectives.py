"""
objective functions for search
"""

import pandas,numpy
from sklearn.linear_model import LinearRegression,Lasso,LassoCV,MultiTaskLassoCV,RandomizedLasso
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import scale
from sklearn.decomposition import FactorAnalysis


def get_subset_corr(ct,taskdata,targetdata):
    subcorr=numpy.corrcoef(scale(targetdata.values))[numpy.triu_indices(targetdata.shape[0],1)]

    tasknames=[i.split('.')[0] for i in taskdata.columns]
    tasks=list(set(tasknames))
    tasks.sort()
    chosen_vars=[]
    for i in ct:
        vars=[j for j in range(len(tasknames)) if tasknames[j].split('.')[0]==tasks[i]]
        chosen_vars+=vars
    chosen_data=taskdata.ix[:,chosen_vars].values
    chosen_data=scale(chosen_data)
    subcorr_subset=numpy.corrcoef(chosen_data)[numpy.triu_indices(chosen_data.shape[0],1)]
    return(numpy.corrcoef(subcorr,subcorr_subset)[0,1])

def get_reconstruction_error(ct,taskdata,targetdata_orig,params):
    targetdata=targetdata_orig.copy()
    tasknames=[i.split('.')[0] for i in taskdata.columns]
    tasks=list(set(tasknames))
    tasks.sort()
    chosen_vars=[]
    varnames=taskdata.columns
    if params.verbose>2:
        print('selected tasks:',[tasks[i] for i in ct])
        print(ct)

    for i in ct:
        vars=[j for j in range(len(tasknames)) if tasknames[j].split('.')[0]==tasks[i]]
        chosen_vars+=vars
    if params.verbose>2:
        print('selected vars:',[varnames[j] for j in chosen_vars])
    # remove chosen vars from test set
    if params.remove_chosen_from_test:
        delnames=[]
        for t in chosen_vars:
            if varnames[t] in targetdata:
                del targetdata[varnames[t]]
                delnames.append(varnames[t])
        if params.verbose>2:
            print('removed %d chosen vars'%len(delnames),delnames)
    taskdata=taskdata.values
    targetdata=targetdata.values

    kf = KFold(n_splits=params.nsplits,shuffle=True)
    #subdata=data.ix[:,chosen_vars].values
    if params.clf=='kridge':
        linreg=KernelRidge(alpha=params.kridge_alpha)
    elif params.clf=='elasticnetcv':
        linreg=MultiTaskElasticNetCV()
    elif params.clf=='lasso':
        linreg=Lasso(alpha=params.lasso_alpha)
    elif params.clf=='rf':
        linreg=RandomForestRegressor()
    else:
       linreg=LinearRegression(n_jobs=params.linreg_n_jobs)

    scaler=StandardScaler()
    predacc_insample=[]
    pred=numpy.zeros(targetdata.shape)
    for train, test in kf.split(targetdata):
        taskdata_train=scaler.fit_transform(taskdata[train,:])
        taskdata_train=taskdata_train[:,chosen_vars]
        taskdata_test=scaler.transform(taskdata[test,:])
        taskdata_test=taskdata_test[:,chosen_vars]
        targetdata_train=scaler.fit_transform(targetdata[train,:])
        targetdata_test=scaler.transform(targetdata[test,:])
        linreg.fit(taskdata_train,targetdata_train)
        pred[test,:]=linreg.predict(taskdata_test)
        predacc_insample.append(numpy.corrcoef(linreg.predict(taskdata_train).ravel(),
                                targetdata_train.ravel())[0,1])
        #print(linreg.predict(taskdata_train))
    cc=numpy.corrcoef(scaler.transform(targetdata).ravel(),pred.ravel())[0,1]
    return cc #,numpy.mean(predacc_insample)


# functions for variable rather than task selection

def get_reconstruction_error_vars(chosen_vars,taskdata,targetdata,nsplits=4,clf='kridge',n_jobs=1):
    """
    get reconstruction error for chosen vars (not tasks)
    """

    kf = KFold(n_splits=nsplits,shuffle=True)
    taskdata=taskdata.values
    targetdata=targetdata.values

    if clf=='kridge':
        linreg=KernelRidge(alpha=0.5)
    elif clf=='rf':
        linreg=RandomForestRegressor()
    else:
       linreg=LinearRegression(n_jobs=n_jobs)
    scaler=StandardScaler()
    pred=numpy.zeros(targetdata.shape)
    for train, test in kf.split(targetdata):
        # fit scaler to train data and apply to test
        taskdata_train=scaler.fit_transform(taskdata[train,chosen_vars])
        taskdata_test=scaler.transform(taskdata[test,chosen_vars])
        targetdata_train=scaler.fit_transform(targetdata[train,:])
        targetdata_test=scaler.transform(targetdata[test,:])
        linreg.fit(taskdata_train,targetdata_train)
        pred[test,:]=linreg.predict(taskdata_test)
    cc=numpy.corrcoef(scaler.transform(targetdata).ravel(),pred.ravel())[0,1]
    return cc

def get_subset_corr_vars(chosen_vars,data):
    """
    get subset correlation for chosen vars (not tasks)
    """

    subcorr=numpy.corrcoef(scale(data.values))[numpy.triu_indices(data.shape[0],1)]
    chosen_data=data.ix[:,chosen_vars].values
    chosen_data=scale(chosen_data)
    subcorr_subset=numpy.corrcoef(chosen_data)[numpy.triu_indices(data.shape[0],1)]
    return(numpy.corrcoef(subcorr,subcorr_subset)[0,1])
