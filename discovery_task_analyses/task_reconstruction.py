"""
assess the ability to reconstruct data from a subset of variables
"""


import os,glob,sys,itertools
import numpy,pandas
import json
from joblib import Parallel, delayed
import multiprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

num_cores = multiprocessing.cpu_count()
print('using %d cores'%num_cores)

from sklearn.preprocessing import scale
from sklearn.decomposition import FactorAnalysis
from sklearn import cross_validation

# this is kludgey but it works
sys.path.append('../utils')
from utils import get_info,get_behav_data

dataset=get_info('dataset')
print('using dataset:',dataset)
basedir=get_info('base_directory')
derived_dir=os.path.join(basedir,'Data/Derived_Data/%s'%dataset)


data=pandas.read_csv(os.path.join(derived_dir,'taskdata_clean_cutoff3.00IQR_imputed.csv'))

cdata=scale(data.values)
nsubs=data.shape[0]
subcorr=numpy.corrcoef(cdata)[numpy.triu_indices(nsubs,1)]

# get task names and indicator
tasknames=[i.split('.')[0] for i in data.columns]
tasks=list(set(tasknames))
tasks.sort()

ntasks=8 # number of tasks to select - later include time

tasknums=[i for i in range(len(tasks))]

allcombs=[i for i in itertools.combinations(range(32),8)]

cc=numpy.zeros(len(allcombs))
chosen_tasks={}

def get_reconstruction_error(x,ct,data):
    tasknames=[i.split('.')[0] for i in data.columns]
    tasks=list(set(tasknames))
    tasks.sort()
    chosen_vars=[]
    for i in ct:
        vars=[j for j in range(len(tasknames)) if tasknames[j].split('.')[0]==tasks[i]]
        chosen_vars+=vars
    kf = KFold(n_splits=4,shuffle=True)
    fulldata=data.values
    #subdata=data.ix[:,chosen_vars].values
    linreg=LinearRegression()
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
    return(numpy.corrcoef(fulldata.ravel(),pred.ravel())[0,1])

use_parallel=True
if use_parallel:
    cc = Parallel(n_jobs=num_cores)(delayed(get_reconstruction_error)(x,ct,data) for x,ct in enumerate(allcombs))

else:
    for x,ct in enumerate(allcombs):
        cc[x]=get_reconstruction_error(x,ct,data)
        if x>4:
            break

numpy.save('reconstruction_cc.npy',cc)
