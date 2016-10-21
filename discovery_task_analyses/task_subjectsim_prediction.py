"""
assess the effect of dropping a particular task on the similarity across subjects
"""


import os,glob,sys
import numpy,pandas
import json

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

ntasks=8 # number of tasks to select - later include time

tasknums=[i for i in range(len(tasks))]

nruns=100000
cc=numpy.zeros((len(tasks),nruns))
chosen_tasks={}
for ntasks in range(2,len(tasks)):
    for x in range(nruns):
        numpy.random.shuffle(tasknums)
        chosen_vars=[]
        ct=tasknums[:ntasks]
        for i in ct:
            vars=[j for j in range(len(tasknames)) if tasknames[j].split('.')[0]==tasks[i]]
            chosen_vars+=vars
            #print([tasknames[t] for t in vars])

        chosen_data=data.ix[:,chosen_vars]
        subcorr_subset=numpy.corrcoef(chosen_data.values)[numpy.triu_indices(nsubs,1)]
        cc[ntasks,x]=numpy.corrcoef(subcorr,subcorr_subset)[0,1]
        if x>1:
            if cc[ntasks,x]>numpy.max(cc[ntasks,:(x-1)]):
                chosen_tasks[ntasks]=tasknums[:ntasks]

plt.plot(numpy.max(cc,1))
plt.show()
