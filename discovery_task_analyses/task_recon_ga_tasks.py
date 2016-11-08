"""
assess the ability to reconstruct data from a subset of variables
choose variables instead of tasks
"""


import os,glob,sys,itertools,time,pickle
import numpy,pandas
import json
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import FactorAnalysis
import fancyimpute

from genetic_algorithm import get_initial_population_tasks,get_population_fitness_tasks
from genetic_algorithm import select_parents_tasks,crossover_tasks,immigrate_tasks

# this is kludgey but it works
sys.path.append('../utils')
from utils import get_info,get_behav_data
#from r_to_py_utils import missForest

if __name__ == '__main__':

    if len(sys.argv)>1:
       clf=sys.argv[1]
       print('using ',clf)
    else:
        clf='linear'

    if len(sys.argv)>2:
       nsplits=int(sys.argv[2])
       print('nsplits=',nsplits)
    else:
        nsplits=8

    target='all' # or 'all' or 'task'
    objective_weights=[1.0,0.0] # weights for reconstruction and correlation

    assert target in ['all','task']

    dataset=get_info('dataset')
    print('using dataset:',dataset)
    basedir=get_info('base_directory')
    derived_dir=os.path.join(basedir,'Data/Derived_Data/%s'%dataset)
    data_dir=os.path.join(basedir,'Data/%s'%dataset)


    data=get_behav_data('Discovery_11-07-2016', file = 'taskdata_imputed.csv')
    drop_tasks=['cognitive_reflection_survey','writing_task']
    for c in data.columns:
        taskname=c.split('.')[0]
        if taskname in drop_tasks:
            print('dropping',c)
            del data[c]

    taskvars=list(data.columns)
    tasknames=[i.split('.')[0] for i in data.columns]
    tasks=list(set(tasknames))
    tasks.sort()

    if target=='all':
        print('targeting all variables')
        alldata=get_behav_data(dataset)
        surveydata=pandas.DataFrame()
        for k in alldata.columns:
            if k.find('survey')>-1:
                surveydata[k]=alldata[k]

        assert all(data.index == surveydata.index)

        alldata = surveydata.merge(data,'inner',right_index=True,left_index=True)

        taskvaridx=[i for i in range(len(alldata.columns)) if alldata.columns[i] in taskvars]

        # there are very few missing values, so just use a fast but dump imputation
        #data=missForest(alldata)[0]
        if numpy.sum(numpy.isnan(alldata.values))>0:
            data_imp=fancyimpute.SimpleFill().complete(alldata.values)
            data=pandas.DataFrame(data_imp,index=alldata.index,columns=alldata.columns)
    else:
        print('targeting task variables')
        taskvaridx=[i for i in range(data.shape[1])]


    # set up genetic algorithm
    nvars=8
    ntasks=len(tasks)

    ngen=2500
    initpopsize=500
    nselect=50
    nimmigrants=500
    nbabies=4
    mutation_rate=1/nvars
    ccmax={}
    bestp_saved={}
    bestctr=0
    clf='kridge'

    start_time = time.time()

    population=get_initial_population_tasks(initpopsize,nvars,data,ntasks)

    for generation in range(ngen):
        roundstart=time.time()
        population,cc,maxcc=select_parents_tasks(population,data,nselect,clf,
                                                obj_weight=objective_weights)
        ccmax[generation]=[numpy.max(cc)]+maxcc
        bestp=population[numpy.where(cc==numpy.max(cc))[0][0]]
        bestp.sort()
        bestp_saved[generation]=bestp
        if generation>1 and bestp_saved[generation]==bestp_saved[generation-1]:
            bestctr+=1
        else:
            bestctr=0
        #for i in bestp:
        #    print(i,data.columns[i])
        print(bestp,bestctr)
        if bestctr>10:
            break
        population=crossover_tasks(population,data,ntasks,nbabies=nbabies)
        population=immigrate_tasks(population,nimmigrants,nvars,data,ntasks)
        print(generation,ccmax[generation])
        print('Time elapsed (secs):', time.time()-roundstart)


    bestp.sort()
    print('best set')
    for i in bestp:
        print(i,tasks[i])

    print('Time elapsed (secs):', time.time()-start_time)

    saved_data={}
    saved_data['bestp_saved']=bestp_saved
    saved_data['ccmax']=ccmax
    saved_data['population']=population
    saved_data['dataset']=dataset

    pickle.dump(saved_data,open("ga_results_tasks_%s_%s.pkl"%(clf,target),'wb'))

    #numpy.save('reconstruction_%s_cc.npy'%clf,cc)
    #numpy.save('reconstruction_%s_sse.npy'%clf,sse)
