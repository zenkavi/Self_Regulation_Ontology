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
from utils import get_info,get_behav_data,get_demographics
#from r_to_py_utils import missForest


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

targets=['survey','demog','task'] #'survey','task'

objective_weights=[0.5,0.5] # weights for reconstruction and correlation

dataset=get_info('dataset')
print('using dataset:',dataset)
basedir=get_info('base_directory')
derived_dir=os.path.join(basedir,'Data/Derived_Data/%s'%dataset)
data_dir=os.path.join(basedir,'Data/%s'%dataset)


taskdata=get_behav_data('Discovery_11-07-2016', file = 'taskdata_imputed.csv')
drop_tasks=['cognitive_reflection_survey','writing_task']
for c in taskdata.columns:
    taskname=c.split('.')[0]
    if taskname in drop_tasks:
        print('dropping',c)
        del taskdata[c]
print('taskdata: %d variables'%taskdata.shape[1])
taskvars=list(taskdata.columns)
tasknames=[i.split('.')[0] for i in taskdata.columns]
tasks=list(set(tasknames))
tasks.sort()

if 'task' in targets:
    targetdata=get_behav_data('Discovery_11-07-2016', file = 'taskdata_imputed.csv')
    assert all(taskdata.index == targetdata.index)
    print('target: task, %d variables'%taskdata.shape[1])
else:
    targetdata=None

if 'survey' in targets:
    alldata=get_behav_data(dataset)
    surveydata=pandas.DataFrame()
    for k in alldata.columns:
        if k.find('survey')>-1:
            surveydata[k]=alldata[k]
    print('target: survey, %d variables'%surveydata.shape[1])
    print('%d missing values'%numpy.sum(numpy.isnan(surveydata.values)))
    if not targetdata is None:
        assert all(taskdata.index == surveydata.index)
        targetdata = surveydata.merge(targetdata,'inner',right_index=True,left_index=True)
    else:
        targetdata=surveydata

if 'demog' in targets:
    demogvars=['BMI','Age','Sex','RetirementAccount','ChildrenNumber',
                'CreditCardDebt','TrafficTicketsLastYearCount',
                'TrafficAccidentsLifeCount','ArrestedChargedLifeCount',
                'LifetimeSmoke100Cigs','AlcoholHowManyDrinksDay',
                'CannabisPast6Months','Nervous',
                'Hopeless', 'RestlessFidgety', 'Depressed',
                'EverythingIsEffort','Worthless']
    #demogvars=['Age','BMI']
    demogdata=get_demographics(dataset,var_subset=demogvars)
    print('target: demog, %d variables'%demogdata.shape[1])
    print('%d missing values'%numpy.sum(numpy.isnan(demogdata.values)))
    if not targetdata is None:
        assert all(taskdata.index == demogdata.index)
        targetdata = demogdata.merge(targetdata,'inner',right_index=True,left_index=True)
    else:
        targetdata=demogdata
# there are very few missing values, so just use a fast but dumb imputation here
if numpy.sum(numpy.isnan(targetdata.values))>0:
    targetdata_imp=fancyimpute.SimpleFill().complete(targetdata.values)
    targetdata=pandas.DataFrame(targetdata_imp,index=targetdata.index,columns=targetdata.columns)

print('using targets:',targets)
print('%d variables in taskdata'%taskdata.shape[1])
print('%d variables in targetdata'%targetdata.shape[1])


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
clf='lasso'

start_time = time.time()

population=get_initial_population_tasks(initpopsize,nvars,ntasks)

for generation in range(ngen):
    roundstart=time.time()
    population,cc,maxcc=select_parents_tasks(population,taskdata,targetdata,nselect,clf,
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
    population=crossover_tasks(population,ntasks,nbabies=nbabies)
    population=immigrate_tasks(population,nimmigrants,nvars,ntasks)
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

pickle.dump(saved_data,open("ga_results_tasks_%s_%s.pkl"%(clf,'-'.join(targets)),'wb'))

#numpy.save('reconstruction_%s_cc.npy'%clf,cc)
#numpy.save('reconstruction_%s_sse.npy'%clf,sse)
