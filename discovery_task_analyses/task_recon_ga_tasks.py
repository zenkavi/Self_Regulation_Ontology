"""
assess the ability to reconstruct data from a subset of variables
choose variables instead of tasks
"""


import os,glob,sys,itertools,time,pickle
import numpy,pandas
import json
import binascii

from genetic_algorithm import get_initial_population_tasks,get_population_fitness_tasks
from genetic_algorithm import select_parents_tasks,crossover_tasks,immigrate_tasks
from genetic_algorithm import get_taskdata,load_targetdata,impute_targetdata

# this is kludgey but it works
sys.path.append('../utils')
from utils import get_info,get_behav_data,get_demographics
#from r_to_py_utils import missForest

# set up variables

class Params:
    def __init__(self):
        self.targets=['survey','demog','task']  # targets for reconstruction and correlation
        self.objective_weights=[1,0] # weights for reconstruction and correlation respectively
        self.nvars=8  # number of selected tasks
        self.ngen=2500  # maximum number of GA generations
        self.initpopsize=500  # initial population size
        self.nselect=50  #number selected to survive at each generation
        self.nimmigrants=500   # number of new immigrants in each generation
        self.nbabies=4    # number of offspring of each survivor
        self.mutation_rate=1/self.nvars   # mutation rate for offspring
        self.convergence_threshold=2  # number of stable generations for convergence
        self.clf='lasso'
        self.nsplits=8
        self.dataset=get_info('dataset')
        self.basedir=get_info('base_directory')


params=Params()

# load info
print('using dataset:',params.dataset)
derived_dir=os.path.join(params.basedir,'Data/Derived_Data/%s'%params.dataset)
data_dir=os.path.join(params.basedir,'Data/%s'%params.dataset)

taskdata,taskvars,tasks=get_taskdata(params.dataset)
ntasks=len(tasks)
targetdata=load_targetdata(params.dataset,params.targets,taskdata)
targetdata=impute_targetdata(targetdata)

print('using targets:',params.targets)
print('%d variables in taskdata'%taskdata.shape[1])
print('%d variables in targetdata'%targetdata.shape[1])

start_time = time.time()
ccmax={}
bestp_saved={}
bestctr=0

# get initial population
population=get_initial_population_tasks(params.initpopsize,params.nvars,ntasks)

# perform selection for maximum of ngen generations
for generation in range(params.ngen):
    roundstart=time.time()
    # cc here is the average z-score for the mulitobjective
    # maxcc contains the max for each of the individual objectives
    population,cc,maxcc=select_parents_tasks(population,taskdata,targetdata,params.nselect,params.clf,
                                            obj_weight=params.objective_weights)
    ccmax[generation]=[numpy.max(cc)]+maxcc

    # store the best scoring set of tasks and count to see if we have
    # exceeded convergence criterion
    bestp=population[numpy.where(cc==numpy.max(cc))[0][0]]
    bestp.sort()
    bestp_saved[generation]=bestp
    if generation>1 and bestp_saved[generation]==bestp_saved[generation-1]:
        bestctr+=1
    else:
        bestctr=0
    print('best set:',bestp,bestctr)
    if bestctr>params.convergence_threshold:
        break

    # crossover and immigrate
    population=crossover_tasks(population,ntasks,nbabies=params.nbabies)
    population=immigrate_tasks(population,params.nimmigrants,params.nvars,ntasks)

    print('gen ',generation,'(Z,recon,subcorr):',ccmax[generation])
    print('Time elapsed (secs):', time.time()-roundstart)

# print best outcome
bestp.sort()
print('best set')
for i in bestp:
    print(i,tasks[i])

print('Time elapsed (secs):', time.time()-start_time)

saved_data={}
saved_data['bestp_saved']=bestp_saved
saved_data['ccmax']=ccmax
saved_data['population']=population
saved_data['start_time']=start_time
saved_data['params']=params

# add a random hash to the file name so that we can run it multiple times
# and it will save to different files
hash=binascii.hexlify(os.urandom(4)).decode('utf-8')

pickle.dump(saved_data,open("ga_results_tasks_%s_%s_%s.pkl"%(clf,'-'.join(targets),hash),'wb'))
