"""
assess the ability to reconstruct data from a subset of variables
choose variables instead of tasks
"""


import os,glob,sys,itertools,time,pickle
import numpy,pandas
import json
import binascii
import fancyimpute
from joblib import Parallel, delayed
import multiprocessing
from sklearn.preprocessing import scale

from search_objectives import get_reconstruction_error,get_subset_corr

# this is kludgey but it works
sys.path.append('../utils')
from utils import get_info,get_behav_data,get_demographics
#from r_to_py_utils import missForest

# set up variables

class GASearchParams:
    def __init__(self):
        self.targets=['survey','demog','task']  # targets for reconstruction and correlation
        self.objective_weights=[0,1] # weights for reconstruction and correlation respectively
        assert numpy.sum(self.objective_weights)==1
        self.nvars=8  # number of selected tasks
        self.ngen=2500  # maximum number of GA generations
        self.initpopsize=500  # initial population size
        self.nselect=50  #number selected to survive at each generation
        self.nimmigrants=500   # number of new immigrants in each generation
        self.nbabies=4    # number of offspring of each survivor
        self.mutation_rate=1/self.nvars   # mutation rate for offspring
        self.convergence_threshold=2  # number of stable generations for convergence
        self.clf='lasso'
        self.num_cores=1
        self.nsplits=8
        self.dataset=get_info('dataset')
        self.basedir=get_info('base_directory')
        self.ntasks=None
        self.n_jobs=1
        self.remove_chosen_from_test=True
        self.verbose=1  # minimal level of verbosity
        self.lasso_alpha=0.1
        self.linreg_n_jobs=-1
        self.derived_dir=os.path.join(self.basedir,'Data/Derived_Data/%s'%self.dataset)
        self.data_dir=os.path.join(self.basedir,'Data/%s'%self.dataset)
        self.taskdatafile= 'taskdata_imputed.csv'
        self.drop_tasks=['cognitive_reflection_survey','writing_task']
        if self.verbose>0:
            print('using dataset:',self.dataset)
        self.start_time = None
        self.demogvars=['BMI','Age','Sex','RetirementAccount','ChildrenNumber',
                        'CreditCardDebt','TrafficTicketsLastYearCount',
                        'TrafficAccidentsLifeCount','ArrestedChargedLifeCount',
                        'LifetimeSmoke100Cigs','AlcoholHowManyDrinksDay',
                        'CannabisPast6Months','Nervous',
                        'Hopeless', 'RestlessFidgety', 'Depressed',
                        'EverythingIsEffort','Worthless']

class GASearch:

    def __init__(self):
        self.params=GASearchParams()
        self.ccmax={}
        self.cc_sorted=None
        self.cc=None
        self.bestp_saved={}
        self.bestctr=0
        self.population=[]
        self.targetdata=None
        self.__USE_MULTIPROC__=True

        if self.__USE_MULTIPROC__:
            if 'NUMCORES' in os.environ:
                self.params.num_cores=int(os.environ['NUMCORES'])
            else:
                self.params.num_cores = multiprocessing.cpu_count()
            if self.params.num_cores==0:
                self.__USE_MULTIPROC__=False
            else:
                print('multiproc: using %d cores'%self.params.num_cores)


    def get_taskdata(self):
        self.taskdata=get_behav_data(self.params.dataset,self.params.taskdatafile)
        for c in self.taskdata.columns:
            taskname=c.split('.')[0]
            if taskname in self.params.drop_tasks:
                if self.params.verbose>0:
                    print('dropping',c)
                del self.taskdata[c]
        if self.params.verbose>0:
            print('taskdata: %d variables'%self.taskdata.shape[1])
        taskvars=list(self.taskdata.columns)
        tasknames=[i.split('.')[0] for i in self.taskdata.columns]
        self.tasks=list(set(tasknames))
        self.params.ntasks=len(self.tasks)
        self.tasks.sort()


    def load_targetdata(self):

        if 'task' in self.params.targets:
            self.targetdata=get_behav_data(self.params.dataset,self.params.taskdatafile )
            assert all(self.taskdata.index == self.targetdata.index)
            if self.params.verbose>0:
                print('target: task, %d variables'%self.taskdata.shape[1])
                print('%d missing values'%numpy.sum(numpy.isnan(self.taskdata.values)))

        if 'survey' in self.params.targets:
            alldata=get_behav_data(self.params.dataset)
            self.surveydata=pandas.DataFrame()
            for k in alldata.columns:
                if k.find('survey')>-1:
                    self.surveydata[k]=alldata[k]
            if self.params.verbose>0:
                print('target: survey, %d variables'%self.surveydata.shape[1])
                print('%d missing values'%numpy.sum(numpy.isnan(self.surveydata.values)))
            if not self.targetdata is None:
                assert all(self.taskdata.index == self.surveydata.index)
                self.targetdata = self.surveydata.merge(self.targetdata,'inner',right_index=True,left_index=True)
            else:
                self.targetdata=self.surveydata

        if 'demog' in self.params.targets:
            self.demogdata=get_demographics(self.params.dataset,var_subset=self.params.demogvars)
            if self.params.verbose>0:
                print('target: demog, %d variables'%self.demogdata.shape[1])
                print('%d missing values'%numpy.sum(numpy.isnan(self.demogdata.values)))
            if not self.targetdata is None:
                assert all(self.taskdata.index == self.demogdata.index)
                self.targetdata = self.demogdata.merge(self.targetdata,'inner',right_index=True,left_index=True)
            else:
                self.targetdata=self.demogdata

    def impute_targetdata(self):
        """
        there are very few missing values, so just use a fast but dumb imputation here
        """
        if numpy.sum(numpy.isnan(self.targetdata.values))>0:
            targetdata_imp=fancyimpute.SimpleFill().complete(self.targetdata.values)
            self.targetdata=pandas.DataFrame(targetdata_imp,
                                index=self.targetdata.index,
                                columns=self.targetdata.columns)


    def get_initial_population_tasks(self):
        idx=[i for i in range(self.params.ntasks)]
        for i in range(self.params.initpopsize):
            numpy.random.shuffle(idx)
            self.population.append(idx[:self.params.nvars])

    def select_parents_tasks(self):
        maxcc=self.get_population_fitness_tasks()
        idx=numpy.argsort(self.cc)[::-1]
        self.population=[self.population[i] for i in idx[:self.params.nselect]]
        self.cc_sorted=[self.cc[i] for i in idx[:self.params.nselect]]
        return maxcc

    def get_population_fitness_tasks(self):
        # first get cc for each item in population
        cc_recon=numpy.zeros(len(self.population))
        if self.params.objective_weights[0]>0:
            if self.__USE_MULTIPROC__:
                cc_recon=Parallel(n_jobs=self.params.num_cores)(delayed(get_reconstruction_error)(ct,self.taskdata,self.targetdata,self.params) for ct in self.population)
            else:
                cc_recon=[get_reconstruction_error(ct,self.taskdata,self.targetdata,self.params) for ct in pop]
        else:
            cc_recon=[0]
        if self.params.objective_weights[1]>0:
            if self.__USE_MULTIPROC__:
                cc_subsim=Parallel(n_jobs=self.params.num_cores)(delayed(get_subset_corr)(ct,self.taskdata,self.targetdata) for ct in self.population)
            else:
                cc_subsim=[get_subset_corr(ct,self.taskdata,self.targetdata) for ct in self.population]
        else:
            cc_subsim=[0]
        maxcc=[numpy.max(cc_recon),numpy.max(cc_subsim)]
        cc_recon=scale(cc_recon)
        cc_subsim=scale(cc_subsim)
        try:
            print('corr recon-subsim:',numpy.corrcoef(cc_recon,cc_subsim)[0,1])
        except:
            pass
        self.cc=cc_recon*self.params.objective_weights[0] + cc_subsim*self.params.objective_weights[1]
        return maxcc


    def crossover_tasks(self):
        if self.params.mutation_rate is None:
            self.params.mutation_rate=1/len(self.population[0])
        # assortative mating - best parents mate
        families=numpy.kron(numpy.arange(numpy.floor(len(self.population)/2)),[1,1])
        numpy.random.shuffle(families)
        for f in numpy.unique(families):
            famidx=[i for i in range(len(families)) if families[i]==f]
            if len(famidx)!=2:
                continue
            try:
                subpop=[self.population[i] for i in famidx]
            except:
                print('oops...')
                print(len(self.population))
                print(famidx)
                raise Exception('breaking')
            parents=list(numpy.unique(numpy.hstack((subpop[0],subpop[1]))))
            if len(set(parents))<len(subpop[1]):
                continue
            for b in range(self.params.nbabies):
                numpy.random.shuffle(parents)
                baby=parents[:len(subpop[1])]
                nmutations=numpy.floor(len(baby)*numpy.random.rand()*self.params.mutation_rate).astype('int')
                alts=[i for i in range(self.params.ntasks) if not i in baby]
                numpy.random.shuffle(alts)
                for m in range(nmutations):
                    mutpos=numpy.random.randint(len(baby))
                    baby[mutpos]=alts[m]
                self.population.append(baby)

    def immigrate_tasks(self):
        immigrants=[]
        idx=[i for i in range(self.params.ntasks)]
        for i in range(self.params.nimmigrants):
            numpy.random.shuffle(idx)
            immigrants.append(idx[:self.params.nvars])
        return self.population+immigrants


if __name__=='main':
    gasearch=GASearch()
    gasearch.get_taskdata()
    gasearch.load_targetdata()
    gasearch.impute_targetdata()
    print('using targets:',gasearch.params.targets)
    print('%d variables in taskdata'%gasearch.taskdata.shape[1])
    print('%d variables in targetdata'%gasearch.targetdata.shape[1])

    # get initial population
    gasearch.params.start_time=time.time()
    gasearch.get_initial_population_tasks()

    # perform selection for maximum of ngen generations

    for generation in range(gasearch.params.ngen):
        roundstart=time.time()

        maxcc=gasearch.select_parents_tasks()

        gasearch.ccmax[generation]=[numpy.max(gasearch.cc)]+maxcc

        # store the best scoring set of tasks and count to see if we have
        # exceeded convergence criterion
        assert len(gasearch.population)==len(gasearch.cc_sorted)
        bestp=gasearch.population[numpy.where(gasearch.cc_sorted==numpy.max(gasearch.cc_sorted))[0][0]]
        bestp.sort()
        gasearch.bestp_saved[generation]=bestp
        if generation>1 and gasearch.bestp_saved[generation]==gasearch.bestp_saved[generation-1]:
            bestctr+=1
        else:
            bestctr=0
        print('best set:',bestp,bestctr)
        if bestctr>gasearch.params.convergence_threshold:
            break

        # crossover and immigrate
        gasearch.crossover_tasks()
        gasearch.immigrate_tasks()

        print('gen ',generation,'(Z,recon,subcorr):',gasearch.ccmax[generation])
        print('Time elapsed (secs):', time.time()-roundstart)

    # print best outcome
    bestp.sort()
    print('best set')
    for i in bestp:
        print(i,gasearch.tasks[i])

    print('Time elapsed (secs):', time.time()-gasearch.params.start_time)

    gasearch.params.hash=binascii.hexlify(os.urandom(4)).decode('utf-8')

    # add a random hash to the file name so that we can run it multiple times
    # and it will save to different files

    pickle.dump(gasearch,
        open("ga_results_tasks_%s_%s_%s_%s.pkl"%(gasearch.params.clf,'-'.join(gasearch.params.targets),
            '-'.join(['%s'%i for i in gasearch.params.objective_weights]),gasearch.params.hash),'wb'))
