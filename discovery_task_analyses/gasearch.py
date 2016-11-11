"""
class definitions for GASearch and GASearchParams
"""

from search_objectives import get_reconstruction_error,get_subset_corr
import os,glob,sys,itertools,time,pickle
import numpy,pandas
import fancyimpute
from joblib import Parallel, delayed
import multiprocessing
from sklearn.preprocessing import scale

from compute_scores import compute_pca_cval

sys.path.append('../utils')
from utils import get_info,get_behav_data,get_demographics



class GASearchParams:
    def __init__(self,
        targets=['survey','demog','task'],  # targets for reconstruction and correlation
        usepca=True, # should we collapse targetdata into PCs?
        objective_weights=[0,1], # weights for reconstruction and correlation respectively
        nvars=8,  # number of selected tasks
        ngen=2500,  # maximum number of GA generations
        initpopsize=500,  # initial population size
        nselect=50,  #number selected to survive at each generation
        nimmigrants=500,   # number of new immigrants in each generation
        nbabies=4,    # number of offspring of each survivor
        mutation_rate=None,   # mutation rate for offspring
        convergence_threshold=10,  # number of stable generations for convergence
        clf='lasso',
        num_cores=1,
        nsplits=8,
        dataset=None,
        n_jobs=1,
        remove_chosen_from_test=True,
        verbose=1,  # minimal level of verbosity
        lasso_alpha=0.1,
        linreg_n_jobs=-1,
        taskdatafile= 'taskdata_imputed.csv',
        behavdatafile= 'meaningful_variables_imputed.csv',
        drop_tasks=['writing_task'],
        demogvars=['BMI','Age','Sex','RetirementAccount','ChildrenNumber',
                                'CreditCardDebt','TrafficTicketsLastYearCount',
                                'TrafficAccidentsLifeCount','ArrestedChargedLifeCount',
                                'LifetimeSmoke100Cigs','AlcoholHowManyDrinksDay',
                                'CannabisPast6Months','Nervous',
                                'Hopeless', 'RestlessFidgety', 'Depressed',
                                'EverythingIsEffort','Worthless']):

        self.targets=targets
        self.usepca=usepca
        if self.usepca:
            self.remove_chosen_from_test=False
        else:
            self.remove_chosen_from_test=remove_chosen_from_test
        self.objective_weights=objective_weights
        assert numpy.sum(self.objective_weights)==1
        self.nvars=nvars
        self.behavdatafile=behavdatafile
        self.ngen=ngen
        self.initpopsize=initpopsize
        self.nselect=nselect
        self.nimmigrants=nimmigrants
        self.nbabies=nbabies
        if mutation_rate is None:
            self.mutation_rate=1/self.nvars
        else:
            self.mutation_rate=mutation_rate
        self.convergence_threshold=convergence_threshold
        self.clf=clf
        self.num_cores=num_cores
        self.nsplits=nsplits
        self.n_jobs=n_jobs
        self.verbose=verbose  # minimal level of verbosity
        self.lasso_alpha=lasso_alpha
        self.linreg_n_jobs=linreg_n_jobs
        if dataset is None:
            self.dataset=get_info('dataset')
        else:
            self.dataset=dataset
        if self.verbose>0:
            print('using dataset:',self.dataset)
        self.basedir=get_info('base_directory')
        self.derived_dir=os.path.join(self.basedir,'Data/Derived_Data/%s'%self.dataset)
        self.data_dir=os.path.join(self.basedir,'Data/%s'%self.dataset)
        self.taskdatafile= taskdatafile
        self.drop_tasks=drop_tasks
        self.demogvars=demogvars

        self.start_time = None
        self.ntasks=None

class GASearch:

    def __init__(self,**kwargs):
        self.params=GASearchParams(**kwargs)
        self.ccmax={}
        self.cc_sorted=None
        self.cc=None
        self.bestp_saved={}
        self.bestctr=0
        self.population=[]
        self.tasks=None
        self.targetdata=None
        self.targetdata__pca_varexplained=None
        self.targetdata_source=None # 0=task, 1=survey, 2=demog
        self.varexp={}

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
        # could use pandas filter
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

    def decimate_task_data(self,tasks_to_keep=None):
        """
        choose 8 tasks and replace all other tasks with random noise
        this is mean for testing only
        """
        print('DANGER: Replacing all variables except for these with noise:')
        if tasks_to_keep is None:
            idx=[i for i in range(self.params.ntasks)]
            numpy.random.shuffle(idx)
            tasks_to_keep=idx[:self.params.nvars]
        tasks_to_keep.sort()
        print(tasks_to_keep)
        tasknames_to_keep=[self.tasks[i] for i in tasks_to_keep]
        for c in self.taskdata.columns:
            task=c.split('.')[0]
            if not task in tasknames_to_keep:
                self.taskdata[c]=numpy.random.randn(self.taskdata[c].shape[0])
        return tasks_to_keep

    def load_targetdata(self):

        if 'task' in self.params.targets:
            self.targetdata=get_behav_data(self.params.dataset,self.params.taskdatafile )
            assert all(self.taskdata.index == self.targetdata.index)
            if self.params.verbose>0:
                print('target: task, %d variables'%self.taskdata.shape[1])
                print('%d missing values'%numpy.sum(numpy.isnan(self.taskdata.values)))
            if self.params.usepca:
                self.targetdata,self.varexp['task']=compute_pca_cval(self.targetdata,flag='task')
                print('using PCA for task: %d dims, %f variance explained'%(self.targetdata.shape[1],
                                            numpy.sum(self.varexp['task'])))
            self.targetdata_source=numpy.zeros(self.targetdata.shape[1])

        if 'survey' in self.params.targets:
            alldata=get_behav_data(self.params.dataset,self.params.behavdatafile)
            self.surveydata=pandas.DataFrame()
            for k in alldata.columns:
                if k.find('survey')>-1:
                    self.surveydata[k]=alldata[k]
            if self.params.verbose>0:
                print('target: survey, %d variables'%self.surveydata.shape[1])
                print('%d missing values'%numpy.sum(numpy.isnan(self.surveydata.values)))
            if self.params.usepca:
                self.surveydata,self.varexp['survey']=compute_pca_cval(self.surveydata,flag='survey')
                print('using PCA for survey: %d dims, %f variance explained'%(self.surveydata.shape[1],
                                            numpy.sum(self.varexp['survey'])))
            if not self.targetdata is None:
                assert all(self.taskdata.index == self.surveydata.index)
                self.targetdata = self.surveydata.merge(self.targetdata,'inner',right_index=True,left_index=True)
                self.targetdata_source=numpy.hstack((self.targetdata_source,numpy.ones(self.surveydata.shape[1])))
            else:
                self.targetdata=self.surveydata
                self.targetdata_source=numpy.zeros(self.surveydata.shape[1])

        if 'demog' in self.params.targets:
            self.demogdata=get_demographics(self.params.dataset,var_subset=self.params.demogvars)
            if self.params.verbose>0:
                print('target: demog, %d variables'%self.demogdata.shape[1])
                print('%d missing values'%numpy.sum(numpy.isnan(self.demogdata.values)))
            if numpy.sum(numpy.isnan(self.demogdata.values))>0:
                demogdata_imp=fancyimpute.SimpleFill().complete(self.demogdata.values)
                self.demogdata=pandas.DataFrame(demogdata_imp,
                                    index=self.demogdata.index,
                                    columns=self.demogdata.columns)

            if self.params.usepca:
                self.demogdata,self.varexp['demog']=compute_pca_cval(self.demogdata,flag='demog')
                print('using PCA for demographics: %d dims, %f variance explained'%(self.demogdata.shape[1],
                                            numpy.sum(self.varexp['demog'])))
            if not self.targetdata is None:
                assert all(self.taskdata.index == self.demogdata.index)
                self.targetdata = self.demogdata.merge(self.targetdata,'inner',right_index=True,left_index=True)
                self.targetdata_source=numpy.hstack((self.targetdata_source,numpy.ones(self.demogdata.shape[1])))
            else:
                self.targetdata=self.demogdata
                self.targetdata_source=numpy.zeros(self.demogdata.shape[1])

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
        """
        simple truncation selection scheme
        """
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
        if self.params.verbose>0:
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

            p1=subpop[0]
            p2=subpop[1]
            parents=list(numpy.unique(numpy.hstack((subpop[0],subpop[1]))))
            if len(set(parents))<len(subpop[1]):
                continue
            for b in range(self.params.nbabies):
                numpy.random.shuffle(parents)
                recomb_location=numpy.random.randint(self.params.nvars)
                baby=p1[:recomb_location]+p2[recomb_location:self.params.nvars]
                assert len(baby)==self.params.nvars
                baby=parents[:len(subpop[1])]
                nmutations=numpy.floor(len(baby)*numpy.random.rand()*self.params.mutation_rate).astype('int')
                alts=[i for i in range(self.params.ntasks) if not i in baby]
                numpy.random.shuffle(alts)
                nmissing=self.params.nvars-len(set(baby))
                if nmissing>0:  # duplicate found
                    baby=list(set(baby))
                    baby[len(set(baby)):self.params.nvars]=alts[:nmissing]
                    assert len(baby)==self.params.nvars
                    alts=[i for i in range(self.params.ntasks) if not i in baby]
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
