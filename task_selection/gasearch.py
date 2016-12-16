"""
class definitions for GASearch and GASearchParams
"""

from search_objectives import get_reconstruction_error,get_subset_corr,get_time_fitness
import os,glob,sys,itertools,time,pickle
import numpy,pandas
import fancyimpute
from joblib import Parallel, delayed
import multiprocessing
from sklearn.preprocessing import scale
from sklearn.linear_model import MultiTaskLassoCV

from compute_scores import compute_pca_cval

sys.path.append('../utils')
from utils import get_info,get_behav_data,get_demographics

class GASearchParams:
    def __init__(self,
        targets=['survey','demog','task'],  # targets for reconstruction and correlation
        usepca=True, # should we collapse targetdata into PCs?
        use_full_dataset=True, # combine discovery and validation datasets
        objective_weights=[0,1], # weights for reconstruction and correlation respectively
        target_weights=[1/3,1/3,1/3],
        normalize_weights_by_numvars=True,
        weight_by_variance=False,
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
        fit_thresh=0.1,
        dataset=None,
        n_jobs=1,
        max_task_time=100, # set to >500 to turn off penalty
        tasktimefile='time_estimates.csv',
        remove_chosen_from_test=True,
        constrain_single_stop_task=True,
        verbose=1,  # minimal level of verbosity
        lasso_alpha=0.1,
        linreg_n_jobs=-1,
        taskdatafile= 'taskdata_imputed_for_task_selection.csv',
        behavdatafile= 'meaningful_variables_imputed_for_task_selection.csv',
        drop_tasks=['writing_task','simple_reaction_time'],
        drop_vars=[],
        demogvars=['BMI','RetirementAccount','ChildrenNumber','DivorceCount',
                                'HouseholdIncome','SmokeEveryDay','CigsPerDay',
                                'CreditCardDebt','TrafficTicketsLastYearCount',
                                'TrafficAccidentsLifeCount','ArrestedChargedLifeCount',
                                'LifetimeSmoke100Cigs','AlcoholHowManyDrinksDay',
                                'AlcoholHowOften','AlcoholHowOften6Drinks',
                                'HowOftenCantStopDrinking','CannabisHowOften',
                                'CannabisPast6Months']):

        self.targets=targets
        # default to equal weighting across targets
        self.target_weights=target_weights
        self.usepca=usepca
        self.use_full_dataset=use_full_dataset
        self.normalize_weights_by_numvars=normalize_weights_by_numvars
        if self.usepca:
            self.remove_chosen_from_test=False
            self.weight_by_variance=weight_by_variance
        else:
            self.remove_chosen_from_test=remove_chosen_from_test
            self.weight_by_variance=False
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
        self.max_task_time=max_task_time
        self.tasktimefile=tasktimefile
        self.constrain_single_stop_task=constrain_single_stop_task
        self.nsplits=nsplits
        self.fit_thresh=fit_thresh
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
        self.drop_vars=drop_vars
        self.demogvars=demogvars

        self.start_time = None
        self.ntasks=None
        self.tasktime=None

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
        self.varexp_weights=None

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
        self.taskdata=get_behav_data(self.params.dataset,self.params.taskdatafile,
                                    full_dataset = self.params.use_full_dataset)
        # could use pandas filter
        for c in self.taskdata.columns:
            if c in self.params.drop_vars:
                #if self.params.verbose>0:
                print('dropping',c)
                del self.taskdata[c]
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
        self.params.stoptasks=[i for i,t in enumerate(self.tasks) if t.find('stop_signal')>-1]

    def get_tasktimes(self):
        """
        load task timing data
        """
        df=pandas.DataFrame.from_csv(self.params.tasktimefile,header=None)
        self.params.tasktime=[]
        for t in self.tasks:
            self.params.tasktime.append(df.loc[t].values[0])

    def estimate_lasso_param(self):
        if self.params.verbose>0:
            print('estimating lasso param using CV')
        lasso=MultiTaskLassoCV(alphas=[10**x for x in numpy.arange(-6,8,0.5)])
        lasso.fit(self.taskdata,self.targetdata)
        self.params.lasso_alpha=lasso.alpha_
        score=lasso.score(self.taskdata,self.targetdata)
        if self.params.verbose>0:
            print('alpha=%f, r^2=%f'%(lasso.alpha_,score))
        if score<0.5:
            print('WARNING: poor fit to training data')

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
            self.targetdata=get_behav_data(self.params.dataset,self.params.taskdatafile,
                                            full_dataset=self.params.use_full_dataset)
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
            alldata=get_behav_data(self.params.dataset,self.params.behavdatafile,
                                                full_dataset=self.params.use_full_dataset)
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
            self.demogdata=get_demographics(self.params.dataset,var_subset=self.params.demogvars,
                                                full_dataset=self.params.use_full_dataset)
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
        # make weights sum to one so that correlations are interpretable later
        #TODO: add an index for datasets and move the varexp_Weights computation down here

        if self.params.normalize_weights_by_numvars:
            if 'task' in self.params.targets:
                self.params.target_weights[0]=(self.targetdata.shape[1]/self.taskdata.shape[1])*self.params.target_weights[0]
            if 'survey' in self.params.targets:
                self.params.target_weights[1]=(self.targetdata.shape[1]/self.surveydata.shape[1])*self.params.target_weights[1]
            if 'demog' in self.params.targets:
                self.params.target_weights[2]=(self.targetdata.shape[1]/self.demogdata.shape[1])*self.params.target_weights[2]

        if self.params.usepca:
            if 'task' in self.params.targets:
                    self.varexp_weights=self.varexp['task']*self.params.target_weights[0]

            if 'survey' in self.params.targets:
                    if self.varexp_weights is None:
                        self.varexp_weights=self.varexp['survey']*self.params.target_weights[1]
                    else:
                        self.varexp_weights=numpy.hstack((self.varexp_weights,self.varexp['survey']*self.params.target_weights[1]))

            if 'demog' in self.params.targets:
                    if self.varexp_weights is None:
                        self.varexp_weights=self.varexp['demog']*self.params.target_weights[2]
                    else:
                        self.varexp_weights=numpy.hstack((self.varexp_weights,self.varexp['demog']*self.params.target_weights[2]))

            # normalize weights
            if self.varexp_weights is not None:
                self.varexp_weights=self.varexp_weights/numpy.sum(self.varexp_weights)
        else:
            print('WARNING: Target weighting not yet implemented for non-pca case')
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
            badlength=1
            while badlength:
                numpy.random.shuffle(idx)
                time_penalty,totaltime=get_time_fitness(idx[:self.params.nvars],self.params)
                if time_penalty>0:
                    badlength=0
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
        if self.params.constrain_single_stop_task:
            for ct in self.population:
                if len(set(self.params.stoptasks).intersection(ct))>1:
                    self.population.remove(ct)
        if self.params.objective_weights[0]>0:
            if self.__USE_MULTIPROC__:
                cc_recon=Parallel(n_jobs=self.params.num_cores)(delayed(get_reconstruction_error)(ct,self.taskdata,self.targetdata,self.params) for ct in self.population)
            else:
                cc_recon=[get_reconstruction_error(ct,self.taskdata,self.targetdata,self.params) for ct in pop]
            cc_recon=numpy.array(cc_recon)
            if self.params.weight_by_variance:
                cc_recon=cc_recon.dot(self.varexp_weights)
            else:
                cc_recon=numpy.mean(cc_recon,1)
        else:
            cc_recon=[0]
        if self.params.verbose>9:
            print("ccrecon shape",numpy.array(cc_recon).shape)
        if self.params.objective_weights[1]>0:
            if self.__USE_MULTIPROC__:
                cc_subsim=Parallel(n_jobs=self.params.num_cores)(delayed(get_subset_corr)(ct,self.taskdata,self.targetdata) for ct in self.population)
            else:
                cc_subsim=[get_subset_corr(ct,self.taskdata,self.targetdata) for ct in self.population]
        else:
            cc_subsim=[0]
        if self.params.verbose>0:
            try:
                print('corr recon-subsim:',numpy.corrcoef(cc_recon,cc_subsim)[0,1])
            except:
                pass

        # penalize any that are over the time limit
        cc_time=[]
        for ct in self.population:
            time_penalty,totaltime=get_time_fitness(ct,self.params)
            cc_time.append(time_penalty*2)
        maxcc=[numpy.max(cc_recon),numpy.max(cc_subsim)]
        cc_recon=scale(cc_recon)
        cc_subsim=scale(cc_subsim)
        self.cc=cc_recon*self.params.objective_weights[0] + cc_subsim*self.params.objective_weights[1]+cc_time
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
            badlength=True
            while badlength:
                numpy.random.shuffle(idx)
                time_penalty,totaltime=get_time_fitness(idx[:self.params.nvars],self.params)
                if time_penalty>0:
                    badlength=0
            immigrants.append(idx[:self.params.nvars])
        return self.population+immigrants

    def get_final_pred_accuracy(self):
        lasso=Lasso(alpha=self.params.lasso_alpha)
