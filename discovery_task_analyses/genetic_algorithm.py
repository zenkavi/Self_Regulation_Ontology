"""
functions for genetic algorithm search
"""
import os
import numpy
from sklearn.preprocessing import scale

from search_objectives import get_reconstruction_error_vars,get_subset_corr_vars
from search_objectives import get_reconstruction_error,get_subset_corr

__USE_MULTIPROC__=True

if __USE_MULTIPROC__:
    from joblib import Parallel, delayed
    import multiprocessing
    if 'NUMCORES' in os.environ:
        num_cores=int(os.environ['NUMCORES'])
    else:
        num_cores = multiprocessing.cpu_count()
    print('multiproc: using %d cores'%num_cores)

def get_initial_population_vars(popsize,nvars,data,taskvaridx):
    poplist=[]
    for i in range(popsize):
        numpy.random.shuffle(taskvaridx)
        poplist.append(taskvaridx[:nvars])
    return(poplist)

def get_initial_population_tasks(popsize,nvars,data,ntasks):
    poplist=[]
    idx=[i for i in range(ntasks)]
    for i in range(popsize):
        numpy.random.shuffle(idx)
        poplist.append(idx[:nvars])
    return poplist

def get_population_fitness_vars(pop,data,nsplits,clf,obj_weight):
    # first get cc for each item in population
    cc_recon=[get_reconstruction_error_vars(cv,data,nsplits,clf) for cv in pop]
    cc_subsim=[get_subset_corr_vars(cv,data) for cv in pop]
    maxcc=[numpy.max(cc_recon),numpy.max(cc_subsim)]
    cc_recon=scale(cc_recon)
    cc_subsim=scale(cc_subsim)
    cc=cc_recon*obj_weight[0] + cc_subsim*obj_weight[1]
    return cc,maxcc

def get_population_fitness_tasks(pop,data,nsplits,clf,obj_weight):
    # first get cc for each item in population
    if __USE_MULTIPROC__:
        cc_recon = Parallel(n_jobs=num_cores)(delayed(get_reconstruction_error)(ct,data,nsplits,clf) for ct in pop)
    else:
        cc_recon=[get_reconstruction_error(ct,data,nsplits,clf) for ct in pop]
    cc_subsim=[get_subset_corr(ct,data) for ct in pop]
    maxcc=[numpy.max(cc_recon),numpy.max(cc_subsim)]
    cc_recon=scale(cc_recon)
    cc_subsim=scale(cc_subsim)
    cc=cc_recon*obj_weight[0] + cc_subsim*obj_weight[1]
    return cc,maxcc

def select_parents_vars(pop,data,nsel,clf,nsplits=4,obj_weight=[0.5,0.5]):
    cc,maxcc=get_population_fitness_vars(pop,data,nsplits,clf,obj_weight)
    idx=numpy.argsort(cc)[::-1]
    pop_sorted=[pop[i] for i in idx[:nsel]]
    cc_sorted=[cc[i] for i in idx[:nsel]]
    return(pop_sorted,cc_sorted,maxcc)

def select_parents_tasks(pop,data,nsel,clf,nsplits=4,obj_weight=[0.5,0.5]):
    cc,maxcc=get_population_fitness_tasks(pop,data,nsplits,clf,obj_weight)
    idx=numpy.argsort(cc)[::-1]
    pop_sorted=[pop[i] for i in idx[:nsel]]
    cc_sorted=[cc[i] for i in idx[:nsel]]
    return(pop_sorted,cc_sorted,maxcc)

def crossover_vars(pop,data,taskvaridx,nbabies=2,
                mutation_rate=None):
    if mutation_rate is None:
        mutation_rate=1/len(pop[0])
    # assortative mating - best parents mate
    families=numpy.kron(numpy.arange(numpy.floor(len(pop)/2)),[1,1])
    numpy.random.shuffle(families)
    for f in numpy.unique(families):
        famidx=[i for i in range(len(families)) if families[i]==f]
        if len(famidx)!=2:
            continue
        try:
            subpop=[pop[i] for i in famidx]
        except:
            print('oops...')
            print(len(pop))
            print(famidx)
            raise Exception('breaking')
        parents=list(numpy.unique(numpy.hstack((subpop[0],subpop[1]))))
        if len(set(parents))<len(subpop[1]):
            continue
        for b in range(nbabies):
            numpy.random.shuffle(parents)
            baby=parents[:len(subpop[1])]
            if numpy.random.randn()<mutation_rate:
                alts=[i for i in taskvaridx if not i in baby]
                numpy.random.shuffle(alts)
                mutpos=numpy.random.randint(len(baby))
                baby[mutpos]=alts[0]
            pop.append(baby)
    return pop

def crossover_tasks(pop,data,ntasks,nbabies=2,
                mutation_rate=None):
    if mutation_rate is None:
        mutation_rate=1/len(pop[0])
    # assortative mating - best parents mate
    families=numpy.kron(numpy.arange(numpy.floor(len(pop)/2)),[1,1])
    numpy.random.shuffle(families)
    for f in numpy.unique(families):
        famidx=[i for i in range(len(families)) if families[i]==f]
        if len(famidx)!=2:
            continue
        try:
            subpop=[pop[i] for i in famidx]
        except:
            print('oops...')
            print(len(pop))
            print(famidx)
            raise Exception('breaking')
        parents=list(numpy.unique(numpy.hstack((subpop[0],subpop[1]))))
        if len(set(parents))<len(subpop[1]):
            continue
        for b in range(nbabies):
            numpy.random.shuffle(parents)
            baby=parents[:len(subpop[1])]
            if numpy.random.randn()<mutation_rate:
                alts=[i for i in range(ntasks) if not i in baby]
                numpy.random.shuffle(alts)
                mutpos=numpy.random.randint(len(baby))
                baby[mutpos]=alts[0]
            pop.append(baby)
    return pop


def immigrate_vars(pop,n,nvars,data,taskvaridx):
    immigrants=[]
    for i in range(n):
        numpy.random.shuffle(taskvaridx)
        immigrants.append(taskvaridx[:nvars])
    return pop+immigrants

def immigrate_tasks(pop,n,nvars,data,ntasks):
    immigrants=[]
    idx=[i for i in range(ntasks)]
    for i in range(n):
        numpy.random.shuffle(idx)
        immigrants.append(idx[:nvars])
    return pop+immigrants
