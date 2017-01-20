from neurodesign import geneticalgorithm, generate, msequence 
import sys

design_i = sys.argv[1]

EXP = geneticalgorithm.experiment( 
    TR = 0.68, 
    P = [0.17, 0.17, 0.17, 0.17, 0.17, 0.17], 
    C = [[.25, .25, -.25, -.25, 0, 0], [-.25, -.25, 0, 0, .25, .25], 
        [1.0/6,-1.0/6,1.0/6,-1.0/6,1.0/6,-1.0/6]]
    rho = 0.3, 
    n_stimuli = 6, 
    n_trials = 192, 
    duration = 211.2, 
    resolution = 0.1, 
    stim_duration = 2.2, 
    t_pre = 0.0, 
    t_post = 0.4, 
    maxrep = 6, 
    hardprob = False, 
    confoundorder = 3, 
    ITImodel = 'exponential', 
    ITImin = 0.0, 
    ITImean = 0.26, 
    ITImax = 10.0, 
    restnum = 0, 
    restdur = 0.0) 


POP = geneticalgorithm.population( 
    experiment = EXP, 
    G = 20, 
    R = [0.4, 0.4, 0.2], 
    q = 0.01, 
    weights = [0.0, 0.1, 0.4, 0.5], 
    I = 4, 
    preruncycles = 1000, 
    cycles = 2000, 
    convergence = 1000, 
    seed = 3281, 
    outdes = 2, 
    folder = '../fmri_experiments/design_files/stroop/stroop_designs_'+design_i) 


POP.naturalselection()
POP.download()
