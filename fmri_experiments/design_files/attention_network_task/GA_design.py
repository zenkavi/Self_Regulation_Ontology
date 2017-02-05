from neurodesign import geneticalgorithm, generate, msequence 
import sys

design_i = sys.argv[1]

EXP = geneticalgorithm.experiment( 
    TR = 0.68, 
    P = [1.0/6,1.0/6,1.0/6,1.0/6,1.0/6,1.0/6], 
    C = [[0.25,0.25,-0.25,-0.25,0,0],[-0.25,-0.25,0,0,0.25,0.25],
        [1.0/6,-1.0/6,1.0/6,-1.0/6,1.0/6,-1.0/6]], 
    rho = 0.3, 
    n_stimuli = 6, 
    n_trials = 192, 
    duration = 550, 
    resolution = 0.1, 
    stim_duration = 2.2, 
    t_pre = 0.0, 
    t_post = .4, 
    maxrep = 6, 
    hardprob = False, 
    confoundorder = 3, 
    ITImodel = 'exponential', 
    ITImin = 0.0, 
    ITImean = 0.26, 
    ITImax = 6.0, 
    restnum = 0, 
    restdur = 0.0) 


POP = geneticalgorithm.population( 
    experiment = EXP, 
    G = 20, 
    R = [0.4, 0.4, 0.2], 
    q = 0.01, 
    weights = [0.0, 0.1, 0.4, 0.5], 
    I = 4, 
    preruncycles = 2000, 
    cycles = 5000, 
    convergence = 1000, 
    outdes = 4, 
    folder = '../fmri_experiments/design_files/attention_network_task/attention_network_task_designs_'+design_i) 


POP.naturalselection()
POP.download()
