from expanalysis.experiments.ddm_utils import get_HDDM_fun
from expanalysis.experiments.jspsych_processing import group_decorate
from glob import glob
import hddm
from kabuki.analyze import gelman_rubin
from multiprocessing import Pool 

from os import path, rename
from selfregulation.utils.utils import get_behav_data


samples=100
hddm_fun_dict = get_HDDM_fun(None, samples)
hddm_fun_dict.pop('twobytwo')

gelman_vals = {}
def assess_convergence(task, reps=5):
    # load data
    data = get_behav_data(file='Individual_Measures/%s.csv.gz' % task)
    data = data.query('worker_id in %s' % list(data.worker_id.unique()[0:4]))
    outputs = []
    def run_model():
        # compute DVs (create models)
        group_dvs = hddm_fun_dict[task](data)
        # load models
        base_files = glob('%s*_base.model' % task)
        m_base = hddm.load(base_files[0])
        m_condition = None
        condition_files = glob('%s*_condition.model' % task)
        if len(condition_files)>0:
            m_condition = hddm.load(condition_files[0])
        return (m_base, m_condition)
    
    for _ in range(reps):
        output = run_model()
        outputs.append(output)
    return {task: outputs}
        
# create group maps
pool = Pool()
mp_results = pool.map(assess_convergence, hddm_fun_dict.keys())
pool.close() 
pool.join()

results = {}
for d in mp_results:
    results.update(d)

for k,v in results.items():
    gelman_vals[k+'_base'] = gelman_rubin([i[0] for i in v])
    # plot posteriors
    v[0][0].plot_posteriors(['a', 't', 'v', 'a_std'], save=True)
    plots = glob('*png')
    for p in plots:
        rename(p, path.join('Plots', '%s_base_%s' % (k,p)))
    
    if v[0][1] is not None:
        gelman_vals[k+'_condition'] = gelman_rubin([i[1] for i in v])
        
        v[0][1].plot_posteriors(['a', 't', 'v', 'a_std'], save=True)
        plots = glob('*png')
        for p in plots:
            rename(p, path.join('Plots', '%s_condition_%s' % (k,p)))


