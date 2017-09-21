import hddm
from os import path




from selfregulation.utils.utils import get_behav_data
data = get_behav_data(file='Individual_Measures/stroop.csv.gz')
data = data.query('worker_id in %s' % list(data.worker_id.unique()[0:3]))
df = data
condition = 'condition'
response_col = 'correct'
fixed= ['t','a']
estimate_task_vars = True
outfile='stroop'
samples = 1000

from expanalysis.experiments.jspsych_processing import fit_HDDM
group_dvs = fit_HDDM(data, condition = 'condition', outfile = 'stroop', samples=1000)



task = 'stroop'
directory = '/mnt/Sherlock_Scratch'
hddm.load(path.join(directory,'%s_base.model' % task))
hddm.models.