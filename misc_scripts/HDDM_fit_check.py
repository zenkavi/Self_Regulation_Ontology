import hddm
from os import path




from selfregulation.utils.utils import get_behav_data
data = get_behav_data(file='Individual_Measures/stroop.csv.gz')
data = data.query('worker_id in %s' % list(data.worker_id.unique()[0:4]))
df = data
condition = 'condition'
response_col = 'correct'
fixed= ['t','a']
estimate_task_vars = True
outfile='stroop'
samples = 1000

assert estimate_task_vars or condition != None, "Condition must be defined or estimate_task_vars must be set to true"
variable_conversion = {'a': ('thresh', 'Pos'), 'v': ('drift', 'Pos'), 't': ('non_decision', 'NA')}
# set up condition variables
if condition:
    condition_vars = [var for var in ['a','v','t'] if var not in fixed]
    depends_dict = {var: 'condition' for var in condition_vars}
else:
    condition_vars = []
    depends_dict = {}
# set up data
data = (df.loc[:,'rt']/1000).astype(float).to_frame()
data.insert(0, 'response', df[response_col].astype(float))
if condition:
    data.insert(0, 'condition', df[condition])
    conditions = [i for i in data.condition.unique() if i]
    
# add subject ids 
data.insert(0,'subj_idx', df['worker_id'])
# remove missed responses and extremely short response
data = data.query('rt > .05')
subj_ids = data.subj_idx.unique()
ids = {subj_ids[i]:int(i) for i in range(len(subj_ids))}
data.replace(subj_ids, [ids[i] for i in subj_ids],inplace = True)
if outfile:
    data.to_csv(outfile + '_data.csv')
    database = outfile + '_traces.db'
else:
    database = 'traces.db'
# extract dvs pip install -U --no-deps kabuki
group_dvs = {}
dvs = {}
# run if estimating variables for the whole task
if estimate_task_vars:
    # run hddm
    m = hddm.HDDM(data)
    # find a good starting point which helps with the convergence.
    m.find_starting_values()
    # start drawing 10000 samples and discarding 1000 as burn-in
    m.sample(samples, burn=samples/10, thin = 5)
    dvs = {var: m.nodes_db.loc[m.nodes_db.index.str.contains(var + '_subj'),'mean'] for var in ['a', 'v', 't']} 


















from expanalysis.experiments.jspsych_processing import fit_HDDM
group_dvs = fit_HDDM(data, condition = 'condition', outfile = 'stroop', samples=1000)



task = 'stroop'
directory = '/mnt/Sherlock_Scratch'
hddm.load(path.join(directory,'%s_base.model' % task))
hddm.models.