import json
import matplotlib.pyplot as plt
from os import path
import pandas as pd
import seaborn as sns
from selfregulation.utils.utils import get_info
sns.set_palette("Set1", 8, .75)

base_dir = get_info('base_directory')
ext = 'png'
"""
# Load data if plots need to be regenerated

post_process_data_loc = ''
data = pd.load_pickle(post_process_data_loc)
"""

# plt total time on tasks
(data.groupby('worker_id').ontask_time.sum()/3600).hist(bins=40, 
                                                        grid=False, 
                                                        density=True,
                                                        figsize=(12,8))
plt.xlabel('Time (Hours)')
plt.title('Total Time on Tasks', weight='bold')



# plot distribution of times per task
tasks = data.experiment_exp_id.unique()
N = len(tasks)

f, axes = plt.subplots(3,1,figsize=(16,20))
for i in range(3):
    for exp in tasks[i*N//3: (i+1)*N//3]:
        task_time = data.query('experiment_exp_id == "%s"' % exp).ontask_time/3600
        task_time.name = ' '.join(exp.split('_'))
        if not pd.isnull(task_time.sum()):
            sns.kdeplot(task_time, linewidth=3, ax=axes[i])
    axes[i].set_xlim(0,1)
    axes[i].legend(ncol=3)
plt.xlabel('Time (Hours)')


"""
# Load worker completions if plot needs to be regenerated
worker_completion_loc = ''
worker_completions = json.load(open(worker_completion_loc, 'r'))
"""
save_dir = path.join(base_dir, 'Data', 'Plots', 'worker_completions.%s' % ext)
completion_rate = np.mean(np.array(list(worker_completions.values())) ==63)
completion_rate = "{0:0.1f}%".format(completion_rate*100)
plt.figure(figsize=(12,8))
plt.hist(worker_completions.values(), bins=40, width=5)
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.text(5, 400, 'Completion Rate: %s' % completion_rate, size=20)
plt.xlabel('Number of Tasks Completed')
plt.savefig(save_dir, dpi=300, bbox_inches='tight')