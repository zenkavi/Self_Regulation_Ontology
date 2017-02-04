from selfregulation.utils.utils import get_behav_data
from selfregulation.utils.plot_utils import dendroheatmap

data = get_behav_data(dataset = 'Complete_01-16-2017')
tasks = ['attention_network_task','columbia_card_task_cold',
         'columbia_card_task_hot', 'dot_pattern_expectancy', 'kirby',
         'motor_selective_stop_signal','stop_signal','stroop','threebytwo',
         'tower_of_london']
task_data = data.filter(regex = '|'.join(tasks))

fig, column_order = dendroheatmap(task_data.corr(), labels = True, label_fontsize = 14)
fig.savefig('Plots/selected_tasks_clustered_heatmap.pdf')
