import numpy

taskset = raw_input('Enter task set (1 or 2)')
if taskset == '1':
	tasks = ['stop_signal','attention_network_task','twobytwo',
			'ward_and_allport','discount_adjusted',
			'columbia_card_task_hot']
else:
	tasks = ['motor_selective_stop_signal','stroop',
			'dot_pattern_expectancy','survey_medley',
			'discount_fixed']
numpy.random.shuffle(tasks)
print('\n'.join(tasks))