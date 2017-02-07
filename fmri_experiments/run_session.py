import json
import numpy
import subprocess
import time

subid = raw_input('Enter subject id (i.e. s999): ')
scanner_set = raw_input('Enter order set (1-4): ')
taskset = raw_input('Enter task day (1 or 2): ')

if taskset == '1':
	tasks = ['stop_signal','attention_network_task','twobytwo',
			'ward_and_allport', #'discount_adjusted',
			'columbia_card_task_hot']
else:
	tasks = ['motor_selective_stop_signal','stroop',
			'dot_pattern_expectancy','survey_medley',
			'discount_fixed']
numpy.random.shuffle(tasks)
print('\n'.join(tasks))
json.dump(tasks, open('temp_tasklist.json','w'))

for task in tasks:
	print('***************************************************************')
	subprocess.call("bash run_task.sh -i %s -s %s -t %s &" % (subid,scanner_set,task), shell=True)
	time.sleep(20)