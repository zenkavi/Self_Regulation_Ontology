# -*- coding: utf-8 -*-

from expanalysis.experiments.ddm_utils import get_HDDM_fun
from selfregulation.utils.utils import get_behav_data

task = 'stim_selective_stop_signal'
data = get_behav_data(file='Individual_Measures/%s.csv.gz' % task)
data = data.query('worker_id in %s' % list(data.worker_id.unique()[0:10]))
fun = get_HDDM_fun(task, 200)
out = fun(data)