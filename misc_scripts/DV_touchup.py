#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:56:41 2018

@author: ian
"""
# fix threebytwo
import pandas as pd
behav_path = '/mnt/temp/mturk_retest_output/threebytwo_mturk_retest_DV.json'

dvs = pd.read_json(behav_path)

param = 'drift'
dvs['task_switch_cost_hddm_' + param] = dvs['hddm_' + param + '_task_switch'] - dvs['hddm_' + param + '_task_stay']
dvs.to_json(behav_path)

# update motor selective
task = 'motor_selective_stop_signal'
behav_path = '/mnt/temp/mturk_retest_output/%s_mturk_retest_DV.json' % task

dvs = pd.read_json(behav_path)
dvs.loc[:, 'proactive_control_rt'] = dvs.selective_proactive_control
dvs.loc[:,'proactive_control_hddm_drift'] = dvs.condition_sensitivity_hddm_drift
dvs.loc[:, 'reactive_control_rt'] = dvs.reactive_control
dvs.drop(['selective_proactive_control','condition_sensitivity_hddm_drift','reactive_control'], axis=1, inplace=True)
dvs.to_json(behav_path)

# update stim selective
task = 'stim_selective_stop_signal'
behav_path = '/mnt/temp/mturk_complete_output/%s_mturk_complete_DV.json' % task

dvs = pd.read_json(behav_path)
dvs.loc[:,'reactive_control_hddm_drift'] = dvs.condition_sensitivity_hddm_drift
dvs.drop(['condition_sensitivity_hddm_drift'], axis=1, inplace=True)
dvs.to_json(behav_path)

# update stop signal
task = 'stop_signal'
behav_path = '/mnt/temp/mturk_retest_output/%s_mturk_retest_DV.json' % task

dvs = pd.read_json(behav_path)
dvs.loc[:,'proactive_slowing_hddm_drift'] = dvs.condition_sensitivity_hddm_drift
dvs.loc[:,'proactive_slowing_hddm_thresh'] = dvs.condition_sensitivity_hddm_thresh

dvs.loc[:,'proactive_slowing_rt'] = dvs.proactive_slowing

dvs.drop(['condition_sensitivity_hddm_drift', 'condition_sensitivity_hddm_thresh', 'proactive_slowing'], axis=1, inplace=True)
dvs.to_json(behav_path)

# larger update
fullDV_loc = '/mnt/temp/mturk_complete_DV.json'
fullvalence_loc = '/mnt/temp/mturk_complete_DV_valence.json'
dv = pd.read_json(fullDV_loc)
valence = pd.read_json(fullvalence_loc)

# update
updateDV = '/mnt/temp/mturk_complete_fix/mturk_complete_DV.json'
updatevalence = '/mnt/temp/mturk_complete_fix/mturk_complete_DV_valence.json'
dv_fix = pd.read_json(updateDV)
valence_fix = pd.read_json(updatevalence)


overlap = pd.concat([dv[dv_fix.columns], dv_fix])
