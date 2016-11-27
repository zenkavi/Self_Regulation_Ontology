#!/usr/bin/env python

import sys
sys.path.append('/corral-repl/utexas/poldracklab/users/zenkavi/expfactory-analysis')
import json
import numpy as np
from os import path
import os
import pandas as pd
import time

sys.path.append('/corral-repl/utexas/poldracklab/users/zenkavi/Self_Regulation_Ontology/utils')
from data_preparation_utils import anonymize_data, calc_trial_order, convert_date, download_data, get_bonuses, get_pay,  remove_failed_subjects
from utils import get_info

if len(sys.argv) < 4:
    sys.exit("Usage: make_commands_for_download_retest_data_tacc.py start_page end_page split_by out_file")

start_page = sys.argv[1]
end_page = sys.argv[2]
split_by = sys.argv[3]
out_file = sys.argv[4]
    
out = '#!/bin/bash'

num_commands = ((int(end_page) - int(start_page))/int(split_by)) + 1

command_start_page = 'http://expfactory.org/api/results/?page=' + start_page
if start_page == '1':
    command_start_page = 'http://expfactory.org/api/results'
command_end_page = 'http://expfactory.org/api/results/?page=' + str(int(start_page)+int(split_by))

for i in range(num_commands):
    command = "python /corral-repl/utexas/poldracklab/users/zenkavi/Self_Regulation_Ontology/data_preparation/download_retest_data_tacc.py 
'Self Regulation Retest Battery'" + " " + command_start_page 
    out += command
    command_start_page = command_end_page
    command_end_page = command_end

"python /corral-repl/utexas/poldracklab/users/zenkavi/Self_Regulation_Ontology/data_preparation/download_retest_data_tacc.py 
'Self Regulation Retest Battery' 
'http://expfactory.org/api/results' 
'http://expfactory.org/api/results/?page=3' 
'1'"