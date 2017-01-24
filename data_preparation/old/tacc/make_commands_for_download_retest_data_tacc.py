#!/usr/bin/env python

import sys
sys.path.append('/corral-repl/utexas/poldracklab/users/zenkavi/expfactory-analysis')
from os import path
import os

if len(sys.argv) < 3:
    sys.exit("Usage: make_commands_for_download_retest_data_tacc.py start_page end_page split_by num_commands_in_file")

start_page = sys.argv[1]
end_page = sys.argv[2]
split_by = sys.argv[3]
num_commands_in_file = sys.argv[4]
    
out = ''

num_commands = ((int(end_page) - int(start_page))/int(split_by)) + 1

command_head = 'python /corral-repl/utexas/poldracklab/users/zenkavi/Self_Regulation_Ontology/data_preparation/download_retest_data_tacc.py "Self Regulation Retest Battery"'

command_start_page = '"http://expfactory.org/api/results/?page=' + start_page + '"'
if start_page == '1':
    command_start_page = '"http://expfactory.org/api/results"'
command_end_page = '"http://expfactory.org/api/results/?page=' + str(int(start_page)+int(split_by)-1)+ '"'

#Make all commands
for i in range(num_commands):
    command = command_head + " " + command_start_page + " " + command_end_page + ' "' + str(i+1) + '"'+ "\n"
    out += command
    command_start_page = command_end_page
    command_end_page = command_start_page.split('=')[0]+"="+str(int(command_start_page.split('=')[1].split('"')[0])+int(split_by))+'"'
    
#Split in to exec files
for j in range(int(num_commands)/int(num_commands_in_file)+1):
    f = open('./launch_scripts/exec_download_retest_data_tacc_'+str(j+1)+'.sh', 'w')
    f.write('#!/bin/bash' + "\n")
    cut = '\n'.join(out.splitlines()[j*int(num_commands_in_file): ((j+1)*int(num_commands_in_file))+1])
    f.write(cut)