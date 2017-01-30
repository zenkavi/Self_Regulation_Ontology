#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:50:26 2017

@author: ian
"""
from glob import glob
import numpy as np
import os
import seaborn

for task in ['attention_network_task', 'dot_pattern_expectancy', 
             'motor_selective_stop_signal', 'stop_signal', 'stroop',
             'twobytwo', 'ward_and_allport']:
    design_dir = os.path.join(task,task+'_designs_1')
    if os.path.exists(design_dir):
	    
	    def get_blocks(order):
	        blocks = []
	        previous_stim = None
	        block_len = 0
	        for i in stim_order:
	            if i != previous_stim and previous_stim != None:
	                blocks.append(block_len)
	                block_len = 1
	            else:
	                block_len+=1
	            previous_stim = i
	        return blocks
	    
	    block_counts = []
	    stim_orders = []
	    for directory in glob(os.path.join(design_dir,'design*')):
	        stim_onsets = []
	        stims = []
	        stim_i = 0
	        for stim_file in glob(os.path.join(directory,'stimulus*')):
	            stim_onset=list(np.loadtxt(stim_file))
	            stim_onsets+=stim_onset
	            stims+= [stim_i]*len(stim_onset)
	            stim_i+=1
	        sort_index = np.argsort(stim_onsets)
	        stim_order = [str(stims[i]) for i in sort_index]
	        stim_orders.append(stim_order)
	        # write stim_order
	        stim_order_file = open(os.path.join(directory,'stim_order.txt'), "w")
	        stim_order_file.write(','.join(map(str,stim_order)))
	        stim_order_file.close()
	        # write ITI in an easier to copy form
	        ITIs = np.loadtxt(os.path.join(directory,'ITIs.txt'))
	        ITI_file = open(os.path.join(directory,'ITIs_to_copy.txt'), "w")
	        ITI_file.write(','.join(map(str,ITIs)))
	        ITI_file.close()
	        blocks = get_blocks(stim_order)
	        block_counts.append(blocks)
	        print('task: %s, length: %s, ITIs mean: %s' %(task,len(stim_onsets),np.mean(ITIs)))
	    
	  
	    f = seaborn.plt.figure()
	    for i,block in enumerate(block_counts):
	        seaborn.plt.subplot(3,2,i+1)
	        seaborn.plt.hist(block)
	    f.suptitle(task + ' Block Histogram', fontsize = 16)
	    f.savefig(os.path.join(design_dir,'task_block_histogram.pdf'))
