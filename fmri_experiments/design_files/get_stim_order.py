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
import sys

task = sys.argv[1]
design_dir = os.path.join(task,task+'_designs_short')

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
    stim_order = [stims[i] for i in sort_index]
    stim_orders.append(stim_order)
    np.savetxt(os.path.join(directory,'stim_order.txt'),stim_order)
    blocks = get_blocks(stim_order)
    block_counts.append(blocks)



f = seaborn.plt.figure()
for i,block in enumerate(block_counts):
    seaborn.plt.subplot(3,2,i+1)
    seaborn.plt.hist(block)
f.suptitle(task + ' Block Histogram', fontsize = 16)
f.savefig(os.path.join(design_dir,'task_block_histogram.pdf'))