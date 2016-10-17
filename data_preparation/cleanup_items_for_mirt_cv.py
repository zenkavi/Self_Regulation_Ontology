#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
need to clean up individual items that have a small number of partiuclar
responses in order to allow crossvalidation with equated response options

Created on Fri Sep  2 15:04:04 2016

@author: poldrack
"""

import os
import pandas,numpy
import scipy.stats


import os,glob,sys
import pandas
import json

sys.path.append('../utils')

from utils import get_info
basedir=get_info('base_directory')
dataset=get_info('dataset')
print('using dataset:',dataset)
datadir=os.path.join(basedir,'data/%s'%dataset)
derived_dir=os.path.join(basedir,'data/Derived_Data/%s'%dataset)


try:
    data=pandas.read_csv(os.path.join(derived_dir,'surveydata.csv'))
except:
    files=glob.glob(os.path.join(derived_dir,'surveydata/*'))
    files.sort()
    pass

    all_metadata={}
    for f in files:

        d=pandas.read_csv(f,sep='\t')
        code=f.split('/')[-1].replace('.tsv','')
        try:
            data=data.merge(d,on='worker')
        except:
            data=d

def truncate_dist(u,h,d,min_freq=4,verbose=False):
    """
    remove responses with less than min_freq occurences, replacing with
    next less extreme response
    heuristic is that we always move it towards the middle of the scale
    - if they are in teh middle, of if more than two, then drop the column

    returns amended data, and drop flag
    """
    u=numpy.array(u)
    h=numpy.array(h)
    dmode=scipy.stats.mode(d,nan_policy='omit')[0][0]
    if numpy.sum(h<min_freq)==0:
        return d,False
    badvals=numpy.where(h<min_freq)[0]
    if verbose:
        print(c,'found %d bad vals'%len(badvals),'resp',u,u[badvals],'freq',h[badvals])
    if len(badvals)==1:
        if len(u)==2:
            # if it's dichotomous, then drop it
            print('dropping',c)
            return d,True
        # is it at an extreme?
        if u[badvals[0]]==u[0]:
            d.loc[d==u[badvals[0]]]=u[1]

        elif u[badvals[0]]==u[-1]:
            d.loc[d==u[badvals[0]]]=u[-2]
        else:
            if verbose:
                    print('midval - replacing with NaN!')
            d.loc[d==u[badvals[0]]]=dmode
            #raise Exception("Can't deal with sole middle values!")
        return d,False
    elif len(badvals)==2:
        #are they at the extreme?
        if all(u[badvals]==u[:2]):
            d.loc[d==u[badvals[0]]]=u[2]
            d.loc[d==u[badvals[1]]]=u[2]
        elif all(u[badvals]==u[-2:]):
            d.loc[d==u[badvals[0]]]=u[-3]
            d.loc[d==u[badvals[1]]]=u[-3]
        else:
            if verbose:
                    print('midval - replacing with NaN!')
            d.loc[d==u[badvals[0]]]=dmode
            d.loc[d==u[badvals[1]]]=dmode
        return d,False
    else:
        print('dropping',c)
        return d,True


min_freq = 8

fixdata=data.copy()
dropped={}
for c in data.columns:
    if c=='worker':
        continue
    u=data[c].unique()
    u.sort()
    h=[numpy.sum(data[c]==i) for i in u]
    f,dropflag=truncate_dist(u,h,fixdata[c],verbose=True,min_freq=min_freq)
    fixdata[c]=f
    if dropflag:
        del fixdata[c]
        dropped[c]=(u,h)

fixdata.to_csv(os.path.join(derived_dir,'surveydata_fixed_minfreq%d.csv'%min_freq),index=False)
