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


import os,glob
import pandas
import json

basedir='/Users/poldrack/code/Self_Regulation_Ontology/discovery_survey_analyses'


try:
    data=pandas.read_csv('surveydata.csv')
except:
    files=glob.glob(os.path.join(basedir,'surveydata/*'))
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
    if numpy.sum(h<min_freq)==0:
        return d,False
    badvals=numpy.where(h<min_freq)[0]
    if verbose:
        print(c,'found %d bad vals'%len(badvals),'resp',u,u[badvals],'freq',h[badvals])
    if len(badvals)==1:
        # is it at an extreme?
        if u[badvals[0]]==u[0]:
            d.loc[d==u[badvals[0]]]=u[1]
            
        elif u[badvals[0]]==u[-1]:
            d.loc[d==u[badvals[0]]]=u[-2]
        else:
            if verbose:
                    print('midval - replacing with NaN!')
            d.loc[d==u[badvals[0]]]=numpy.nan
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
            d.loc[d==u[badvals[0]]]=numpy.nan
            d.loc[d==u[badvals[1]]]=numpy.nan
        return d,False
    else:
        print('dropping',c)
        return d,True
        
    
fixdata=data.copy()
dropped={}
for c in data.columns:
    if c=='worker':
        continue
    u=data[c].unique()
    u.sort()
    h=[numpy.sum(data[c]==i) for i in u]
    f,dropflag=truncate_dist(u,h,fixdata[c],verbose=True)
    fixdata[c]=f
    if dropflag:
        del fixdata[c]
        dropped[c]=(u,h)
    
fixdata.to_csv('surveydata_fixed.csv',index=False)         
