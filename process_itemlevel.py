# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy,pandas
import os

basedir='/Users/poldrack/code/Self_Regulation_Ontology'

datafile=os.path.join(basedir,'items.csv')



data=pandas.read_csv(datafile,index_col=0)

