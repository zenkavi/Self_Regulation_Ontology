#!/usr/bin/env python3
import glob
import os
import pandas

#discovery
DVs = pandas.DataFrame()
valence = pandas.DataFrame()
for exp_file in glob.glob(os.path.join('output', '*fmri*DV.json')):
    base_name = os.path.basename(exp_file)
    exp = base_name.replace('_mturk_discovery_DV.json','')
    print('fMRI: Extracting %s DVs' % exp)
    exp_DVs = pandas.read_json(exp_file)
    exp_valence = pandas.read_json(exp_file.replace('.json','_valence.json'))
    exp_DVs.columns = [exp + '.' + c for c in exp_DVs.columns]
    exp_valence.columns = [exp + '.' + c for c in exp_valence.columns]
    DVs = pandas.concat([DVs,exp_DVs], axis = 1)
    valence = pandas.concat([valence,exp_valence], axis = 1)

DVs.to_json('output/fmri_followup_DV.json')
valence.to_json('output/fmri_followup_DV_valence.json')

