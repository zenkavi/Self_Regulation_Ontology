#!/usr/bin/env python3
import glob
import os
import pandas



# concatenate subsets, if necessary
subset1_loc = '/oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_complete_subset1_output'
subset2_loc = '/oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_complete_subset2_output'


#complete
output_loc = '/oak/stanford/groups/russpold/users/ieisenbe/Self_Regulation_Ontology/behavioral_data/mturk_complete_output'

DVs = pandas.DataFrame()
valence = pandas.DataFrame()
for exp_file in glob.glob(os.path.join(output_loc, '*complete*DV.json')):
    base_name = os.path.basename(exp_file)
    exp = base_name.replace('_mturk_complete_DV.json','')
    print('Complete: Extracting %s DVs' % exp)
    exp_DVs = pandas.read_json(exp_file)
    exp_valence = pandas.read_json(exp_file.replace('.json','_valence.json'))
    exp_DVs.columns = [exp + '.' + c for c in exp_DVs.columns]
    exp_valence.columns = [exp + '.' + c for c in exp_valence.columns]
    DVs = pandas.concat([DVs,exp_DVs], axis = 1)
    valence = pandas.concat([valence,exp_valence], axis = 1)

DVs.to_json(os.path.join(output_loc, 'mturk_complete_DV.json'))
valence.to_json(os.path.join(output_loc, 'mturk_complete_DV_valence.json'))