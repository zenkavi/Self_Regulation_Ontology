import glob
import os
import pandas

#discovery
DVs = pandas.DataFrame()
valence = pandas.DataFrame()
for exp_file in glob.glob(os.path.join('output', '*discovery*DV.json')):
    base_name = os.path.basename(exp_file)
    exp = base_name.replace('_discovery_DV.json','')
    print('Discovery: Extracting %s DVs' % exp)
    exp_DVs = pandas.read_json(exp_file)
    exp_valence = pandas.read_json(exp_file.replace('.json','_valence.json'))
    exp_DVs.columns = [exp + '.' + c for c in exp_DVs.columns]
    exp_valence.columns = [exp + '.' + c for c in exp_valence.columns]
    DVs = pandas.concat([DVs,exp_DVs], axis = 1)
    valence = pandas.concat([valence,exp_valence], axis = 1)

DVs.to_json('output/mturk_discovery_DV.json')
valence.to_json('output/mturk_discovery_DV_valence.json')

#validation
DVs = pandas.DataFrame()
valence = pandas.DataFrame()
for exp_file in glob.glob(os.path.join('output', '*validation*DV.json')):
    base_name = os.path.basename(exp_file)
    exp = base_name.replace('_validation_DV.json','')
    print('Validation: Extracting %s DVs' % exp)
    exp_DVs = pandas.read_json(exp_file)
    exp_valence = pandas.read_json(exp_file.replace('.json','_valence.json'))
    exp_DVs.columns = [exp + '.' + c for c in exp_DVs.columns]
    exp_valence.columns = [exp + '.' + c for c in exp_valence.columns]
    DVs = pandas.concat([DVs,exp_DVs], axis = 1)
    valence = pandas.concat([valence,exp_valence], axis = 1)

DVs.to_json('output/mturk_validation_DV.json')
valence.to_json('output/mturk_validation_DV_valence.json')
