import glob
import os
import pandas

#retest
DVs = pandas.DataFrame()
valence = pandas.DataFrame()
for exp_file in glob.glob(os.path.join('/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_11-27-2017/batch_output', '*retest*DV.json')):
    base_name = os.path.basename(exp_file)
    exp = base_name.replace('_retest_DV.json','')
    print('Retest: Extracting %s DVs' % exp)
    exp_DVs = pandas.read_json(exp_file)
    exp_valence = pandas.read_json(exp_file.replace('.json','_valence.json'))
    exp_DVs.columns = [exp + '.' + c for c in exp_DVs.columns]
    exp_valence.columns = [exp + '.' + c for c in exp_valence.columns]
    DVs = pandas.concat([DVs,exp_DVs], axis = 1)
    valence = pandas.concat([valence,exp_valence], axis = 1)

DVs.to_json('/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_11-27-2017/Local/mturk_retest_DV.json')
valence.to_json('/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_11-27-2017/Local/mturk_retest_DV_valence.json')
