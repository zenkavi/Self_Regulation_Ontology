from os import path
import pandas as pd

data_dir = path.join('/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data','Retest_11-27-2017/Local')

#Read in DVs and valence
label = 'hddm_refit'
DV_df = pd.read_json(path.join(data_dir,'mturk_' + label + '_DV.json'))
valence_df = pd.read_json(path.join(data_dir,'mturk_' + label + '_DV_valence.json'))

readme_lines = []

#drop na columns
DV_df.dropna(axis = 1, how = 'all', inplace = True)
DV_df.to_csv(path.join(data_dir, 'hddm_refits_exhaustive.csv'))
readme_lines += ["hddm_refits_exhaustive.csv: all variables for hddm's for retest subjects' t1 data\n\n"]

readme = open(path.join(data_dir, "README.txt"), "a")
readme.writelines(readme_lines)
readme.close()