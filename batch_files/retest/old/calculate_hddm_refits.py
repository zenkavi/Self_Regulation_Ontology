import sys
from expanalysis.experiments.processing import get_exp_DVs
from os import path
import pandas as pd

exp_id = sys.argv[1]
data = sys.argv[2]

out_dir = '/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_01-15-2018/batch_output/hddm_refits'

#load Data
dataset = pd.read_json('/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_01-15-2018/Local/retest_subs_test_data_post.json')

#calculate DVs
DV_df, valence_df, description = get_exp_DVs(dataset, exp_id, use_group_fun = True)
if not DV_df is None:
    DV_df.to_json(path.join(out_dir, exp_id + '_' + data + '_DV.json'))
    valence_df.to_json(path.join(out_dir, exp_id + '_' + data + '_DV_valence.json'))
print('completed %s %s' % (data, exp_id))

