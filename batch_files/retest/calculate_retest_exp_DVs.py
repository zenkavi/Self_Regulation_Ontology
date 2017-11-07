import sys
sys.path.append('/oak/stanford/groups/russpold/users/zenkavi/expfactory-analysis')
from expanalysis.experiments.processing import get_exp_DVs
from os import path
import pandas as pd

data_dir='/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_09-27-2017/Local'

#parse arguments
exp_id = sys.argv[1]
data = sys.argv[2]
#if len(sys.argv) > 3:
#    out_dir = sys.argv[3]
#else:
 #   out_dir = data_dir

out_dir = '/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Retest_09-27-2017/batch_output'

#discovery
#load Data
dataset = pd.read_json(path.join(data_dir,'mturk_' + data + '_data_post.json'))

#calculate DVs
DV_df, valence_df, description = get_exp_DVs(dataset, exp_id, use_group_fun = True)
if not DV_df is None:
    DV_df.to_json(path.join(out_dir, exp_id + '_' + data + '_DV.json'))
    valence_df.to_json(path.join(out_dir, exp_id + '_' + data + '_DV_valence.json'))
print('completed %s %s' % (data, exp_id))
