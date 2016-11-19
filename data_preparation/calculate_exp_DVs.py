from expanalysis.experiments.processing import get_exp_DVs
from os import path
import pandas as pd
import sys

sys.path.append('../utils')
from utils import get_info


#load Data
try:
    data_dir=get_info('data_directory')
except Exception:
    data_dir=path.join(get_info('base_directory'),'Data')
data = pd.read_json(path.join(data_dir,'mturk_discovery_data_post.json'))

#parse arguments
exp_id = sys.argv[1]
if len(sys.argv) > 2:
    out_dir = sys.argv[2]
else:
    out_dir = data_dir

#calculate DVs
DV_df, valence_df, description = get_exp_DVs(data, exp_id, use_group_fun = True)
if not DV_df is None:
    DV_df.to_json(path.join(out_dir, exp_id + '_discovery_DV.json'))
    valence_df.to_json(path.join(out_dir, exp_id + '_discovery_DV_valence.json'))