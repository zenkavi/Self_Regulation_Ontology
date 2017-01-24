from expanalysis.experiments.processing import get_exp_DVs
from os import path
import pandas as pd
import sys

from selfregulation.utils.utils import get_info

try:
    data_dir=get_info('data_directory')
except Exception:
    data_dir=path.join(get_info('base_directory'),'Data')

#parse arguments
exp_id = sys.argv[1]
data = sys.argv[2]
if len(sys.argv) > 3:
    out_dir = sys.argv[3]
else:
    out_dir = data_dir

#discovery
#load Data
dataset = pd.read_json(path.join(data_dir,'mturk_' + data + '_data_post.json'))

#calculate DVs
DV_df, valence_df, description = get_exp_DVs(dataset, exp_id, use_group_fun = True)
if not DV_df is None:
    DV_df.to_json(path.join(out_dir, exp_id + '_' + data + '_DV.json'))
    valence_df.to_json(path.join(out_dir, exp_id + '_' + data + '_DV_valence.json'))
print('completed %s %s' % (data, exp_id))

