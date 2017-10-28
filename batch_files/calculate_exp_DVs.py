import argparse
from expanalysis.experiments.processing import get_exp_DVs
from os import path
import pandas as pd
import sys

from selfregulation.utils.utils import get_info

try:
    data_dir=get_info('data_directory')
except Exception:
    data_dir=path.join(get_info('base_directory'),'Data')


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('exp_id')
parser.add_argument('data')
parser.add_argument('--out_dir', default=data_dir)

args = parser.parse_args()

exp_id = args.exp_id
data = args.data
out_dir = args.out_dir

print(exp_id, data, out_dir)
#load Data
dataset = pd.read_pickle(path.join(data_dir, data + '_data_post.pkl'))

print('loaded dataset %s' % data)
#calculate DVs
DV_df, valence_df, description = get_exp_DVs(dataset, exp_id, use_group_fun = True)
if not DV_df is None:
    DV_df.to_json(path.join(out_dir, exp_id + '_' + data + '_DV.json'))
    valence_df.to_json(path.join(out_dir, exp_id + '_' + data + '_DV_valence.json'))
print('completed %s %s' % (data, exp_id))

