#!/usr/bin/env python3
import argparse
from expanalysis.experiments.processing import get_exp_DVs
from os import path
import pandas as pd

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
parser.add_argument('--no_group', action='store_false')
parser.add_argument('--hddm_model_file', default=None)
parser.add_argument('--hddm_samples', default=None)

args = parser.parse_args()

exp_id = args.exp_id
data = args.data
out_dir = args.out_dir
use_group = args.no_group
model_file = args.hddm_model_file
hddm_samples = args.hddm_samples
if hddm_samples is not None:
    hddm_samples = int(hddm_samples)

#load Data
dataset = pd.read_pickle(path.join(data_dir, data + '_data_post.pkl'))

print('loaded dataset %s' % data)
#calculate DVs
if hddm_samples is None:
    DV_df, valence_df, description = get_exp_DVs(dataset, exp_id, 
                                                 use_group_fun = use_group,
                                                 outfile = path.join(out_dir, exp_id),
                                                 loadfile = model_file)
else:
    burn = min(hddm_samples//10, 10000)
    DV_df, valence_df, description = get_exp_DVs(dataset, exp_id, 
                                                 use_group_fun = use_group,
                                                 outfile = path.join(out_dir, exp_id),
                                                 loadfile = model_file,
                                                 samples = hddm_samples,
                                                 burn = burn)
    
if not DV_df is None:
    DV_df.to_json(path.join(out_dir, exp_id + '_' + data + '_DV.json'))
    valence_df.to_json(path.join(out_dir, exp_id + '_' + data + '_DV_valence.json'))
print('completed %s %s' % (data, exp_id))
