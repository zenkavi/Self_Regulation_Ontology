#!/usr/bin/env python3
from os import path
import pandas as pd
from selfregulation.utils.utils import get_info

try:
    data_dir=get_info('data_directory')
except Exception:
    data_dir=path.join(get_info('base_directory'),'Data')

# concatenate discovery and validation data into one complete
discovery = pd.read_pickle(path.join(data_dir, 'mturk_discovery_data_post.pkl'))
validation = pd.read_pickle(path.join(data_dir, 'mturk_validation_data_post.pkl'))

complete = pd.concat([discovery, validation])
complete.to_pickle(path.join(data_dir, 'mturk_complete_data_post.pkl'))

# separate complete into two data subsets for particularly memory intensive analyses (DDM)

workers = data.worker_id.unique()
mid = len(workers)//2
subset1 = data.query('worker_id in %s' % list(workers)[:mid])
subset2 = data.query('worker_id in %s' % list(workers)[mid:])
subset1.to_pickle(path.join(data_dir, 'mturk_complete_subset1_data_post.pkl'))
subset2.to_pickle(path.join(data_dir, 'mturk_complete_subset2_data_post.pkl'))
