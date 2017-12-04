from os import path
import pandas as pd
from selfregulation.utils.utils import get_info

try:
    data_dir=get_info('data_directory')
except Exception:
    data_dir=path.join(get_info('base_directory'),'Data')

discovery = pd.read_pickle(path.join(data_dir, 'mturk_discovery_data_post.pkl'))
validation = pd.read_pickle(path.join(data_dir, 'mturk_validation_data_post.pkl'))

complete = pd.concat([discovery, validation])
complete.to_pickle(path.join(data_dir, 'mturk_complete_data_post.pkl'))