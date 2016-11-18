from expanalysis.experiments.processing import get_exp_DVs
from os import path
import pandas as pd
import sys

sys.path.append('../utils')
from utils import get_info


#load Data
token = get_info('expfactory_token')
try:
    data_dir=get_info('data_directory')
except Exception:
    data_dir=path.join(get_info('base_directory'),'Data')
print(data_dir)  
#parse arguments
exp_id = sys.argv[1]
if len(sys.argv) > 2:
    out_dir = sys.argv[2]
else:
    out_dir = data_dir

# load data
data = pd.read_json(data_dir + 'mturk_discovery_data_post.json')
#calculate DVs
DV_df, valence_df = get_exp_DVs(data, exp_id)
DV_df.to_json(path.join(out_dir, 'mturk_discovery_DV.json'))
valence_df.to_json(path.join(out_dir, 'mturk_discovery_DV_valence.json'))