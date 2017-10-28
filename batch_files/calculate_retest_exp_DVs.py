import argparse
import sys
sys.path.append('/scratch/users/zenkavi/expfactory-analysis')
from expanalysis.experiments.processing import get_exp_DVs
from os import path
import pandas as pd

data_dir='/scratch/users/zenkavi/Self_Regulation_Ontology/Data/Retest_02-11-2017/Local'

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('exp_id')
parser.add_argument('data')

exp_id = args.exp_id
data = args.data
out_dir = '/scratch/users/zenkavi/Self_Regulation_Ontology/Data/Retest_02-11-2017/batch_output'

#discovery
#load Data
dataset = pd.read_json(path.join(data_dir,'mturk_' + data + '_data_manual_post.json'))

#calculate DVs
DV_df, valence_df, description = get_exp_DVs(dataset, exp_id, use_group_fun = True)
if not DV_df is None:
    DV_df.to_json(path.join(out_dir, exp_id + '_' + data + '_DV.json'))
    valence_df.to_json(path.join(out_dir, exp_id + '_' + data + '_DV_valence.json'))
print('completed %s %s' % (data, exp_id))

