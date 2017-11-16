import sys
#sys.path.append('/oak/stanford/groups/russpold/users/zenkavi/expfactory-analysis')
#sys.path.append('/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/data_preparation')
import numpy as np
from os import path, chdir, makedirs
import pandas as pd
from glob import glob
from selfregulation.utils.data_preparation_utils import convert_var_names, drop_failed_QC_vars, drop_vars, remove_correlated_task_variables, remove_outliers, transform_remove_skew
from selfregulation.utils.r_to_py_utils import missForest

release_date = '10-27-2017'

data_dir = '/oak/stanford/groups/russpold/users/zenkavi/Self_Regulation_Ontology/Data/Complete_10-27-2017'

meta_dir = path.join(data_dir,'metadata')
reference_dir = path.join(data_dir,'references')
if not path.exists(meta_dir):
    makedirs(meta_dir)
if not path.exists(reference_dir):
    makedirs(reference_dir)

#Read in DVs and valence
label = 'complete'
DV_df = pd.read_json(path.join(data_dir,'Local/mturk_' + label + '_DV.json'))
valence_df = pd.read_json(path.join(data_dir,'Local/mturk_' + label + '_DV_valence.json'))
data = pd.read_pickle(path.join(data_dir, 'Local/mturk_'+label+'_data_post.pkl')).reset_index(drop = True)
#data = pd.read_json(path.join(data_dir,'Local/mturk_' + label + '_data_manual_post.json')).reset_index(drop = True)

# drop failed QC vars
drop_failed_QC_vars(DV_df,data)

#save valence
def get_flip_list(valence_df):
        #flip negative signed valence DVs
        valence_df = valence_df.replace(to_replace={np.nan: 'NA'})
        flip_df = np.floor(valence_df.replace(to_replace ={'Pos': 1, 'NA': 1, 'Neg': -1}).mean())
        valence_df = pd.Series(data = [col.unique()[0] for i,col in valence_df.iteritems()], index = valence_df.columns)
        return flip_df, valence_df
flip_df, valence_df = get_flip_list(valence_df)
flip_df.to_csv(path.join(data_dir, 'DV_valence.csv'))

readme_lines = []
readme_lines += ["DV_valence.csv: Subjective assessment of whether each variable's 'natural' direction implies 'better' self regulation\n\n"]

#variables_exhaustive.csv
#drop na columns
DV_df.dropna(axis = 1, how = 'all', inplace = True)
DV_df.to_csv(path.join(data_dir, 'variables_exhaustive.csv'))