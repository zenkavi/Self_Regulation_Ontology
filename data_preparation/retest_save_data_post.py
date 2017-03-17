import sys
sys.path.append('/Users/zeynepenkavi/Dropbox/PoldrackLab/expfactory-analysis')
sys.path.append('/Users/zeynepenkavi/Documents/PoldrackLabLocal/Self_Regulation_Ontology/data_preparation')
import json
import numpy as np
from os import path
import pandas as pd
from glob import glob
from selfregulation.utils.data_preparation_utils import convert_var_names, drop_failed_QC_vars, drop_vars, remove_correlated_task_variables, remove_outliers, transform_remove_skew
from selfregulation.utils.r_to_py_utils import missForest

try: 
    release_date
except NameError:
    release_date = input('Enter release_ date:')
    

data_dir = data_dir=path.join('/Users/zeynepenkavi/Documents/PoldrackLabLocal/Self_Regulation_Ontology/Data/','Retest_'+release_date)

meta_dir = path.join(data_dir,'metadata')
reference_dir = path.join(data_dir,'references')
if not path.exists(meta_dir):
    makedirs(meta_dir)
if not path.exists(reference_dir):
    makedirs(reference_dir)

#Read in DVs and valence
label = 'retest'
DV_df = pd.read_json(path.join(data_dir,'Local/mturk_' + label + '_DV.json'))
valence_df = pd.read_json(path.join(data_dir,'Local/mturk_' + label + '_DV_valence.json'))
#data = pd.read_json(path.join(data_dir,'Local/mturk_' + label + '_data_post.json')).reset_index(drop = True)
data = pd.read_json(path.join(data_dir,'Local/mturk_' + label + '_data_manual_post.json')).reset_index(drop = True)

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
readme_lines += ["variables_exhaustive.csv: all variables calculated for each measure\n\n"]

# drop other columns of no interest
subset = drop_vars(DV_df, saved_vars = ['simple_reaction_time.avg_rt', 'shift_task.acc'])

#meaningful_variables_noDDM.csv
# make subset without EZ variables
noDDM_subset = drop_vars(DV_df, saved_vars = ["\.acc$", "\.avg_rt$"])
noDDM_subset = drop_vars(noDDM_subset, drop_vars = ['EZ', 'hddm'])
noDDM_subset.to_csv(path.join(data_dir, 'meaningful_variables_noDDM.csv'))
readme_lines += ["meaningful_variables_noDDM.csv: subset of exhaustive data to only meaningful variables with DDM parameters removed\n\n"]

#meaningful_variables_EZ.csv
# make subset without acc/rt vars and just EZ DDM
EZ_subset = drop_vars(subset, drop_vars = ['_acc', '_rt', 'hddm'], saved_vars = ['simple_reaction_time.avg_rt', 'dospert_rt_survey'])
EZ_subset.to_csv(path.join(data_dir, 'meaningful_variables_EZ.csv'))
readme_lines += ["meaningful_variables_EZ.csv: subset of exhaustive data to only meaningful variables with rt/acc parameters removed (replaced by EZ DDM params)\n\n"]
        
#meaningful_variables_hddm.csv
# make subset without acc/rt vars and just hddm DDM
hddm_subset = drop_vars(subset, drop_vars = ['_acc', '_rt', 'EZ'], saved_vars = ['simple_reaction_time.avg_rt', 'dospert_rt_survey'])
hddm_subset.to_csv(path.join(data_dir, 'meaningful_variables_hddm.csv'))
readme_lines += ["meaningful_variables_hddm.csv: subset of exhaustive data to only meaningful variables with rt/acc parameters removed (replaced by hddm DDM params)\n\n"]

#meaningful_variables.csv
# save files that are selected for use
selected_variables = hddm_subset
selected_variables.to_csv(path.join(data_dir, 'meaningful_variables.csv'))
readme_lines += ["meaningful_variables.csv: Same as meaningful_variables_hddm.csv\n\n"]

#meaningful_variables_clean.csv
# clean data
#selected_variables_clean = remove_outliers(selected_variables) #getting some warning
#selected_variables_clean = remove_correlated_task_variables(selected_variables_clean)
#selected_variables_clean = transform_remove_skew(selected_variables_clean)
#selected_variables_clean.to_csv(path.join(data_dir, 'meaningful_variables_clean.csv'))
#readme_lines += ["meaningful_variables_clean.csv: same as meaningful_variables.csv with outliers defined as greater than 2.5 IQR from median removed from each column\n\n"]

# Retest meaningful_variables_clean.csv mimicking the test one
# Instead of cleaning on this sample get variables resulting from test cleaning procedures
# DO NOT remove outliers from this sample and transform based on what was done for test data
meaningful_variables_clean_test = pd.read_csv('/Users/zeynepenkavi/Documents/PoldrackLabLocal/Self_Regulation_Ontology/Data/Complete_01-31-2017/meaningful_variables_clean.csv')
transformed_variables = [col for col in meaningful_variables_clean_test.columns if 'logTr' in col]
signs = ['negative' if 'ReflogTr' in x else 'positive' for x in transformed_variables]
transformed_variables = pd.DataFrame({'var': transformed_variables, 'signs': signs})
del(signs)
transformed_variables['var'] = [x.replace('.logTr','').replace('.ReflogTr','') for x in transformed_variables['var']]
#first transform the subset for retest
for col in selected_variables.columns:
    if col in transformed_variables['var']:
        sign = 

#then drop the columns that are not in meaningful_variables_clean_test
#save to csv

#save selected variables
selected_variables_reference = valence_df
selected_variables_reference.loc[selected_variables.columns].to_csv(path.join(reference_dir, 'selected_variables_reference.csv'))

#meaningful_variables_imputed.csv
# imputed data
selected_variables_imputed, error = missForest(selected_variables_clean)
selected_variables_imputed.to_csv(path.join(data_dir, 'meaningful_variables_imputed.csv'))
readme_lines += ["meaningful_variables_imputed.csv: meaningful_variables_clean.csv after imputation with missForest\n\n"]

#taskdata.csv
#taskdata_clean.csv
#taskdata_imputed.csv
#taskdata_imputed_for_task_selection.csv
# save task data subset
task_data = drop_vars(selected_variables, ['survey'], saved_vars = ['holt','cognitive_reflection'])
task_data.to_csv(path.join(data_dir, 'taskdata.csv'))
task_data_clean = drop_vars(selected_variables_clean, ['survey'], saved_vars = ['holt','cognitive_reflection'])
task_data_clean.to_csv(path.join(data_dir, 'taskdata_clean.csv'))
task_data_imputed = drop_vars(selected_variables_imputed, ['survey'], saved_vars = ['holt','cognitive_reflection'])
task_data_imputed.to_csv(path.join(data_dir, 'taskdata_imputed.csv'))
readme_lines += ["taskdata*.csv: taskdata are the same as meaningful_variables excluded surveys. Note that imputation is performed on the entire dataset including surveys\n\n"]


#meaningful_variables_imputed_for_task_selection.csv
# create task selection dataset
task_selection_data = drop_vars(selected_variables_imputed, ['stop_signal.SSRT_low', '^stop_signal.proactive'])
task_selection_data.to_csv(path.join(data_dir,'meaningful_variables_imputed_for_task_selection.csv'))
task_selection_taskdata = drop_vars(task_data_imputed, ['stop_signal.SSRT_low', '^stop_signal.proactive'])
task_selection_taskdata.to_csv(path.join(data_dir,'taskdata_imputed_for_task_selection.csv'))
        
#save selected variables
selected_variables_reference.loc[task_selection_data.columns].to_csv(path.join(reference_dir, 'selected_variables_for_task_selection_reference.csv'))

#short_DV_valence.csv
#short_meaningful_variables.csv
#short_meaningful_variables_EZ.csv
#short_meaningful_variables_clean.csv
#short_meaningful_variables_hddm.csv
#short_meaningful_variables_imputed.csv
#short_meaningful_variables_imputed_for_task_selection.csv
#short_meaningful_variables_noDDM.csv
#short_subject_x_items.csv
#short_taskdata.csv
#short_taskdata_clean.csv
#short_taskdata_imputed.csv
#short_taskdata_imputed_for_task_selection.csv
#short_variables_exhaustive.csv

files = glob(path.join(data_dir,'*csv'))
files = [f for f in files if not any(i in f for i in ['demographic','health','alcohol_drug'])]
from selfregulation.utils.data_preparation_utils import convert_var_names
for f in files:
    name = f.split('/')[-1]
    df = pd.DataFrame.from_csv(f)
    convert_var_names(df)
    df.to_csv(path.join(data_dir, 'short_' + name))
    print('short_' + name)
readme_lines += ["short*.csv: short versions are the same as long versions with variable names shortened using variable_name_lookup.csv\n\n"]

readme = open(path.join(data_dir, "README.txt"), "a")
readme.writelines(readme_lines)
readme.close()