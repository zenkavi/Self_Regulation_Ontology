#Read in DVs and valence
label = 'retest'
DVs = pd.read_json(path.join(data_dir,'Local/mturk_' + label + '_DV.json'))
DVs_valence = pd.read_json(path.join(data_dir,'Local/mturk_' + label + '_DV_valence.json'))
data = pd.read_json(path.join(data_dir,'Local/mturk_' + label + '_data_manual_post.json')).reset_index(drop = True)

DV_df = DVs
valence_df = DVs_valence

del DVs, DVs_valence

# drop failed QC vars
from selfregulation.utils.data_preparation_utils import drop_failed_QC_vars
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
readme_lines += ["DV_valence.csv: Subjective assessment of whether each variable's 'natural' direction implies 'better' self regulation\n\n"]

#variables_exhaustive.csv
#drop na columns
DV_df.dropna(axis = 1, how = 'all', inplace = True)
DV_df.to_csv(path.join(data_dir, 'variables_exhaustive.csv'))
readme_lines += ["variables_exhaustive.csv: all variables calculated for each measure\n\n"]

# drop other columns of no interest
from selfregulation.utils.data_preparation_utils import drop_vars
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
from selfregulation.utils.data_preparation_utils import remove_correlated_task_variables, remove_outliers, transform_remove_skew
selected_variables_clean = remove_outliers(selected_variables) #getting some warning
selected_variables_clean = remove_correlated_task_variables(selected_variables_clean)
selected_variables_clean = transform_remove_skew(selected_variables_clean)
selected_variables_clean.to_csv(path.join(data_dir, 'meaningful_variables_clean.csv'))
readme_lines += ["meaningful_variables_clean.csv: same as meaningful_variables.csv with outliers defined as greater than 2.5 IQR from median removed from each column\n\n"]

#save selected variables
selected_variables_reference = valence_df
selected_variables_reference.loc[selected_variables.columns].to_csv(path.join(reference_dir, 'selected_variables_reference.csv'))

#meaningful_variables_imputed.csv
# imputed data
from selfregulation.utils.r_to_py_utils import missForest
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
from glob import glob
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