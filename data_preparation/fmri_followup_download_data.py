from expanalysis.experiments.jspsych import calc_time_taken, get_post_task_responses
from expanalysis.experiments.processing import post_process_data
from expanalysis.results import get_filters
from os import path
import pandas as pd
from selfregulation.utils.data_preparation_utils import calc_trial_order, \
    convert_date, convert_fmri_ids, download_data, get_bonuses, get_fmri_pay
from selfregulation.utils.utils import get_info

#load Data
token = get_info('expfactory_token')
try:
    data_dir=get_info('data_directory')
except Exception:
    data_dir=path.join(get_info('base_directory'),'Data')

#***************************************************
# ********* Load Data **********************
#**************************************************        
pd.set_option('display.width', 200)
figsize = [16,12]
#set up filters
filters = get_filters()
drop_columns = ['battery_description', 'experiment_reference', 'experiment_version', \
         'experiment_name','experiment_cognitive_atlas_task']
for col in drop_columns:
    filters[col] = {'drop': True}

#***************************************************
# ********* Download Data**********************
#**************************************************  
#load Data
f = open(token)
access_token = f.read().strip()  
data = download_data(data_dir, access_token, filters = filters,  
                     battery = 'Self Regulation fMRI Battery',
                     url = 'http://www.expfactory.org/new_api/results/63/',
                     file_name = 'fmri_followup_data.json')

data.reset_index(drop = True, inplace = True)
    
#***************************************************
# ********* Add extras to data **********************
#**************************************************  
#anonymize data
convert_fmri_ids(data)

# record subject completion statistics
(data.groupby('worker_id').count().finishtime).to_json(path.join(data_dir, 'admin', 'fmri_followup_worker_counts.json'))

# add a few extras
convert_date(data)
bonuses = get_bonuses(data)
calc_time_taken(data)
get_post_task_responses(data)   
calc_trial_order(data)

# save data
data.to_pickle(path.join(data_dir, 'fmri_followup_data_extras.pkl'))

# calculate pay
pay = get_fmri_pay(data)
pay.to_json(path.join(data_dir, 'admin', 'fmri_followup_worker_pay.json'))
print('Finished saving worker pay')

#***************************************************
# ********* Post process data **********************
#************************************************** 

post_process_data(data)
data.to_pickle(path.join(data_dir,'fmri_followup_data_post.pkl'))
