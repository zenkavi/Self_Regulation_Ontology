from expanalysis.results import get_filters
import pandas as pd
from util import load_data

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
f = open('/home/ian/Experiments/expfactory/docs/expfactory_token.txt')
access_token = f.read().strip()      
data_loc = '/home/ian/Experiments/expfactory/Self_Regulation_Ontology/Data/Battery_Results'     
data_source = load_data(access_token, data_loc, filters = filters, source = 'web', battery = 'Self Regulation Battery')