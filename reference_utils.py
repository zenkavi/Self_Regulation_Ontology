import numpy as np
import os
import pandas as pd
from selfregulation.utils.utils import  get_info

def gen_reference_item_text(items_df):
    base_directory = get_info('base_directory')
    reference_location = os.path.join(base_directory,'references','variable_name_lookup.csv')
    ref = pd.read_csv(reference_location)
    lookup = items_df.groupby('item_ID').item_text.unique().apply(lambda x: x[0]).to_dict()
    item_text = [lookup[i] if i in lookup.keys() else np.nan for i in ref['Variable Name']]
    ref.loc[:,'Question'] = item_text
    ref.to_csv(reference_location)
