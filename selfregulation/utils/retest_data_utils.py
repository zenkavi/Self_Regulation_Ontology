import pandas as pd
from selfregulation.utils.utils import get_behav_data
from selfregulation.utils.data_preparation_utils import remove_outliers
from selfregulation.utils.data_preparation_utils import transform_remove_skew

def get_retest_comparison_data():
    subsets = ['meaningful_variables_noDDM.csv', 'meaningful_variables_EZ.csv',
               'meaningful_variables_hddm.csv']
    dataset = pd.DataFrame()
    for subset in subsets:
        df = get_behav_data(file=subset)
        df_clean = remove_outliers(df)
        df_clean = transform_remove_skew(df_clean)
        drop_columns = set(dataset) & set(df_clean)
        df_clean.drop(labels=drop_columns, axis=1, inplace=True)
        dataset = pd.concat([dataset, df_clean], axis=1)
    return dataset
