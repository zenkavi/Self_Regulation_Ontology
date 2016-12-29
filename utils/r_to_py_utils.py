import numpy as np
import pandas as pd
import readline
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

pandas2ri.activate()
def missForest(data):
    missForest = importr('missForest')
    data_complete, error = missForest.missForest(data)
    imputed_df = pd.DataFrame(np.matrix(data_complete).T, index=data.index, columns=data.columns)
    return imputed_df, error
    

def GPArotation(data, method='varimax', normalize=True):
    GPArotation = importr('GPArotation')
    rotated_data = GPArotation.GPForth(data.values, method = method, normalize=normalize)[0]
    rotated_data = pd.DataFrame(data = np.matrix(rotated_data), index=data.index, columns=data.columns)
    return rotated_data

def psychFA(data, n_components, return_attrs = ['BIC', 'RMSEA']):
    def get_attr(attr):
        try:
            index = list(fa.names).index(attr)
            val = list(fa.items())[index][1]
            if len(val) == 1:
                val = val[0]
            return val
        except ValueError:
            print('Did not pass a valid attribute')
    psych = importr('psych')
    fa = psych.fac(data, n_components)
    attr_dic = {}
    attr_dic['loadings'] = np.matrix(get_attr('loadings'))
    for attr in return_attrs:
        attr_dic[attr] = get_attr(attr)
    return attr_dic
    
