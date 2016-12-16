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
    rotated_data = GPArotation.GPForth(data, method = method, normalize=normalize)[0]
    rotated_data = pd.DataFrame(data = np.matrix(rotated_data), index=data.index, columns=data.columns)
    return rotated_data
    
    
