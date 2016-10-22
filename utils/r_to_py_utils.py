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
    imputed_df = pd.DataFrame(np.matrix(data_complete).T, index = data.index, columns = data.columns)
    return imputed_df, error
    
    
    import readline
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

"""
not finished

def polycor(data):
    polycor = importr('polycor')
    polycor_out = polycor.hetcor(data)
    return polycor_out
"""
    
    
    
