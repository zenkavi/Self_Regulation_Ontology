import numpy as np
import pandas as pd
import readline
import rpy2.robjects
from rpy2.robjects import pandas2ri, Formula
from rpy2.robjects.packages import importr

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
    
def glmer(data, formula):
    base = importr('base')
    lme4 = importr('lme4')
    rs = lme4.glmer(Formula(formula), data, family = 'binomial')
    
    fixed_effects = lme4.fixed_effects(rs)
    fixed_effects = {k:v for k,v in zip(fixed_effects.names, list(fixed_effects))}
                                  
    random_effects = lme4.random_effects(rs)[0]
    random_effects = pd.DataFrame([list(lst) for lst in random_effects], index = list(random_effects.colnames)).T
    print(base.summary(rs))
    return fixed_effects, random_effects

def psychICC(df):
    psych = importr('psych')
    rs = psych.ICC(df)
    return rs
