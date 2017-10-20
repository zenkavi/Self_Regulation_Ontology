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

def get_Rpsych():
    psych = importr('psych')
    return psych

def psychFA(data, n_components, return_attrs=['BIC', 'SABIC', 'RMSEA'], 
            rotate='oblimin', method='ml', verbose=False):
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
    fa = psych.fa(data, n_components, rotate=rotate, fm=method)
    attr_dic = {}
    # loadings are roughly equivalent to the correlation between each variable
    # and the factor scores
    attr_dic['loadings'] = np.matrix(get_attr('loadings'))
    # scores are the the factors
    attr_dic['scores'] = np.matrix(get_attr('scores'))
    # weights are the "mixing matrix" such that the final data is
    # S * W
    attr_dic['weights'] = np.matrix(get_attr('weights'))
    for attr in return_attrs:
        attr_dic[attr] = get_attr(attr)
    if verbose:
        print(fa)
    return fa, attr_dic
    
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

def qgraph_cor(data, glasso=False, gamma=.25):
    qgraph = importr('qgraph')
    cors = qgraph.cor_auto(data)
    if glasso==True:
        EBICglasso = qgraph.EBICglasso(cors, data.shape[0],
                                       returnAllResults=True,
                                       gamma=gamma)
        # figure out the index for the lowest EBIC
        best_index = np.argmin(EBICglasso[1])
        tuning_param = EBICglasso[4][best_index]
        glasso_cors = np.array(EBICglasso[0][0])[:,:,best_index]
        glasso_cors_df = pd.DataFrame(np.matrix(glasso_cors), 
                           index=data.columns, 
                           columns=data.columns)
        return glasso_cors_df, tuning_param
    else:
        cors_df = pd.DataFrame(np.matrix(cors), 
                           index=data.columns, 
                           columns=data.columns)
        return cors_df