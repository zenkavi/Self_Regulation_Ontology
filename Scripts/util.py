'''
Utility functions for the ontology project
'''

import numpy as np
import pandas as pd

def set_discovery_sample(n, discovery_n, seed = None):
    """
    :n: total size of sample
    :discovery_n: number of discovery subjects
    :param seed: if set, use as the seed for randomization
    :return array: array specifying which subjects, in order, are discovery/validation
    """
    if seed:
        np.random.seed(seed)
    sample = ['discovery']*discovery_n + ['validation']*(n-discovery_n)
    np.random.shuffle(sample)
    return sample

seed = 1960
n = 500
discovery_n = 200
subjects = ['s' + str(i).zfill(3) for i in range(1,n+1)]
subject_order = set_discovery_sample(n, discovery_n, seed)
subject_assignment_df = pd.DataFrame({'dataset': subject_order}, index = subjects)
subject_assignment_df.to_csv('../subject_assignment.csv')