import numpy as np
import pandas as pd

from dimensional_structure.utils import get_loadings
from ontology_mapping.reconstruction_utils import reorder_FA
from selfregulation.utils.r_to_py_utils import psychFA
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_recent_dataset

results = load_results(get_recent_dataset())

task = results['task']
c = task.EFA.get_c()
orig_loadings = task.EFA.get_loading(c=c)
measures = np.unique([c.split('.')[0] for c in task.data.columns])

def drop_EFA(data, measure, c):
    to_drop = data.filter(regex=measure).columns
    subset = data.drop(to_drop, axis=1)
    fa, output = psychFA(subset, c, method='ml', rotate='oblimin')
    loadings = get_loadings(output, labels=subset.columns)
    return loadings

factor_correlations = {}
for measure in measures:
    data = task.data
    new_loadings = drop_EFA(data, measure, c)
    new_loadings = reorder_FA(orig_loadings, new_loadings, thresh=0)
    
    corr = pd.concat([new_loadings, orig_loadings], axis=1, sort=False) \
            .corr().iloc[:c, c:]
    diag = {c:i for c,i in zip(new_loadings.columns, np.diag(corr))}
    factor_correlations[measure] = diag
