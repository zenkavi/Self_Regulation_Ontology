"""
export data from pickle into text files that can be loaded into R
"""

import pickle
import pandas
import os
from selfregulation.utils.utils import get_info
import selfregulation.prediction.behavpredict as behavpredict

clf='rf'
acc,features=pickle.load(open('singularity_analyses/%s_data_collapsed.pkl'%clf,'rb'))

acc_frames={}

if not os.path.exists('R_exports'):
    os.mkdir('R_exports')

for k in acc:
    acc_frames[k]=acc_df=pandas.DataFrame()
    for v in acc[k]:
        acc_frames[k][v]=acc[k][v]
    acc_frames[k].to_csv('R_exports/%s_%s.csv'%(clf,k))

# get variable names for each dataset

bp=behavpredict.BehavPredict(verbose=True,
     drop_na_thresh=100,n_jobs=1,
     skip_vars=['RetirementPercentStocks'])
bp.load_demog_data()
bp.get_demogdata_vartypes()
bp.load_behav_data('task')
taskvars=list(bp.behavdata.columns)
bp.load_behav_data('survey')
surveyvars=list(bp.behavdata.columns)
bp.load_behav_data('baseline')
baselinevars=list(bp.behavdata.columns)

with open('R_exports/baseline_varnames.txt','w') as f:
    for v in baselinevars:
        f.write(v+'\n')
with open('R_exports/task_varnames.txt','w') as f:
    for v in taskvars:
        f.write(v+'\n')
with open('R_exports/survey_varnames.txt','w') as f:
    for v in surveyvars:
        f.write(v+'\n')
