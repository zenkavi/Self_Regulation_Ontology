"""
export data from pickle into text files that can be loaded into R
"""

import pickle,numpy
import pandas
import os
from selfregulation.utils.utils import get_info
import selfregulation.prediction.behavpredict as behavpredict

clf='lasso'
data=pickle.load(open('singularity_analyses/%s_data.pkl'%clf,'rb'))

if not os.path.exists('R_exports'):
    os.mkdir('R_exports')
if not os.path.exists('R_exports/features'):
    os.mkdir('R_exports/features')

# get variable names for each dataset

# bp=behavpredict.BehavPredict(verbose=True,
#      drop_na_thresh=100,n_jobs=1,
#      skip_vars=['RetirementPercentStocks',
#      'HowOftenFailedActivitiesDrinking',
#      'HowOftenGuiltRemorseDrinking',
#      'AlcoholHowOften6Drinks'],
#      add_baseline_vars=True,
#      smote_cutoff=0.05,
#      freq_threshold=0.04)
#
# bp.load_demog_data()
# bp.get_demogdata_vartypes()
# bp.remove_lowfreq_vars()
# bp.binarize_ZI_demog_vars()
#
# bp.load_behav_data('task')
# bp.add_varset('discounting',[v for v in list(bp.behavdata.columns) if v.find('discount')>-1])
# bp.add_varset('stopping',[v for v in list(bp.behavdata.columns) if v.find('stop_signal')>-1 or v.find('nogo')>-1])
# bp.add_varset('intelligence',[v for v in list(bp.behavdata.columns) if v.find('raven')>-1 or v.find('cognitive_reflection')>-1])
# bp.load_behav_data('task')
# bp.filter_by_icc(0.25)
#
#
# varnames['task']=list(bp.behavdata.columns)+['Age','Sex']
# bp.load_behav_data('survey')
# bp.filter_by_icc(0.25)
# varnames['survey']=list(bp.behavdata.columns)+['Age','Sex']
# bp.load_behav_data('baseline')
# varnames['baseline']=list(bp.behavdata.columns)
# for i in ['discounting','stopping','intelligence']:
#     bp.load_behav_data(i)
#     bp.filter_by_icc(0.25)
#     varnames[i]=list(bp.behavdata.columns)+['Age','Sex']

predvars={}

acc_frames={}
feat_frames={}

maxdata=10
accvars=['scores_cv','scores_insample','scores_insample_unbiased']
for k in data.keys():
    acc_frames[k]={}
    feat_frames[k]={}
    for v in data[k]:
        if not k in predvars:
            predvars[k]=data[k][v][0]['predvars']
            if len(predvars[k])==(data[k][v][0]['importances'].shape[1]-2):
                # add baseline vars
                predvars[k]=predvars[k]+['Age', 'Sex']
            assert len(predvars[k])==data[k][v][0]['importances'].shape[1]
        if not v in acc_frames[k]:
            acc_frames[k][v]={}
            feat_frames[k][v]=pandas.DataFrame()
            for accvar in accvars:
                acc_frames[k][v][accvar]=pandas.DataFrame()
        if len(data[k][v])>maxdata:
            data[k][v]=data[k][v][:maxdata]
        for accvar in accvars:
            if not accvar in data[k][v][0]:
                continue
            for i in range(len(data[k][v])):
                if len(data[k][v][i][accvar])==1:
                    df=pandas.DataFrame(data[k][v][i][accvar],
                        index=['AUROC']).T
                else:
                    df=pandas.DataFrame(data[k][v][i][accvar],
                        index=['r2','MAE']).T
                acc_frames[k][v][accvar]=acc_frames[k][v][accvar].append(df)
                if accvar==accvars[0]:
                    feats=pandas.DataFrame(data[k][v][i]['importances'],
                        columns=predvars[k])
                    feat_frames[k][v]=feat_frames[k][v].append(feats)

# now reformat so that frames contain all vars for each output type
output_frames={}
insample_frames={}

for k in data.keys():
    output_frames[k]={}
    for v in acc_frames[k].keys():
        if not k in insample_frames:
            insample_frames[k]=pandas.DataFrame([])
        outvars=list(acc_frames[k][v]['scores_cv'].columns)
        for ov in outvars:
            if not ov in output_frames[k]:
                output_frames[k][ov]={}

            for accvar in accvars:
                if not ov in acc_frames[k][v][accvar]:
                    continue
                if not accvar in output_frames[k][ov]:
                    output_frames[k][ov][accvar]=pandas.DataFrame()
                    output_frames[k][ov][accvar][v]=acc_frames[k][v][accvar][ov].values
                else:
                    output_frames[k][ov][accvar][v]=acc_frames[k][v][accvar][ov].values


for k in output_frames.keys():
    for ov in output_frames[k].keys():
        for accvar in accvars:
            index=False
            if accvar.find('insample')>-1:
                tmp=output_frames[k][ov][accvar].mean(0)
                index=True
            else:
                tmp=output_frames[k][ov][accvar]
            tmp.to_csv('R_exports/%s_%s_%s.csv'%(k,ov,accvar.replace('scores_','')),index=index)
    for v in feat_frames[k]:
        feat_frames[k][v].to_csv('R_exports/features/%s_%s_features.csv'%(k,v),index=False)


    #     tmp=pandas.DataFrame(numpy.array(features[k][v]).squeeze())
    #     tmp['varname']=v
    #     tmp.index=tmp['varname']
    #     del tmp['varname']
    #     tmp.columns=varnames[k.replace('_shuffle','')]
    #     feat_frames[k]=feat_frames[k].append(tmp)
    #
    # acc_frames[k].to_csv('R_exports/%s_%s.csv'%(clf,k))
    # feat_frames[k].to_csv('R_exports/%s_%s_features.csv'%(clf,k))
