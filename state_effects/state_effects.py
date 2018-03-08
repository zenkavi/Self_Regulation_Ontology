import pandas as pd
from rpy2.robjects import pandas2ri
from sklearn.preprocessing import scale

from selfregulation.utils.r_to_py_utils import lmer, get_Rpsych
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import filter_behav_data, get_behav_data, get_info, get_recent_dataset

# get dataset of interest
basedir=get_info('base_directory')
dataset = get_recent_dataset()
# get datasets
data1 = get_behav_data(dataset,
                       file='meaningful_variables_imputed.csv',
                       verbose=True)
data2 = get_behav_data(dataset=dataset.replace('Complete', 'Retest'),
                       file='meaningful_variables_imputed.csv',
                       verbose=True)
overlap_index = set(data1.index) & set(data2.index)
overlap_columns = set(data1.columns) & set(data2.columns)
data1 = data1.loc[overlap_index, overlap_columns]
data2 = data2.loc[overlap_index, overlap_columns]
# calculate change and standardize
change = pd.DataFrame(scale(data2-data1, with_mean=False),
                      index=data2.index,
                      columns=data2.columns)
# get reliability
reliabilities = get_behav_data(dataset=dataset.replace('Complete', 'Retest'),
                               file='bootstrap_merged.csv.gz')
reliabilities = reliabilities.groupby('dv').icc.mean()
# threshold by reliabilities
change = change.loc[:, reliabilities[change.columns]>.2]
                               
# ANOVA of task change
def cluster_state_analysis(change, subset='task'):
    change = abs(filter_behav_data(change, subset)).assign(Subject=change.index)
    results = load_results(dataset)[subset]
    c = results.EFA.results['num_factors']
    clusters = results.HCA.get_cluster_labels(inp='EFA%s' % c)
    cluster_IDs={}
    for c in change.columns[:-1]:
        cluster_IDs[c] = [i for i, cluster in enumerate(clusters) if c in cluster][0]
    # convert to absolute change between two conditions
    change = change.melt(id_vars=['Subject'])
    row_IDs = change.apply(lambda x: str(cluster_IDs[x['variable']]), axis=1)
    change.insert(0, 'cluster_ID', row_IDs)
    # analyze!
    return lmer(change, 'value ~ (1|cluster_ID) + (1|Subject) + (1|variable)')
    
    
    
# model based ICC
# add timepoints and concatenate
data1.insert(0, 'timepoint', 1)
data2.insert(0, 'timepoint', 2)
data = pd.concat([data1, data2])
data.insert(0, 'Subject', data.index)
data = data.melt(id_vars = ['Subject','timepoint'])

variables = ['digit_span.forward_span',
             "ten_item_personality_survey.emotional_stability",
             "ravens.score",
             'two_stage_decision.model_free'
             ]

for v in variables:
    # calculate ICC using mixed model
    var = data.query('variable=="%s"' % v)
    # find subjects that have NA for a timepoint
    bad_subjects = var.groupby('Subject').value.count()<2
    bad_subjects = var.Subject.iloc[bad_subjects.tolist()]
    var = var.query('Subject not in %s' % list(bad_subjects))
    rs, variance, a, b =lmer(var, 'value ~ (timepoint|group) + (1|Subject)')
    mixed_ICC = (variance.query('grp=="Subject"')['vcov']/variance.vcov.sum()).iloc[0]
    # calculate ICC using ICC function
    ICC_df = var.drop('variable', axis=1).pivot(columns='timepoint', index="Subject")
    ICCfun = get_Rpsych().ICC
    ICC_df=pandas2ri.ri2py(ICCfun(ICC_df)[0])
    print("Variable: %s\nMixed ICC: %s\nICC: %s" % (v, mixed_ICC, ICC_df.iloc[0,1]))
