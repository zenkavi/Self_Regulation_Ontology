import numpy as np
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
# add timepoints and concatenate
data1.insert(0, 'timepoint', 1)
data2.insert(0, 'timepoint', 2)
data = pd.concat([data1, data2])
data.insert(0, 'Subject', data.index)
data = data.melt(id_vars = ['Subject','timepoint'])

# model based ICC
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

'diff ~ (1|group)'



# get results (for groups)
results = load_results(dataset)

# check for consistent time
survey = filter_behav_data(change, 'survey')
task = filter_behav_data(change, 'non_decision')


def boot_change(data1, data2, reps=10):
    N = data1.shape[0]//2
    vals = []
    for _ in range(reps):
        random_order = np.random.permutation(data1.shape[0])    
        time1 = pd.concat([data1.iloc[random_order[:N]],
                           data2.iloc[random_order[N:]]])
        time2 = pd.concat([data2.iloc[random_order[:N]],
                           data1.iloc[random_order[N:]]])
        change = time2-time1
        # standardize
        change = change/change.std()
        vals.append(change.max().max())
    return vals