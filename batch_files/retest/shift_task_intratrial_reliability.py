import json
import numpy
import pandas
import statsmodels.formula.api as smf


def calc_shift_DV(df, dvs = {}):
    """ Calculate dv for shift task. I
    :return dv: dictionary of dependent variables
    :return description: descriptor of DVs
    """
    # subset df
    df = df.query('rt != -1').reset_index(drop = True)

    # Calculate basic statistics - accuracy, RT and error RT
    dvs['acc'] = {'value':  df.correct.mean(), 'valence': 'Pos'}
    dvs['avg_rt'] = {'value':  df.rt.median(), 'valence': 'Neg'}

    try:
        rs = smf.glm('correct ~ trials_since_switch+trial_num', data = df, family = sm.families.Binomial()).fit()
        learning_rate = rs.params['trials_since_switch']
        learning_to_learn = rs.params['trial_num']
    except ValueError:
        learning_rate = 'NA'
        learning_to_learn = 'NA'

    dvs['learning_to_learn'] = {'value': learning_to_learn, 'valence':'Pos'}
    dvs['learning_rate'] = {'value':  learning_rate  , 'valence': 'Pos'}

    #conceptual_responses: The CLR score is the total number of consecutive correct responses in a sequence of 3 or more.
    CLR_score = 0
    CLR_thresh_sum= 0 # must be >= 3
    #fail_to_maintain_set: The FTMS score is the number of sequences of 5 correct responses or more,
    #followed by an error, before attaining the 10 necessary for a set change - for us just counting number of streaks of >5 since 10 isn't necessary for set change
    FTMS_score = 0
    FTMS_thresh_sum = 0

    for i, row in df.iterrows():
        if row.correct==True:
            CLR_thresh_sum += 1
            FTMS_thresh_sum += 1
        else:
            if FTMS_thresh_sum >= 5:
                FTMS_score += 1
            CLR_thresh_sum = 0
            FTMS_thresh_sum = 0
        if CLR_thresh_sum>=3:
            CLR_score+=1

    dvs['conceptual_responses'] = {'value': CLR_score, 'valence':'Pos'}
    dvs['fail_to_maintain_set'] = {'value': FTMS_score, 'valence':'Pos'}


    #add last_rewarded_feature column by switching the variable to the feature in the row right before a switch and assigning to the column until there is another switch
    df['last_rewarded_feature'] = "NaN"
    last_rewarded_feature = "NaN"
    for i, row in df.iterrows():
        if row.shift_type != 'stay':
            last_rewarded_feature = df.rewarded_feature.iloc[i-1]
        df.last_rewarded_feature.iloc[i] = last_rewarded_feature

    #perseverative_responses: length of df where the choice_stim includes the last_rewarded_feature
    perseverative_responses = df[df.apply(lambda row: row.last_rewarded_feature in str(row.choice_stim), axis=1)]
    dvs['perseverative_responses'] = {'value': len(perseverative_responses),'valence':'Neg'}
    #perseverative_errors: length of perseverative_responses df that is subsetted by incorrect responses
    dvs['perseverative_errors'] = {'value': len(perseverative_responses.query("correct == 0")),'valence':'Neg'}
    #total_errors
    dvs['total_errors'] = {'value': len(df.query("correct==0")), 'valence':'Neg'}
    #nonperseverative_errors
    dvs['nonperseverative_errors'] = {'value': len(df.query("correct==0")) - dvs['perseverative_errors']['value'], 'valence': 'Neg'}

    description = """
        Shift task has a complicated analysis. Right now just using accuracy and
        slope of learning after switches (which I'm calling "learning rate")
        """
    return dvs, description

#load requested raw data
pandas.read_csv()
#extract retest workers from raw data

#loop through each trial subset for each subject to get dvs (double loop)

#write output data
