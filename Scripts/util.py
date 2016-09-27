'''
Utility functions for the ontology project
'''
from expanalysis.experiments.processing import extract_row, extract_experiment
from expanalysis.results import Result
from expanalysis.experiments.utils import remove_duplicates, result_filter
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from time import time

#***************************************************
# ********* Plotting Functions **********************
#**************************************************
def dendroheatmap(df, labels = True):
    """
    :df: plot hierarchical clustering and heatmap
    """
    #clustering
    
    row_clusters = linkage(df.values, method='complete', metric='euclidean')
    pd.DataFrame(row_clusters, 
                 columns=['row label 1', 'row label 2', 'distance', 'no. of items in clust.'],
                 index=['cluster %d' %(i+1) for i in range(row_clusters.shape[0])])
    
    #dendrogram
    row_dendr = dendrogram(row_clusters, labels=df.columns, no_plot = True)
    df_rowclust = df.ix[row_dendr['leaves'],row_dendr['leaves']]
    sns.set_style("white")
    fig = plt.figure(figsize = [16,16])
    ax = fig.add_axes([.1,.2,.6,.6]) 
    cax = fig.add_axes([0.02,0.3,0.02,0.4]) 
    sns.heatmap(df_rowclust, ax = ax, cbar_ax = cax, xticklabels = False)
    ax.yaxis.tick_right()
    ax.set_yticklabels(df_rowclust.columns[::-1], rotation=0, rotation_mode="anchor", fontsize = 'small', visible = labels)
    ax.set_xticklabels(df_rowclust.columns, rotation=-90, rotation_mode = "anchor", ha = 'left')
    ax1 = fig.add_axes([.1,.8,.6,.2])
    plt.axis('off')
    row_dendr = dendrogram(row_clusters, orientation='top',  
                           count_sort='ascending', ax = ax1) 
    return fig

def dendroheatmap_left(df, labels = True):
    """
    :df: plot hierarchical clustering and heatmap, dendrogram on left
    """
    #clustering
    
    row_clusters = linkage(df.values, method='complete', metric='euclidean')
    pd.DataFrame(row_clusters, 
                 columns=['row label 1', 'row label 2', 'distance', 'no. of items in clust.'],
                 index=['cluster %d' %(i+1) for i in range(row_clusters.shape[0])])
    
    #dendrogram
    row_dendr = dendrogram(row_clusters, labels=df.columns, no_plot = True)
    df_rowclust = df.ix[row_dendr['leaves'],row_dendr['leaves']]
    sns.set_style("white")
    fig = plt.figure(figsize = [16,16])
    ax = fig.add_axes([.16,.3,.62,.62]) 
    cax = fig.add_axes([0.21,0.25,0.5,0.02]) 
    sns.heatmap(df_rowclust, ax = ax, cbar_ax = cax, cbar_kws = {'orientation': 'horizontal'}, xticklabels = False)
    ax.yaxis.tick_right()
    ax.set_yticklabels(df_rowclust.columns[::-1], rotation=0, rotation_mode="anchor", fontsize = 'large', visible = labels)
    ax.set_xticklabels(df_rowclust.columns, rotation=-90, rotation_mode = "anchor", ha = 'left')
    ax1 = fig.add_axes([.01,.3,.15,.62])
    plt.axis('off')
    row_dendr = dendrogram(row_clusters, orientation='right',  
                           count_sort='descending', ax = ax1) 
    return fig


#***************************************************
# ********* Helper Functions **********************
#**************************************************
def anonymize_data(data):
    complete_workers = (data.groupby('worker_id').count().finishtime>=63)
    complete_workers = list(complete_workers[complete_workers].index)
    workers = data.groupby('worker_id').finishtime.max().sort_values().index
    # make new ids
    new_ids = []
    id_index = 1
    for worker in workers:
        if worker in complete_workers:
            new_ids.append('s' + str(id_index).zfill(3))
            id_index += 1
        else:
            new_ids.append(worker)
    data.replace(workers, new_ids,inplace = True)
    return {x:y for x,y in zip(new_ids, workers)}
    
def calc_bonuses(data):
    bonus_experiments = ['angling_risk_task_always_sunny', 'two_stage_decision',
                         'columbia_card_task_hot', 'columbia_card_task_cold', 'hierarchical_rule',
                         'kirby','discount_titrate','bickel_titrator']
    bonuses = []
    for i,row in data.iterrows():
        if row['experiment_exp_id'] in bonus_experiments:
            df = extract_row(row, clean = False)
            bonus = df.iloc[-1].get('performance_var','error')
            if pd.isnull(bonus):
                bonus = df.iloc[-5].get('performance_var','error')
            if not isinstance(bonus,(int,float)):
                bonus = json.loads(bonus)['amount']
            bonuses.append(bonus)
        else:
            bonuses.append(np.nan)
    data.loc[:,'bonus'] = bonuses
    data.loc[:,'bonus_zscore'] = data['bonus']
    means = data.groupby('experiment_exp_id').bonus.mean()
    std = data.groupby('experiment_exp_id').bonus.std()
    for exp in bonus_experiments:
        data.loc[data.experiment_exp_id == exp,'bonus_zscore'] = (data[data.experiment_exp_id == exp].bonus-means[exp])/std[exp]
        
def calc_trial_order(data):
    sorted_data = data.sort_values(by = ['worker_id','finishtime'])
    num_exps = data.groupby('worker_id')['finishtime'].count() 
    order = []    
    for x in num_exps:
        order += range(x)
    data.loc[sorted_data.index, 'trial_order'] = order
    
def check_timing(df):
    df.loc[:, 'time_diff'] = df['time_elapsed'].diff()
    timing_cols = pd.concat([df['block_duration'], df.get('feedback_duration'), df['timing_post_trial'].shift(1)], axis = 1)
    df.loc[:, 'expected_time'] = timing_cols.sum(axis = 1)
    df.loc[:, 'timing_error'] = df['time_diff'] - df['expected_time']
    errors = [df[abs(df['timing_error']) < 500]['timing_error'].mean(), df[df['timing_error'] < 500]['timing_error'].max()]
    return errors
      
def download_data(data_loc, access_token = None, filters = None, battery = None, save = True, url = None):
    start_time = time()
    #Load Results from Database
    results = Result(access_token, filters = filters, url = url)
    data = results.data
    if battery:
        data = result_filter(data, battery = battery)

    # remove duplicates
    remove_duplicates(data)
    
    # remove a few mistakes from data
    data = data.query('worker_id not in ["A254JKSDNE44AM", "A1O51P5O9MC5LX"]') # Sandbox workers
    data.reset_index(drop = True, inplace = True)    
    
    # if saving, save the data and the lookup file for anonymized workers
    if save == True:
        data.to_json(data_loc + 'mturk_data.json')
        print('Finished saving')
    
    finish_time = (time() - start_time)/60
    print('Finished downloading data. Time taken: ' + str(finish_time))
    return data                 

def drop_vars(data, drop_vars = []):
    if len(drop_vars) == 0:
        # variables that are calculated without regar to their actual interest
        basic_vars = ["\.missed_percent$","\.acc$","\.avg_rt_error$","\.std_rt_error$","\.avg_rt$","\.std_rt$"]
        # variables that are of theoretical interest, but we aren't certain enough to include in 2nd stage analysis
        exploratory_vars = ["\.congruency_seq", "\.post_error_slowing$"]
        # task variables that are irrelevent to second stage analysis, either because they are correlated
        # with other DV's or are just of no interest. Each row is a task
        task_vars = ["demographics", # demographics
                    "network_task.EZ_drift_congruent$", "network_task.EZ_thresh_congruent$", "network_task.EZ_non_decision_congruent$", # ANT
                    "\.EZ_drift_incongruent$", "\.EZ_thresh_incongruent$", "\.EZ_non_decision_incongruent$", # ANT, local_global, simon, stroop
                    "\.EZ_drift_con", "\.EZ_drift_neg", "\.EZ_thresh_con", "\.EZ_thresh_neg", "\.EZ_non_decision_con", "\.EZ_non_decision_neg", # directed forgetting
                    "\.EZ_drift_AY", "\.EZ_drift_BX", "\.EZ_drift_BY", "\.EZ_thresh_AY", "\.EZ_thresh_BX", "\.EZ_thresh_BY", # DPX
                    "\.EZ_non_decision_AX", "\.EZ_non_decision_BX", "\.EZ_non_decision_BY", # DPX continued
                    "_total_points$", # IST
                    "\.go_acc$", "\.nogo_acc$", "\.go_rt$", #go_nogo
                    "_large$", "_medium$", "_small$", "\.warnings$", "_notnow$", "_now$", #kirby and delay discounting
                    "letter.EZ_drift_congruent$", "letter.EZ_thresh_congruent$", "letter.EZ_non_decision_congruent$", # local global letter
                    "\.EZ_drift_rec", "\.EZ_drift_xrec", "\.EZ_thresh_rec", "\.EZ_thresh_xrec", "\.EZ_non_decision_rec", "\.EZ_non_decision_xrec", # recent probes
                     "go_acc","stop_acc","go_rt_error","go_rt_std_error", "go_rt","go_rt_std","stop_rt_error","stop_rt_error_std","SS_delay", # stop signal
                     "\.EZ_drift_.*_switch", "\.EZ_thresh_.*_switch", "\.EZ_non_decision_.*_switch", "\.EZ_drift_task_stay", "\.EZ_thresh_task_stay", "\.EZ_non_decision_task_stay", # three by two
                     "sentiment_label" # writing task
                    ]
        drop_vars = basic_vars + exploratory_vars + task_vars
    drop_vars = '|'.join(drop_vars)
    return data.drop(data.filter(regex=drop_vars).columns, axis = 1)
    
    
def get_bonuses(data):
    if 'bonus_zscore' not in data.columns:
        calc_bonuses(data)
    workers_finished = data.groupby('worker_id').count().finishtime==63
    index = list(workers_finished[workers_finished].index)
    tmp_data = data.query('worker_id in %s' % index)
    tmp_bonuses = tmp_data.groupby('worker_id').bonus_zscore.sum()
    min_score = tmp_bonuses.min()
    max_score = tmp_bonuses.max()
    num_tasks_bonused = data.groupby('worker_id').bonus_zscore.count()
    bonuses = data.groupby('worker_id').bonus_zscore.sum()
    bonuses = (bonuses-min_score)/(max_score-min_score)*10+5
    bonuses = bonuses.map(lambda x: round(x,1))*num_tasks_bonused/8
    print('Finished getting bonuses')
    return bonuses

def get_credit(data):
    credit_array = []
    for i,row in data.iterrows():
        if row['experiment_template'] in 'jspsych':
            df = extract_row(row, clean = False)
            credit_var = df.iloc[-1].get('credit_var',np.nan)
            if credit_var != None:
                credit_array.append(float(credit_var))
            else:
                credit_array.append(np.nan)
        else:
            credit_array.append(np.nan)
    data.loc[:,'credit'] = credit_array   
    

def get_demographics(df):
    race = (df.query('text == "What is your race?"').groupby('worker_id')['response'].unique()).value_counts()
    hispanic = (df.query('text == "Are you of Hispanic, Latino or Spanish origin?"'))['response_text'].value_counts() 
    sex = (df.query('text == "What is your sex?"'))['response_text'].value_counts() 
    age_col = df.query('text == "How old are you (in years)?"')['response'].astype(float)
    age_vars = [age_col.min(), age_col.max(), age_col.mean()]    
    return {'age': age_vars, 'sex': sex, 'race': race, 'hispanic': hispanic}  
    
def get_info(item,infile='../Self_Regulation_Settings.txt'):
    """
    get info from settings file
    """
    infodict={}
    try:
        assert os.path.exists(infile)
    except:
        raise Exception('You must first create a Self_Regulation_Settings.txt file')

    with open(infile) as f:
        lines=[i for i in f.readlines() if not i.find('#')==0]
        for l in lines:
            l_s=l.rstrip('\n').split(':')
            infodict[l_s[0]]=l_s[1]
    try:
        assert item in infodict
    except:
        raise Exception('infodict does not include requested item')
    return infodict[item]


def get_items(data):
    excluded_surveys = ['holt_laury_survey', 'selection_optimization_compensation_survey', 'sensation_seeking_survey']
    items = []
    responses = []
    responses_text = []
    options = []
    workers = []
    item_nums = []
    exps = []
    for exp in data.experiment_exp_id.unique():
        if 'survey' in exp and exp not in excluded_surveys:
            survey = extract_experiment(data,exp)
            try:
                responses += list(survey.response.map(lambda x: float(x)))
            except ValueError:
                continue
            items += list(survey.text)
            responses_text += list(survey.response_text)
            options += list(survey.options)
            workers += list(survey.worker_id)
            item_nums += list(survey.question_num)
            exps += [exp] * len(survey.text)
    
    items_df = pd.DataFrame({'survey': exps, 'worker': workers, 'item_text': items, 'coded_response': responses,
                             'response_text': responses_text, 'options': options}, dtype = float)
    items_df.loc[:,'item_num'] = [str(i).zfill(3) for i in item_nums]
    items_df.loc[:,'item_ID'] = items_df['survey'] + '_' + items_df['item_num'].astype(str)
    return items_df
    
    
def get_pay(data):
    assert 'ontask_time' in data.columns, \
        'Task time not found. Must run "calc_time_taken" first.' 
    all_exps = data.experiment_exp_id.unique()
    exps_completed = data.groupby('worker_id').experiment_exp_id.unique()
    exps_not_completed = exps_completed.map(lambda x: list(set(all_exps) - set(x) - set(['selection_optimization_compensation'])))
    completed = exps_completed[exps_completed.map(lambda x: len(x)>=63)]
    almost_completed = exps_not_completed[exps_not_completed.map(lambda x: x == ['angling_risk_task_always_sunny'])]
    not_completed = exps_not_completed[exps_not_completed.map(lambda x: len(x)>0 and x != ['angling_risk_task_always_sunny'])]
    # remove stray completions
    not_completed.loc[[i for i in not_completed.index if 's0' not in i]]
    # calculate time taken
    task_time = data.groupby('experiment_exp_id').ontask_time.mean()/60+2 # +2 for generic instruction time
    time_spent = exps_completed.map(lambda x: np.sum([task_time[i] if task_time[i]==task_time[i] else 3 for i in x])/60)
    time_missed = exps_not_completed.map(lambda x: np.sum([task_time[i] if task_time[i]==task_time[i] else 3 for i in x])/60)
    # calculate pay
    completed_pay = pd.Series(data = 60, index = completed.index)
    prorate_pay = 60-time_missed[almost_completed.index]*6
    reduced_pay = time_spent[not_completed.index]*2 + np.floor(time_spent[not_completed.index])*2
    #remove anyone who was double counted
    reduced_pay.drop(list(completed_pay.index) + list(prorate_pay.index), inplace = True, errors = 'ignore')
    pay= pd.concat([completed_pay, reduced_pay,prorate_pay]).map(lambda x: round(x,1)).to_frame(name = 'base')
    pay['bonuses'] = get_bonuses(data)
    pay['total'] = pay.sum(axis = 1)
    return pay
    
def get_worker_demographics(worker_id, data):
    df = data[(data['worker_id'] == worker_id) & (data['experiment_exp_id'] == 'demographics_survey')]
    if len(df) == 1:
        race = df.query('text == "What is your race?"')['response'].unique()
        hispanic = df.query('text == "Are you of Hispanic, Latino or Spanish origin?"')['response_text']
        sex = df.query('text == "What is your sex?"')['response_text']
        age = float(df.query('text == "How old are you?"')['response'])
        return {'age': age, 'sex': sex, 'race': race, 'hispanic': hispanic}
    else:
        return np.nan
    
def heatmap(df):
    """
    :df: plot heatmap
    """
    plt.Figure(figsize = [16,16])
    sns.set_style("white")
    fig = plt.figure(figsize = [12,12])
    ax = fig.add_axes([.1,.2,.6,.6]) 
    cax = fig.add_axes([0.02,0.3,0.02,0.4]) 
    sns.heatmap(df, ax = ax, cbar_ax = cax, xticklabels = False)
    ax.yaxis.tick_right()
    ax.set_yticklabels(df.columns[::-1], rotation=0, rotation_mode="anchor", fontsize = 'large')
    ax.set_xticklabels(df.columns, rotation=-90, rotation_mode = "anchor", ha = 'left') 
    return fig

def print_time(data, time_col = 'ontask_time'):
    '''Prints time taken for each experiment in minutes
    :param time_col: Dataframe column of time in seconds
    '''
    df = data.copy()    
    assert time_col in df, \
        '"%s" has not been calculated yet. Use calc_time_taken method' % (time_col)
    #drop rows where time can't be calculated
    df = df.dropna(subset = [time_col])
    time = (df.groupby('experiment_exp_id')[time_col].mean()/60.0).round(2)
    print(time)
    return time
    
               
def quality_check(data):
    start_time = time()
    rt_thresh_lookup = {
        'angling_risk_task_always_sunny': 0,
        'simple_reaction_time': 150    
    }
    acc_thresh_lookup = {
        'digit_span': 0,
        'hierarchical_rule': 0,
        'information_sampling_task': 0,
        'probabilistic_selection': 0,
        'ravens': 0,
        'shift_task': 0,
        'spatial_span': 0,
        'tower_of_london': 0
        
    }
    missed_thresh_lookup = {
        'information_sampling_task': 1,
        'go_nogo': 1,
        'tower_of_london': 2
    }
    
    response_thresh_lookup = {
        'angling_risk_task_always_sunny': np.nan,
        'columbia_card_task_cold': np.nan,
        'discount_titrate': np.nan,
        'digit_span': np.nan,
        'go_nogo': .98,
        'kirby': np.nan,
        'simple_reaction_time': np.nan,
        'spatial_span': np.nan,
    }
    
    templates = data.groupby('experiment_exp_id').experiment_template.unique()
    data.loc[:,'passed_QC'] = True
    for exp in data.experiment_exp_id.unique():
        try:
            if templates.loc[exp] == 'jspsych':
                print('Running QC on ' + exp)
                df = extract_experiment(data, exp)
                rt_thresh = rt_thresh_lookup.get(exp,200)
                acc_thresh = acc_thresh_lookup.get(exp,.6)
                missed_thresh = missed_thresh_lookup.get(exp,.25)
                response_thresh = response_thresh_lookup.get(exp,.95)
                
                # special cases...
                if exp == 'information_sampling_task':
                    df.groupby('worker_id').which_click_in_round.value_counts()
                    passed_response = df.groupby('worker_id').which_click_in_round.mean() > 2
                    passed_rt = pd.Series([True] * len(passed_response), index = passed_response.index)
                    passed_miss = pd.Series([True] * len(passed_response), index = passed_response.index)
                    passed_acc = pd.Series([True] * len(passed_response), index = passed_response.index)
                elif exp == 'go_nogo':
                    passed_rt = df.query('rt != -1').groupby('worker_id').rt.median() >= rt_thresh
                    passed_miss = df.groupby('worker_id').rt.agg(lambda x: np.mean(x == -1)) < missed_thresh
                    passed_acc = df.groupby('worker_id').correct.mean() >= acc_thresh
                    passed_response = np.logical_not(df.groupby('worker_id').key_press.agg(
                                                            lambda x: np.any(pd.value_counts(x) > pd.value_counts(x).sum()*response_thresh)))
                elif exp == 'psychological_refractory_period_two_choices':
                    passed_rt = (df.groupby('worker_id').median()[['choice1_rt','choice2_rt']] >= rt_thresh).all(axis = 1)
                    passed_acc = df.query('choice1_rt != -1').groupby('worker_id').choice1_correct.mean() >= acc_thresh
                    passed_miss = ((df.groupby('worker_id').choice1_rt.agg(lambda x: np.mean(x!=-1) >= missed_thresh)) \
                                        + (df.groupby('worker_id').choice2_rt.agg(lambda x: np.mean(x>-1) >= missed_thresh))) == 2
                    passed_response1 = np.logical_not(df.query('choice1_rt != -1').groupby('worker_id').choice1_key_press.agg(
                                                            lambda x: np.any(pd.value_counts(x) > pd.value_counts(x).sum()*response_thresh)))
                    passed_response2 = np.logical_not(df.query('choice2_rt != -1').groupby('worker_id').choice2_key_press.agg(
                                                            lambda x: np.any(pd.value_counts(x) > pd.value_counts(x).sum()*response_thresh)))
                    passed_response = np.logical_and(passed_response1,passed_response2)
                elif exp == 'ravens':
                    passed_rt = df.query('rt != -1').groupby('worker_id').rt.median() >= rt_thresh
                    passed_acc = df.query('rt != -1').groupby('worker_id').correct.mean() >= acc_thresh
                    passed_response = np.logical_not(df.groupby('worker_id').stim_response.agg(
                                                            lambda x: np.any(pd.value_counts(x) > pd.value_counts(x).sum()*response_thresh)))
                    passed_miss = pd.Series([True] * len(passed_rt), index = passed_rt.index)
                elif exp == 'tower_of_london':
                    passed_rt = df.groupby('worker_id').rt.median() >= rt_thresh
                    passed_acc = df.query('trial_id == "feedback"').groupby('worker_id').correct.mean() >= acc_thresh
                    # Labeling someone as "missing" too many problems if they don't make enough moves
                    passed_miss = (df.groupby(['worker_id','problem_id']).num_moves_made.max().reset_index().groupby('worker_id').mean() >= missed_thresh).num_moves_made
                    passed_response = pd.Series([True] * len(passed_rt), index = passed_rt.index)
                elif exp == 'two_stage_decision':
                    passed_rt = (df.groupby('worker_id').median()[['rt_first','rt_second']] >= rt_thresh).all(axis = 1)
                    passed_miss = df.groupby('worker_id').trial_id.agg(lambda x: np.mean(x == 'incomplete_trial')) < missed_thresh
                    passed_acc = pd.Series([True] * len(passed_rt), index = passed_rt.index)
                    passed_response = pd.Series([True] * len(passed_rt), index = passed_rt.index)
                    passed_response1 = np.logical_not(df.query('rt_first != -1').groupby('worker_id').key_press_first.agg(
                                                            lambda x: np.any(pd.value_counts(x) > pd.value_counts(x).sum()*response_thresh)))
                    passed_response2 = np.logical_not(df.query('rt_second != -1').groupby('worker_id').key_press_second.agg(
                                                            lambda x: np.any(pd.value_counts(x) > pd.value_counts(x).sum()*response_thresh)))
                    passed_response = np.logical_and(passed_response1,passed_response2)
                elif exp == 'writing_task':
                    passed_response = df.query('trial_id == "write"').groupby('worker_id').final_text.agg(lambda x: len(x[0]) > 100)
                    passed_acc = pd.Series([True] * len(passed_response), index = passed_rt.index)
                    passed_rt = pd.Series([True] * len(passed_response), index = passed_rt.index)
                    passed_miss = pd.Series([True] * len(passed_response), index = passed_rt.index)
                # everything else
                else:
                    passed_rt = df.query('rt != -1').groupby('worker_id').rt.median() >= rt_thresh
                    passed_miss = df.groupby('worker_id').rt.agg(lambda x: np.mean(x == -1)) < missed_thresh
                    if 'correct' in df.columns:
                        passed_acc = df.query('rt != -1').groupby('worker_id').correct.mean() >= acc_thresh
                    else:
                        passed_acc = pd.Series([True] * len(passed_rt), index = passed_rt.index)
                    if 'mouse_click' in df.columns:
                        passed_response = np.logical_not(df.query('rt != -1').groupby('worker_id').mouse_click.agg(
                                                            lambda x: np.any(pd.value_counts(x) > pd.value_counts(x).sum()*response_thresh)))
                    elif 'key_press' in df.columns:
                        passed_response = np.logical_not(df.query('rt != -1').groupby('worker_id').key_press.agg(
                                                            lambda x: np.any(pd.value_counts(x) > pd.value_counts(x).sum()*response_thresh)))   
                                                            
                passed_df = pd.concat([passed_rt,passed_acc,passed_miss,passed_response], axis = 1).fillna(False, inplace = False)
                passed = passed_df.all(axis = 1)
                failed = passed[passed == False]
                for subj in failed.index:
                    data.loc[(data.experiment_exp_id == exp) & (data.worker_id == subj),'passed_QC'] = False
        except AttributeError as e:
            print('QC could not be run on experiment %s' % exp)
            print(e)
    finish_time = (time() - start_time)/60
    print('Finished QC. Time taken: ' + str(finish_time))

def remove_failed_subjects(data):
    if 'passed_QC' not in data.columns:
        quality_check(data)
    failed_workers = data.groupby('worker_id').passed_QC.sum() < 60
    failed_workers = list(failed_workers[failed_workers].index)
    # drop workers
    failed_data = data[data['worker_id'].isin(failed_workers)]
    data.drop(failed_data.index, inplace = True)
    return failed_data
    
def save_task_data(data_loc, data):
    path = data_loc + 'Individual_Measures/'
    if not os.path.exists(path):
        os.makedirs(path)
    for exp_id in data.experiment_exp_id.unique():
        print('Saving %s...' % exp_id)
        extract_experiment(data,exp_id).to_json(path + exp_id + '.json')
    
    
    
def zscore_data(data):
    return (data - data.mean())/data.std()
    
    
    