'''
Utility functions for the ontology project
'''
import cPickle
import cStringIO
from expanalysis.experiments.processing import extract_row, extract_experiment
from expanalysis.results import Result
from expanalysis.experiments.utils import result_filter
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import sys

# Used to capture print
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = cStringIO.StringIO()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def get_log(self):
        return self.log.getvalue()
        
    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    



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

def anonymize_data(data):
    workers = data.groupby('worker_id').finishtime.min().sort_values().index
    new_ids = ['s' + str(x).zfill(3) for x in range(len(workers))]
    data.replace(workers, new_ids, inplace = True)
    return {x:y for x,y in zip(new_ids, workers)}

#***************************************************
# ********* Helper Functions **********************
#**************************************************
def append_to_json(filey, dic):
    try:
        data = json.load(open(filey,'w'))
        data.update(dic)
    except IOError:
        data = dic
        
    with open(filey, 'w') as f:
        json.dump(data, f)

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
            if isinstance(bonus,unicode):
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

def check_timing(df):
    df.loc[:, 'time_diff'] = df['time_elapsed'].diff()
    timing_cols = pd.concat([df['block_duration'], df.get('feedback_duration'), df['timing_post_trial'].shift(1)], axis = 1)
    df.loc[:, 'expected_time'] = timing_cols.sum(axis = 1)
    df.loc[:, 'timing_error'] = df['time_diff'] - df['expected_time']
    errors = [df[abs(df['timing_error']) < 500]['timing_error'].mean(), df[df['timing_error'] < 500]['timing_error'].max()]
    return errors
      
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
    
def export_to_csv(data, clean = False):
    for exp in np.unique(data['experiment_exp_id']):
        extract_experiment(data,exp, clean = clean).to_csv('/home/ian/' + exp + '.csv')

def get_bonuses(data):
    if 'bonus_zscore' not in data.columns:
        calc_bonuses(data)
    workers_finished = data.groupby('worker_id').count().finishtime==63
    index = list(workers_finished[workers_finished].index)
    tmp_data = data.query('worker_id in %s' % index)
    tmp_bonuses = tmp_data.groupby('worker_id').bonus_zscore.sum()
    min_score = tmp_bonuses.min()
    max_score = tmp_bonuses.max()
    bonuses = data.groupby('worker_id').bonus_zscore.sum()
    bonuses = (bonuses-min_score)/(max_score-min_score)*10+5
    bonuses = bonuses.map(lambda x: round(x,1))
    print('Finished getting bonuses')
    return bonuses

def get_credit(data):
    credit_array = []
    for i,row in data.iterrows():
        if row['experiment_template'] in 'jspsych':
            df = extract_row(row, clean = False)
            credit_var = df.iloc[-1].get('credit_var',999)
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
    age_col = df.query('text == "How old are you?"')['response'].astype(float)
    age_vars = [age_col.min(), age_col.max(), age_col.mean()]    
    return {'age': age_vars, 'sex': sex, 'race': race, 'hispanic': hispanic}  
    
    
def get_dummy_pay(data):
    assert 'ontask_time' in data.columns, \
        'Task time not found. Must run "calc_time_taken" first.' 
    all_exps = data.experiment_exp_id.unique()
    num_completed = data.groupby('worker_id').count().finishtime
    exps_completed = data.groupby('worker_id').experiment_exp_id.unique()
    exps_not_completed = exps_completed.map(lambda x: list(set(all_exps) - set(x) - set(['selection_optimization_compensation'])))
    almost_completed = num_completed[(num_completed<63) & (num_completed >=60)]
    not_completed = num_completed[num_completed < 60]
    task_time = data.groupby('experiment_exp_id').ontask_time.mean()/60+2 # +2 for generic instruction time
    time_spent = exps_completed.map(lambda x: np.sum([task_time[i] if task_time[i]==task_time[i] else 3 for i in x])/60)
    time_missed = exps_not_completed.map(lambda x: np.sum([task_time[i] if task_time[i]==task_time[i] else 3 for i in x])/60)
    prorate_pay = 60-time_missed[almost_completed.index]*6
    reduced_pay = time_spent[not_completed.index]*2 + np.floor(time_spent[not_completed.index])
    pay= pd.concat([reduced_pay,prorate_pay]).map(lambda x: round(x,1)).to_frame(name = 'base')
    pay['bonuses'] = get_bonuses(data)
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
    :df: plot hierarchical clustering and heatmap
    """
    #clustering
    
    #dendrogram
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
 
def load_data(data_loc, access_token = None, action = 'file', filters = None, battery = None, save = True, url = None):
    sys.stdout = Logger()
    files = glob.glob(data_loc + '_data*')
    if action == 'file':
    	data = pd.read_json(data_loc + '_data.json')
    elif action == 'overwrite':
        #Load Results from Database
        results = Result(access_token, filters = filters, url = url)
        data = results.data
        if battery:
            data = result_filter(data, battery = battery)
    elif action == 'append':
        try:
            url = json.load(open('../internal_settings.json','r'))['last_used_url']
            old_data = pd.read_json(data_loc + '_data.json')
            results = Result(access_token, filters = filters, url = url)
            new_data = results.data
            if battery:
                new_data = result_filter(new_data, battery = battery)
            data = pd.concat([old_data,new_data]).drop_duplicates(subset = 'finishtime')
        except IOError:
            print('No url found in internal_settings file. Cannot append')
            data = pd.read_json(data_loc + '_data.json')
    data.reset_index(drop = False, inplace = True)
    if action != 'file':
        if save == True:
            data.to_json(data_loc + '_data.json')
            print('Finished saving')
        final_url = sys.stdout.get_log().split('\n')[-2].split()[1]
        append_to_json('../internal_settings.json', {'last_used_url': final_url})
    return data                 
    
def order_by_time(data):
    data.sort_values(['worker_id','finishtime'], inplace = True)
    num_exps = data.groupby('worker_id')['finishtime'].count() 
    order = []    
    for x in num_exps:
        order += range(x)
    data.insert(data.columns.get_loc('data'), 'experiment_order', order)

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
    passed_QC = []
    rt_thresh_lookup = {
        'simple_reaction_time': 150    
    }
    acc_thresh_lookup = {
        'choice_reaction_time': .8
    }
    missed_thresh_lookup = {
    
    }
    for i,row in data.iterrows():
        QC = True
        if row['experiment_template'] in 'jspsych':
            exp_id = row['experiment_exp_id']
            rt_thresh = rt_thresh_lookup.get(exp_id,300)
            acc_thresh = acc_thresh_lookup.get(exp_id,0)
            missed_thresh = missed_thresh_lookup.get(exp_id,.15)
            df = extract_row(row)
            if exp_id not in ['psychological_refractory_period_two_choices', 'two_stage_decision']:
                if (df['rt'].median < rt_thresh) or \
                   (np.mean(df.get('correct',1)) < acc_thresh) or \
                   (np.mean(df['rt'] == -1) > missed_thresh):
                       QC = False
            else:
                if (df['choice1_rt'].median < rt_thresh) or \
               (np.mean(df.get('choice1_correct',1)) < acc_thresh) or \
               (((df['choice1_rt']==-1) | (df['choice2_rt'] <= -1)).mean() > missed_thresh):
                   QC = False
        passed_QC.append(QC)
    data.loc[:,'passed_QC'] = passed_QC
           
        
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    