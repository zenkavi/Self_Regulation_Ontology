'''
Utility functions for the ontology project
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from expanalysis.results import Result
from expanalysis.experiments.utils import result_filter
from expanalysis.experiments.processing import extract_row, extract_experiment
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

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


#***************************************************
# ********* Helper Functions **********************
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
                           
def load_data(access_token, data_loc, source = 'file', filters = None, battery = None):
    if source == 'file':
        #results = Result(filters = filters)
        #results.load_results(data_loc)
        #data = results.data
        #if battery:
            #data = result_filter(data, battery = battery)
        data = pd.read_json(data_loc + '_data.json')
    elif source == 'web':
        #Load Results from Database
        results = Result(access_token, filters = filters)
        data = results.data
        if battery:
            data = result_filter(data, battery = battery)
        #results.export(data_loc + '.json')
        data.to_json(data_loc + '_data.json')
    data.reset_index(inplace = True)
    return data                    
    
def order_by_time(data):
    data.sort_values(['worker_id','finishtime'], inplace = True)
    num_exps = data.groupby('worker_id')['finishtime'].count() 
    order = []    
    for x in num_exps:
        order += range(x)
    data.insert(data.columns.get_loc('data'), 'experiment_order', order)

def check_timing(df):
    df.loc[:, 'time_diff'] = df['time_elapsed'].diff()
    timing_cols = pd.concat([df['block_duration'], df.get('feedback_duration'), df['timing_post_trial'].shift(1)], axis = 1)
    df.loc[:, 'expected_time'] = timing_cols.sum(axis = 1)
    df.loc[:, 'timing_error'] = df['time_diff'] - df['expected_time']
    errors = [df[abs(df['timing_error']) < 500]['timing_error'].mean(), df[df['timing_error'] < 500]['timing_error'].max()]
    return errors

def export_to_csv(data, clean = False):
    for exp in np.unique(data['experiment_exp_id']):
        extract_experiment(data,exp, clean = clean).to_csv('/home/ian/' + exp + '.csv')

def get_demographics(df):
    race = (df.query('text == "What is your race?"').groupby('worker_id')['response'].unique()).value_counts()
    hispanic = (df.query('text == "Are you of Hispanic, Latino or Spanish origin?"'))['response_text'].value_counts() 
    sex = (df.query('text == "What is your sex?"'))['response_text'].value_counts() 
    age_col = df.query('text == "How old are you?"')['response'].astype(float)
    age_vars = [age_col.min(), age_col.max(), age_col.mean()]    
    return {'age': age_vars, 'sex': sex, 'race': race, 'hispanic': hispanic}  
    
    
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

def calc_bonuses(data):
    bonus_experiments = ['angling_risk_task_always_sunny', 'two_stage_decision',
                         'columbia_card_task_hot', 'columbia_card_task_cold', 'hierarchical_rule']
    bonuses = []
    for i,row in data.iterrows():
        if row['experiment_exp_id'] in bonus_experiments:
            bonus = extract_row(row, clean = False).iloc[-1].get('performance_var','error')
            bonuses.append(bonus)
        else:
            bonuses.append(np.nan)
    data['bonus'] = bonuses
    