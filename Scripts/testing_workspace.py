from expanalysis.results import Result, get_filters
from expanalysis.experiments.processing import extract_row, post_process_data, post_process_exp, extract_experiment, calc_DVs, extract_DVs,flag_data,  generate_reference
from expanalysis.experiments.stats import results_check
from expanalysis.experiments.utils import result_filter, anonymize_data
from expanalysis.experiments.jspsych import calc_time_taken, print_time, get_post_task_responses
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#***************************************************
# ********* Helper Functions **********************
#**************************************************
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

def dendroheatmap(df):
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
    ax.set_yticklabels(df_rowclust.columns[::-1], rotation=0, rotation_mode="anchor", fontsize = 'medium')
    ax.set_xticklabels(df_rowclust.columns, rotation=-90, rotation_mode = "anchor", ha = 'left')
    ax1 = fig.add_axes([.1,.8,.6,.2])
    plt.axis('off')
    row_dendr = dendrogram(row_clusters, orientation='top',  
                           count_sort='ascending', ax = ax1) 
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
                           
def load_data(access_token, data_loc, source = 'file', filters = None):
    if source == 'file':
        results = Result(filters = filters)
        results.load_results(data_loc)
    elif source == 'web':
        #Load Results from Database
        results = Result(access_token, filters = filters)
        results.export(data_loc + '.json')
    data = results.data
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
    df.loc[:, 'timing_error'] = abs(df['time_diff'] - df['expected_time'])
    errors = [df[df['timing_error'] < 500]['timing_error'].mean(), df[df['timing_error'] < 500]['timing_error'].max()]
    return errors

def export_to_csv(data, clean = False):
    for exp in np.unique(data['experiment_exp_id']):
        extract_experiment(data,exp, clean = clean).to_csv('/home/ian/' + exp + '.csv')
        
#***************************************************
# ********* Load Data **********************
#**************************************************        
pd.set_option('display.width', 200)
figsize = [16,12]
#set up filters
filters = get_filters()
drop_columns = ['battery_description', 'experiment_reference', 'experiment_version', \
         'experiment_name','experiment_cognitive_atlas_task']
for col in drop_columns:
    filters[col] = {'drop': True}

                  
f = open('/home/ian/Experiments/expfactory/docs/expfactory_token.txt')
access_token = f.read().strip()      
data_loc = '/home/ian/Experiments/expfactory/Self_Regulation_Ontology/Data/Pilot_Results'     
source_data = load_data(access_token, data_loc, filters = filters, source = 'file')

#filter and process data
first_update_time = '2016-04-17T04:24:37.041870Z'
second_update_time = '2016-05-14T04:24:37.041870Z'
third_update_time = '2016-06-01T04:24:37.041870Z'

data = result_filter(source_data, battery = 'Self Regulation Pilot', finishtime = second_update_time)
worker_lookup = anonymize_data(data)
calc_time_taken(data)
get_post_task_responses(data)
post_process_data(data)
flag_data(data,'/home/ian/Experiments/expfactory/Self_Regulation_Ontology/post_process_reference.pkl')



# ************************************
# ********* DVs **********************
# ************************************
DV_df = extract_DVs(data)
DV_df.drop(DV_df.filter(regex='missed_percent').columns, axis = 1, inplace = True)

subset = DV_df.drop(DV_df.filter(regex='avg_rt|std_rt|overall_accuracy|EZ').columns, axis = 1)

EZ_df = DV_df.filter(regex = 'thresh|drift')
#rt!
rt_df = DV_df.filter(regex = 'avg_rt')

#drift
drift_df = DV_df.filter(regex = 'drift')

#thresh
thresh_df = DV_df.filter(regex = 'thresh')

#memory
memory_df = DV_df.filter(regex = '.*span.*[^avg_rt]$|.*n_back.*[^avg_rt]$|keep_track')

plot_df = subset
plot_df.columns = [' '.join(x.split('_')) for x in  plot_df.columns]
fig = dendroheatmap(plot_df.corr())
fig.savefig('/home/ian/rt_df.png')
np.mean(np.mean(plot_df.corr().mask(np.triu(np.ones(plot_df.corr().shape)).astype(np.bool))))

# ***************************
#PCA
# ***************************

X = DV_df.corr()
pca = PCA(n_components = 'mle')
pca.fit(X)
Xt = pca.transform(X)
[abs(np.corrcoef(pca.components_[0,:],X.iloc[i,:]))[0,1] for i in range(len(X))]

#PCA plotting
selection = 'EZ'
fig, ax = sns.plt.subplots()
ax.scatter(Xt[:,0],Xt[:,1],100, c = ['r' if selection in x else 'b' for x in X.columns])

for i, txt in enumerate(X.columns):
    if selection in txt:
        ax.annotate(txt, (Xt[i,0],Xt[i,1]))

fig = plt.figure(1, figsize=(12, 9))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(Xt[:, 0], Xt[:, 1], Xt[:, 2], c = ['r' if selection in x else 'b' for x in X.columns], cmap=plt.cm.spectral)
    



sns.plt.plot(pca.explained_variance_ratio_)

summary = results_check(data, silent = True, plot = True)


# ************************************
# ********* Misc Code for Reference **********************
# ************************************
data.groupby('worker_id')['finishtime'].count()
flagged = data.query('flagged == True')

#generate reference
ref_worker = 's028' #this guy works for everything except shift task
file_base = '/home/ian/Experiments/expfactory/Self_Regulation_Ontology/post_process_reference'
generate_reference(result_filter(data, worker = ref_worker), file_base)
exp_dic = pd.read_pickle(file_base + '.pkl')
pd.to_pickle(exp_dic, file_base + '.pkl')

    
  

