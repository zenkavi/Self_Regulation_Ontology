from expanalysis.results import get_filters
from expanalysis.experiments.processing import extract_row, post_process_data, post_process_exp, extract_experiment, calc_DVs, extract_DVs,flag_data,  get_DV, generate_reference
from expanalysis.experiments.stats import results_check
from expanalysis.experiments.utils import result_filter
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from util import *
import json

#***************************************************
# ********* Load Data **********************
#**************************************************        
#load Data
token, data_dir = [line.rstrip('\n').split()[1] for line in open('../Self_Regulation_Settings.txt')]
data_file = data_dir + 'Battery_Results'
data_source = load_data(data_file, source = 'file', battery = 'Self Regulation Battery')
data = data_source.query('worker_id not in ["A254JKSDNE44AM", "A1O51P5O9MC5LX"]') # Sandbox workers
data.reset_index(drop = False, inplace = True)

# read preprocessed data
data = pd.read_json(data_file + '_data_post.json')


# calculate DVs and save
DV_df = extract_DVs(data)
DV_df.to_json(data_file + '_DV.json')


#anonymize data and write anonymize lookup
worker_lookup = anonymize_data(data)
json.dump(worker_lookup, open(data_dir + 'worker_lookup.json','w'))
all_data = data # validation and discovery

# only get discovery data
subject_assignment = pd.read_csv('/home/ian/Experiments/expfactory/Self_Regulation_Ontology/subject_assignment.csv')
discovery_sample = list(subject_assignment.query('dataset == "discovery"').iloc[:,0])
data = data.query('worker_id in %s' % discovery_sample)
#flag_data(data,'/home/ian/Experiments/expfactory/Self_Regulation_Ontology/post_process_reference.pkl')

# ************************************
# ********* Save Components of Data **
# ************************************
items = []
exps = []
for exp in data.experiment_exp_id.unique():
    if 'survey' in exp:
        survey = extract_experiment(data,exp)
        items += list(survey.text.unique())
        exps += [exp] * len(survey.text.unique())
items_df = pd.DataFrame({'survey': exps, 'items': items})
items_df.to_csv('/home/ian/tmp/items.csv')

# ************************************
# ********* DVs **********************
# ************************************
exp = data.experiment_exp_id.unique()[5]
print exp 
dv=get_DV(data,exp)
np.mean([i['Release_clicks'] for i in dv[0].values()])
sns.plt.hist([i['alerting_rt'] for i in dv[0].values()])

# get all DVs

subset = DV_df.drop(DV_df.filter(regex='missed_percent|avg_rt|std_rt|overall_accuracy').columns, axis = 1)
survey_df = subset.filter(regex = 'survey')

EZ_df = subset.filter(regex = 'thresh|drift')
rt_df = DV_df.filter(regex = 'avg_rt')

plot_df = rt_df
plot_df.columns = [' '.join(x.split('_')) for x in  plot_df.columns]
fig = dendroheatmap(plot_df.corr(), labels = True)
fig.savefig('/home/ian/EZ_df.png')
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
worker_count = pd.concat([data.groupby('worker_id')['finishtime'].count(), \
    data.groupby('worker_id')['battery_name'].unique()], axis = 1)
flagged = data.query('flagged == True')

#generate reference
ref_worker = 's028' #this guy works for everything except shift task
file_base = '/home/ian/Experiments/expfactory/Self_Regulation_Ontology/post_process_reference'
generate_reference(result_filter(data, worker = ref_worker), file_base)
exp_dic = pd.read_pickle(file_base + '.pkl')
pd.to_pickle(exp_dic, file_base + '.pkl')

    
  

