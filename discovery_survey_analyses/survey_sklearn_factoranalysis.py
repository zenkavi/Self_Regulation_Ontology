"""
based loosely on PyIBP/example/example.py
using full LA2K dataset
"""

import os,glob
import numpy,pandas
import json

from sklearn.preprocessing import scale
from sklearn.decomposition import FactorAnalysis
from sklearn import cross_validation

data=pandas.read_csv('surveydata.csv',index_col='worker')
with open('all_survey_metadata.json', encoding='utf-8') as data_file:
        md = json.loads(data_file.read())
cdata=data.values

all_metadata={}
for measure in md.keys():
    for k in md[measure].keys():
        all_metadata[k]=md[measure][k]


kf = cross_validation.KFold(cdata.shape[0], n_folds=4)

max_components=20

sc=numpy.zeros((max_components,4))

for n_components in range(1,max_components):
    fa=FactorAnalysis(n_components=n_components)
    fold=0
    for train,test in kf:
        train_data=cdata[train,:]
        test_data=cdata[test,:]
    
        fa.fit(train_data)
        sc[n_components,fold]=fa.score(test_data)
        fold+=1
        
meanscore=numpy.mean(sc,1)
meanscore[0]=-numpy.inf
maxscore=numpy.argmax(meanscore)
print ('crossvalidation suggests %d components'%maxscore)

# now run it on full dataset to get components
fa=FactorAnalysis(n_components=maxscore)
fa.fit(cdata)

for c in range(maxscore):
    s=numpy.argsort(fa.components_[c,:])
    print('')
    print('component %d'%c)
    for i in range(3):
        print('%f: %s %s'%(fa.components_[c,s[i]],data.columns[s[i]],all_metadata[data.columns[s[i]]]['Description']))
    for i in range(len(s)-4,len(s)-1):
        print('%f: %s %s'%(fa.components_[c,s[i]],data.columns[s[i]],all_metadata[data.columns[s[i]]]['Description']))
    
with open('variable_key.txt','w') as f:   
    for k in all_metadata.keys():
        if 'Derivative' in all_metadata[k]:
            continue
        f.write('%s\t%s\n'%(k,all_metadata[k]['Description'].replace('"','')))
    

