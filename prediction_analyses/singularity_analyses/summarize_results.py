import os,glob,pickle
import numpy,pandas

files=glob.glob('/work/01329/poldrack/stampede2/analyses/SRO_prediction/results/prediction_outputs/*pkl')
files.sort()
datasets={}
accuracy=pandas.DataFrame()

for f in files:
    d=pickle.load(open(f,'rb'))
    l_s=os.path.basename(f).replace('.pkl','').split('_')  
    if l_s[2]=='shuffle':
        l_s[2]=l_s[3]
        l_s[1]=l_s[1]+'_shuffle'
    data=
    if not l_s[1] in datasets:
        acc[l_s[1]]={}
    if not l_s[3] in datasets[l_s[1]]:
        acc[l_s[1]][l_s[3]]=1
    else:
        acc[l_s[1]][l_s[3]]+=1

for t in datasets:
    print('')
    print(t)
    for v in datasets[t]:
        print(v,datasets[t][v])
