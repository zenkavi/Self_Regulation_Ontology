import os,glob
import pickle
clf='lasso'
minsize=10

indir='/work/01329/poldrack/stampede2/code/Self_Regulation_Ontology/prediction_analyses/singularity_analyses/ls5/results/prediction_outputs'
indir='/data/01329/poldrack/SRO/lasso/prediction_outputs'
indir='/Users/poldrack/code/Self_Regulation_Ontology/results/prediction_outputs'
files=glob.glob(os.path.join(indir,'pred*pkl'))
files.sort()
datasets={}
for f in files:
    l_s=os.path.basename(f).replace('.pkl','').split('_')
    if l_s[3]=='shuffle':
        l_s[1]=l_s[1]+'_shuffle'
    if not l_s[1] in datasets:
        datasets[l_s[1]]=[]
    datasets[l_s[1]].append(f)
counter={}
completed={}
incomplete={}
allkeys=[]
allsets=['baseline','task','survey','baseline_shuffle','discounting','intelligence','stopping']

if os.path.exists('../lasso_data.pkl'):
    print('loading existing data')
    data=pickle.load(open('../lasso_data.pkl','rb'))
else:
    data={}

for t in allsets:
    if not t in datasets:
        datasets[t]={}
    if not t in data:
        data[t]={}
    if not t in counter:
        counter[t]={}
        incomplete[t]={}
        completed[t]=[]
    print('')
    print(t,len(datasets[t]))
    for v in datasets[t]:

        d=pickle.load(open(v,'rb'))
        for k in d['data'].keys():
            if not k in data[t]:
                data[t][k]=[d['data'][k]]
            else:
                data[t][k].append(d['data'][k])
            if not k in counter[t]:
                allkeys.append(k)
                counter[t][k]=1
            else:
                counter[t][k]+=1
pickle.dump(data,open('%s_data.pkl'%clf,'wb'))

allkeys=list(set(allkeys))
for t in allsets:
    for k in allkeys:
        if not k in counter[t]:
             incomplete[t][k]=minsize
             continue
        if counter[t][k]>=minsize:
               completed[t].append(k)
        else:
               incomplete[t][k]=minsize-counter[t][k]

    print(t,len(completed[t]),'completed',len(incomplete[t]),'incomplete')
