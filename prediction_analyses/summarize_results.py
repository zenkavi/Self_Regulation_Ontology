import os,glob
import pickle
import numpy

analysis='prediction'

datadir='%s_outputs'%analysis
if os.path.exists('%s_data.pkl'%analysis):
    truedata,permuted=pickle.load('%s_data.pkl'%analysis,'rb')
else:
    for clf in ['lasso','forest']:
        for ds in ['all','survey','task']:
            allfiles=glob.glob(os.path.join(datadir,'prediction_%s_%s*.pkl'%(ds,clf)))
            if len(allfiles)==0:
                continue
            print(clf,ds,'found %d files'%len(allfiles))
            true=[i for i in allfiles if not i.find('shuffle')>-1]
            shuf=[i for i in allfiles if i.find('shuffle')>-1]
            print(true,shuf)
            truedata=pickle.load(open(true[0],'rb'))
            permuted={}
            for k in truedata[0].keys():
                permuted[k]=[]
            for f in shuf:
                tmp=pickle.load(open(f,'rb'))
                for k in tmp[0]:
                    permuted[k].append(tmp[0][k])

    pickle.dump((truedata,permuted),open('%s_data.pkl'%analysis,'wb'))

# put into a matrix
keys=list(truedata[0].keys())
keys.sort()

permuted_data=numpy.zeros((len(keys),len(permuted[k])))

for i,k in enumerate(keys):
    permuted_data[i,:]=permuted[k]

perm_max=numpy.nanmax(permuted_data,1)
