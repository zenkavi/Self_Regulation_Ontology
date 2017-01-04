import os,glob
import pickle
import numpy

files=glob.glob('outputs/*.pkl')
files.sort()
idx=[]
for f in files:
    d=pickle.load(open(f,'rb'))
    print(d.params.objective_weights,d.bestp_saved[len(d.bestp_saved)-1])
    bestp=d.bestp_saved[len(d.bestp_saved)-1]
    bestp.sort()
    if not numpy.prod(numpy.array(bestp)) in idx:
        idx.append(numpy.prod(numpy.array(bestp)))
        print('best set: task (time)')
        totaltasktime=0
        for i in bestp:
            print(i,d.tasks[i],'(%f)'%d.params.tasktime[i])
            totaltasktime+=d.params.tasktime[i]
        print('total task time:',totaltasktime)
