import os,glob,pickle
from joblib import Parallel, delayed
import time


basedir='/scratch/01329/poldrack/SRO/rf'
#basedir='/Users/poldrack/Downloads'
#clf='lasso'
clf='rf'
files=glob.glob(os.path.join(basedir,'prediction_outputs/*pkl'))
files=[i for i in files if i.find('_%s_'%clf)>-1]
files.sort()
print('found %d files'%len(files))

start = time.time()
def load_data(f):
    acc={}
    features={}
    d=pickle.load(open(f,'rb'))
    l_s=os.path.basename(f).replace('.pkl','').split('_')
    if l_s[3]=='shuffle':
        l_s[3]=l_s[4]
        l_s[1]=l_s[1]+'_shuffle'
    #data=
    if not l_s[1] in acc:
        acc[l_s[1]]={}
        features[l_s[1]]={}

    if not l_s[3] in acc[l_s[1]]:
        acc[l_s[1]][l_s[3]]=[d[0]]
        features[l_s[1]][l_s[3]]=[d[1]]
    else:
        acc[l_s[1]][l_s[3]].append(d[0])
        features[l_s[1]][l_s[3]].append(d[1])
    return (acc,features)

if os.path.exists('%s_data.pkl'%clf):
  output=pickle.load(open('%s_data.pkl'%clf,'rb'))
else:
  output=Parallel(n_jobs=60)(delayed(load_data)(f) for f in files)
  print('saving data to pickle')
  pickle.dump(output,open('%s_data.pkl'%clf,'wb'))

end = time.time()
print('elapsed time:',end - start)
