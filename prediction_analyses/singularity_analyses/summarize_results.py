import os,glob,pickle
import numpy,pandas
import scipy.stats
import selfregulation.prediction.behavpredict as behavpredict

basedir='/work/01329/poldrack/stampede2/analyses/SRO_prediction/results'
basedir='/Users/poldrack/Downloads'

files=glob.glob(os.path.join(basedir,'prediction_outputs/*pkl'))
files.sort()
acc={}
features={}
accuracy=pandas.DataFrame()

for f in files:
    d=pickle.load(open(f,'rb'))
    l_s=os.path.basename(f).replace('.pkl','').split('_')
    if l_s[2]=='shuffle':
        l_s[2]=l_s[3]
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

bp=behavpredict.BehavPredict(verbose=True,
     drop_na_thresh=100,n_jobs=1,
     skip_vars=['RetirementPercentStocks'])
bp.load_demog_data()
bp.get_demogdata_vartypes()
data=[]

for t in acc:
    print('')
    bp.load_behav_data(t)
    print(t)
    for v in acc[t]:
        meanacc=numpy.mean(acc[t][v])
        lower5pct=scipy.stats.scoreatpercentile(acc[t][v],5)
        upper5pct=scipy.stats.scoreatpercentile(acc[t][v],95)
        meanfeaturevals=numpy.mean(features[t][v],0)
        maxfeatidx=numpy.argmax(meanfeaturevals)
        minfeatidx=numpy.argmin(meanfeaturevals)
        print(v,bp.data_models[v],meanacc,lower5pct,
            upper5pct,bp.behavdata.columns[maxfeatidx],bp.behavdata.columns[minfeatidx])
        data.append([t,v,bp.data_models[v],meanacc,lower5pct,upper5pct,
            bp.behavdata.columns[maxfeatidx],meanfeaturevals[0,maxfeatidx],
            bp.behavdata.columns[minfeatidx],meanfeaturevals[0,minfeatidx]])
