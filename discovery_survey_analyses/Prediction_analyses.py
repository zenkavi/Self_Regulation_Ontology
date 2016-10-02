
# coding: utf-8

# This notebook assesses the ability to predict demographic outcomes from survey data.  

# In[1]:

import os,glob,sys
import numpy,pandas
from sklearn.svm import LinearSVC,SVC,OneClassSVM
from sklearn.linear_model import LinearRegression,LogisticRegressionCV,RandomizedLogisticRegression,ElasticNet,ElasticNetCV,Ridge,RidgeCV
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold,KFold
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,classification_report,confusion_matrix
from imblearn.over_sampling import SMOTE 
from imblearn.combine import SMOTETomek
from costcla.models import CostSensitiveRandomForestClassifier,CostSensitiveDecisionTreeClassifier,CostSensitiveRandomPatchesClassifier

get_ipython().magic('matplotlib inline')

get_ipython().magic('load_ext rpy2.ipython')
get_ipython().magic('R require(mirt)')

# this is kludgey but it works
sys.path.append('../utils')
from utils import get_info,get_survey_data


dataset='Discovery_9-26-16'
basedir=get_info('base_directory')
derived_dir=os.path.join(basedir,'data/Derived_Data/%s'%dataset)



# In[2]:

get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.auto_scroll_threshold = 9999;')


# In[6]:

binary_vars=["Sex","ArrestedChargedLifeCount","DivorceCount","GamblingProblem","ChildrenNumber",
            "CreditCardDebt","RentOwn","RetirementAccount","TrafficTicketsLastYearCount","Obese",
             "TrafficAccidentsLifeCount","CaffienatedSodaCansPerDay"]

def get_demog_data(binarize=True):
    demogdata=pandas.read_csv(os.path.join(derived_dir,'surveydata/demographics.tsv'),index_col=0,delimiter='\t')
    # remove a couple of outliers
    demogdata=demogdata.query('WeightPounds>50')
    demogdata=demogdata.query('HeightInches>36')
    demogdata=demogdata.query('CaffienatedSodaCansPerDay>-1')
    demogdata=demogdata.assign(BMI=demogdata['WeightPounds']*0.45 / (demogdata['HeightInches']*0.025)**2)
    demogdata=demogdata.assign(Obese=(demogdata['BMI']>30).astype('int'))

    if binarize:
        demogdata=demogdata[binary_vars]
        demogdata=demogdata.loc[demogdata.isnull().sum(1)==0]

        for i in range(len(binary_vars)):
            v=binary_vars[i]
            if not demogdata[v].min()==0:
                demogdata.loc[demogdata[v]==demogdata[v].min(),v]=0
            demogdata.loc[demogdata[v]>demogdata[v].min(),v]=1
            assert demogdata[v].isnull().sum()==0
    return demogdata

def get_subscale_data():
    subscale_data=pandas.read_csv('survey_subscales.csv',index_col=0)
    subscale_data=subscale_data.ix[subscale_data.isnull().sum(1)==0]
    return subscale_data


def get_joint_dataset(d1,d2):
    d1_index=set(d1.index)
    d2_index=set(d2.index)
    inter=list(d1_index.intersection(d2_index))
    return d1.ix[inter],d2.ix[inter]
    return inter
surveydata_orig,surveykeys=get_survey_data('Discovery_9-26-16')
demogdata,surveydata=get_joint_dataset(get_demog_data(),surveydata_orig)
assert list(demogdata.index)==list(surveydata.index)
print('%d joint subjects found'%demogdata.shape[0])
surveyvars=list(surveydata.columns)
print('%d survey items found'%len(surveyvars))
print('Demographic variables to test:')
print(list(demogdata.columns))


# First get binary variables and test classification based on survey data.  Only include variables that have at least 10% of the infrequent category. Some of these were not collected as binary variables, but we binarize by calling anything above the minimum value a positive outcome.

# In[8]:

nfeatures=5 # number of features to show
nfolds=8
classifier='csrp'
degree=2
kernel='poly'
oversample=False

    
def print_confusion_matrix(y_true,y_pred,labels=[0,1]):
    cm=confusion_matrix(y_true,y_pred)
    print('Confusion matrix')
    print('\t\tPredicted')
    print('\t\t0\t1')
    print('Actual\t0\t%d\t%d'%(cm[0,0],cm[0,1]))
    print('\t1\t%d\t%d'%(cm[1,0],cm[1,1]))

bvardata=numpy.array(demogdata)
sdata=scale(numpy.array(surveydata))

for i in range(len(binary_vars)):
    print('')
    print('%s:'%binary_vars[i])

    y=bvardata[:,i]
    X=sdata.copy()
    kf=StratifiedKFold(y,n_folds=nfolds) # use stratified K-fold CV to get roughly equal folds
    # we use an inner CV loop on training data to estimate the best penalty value
    predlabels=[0,1]
    # define cost for 0 and 1 respectively
    #Cost matrix of the classification problem W
    #here the columns represents the costs of: 
    #false positives, false negatives, true positives and true negatives, for each example.

    cost=numpy.array([[1,0,0,0],[0,0.5/numpy.mean(y),0,0]])



    if classifier=='RandomForest':
        clf=RandomForestClassifier()
    elif classifier=='knn':
        clf=KNN()
    elif classifier=='csrf':
        clf=CostSensitiveRandomForestClassifier()
    elif classifier=='csrp':
        clf=CostSensitiveRandomPatchesClassifier()
    elif classifier=='csdt':
        clf=CostSensitiveDecisionTreeClassifier()
    elif classifier=='oneclasssvc':
        clf=OneClassSVM() 
        y[y==0]=-1
        predlabels=[-1,1]
    else:
        clf=SVC(probability=True,kernel=kernel,degree=degree) #LogisticRegressionCV(solver='liblinear',penalty='l1')  #LinearSVC()

    
    pred=numpy.zeros(len(y))
    pred_prob=numpy.zeros(len(y))

    trainpredroc=[]
    for train,test in kf:
        Xtrain=sdata[train,:].copy()
        Ytrain=y[train].copy()
        if oversample:
            sm = SMOTETomek(random_state=42)
            Xtrain, Ytrain = sm.fit_sample(Xtrain, Ytrain)
        if classifier in ['csrf','csdt','csrp']:
            cost_mat=numpy.zeros((len(Ytrain),4))
            cost_mat[Ytrain==0,:]=cost[0,:]
            cost_mat[Ytrain==1,:]=cost[1,:]

            clf.fit(Xtrain,Ytrain,cost_mat=cost_mat)
        else:
            clf.fit(Xtrain,Ytrain)
        if hasattr(clf,'predict_proba'):
            pred_prob.flat[test]=clf.predict_proba(sdata[test,:])
        pred.flat[test]=clf.predict(sdata[test,:])
        trainpredroc.append(roc_auc_score(Ytrain,clf.predict(Xtrain)))
    rocauc=roc_auc_score(y,pred_prob)
    print('Training accuracy: %f'%numpy.mean(trainpredroc))
    if numpy.var(pred)==0:
        print('WARNING: no variance in classifier output, degenerate model fit')
    if hasattr(clf,'predict_proba'):
        print('predictive accuracy (AUC: chance = 0.5) = %0.3f'%rocauc)
    else:
        print('Predictive accuracy')
    print(classification_report(y,pred,labels=predlabels))
    print_confusion_matrix(y,pred)
    print("Features sorted by their absolute correlation with outcome (top %d):"%nfeatures)
    featcorr=numpy.array([numpy.corrcoef(sdata[:,x],y)[0,1] for x in range(sdata.shape[1])])
    idx=numpy.argsort(numpy.abs(featcorr))[::-1]
    for i in range(nfeatures):
        print('%f: %s'%(featcorr[idx[i]],surveykeys[surveyvars[idx[i]]]))


# In[14]:

pred


# In[ ]:




# In[ ]:

OLDER stuff below


# In[ ]:

get_ipython().run_cell_magic('R', '-i workers', "compnums=c(3:10)\nfor (i in 1:length(compnums)) {\n  ncomps=compnums[i]\n  load(sprintf('rdata_files_wrangler/mirt_%ddims.Rdata',ncomps))\n  scores=fscores(m,full.scores = TRUE,method='MAP')\n  scores=data.frame(scores)\n  row.names(scores)=workers\n  write.table(scores,file=sprintf('factor_scores/factor_scores_%ddims.tsv',ncomps),sep='\\t',quote=FALSE,col.names=FALSE)\n}")


# Now test using scores from MIRT - note that there is a bit of leakage here because the full dataset was used to estimate the MIRT models.  Ultimately we want to fit to discovery set and test on validation set.

# In[148]:

ncomps=10
def get_mirt_data(ncomps=10):
    scoredata=pandas.read_csv('factor_scores/factor_scores_%ddims.tsv'%ncomps,delimiter='\t',index_col=0,header=None)
    scoredata=scoredata.loc[scoredata.isnull().sum(1)==0]
    return scoredata
demogdata,mirt_data=get_joint_dataset(get_demog_data(),get_mirt_data())
assert list(demogdata.index)==list(mirt_data.index)
mirt_vars=list(mirt_data.columns)
print(demogdata.shape)
print(mirt_data.shape)


# In[149]:


for i in range(len(binary_vars)):
    print('')
    y=demogdata.loc[:,binary_vars[i]].values
    kf=StratifiedKFold(y,n_folds=8) # use stratified K-fold CV to get roughly equal folds
    # we use an inner CV loop on training data to estimate the best penalty value
    if classifier=='RandomForest':
        clf=RandomForestClassifier()
    elif classifier=='knn':
        clf=KNN()
    else:
        clf=SVC(probability=True) #LogisticRegressionCV(solver='liblinear',penalty='l1')  #LinearSVC()
    
    pred=numpy.zeros(len(y))
    pred_prob=numpy.zeros(len(y))


    for train,test in kf:
        clf.fit(mirt_data.iloc[train,:].values,y[train])
        if hasattr(clf,'predict_proba'):
            pred_prob.flat[test]=clf.predict_proba(mirt_data.iloc[test,:].values)
        pred.flat[test]=clf.predict(mirt_data.iloc[test,:].values)
    
    rocauc=roc_auc_score(y,pred_prob)
    print('%s:'%binary_vars[i])

    if numpy.var(pred)==0:
        print('WARNING: no variance in classifier output, degenerate model fit')
    if hasattr(clf,'predict_proba'):
        print('predictive accuracy (AUC: chance = 0.5) = %0.3f'%rocauc)
    else:
        print('Predictive accuracy')
    print(classification_report(y,pred,labels=[0,1]))
    print_confusion_matrix(y,pred)
    print("Features sorted by their absolute correlation with outcome (top %d):"%nfeatures)
    featcorr=numpy.array([numpy.corrcoef(mirt_data.iloc[:,x],y)[0,1] for x in range(mirt_data.shape[1])])
    idx=numpy.argsort(numpy.abs(featcorr))[::-1]
    for i in range(nfeatures):
        print('%f: %s'%(featcorr[idx[i]],mirt_vars[idx[i]]))

