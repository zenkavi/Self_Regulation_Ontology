
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
from sklearn.decomposition import FactorAnalysis
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,classification_report,confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

# this is kludgey but it works
sys.path.append('../utils')
from utils import get_info,get_survey_data


dataset='Discovery_9-26-16'
basedir=get_info('base_directory')
derived_dir=os.path.join(basedir,'Data/Derived_Data/%s'%dataset)



binary_vars=["Sex","ArrestedChargedLifeCount","DivorceCount","GamblingProblem","ChildrenNumber",
            "CreditCardDebt","RentOwn","RetirementAccount","TrafficTicketsLastYearCount","Obese",
             "TrafficAccidentsLifeCount","CaffienatedSodaCansPerDay"]
# note: focusing here on a subset that seem predictable, for testing of classifiers
binary_vars=['Sex','ChildrenNumber','Obese','TrafficAccidentsLifeCount']
binary_vars=['Nervous']
item_thresholds={'Nervous':1}
def get_demog_data(binary_vars=binary_vars,ordinal_vars=[],item_thresholds={},binarize=True):
    demogdata=pandas.read_csv(os.path.join(derived_dir,'surveydata/demographics_ordinal.tsv'),index_col=0,delimiter='\t')
    healthdata=pandas.read_csv(os.path.join(derived_dir,'surveydata/health_ordinal.tsv'),index_col=0,delimiter='\t')
    alcdrugs=pandas.read_csv(os.path.join(derived_dir,'surveydata/alcohol_drugs_ordinal.tsv'),index_col=0,delimiter='\t')
    assert all(demogdata.index==healthdata.index)
    assert all(demogdata.index==alcdrugs.index)
    demogdata=demogdata.merge(healthdata,left_index=True,right_index=True)
    demogdata=demogdata.merge(alcdrugs,left_index=True,right_index=True)
    # remove a couple of outliers - this is only for cases when we include BMI/obesity
    if 'BMI' in ordinal_vars:
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
            if v in item_thresholds:
                threshold=item_thresholds[v]
            else:
                threshold=demogdata[v].min()
            demogdata.loc[demogdata[v]>threshold,v]=1
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
surveykeys['HeightInches']='HeightInches'
surveykeys['WeightPounds']='WeightPounds'
surveykeys['Sex']='Sex'

demogdata,surveydata=get_joint_dataset(get_demog_data(),surveydata_orig)
assert list(demogdata.index)==list(surveydata.index)
print('%d joint subjects found'%demogdata.shape[0])
surveyvars=list(surveydata.columns)
print('%d survey items found'%len(surveyvars))
print('Demographic variables to test:')
print(list(demogdata.columns))


# First get binary variables and test classification based on survey data.  Only include variables that have at least 10% of the infrequent category. Some of these were not collected as binary variables, but we binarize by calling anything above the minimum value a positive outcome.

nfeatures=5 # number of features to show
nfolds=8
degree=3
kernel='rbf'
shuffle=False
add_cheat_vars=False
verbose=False
use_fa='fa'

def print_confusion_matrix(y_true,y_pred,labels=[0,1]):
    cm=confusion_matrix(y_true,y_pred)
    print('Confusion matrix')
    print('\t\tPredicted')
    print('\t\t0\t1')
    print('Actual\t0\t%d\t%d'%(cm[0,0],cm[0,1]))
    print('\t1\t%d\t%d'%(cm[1,0],cm[1,1]))

bvardata=numpy.array(demogdata)
sdata=numpy.array(surveydata) #scale(numpy.array(surveydata))
fa=FactorAnalysis(20)  # reduce to 20 dimensions

results=pandas.DataFrame(columns=['variable','kernel','fa','smote','smvC','trainf1','testf1'])

try:
    clf_result
except:
    clf_result={}

ctr=0

classifier='svm'
gamma='auto'

for i in range(len(binary_vars)):
    y=numpy.array(demogdata[binary_vars[i]])
    if shuffle:
        numpy.random.shuffle(y)
    kf=StratifiedKFold(y,n_folds=nfolds) # use stratified K-fold CV to get roughly equal folds

    for kernel in ['linear','rbf','poly']:
        for use_fa in ['fa','nofa']:
          for svmC in [0.1,0.5,1,5,10,50,100]:
            if numpy.abs(numpy.mean(y)-0.5)>0.1:
                oversample='smote'
            else:
                oversample='none'
            print('%s\t%s\t%s\t%f\t%s:'%(kernel,use_fa,oversample,svmC,binary_vars[i]))

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
            elif classifier=='oneclasssvc':
                clf=OneClassSVM()
                y[y==0]=-1
                predlabels=[-1,1]
            else:
                clf=SVC(kernel=kernel,degree=degree,gamma=gamma,C=svmC) #LogisticRegressionCV(solver='liblinear',penalty='l1')  #LinearSVC()


            pred=numpy.zeros(len(y))
            pred_prob=numpy.zeros(len(y))

            trainpredroc=[]
            for train,test in kf:
                if use_fa=='fa':
                    Xtrain=fa.fit_transform(sdata[train,:])
                    Xtest=fa.transform(sdata[test,:])
                else:
                    Xtrain=sdata[train,:]
                    Xtest=sdata[test,:]
                if add_cheat_vars:
                    Xtrain=numpy.hstack((Xtrain,numpy.array(demogdata[['WeightPounds','HeightInches','Sex']].iloc[train,:])))
                    Xtest=numpy.hstack((Xtest,numpy.array(demogdata[['WeightPounds','HeightInches','Sex']].iloc[test,:])))
                Ytrain=y[train]
                if oversample=='smote':
                    sm = SMOTETomek(random_state=42)
                    Xtrain, Ytrain = sm.fit_sample(Xtrain, Ytrain)
                if classifier in ['csrf','csdt','csrp']:
                    cost_mat=numpy.zeros((len(Ytrain),4))
                    cost_mat[Ytrain==0,:]=cost[0,:]
                    cost_mat[Ytrain==1,:]=cost[1,:]
                    clf.fit(Xtrain,Ytrain,cost_mat=cost_mat)
                else:
                    clf.fit(Xtrain,Ytrain)
                pred.flat[test]=clf.predict(Xtest)
                trainpredroc.append(f1_score(Ytrain,clf.predict(Xtrain)))
            cm=confusion_matrix(y,pred)
            results.loc[ctr,:]=[binary_vars[i],kernel,use_fa,oversample,svmC,numpy.mean(trainpredroc),f1_score(y,pred)]
            ctr+=1
            if verbose:
                print('Training accuracy (f-score): %f'%numpy.mean(trainpredroc))
                if numpy.var(pred)==0:
                    print('WARNING: no variance in classifier output, degenerate model fit')
                print('Predictive accuracy')
                print(classification_report(y,pred,labels=predlabels))
                print_confusion_matrix(y,pred)
                if False:
                    print("Features sorted by their absolute correlation with outcome (top %d):"%nfeatures)
                    featcorr=numpy.array([numpy.corrcoef(sdata[:,x],y)[0,1] for x in range(sdata.shape[1])])
                    idx=numpy.argsort(numpy.abs(featcorr))[::-1]
                    for i in range(nfeatures):
                        print('%f: %s'%(featcorr[idx[i]],surveykeys[surveyvars[idx[i]]]))

#print(results[[0,1,2,3,5]])
for v in binary_vars:
    d=results.query('variable=="%s"'%v)
    if d.testf1.max()<0.55:
        print('%s: poor performance (%f max test accuracy)'%(v,d.testf1.max()))
    else:
        print(d)
        #print(d[d.testf1==d.testf1.max()])
    print('')
