"""
some util functions
"""
from glob import glob
import os,json
import pandas
from sklearn.metrics import confusion_matrix
import zipfile

def print_confusion_matrix(y_true,y_pred,labels=[0,1]):
    cm=confusion_matrix(y_true,y_pred)
    print('Confusion matrix')
    print('\t\tPredicted')
    print('\t\t0\t1')
    print('Actual\t0\t%d\t%d'%(cm[0,0],cm[0,1]))
    print('\t1\t%d\t%d'%(cm[1,0],cm[1,1]))

def get_behav_data(dataset = None,file=None):
    basedir=get_info('base_directory')
    if dataset == None:
        files = glob(os.path.join(basedir,'Data/Discovery*'))
        datadir = files[-1]
    else:
        datadir = os.path.join(basedir,'Data',dataset)
    if file == None:
        file = 'meaningful_variables.csv'
    datafile=os.path.join(datadir,file)
    if os.path.exists(datafile):
        df=pandas.read_csv(datafile,index_col=0)
    else:
        with zipfile.ZipFile('../Data/previous_releases.zip') as z:
            try:
                with z.open(os.path.join(dataset, file)) as f:
                    df = pandas.DataFrame.from_csv(f)
            except KeyError:
                print('Error: %s not found in %s' % (file, dataset))
                return
    return df

def get_info(item,infile='../Self_Regulation_Settings.txt'):
    """
    get info from settings file
    """

    infodict={}
    try:
        assert os.path.exists(infile)
    except:
        raise Exception('You must first create a Self_Regulation_Settings.txt file')

    with open(infile) as f:
        lines=[i for i in f.readlines() if not i.find('#')==0]
        for l in lines:
            l_s=l.rstrip('\n').split(':')
            if len(l_s)>1:
                infodict[l_s[0]]=l_s[1]
    try:
        assert item in infodict
    except:
        raise Exception('infodict does not include requested item')
    return infodict[item]

def get_single_dataset(dataset,survey):
    basedir=get_info('base_directory')
    infile=os.path.join(basedir,'data/Derived_Data/%s/surveydata/%s.tsv'%(dataset,survey))
    print(infile)
    assert os.path.exists(infile)
    if survey.find('ordinal')>-1:
        survey=survey.replace('_ordinal','')
    mdfile=os.path.join(basedir,'data/Derived_Data/%s/metadata/%s.json'%(dataset,survey))
    print(mdfile)
    assert os.path.exists(mdfile)
    data=pandas.read_csv(infile,index_col=0,sep='\t')
    metadata=load_metadata(survey,os.path.join(basedir,
        'data/Derived_Data/%s/metadata'%dataset))
    return data,metadata

def get_survey_data(dataset):
    basedir=get_info('base_directory')
    infile=os.path.join(basedir,'Data/Derived_Data/%s/surveydata.csv'%dataset)
    surveydata=pandas.read_csv(infile,index_col=0)
    keyfile=os.path.join(basedir,'Data/Derived_Data/%s/surveyitem_key.txt'%dataset)
    with open(keyfile) as f:
        keylines=[i.strip().split('\t') for i in f.readlines()]
    surveykey={}
    for k in keylines:
        surveykey[k[0]]=k[2]
    return surveydata,surveykey

def load_metadata(variable,basedir):

    with open(os.path.join(basedir,'%s.json'%variable)) as outfile:
            metadata=json.load(outfile)
    return metadata

