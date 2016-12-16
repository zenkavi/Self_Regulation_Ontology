"""
some util functions
"""
from glob import glob
import os,json
import pandas,numpy
import re
from sklearn.metrics import confusion_matrix
import zipfile

def print_confusion_matrix(y_true,y_pred,labels=[0,1]):
    cm=confusion_matrix(y_true,y_pred)
    print('Confusion matrix')
    print('\t\tPredicted')
    print('\t\t0\t1')
    print('Actual\t0\t%d\t%d'%(cm[0,0],cm[0,1]))
    print('\t1\t%d\t%d'%(cm[1,0],cm[1,1]))

def get_behav_data(dataset=None, file=None, full_dataset=False, flip_valence=False):
    '''Retrieves a file from a data release. By default extracts meaningful_variables from
    the most recent Discovery dataset.
    :param dataset: optional, string indicating discovery, validation, or complete dataset of interest
    :param file: optional, string indicating the file of interest
    :full_dataset: bool, default false. If True and either a discovery or validation dataset is specified, retrieve the other as well
    :flip_valence: bool, default false. If true use DV_valence.csv to flip variables based on their subjective valence
    '''
    d = {'Discovery': 'Validation', 'Validation': 'Discovery'}
    basedir=get_info('base_directory')
    pattern = re.compile('|'.join(d.keys()))
    if dataset == None:
        files = glob(os.path.join(basedir,'Data/Discovery*'))
        datadir = files[-1]
    else:
        datadir = os.path.join(basedir,'Data',dataset)
    if full_dataset == True and 'Complete' not in datadir:
        second_datadir = pattern.sub(lambda x: d[x.group()], datadir)
        datadirs = [datadir, second_datadir]
    else:
        datadirs = [datadir]
    print('Getting datasets...:\n', '\n '.join(datadirs))
    if file == None:
        file = 'meaningful_variables.csv'
    data = pandas.DataFrame()
    for datadir in datadirs:
        datafile=os.path.join(datadir,file)
        if os.path.exists(datafile):
            df=pandas.read_csv(datafile,index_col=0)
        else:
            with zipfile.ZipFile('../Data/previous_releases.zip') as z:
                try:
                    with z.open(os.path.join(os.path.basename(datadir), file)) as f:
                        df = pandas.DataFrame.from_csv(f)
                except KeyError:
                    print('Error: %s not found in %s' % (file, datadir))
                    df = pandas.DataFrame()
                    continue
        data = pandas.concat([df,data])
    
    def valence_flip(data, flip_list):
        for c in data.columns:
            try:
                data.loc[:,c] = data.loc[:,c] * flip_list.loc[c]
            except TypeError:
                continue
    if flip_valence==True:
        print('Flipping variables based on valence')
        flip_df = os.path.join(datadirs[0], 'DV_valence.csv')
        valence_flip(data, flip_df)
    return data.sort_index()

def get_info(item,infile=None):
    """
    get info from settings file
    """

    infodict={}
    if not infile:
        s = 'Self_Regulation_Ontology'
        path = os.path.abspath('.')
        infile = os.path.join(path[:path.find(s)+len(s)], 'Self_Regulation_Settings.txt')
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


def get_demographics(dataset,var_subset=None,full_dataset=False):
    """
    misnomer - actually get demographics, alc/drug, and health
    """
    basedir=get_info('base_directory')
    if not full_dataset:
        datasets=[dataset]
    else:
        if dataset.find('Discovery')==0:
            datasets=[dataset,dataset.replace('Discovery','Validation')]
        else:
            datasets=[dataset,dataset.replace('Validation','Discovery')]
        print('using datasets:',datasets)
    ds_all={}
    for ds in datasets:
      for i,survey in enumerate(['demographics_ordinal','alcohol_drugs_ordinal','health_ordinal']):
        infile=os.path.join(basedir,'Data/%s/%s.csv'%(ds,survey))
        if i==0:
            ds_all[ds]=pandas.DataFrame.from_csv(infile,index_col=0,sep=',')
        else:
            data=pandas.DataFrame.from_csv(infile,index_col=0,sep=',')
            ds_all[ds]=ds_all[ds].merge(data,'inner',right_index=True,left_index=True)
    if len(ds_all)==1:
        alldata=ds_all[ds]
    else:
        alldata=pandas.concat([ds_all[ds] for ds in ds_all.keys()])
    badweight=alldata['WeightPounds']<80
    badheight=alldata['HeightInches']<36
    alldata.loc[badweight,'WeightPounds']=numpy.nan
    alldata.loc[badheight,'HeightInches']=numpy.nan
    alldata['BMI']=alldata['WeightPounds']*0.45 / (alldata['HeightInches']*0.025)**2

    if not var_subset is None:
        for c in alldata.columns:
            if not c in var_subset:
                del alldata[c]

    return(alldata)
