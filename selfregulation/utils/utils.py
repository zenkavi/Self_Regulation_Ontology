"""
some util functions
"""
from glob import glob
import os,json
import pandas,numpy
import re
from sklearn.metrics import confusion_matrix
import pkg_resources
from collections import OrderedDict

# Regex filtering helper functions

def not_regex(txt):
        return '^((?!%s).)*$' % txt

def filter_behav_data(data, filter_regex):
    """ filters dataframe using regex
    Args:
        filter_regex: regex expression to filter data columns on. Can also supply
            "survey(s)" or "task(s)" to return measures associated with those
    """
    # filter columns if filter_regex is set
    if filter_regex.rstrip('s').lower() == 'survey':
        regex = not_regex(not_regex('survey')+'|cognitive_reflection|holt')
    elif filter_regex.rstrip('s').lower() == 'task':
        regex = not_regex('survey')+'|cognitive_reflection|holt'
    else:
        regex = filter_regex
    return data.filter(regex=regex)

def get_var_category(var):
    ''' Return "task" or "survey" classification for variable

    var: variable name passed as a string
    '''
    m = re.match(not_regex('survey')+'|cognitive_reflection|holt', var)
    if m is None:
        return 'survey'
    else:
        return 'task'

# Data get methods
def sorting(L):
    date = L.split('_')[-1]
    month,day,year = date.split('-')
    return year, month, day

def get_behav_data(dataset=None, file=None, filter_regex=None,
                flip_valence=False, verbose=False, full_dataset=None):
    '''Retrieves a file from a data release.

    By default extracts meaningful_variables from the most recent Complete dataset.

    Args:
        dataset: optional, string indicating discovery, validation, or complete dataset of interest
        file: optional, string indicating the file of interest
        filter_regex: regex expression to filter data columns on. Can also supply
            "survey(s)" or "task(s)" to return measures associated with those
        flip_valence: bool, default false. If true use DV_valence.csv to flip variables based on their subjective valence
    '''
    if full_dataset is not None:
        print("Full dataset is deprecrated and no longer functional")

    basedir=get_info('base_directory')
    if dataset == None:
        files = glob(os.path.join(basedir,'Data/Complete*'))
        files.sort(key=sorting)
        datadir = files[-1]
    else:
        datadir = os.path.join(basedir,'Data',dataset)
    if file == None:
        file = 'meaningful_variables.csv'
    if verbose:
        print('Getting dataset: %s...:\n' 'file: %s \n ' % (datadir, file))
    datafile=os.path.join(datadir,file)
    if os.path.exists(datafile):
        data=pandas.read_csv(datafile,index_col=0)
        print('Getting %s from %s' % (file, datadir))
    else:
        data = pandas.DataFrame()
        print('Error: %s not found in %s' % (file, datadir))
        return None

    def valence_flip(data, flip_list):
        for c in data.columns:
            try:
                data.loc[:,c] = data.loc[:,c] * flip_list.loc[c]
            except TypeError:
                continue
    if flip_valence==True:
        print('Flipping variables based on valence')
        flip_df = os.path.join(datadir, 'DV_valence.csv')
        valence_flip(data, flip_df)
    if filter_regex is not None:
        data = filter_behav_data(data, filter_regex=filter_regex)
    return data.sort_index()

def get_info(item,infile=None):
    """
    get info from settings file
    """
    config=pkg_resources.resource_string('selfregulation',
                        'data/Self_Regulation_Settings.txt')
    config=str(config,'utf-8').strip()
    infodict={}
    for l in config.split('\n'):
        if l.find('#')==0:
            continue
        l_s=l.rstrip('\n').split(':')
        if len(l_s)>1:
                infodict[l_s[0]]=l_s[1]
    if (item == 'dataset') and (not 'dataset' in infodict):
        files = glob(os.path.join(infodict['base_directory'],'Data/Complete*'))
        files.sort(key=sorting)
        datadir = files[-1]
        return os.path.basename(datadir)
    try:
        assert item in infodict
    except:
        raise Exception('infodict does not include requested item: %s' % item)
    return infodict[item]

def get_item_metadata(survey, dataset=None,verbose=False):
    data = get_behav_data(dataset=dataset, file=os.path.join('Individual_Measures',
                                                             '%s.csv.gz' % survey))

    metadata = []
    for i in data.question_num.unique():
        item = data[data['question_num'] == i].iloc[0].to_dict()
        # drop unnecessary variables
        for drop in ['battery_name', 'finishtime', 'required', 'response',
                     'response_text', 'worker_id','experiment_exp_id']:
            try:
                item.pop(drop)
            except KeyError:
                continue
        if type(item['options']) != list:
            if verbose:
                print(item['options'])
            item['options'] = eval(item['options'])
        # turn options into an ordered dict, indexed by option number
        item['responseOptions']=OrderedDict()
        for o in item['options']:
            option_num=int(o['id'].split('_')[-1])
            o.pop('id')
            o['valueOrig']=option_num
            try:
                v=int(o['value'])
            except ValueError:
                v=o['value']
            if v in item['responseOptions']:
                item['responseOptions'][v]['valueOrig']=[item['responseOptions'][v]['valueOrig'],o['valueOrig']]
                item['responseOptions'][v]['text']=[item['responseOptions'][v]['text'],o['text']]
            else:
                item['responseOptions'][v]=o.copy()
                item['responseOptions'][v].pop('value')
        # scoring
        values = [int(i['value']) for i in item['options']]
        sorted_values = list(range(1,len(values)+1))
        cc=numpy.corrcoef(values,sorted_values)[0,1]
        if cc>0.5:
            item['scoring'] = 'Forward'
        elif cc<0.5:
            item['scoring'] = 'Reverse'
        else:
            item['scoring'] = 'other'
        # convert from numpy.int64 since it's not json serializable
        item['question_num']=int(item['question_num'])
        item_s=item['id'].replace('_options','').split('_')
        item['expFactoryName']='_'.join(item_s[:-1])+'.'+item_s[-1]
        item.pop('id')
        item.pop('options')
        metadata.append(item)
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

def print_confusion_matrix(y_true,y_pred,labels=[0,1]):
    cm=confusion_matrix(y_true,y_pred)
    print('Confusion matrix')
    print('\t\tPredicted')
    print('\t\t0\t1')
    print('Actual\t0\t%d\t%d'%(cm[0,0],cm[0,1]))
    print('\t1\t%d\t%d'%(cm[1,0],cm[1,1]))
