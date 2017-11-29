"""
some util functions
"""
from glob import glob
import os,json
import pandas,numpy
import re
from sklearn.metrics import confusion_matrix
import pkg_resources

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
    def sorting(L):
        date = L.split('_')[-1]
        month,day,year = date.split('-')
        return year, month, day

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
    try:
        assert item in infodict
    except:
        raise Exception('infodict does not include requested item: %s' % item)
    return infodict[item]

def get_item_metadata(survey, dataset=None):
    data = get_behav_data(dataset=dataset, file=os.path.join('Individual_Measures',
                                                             '%s.csv.gz' % survey))

    metadata = []
    for i in data.question_num.unique():
        item = data[data['question_num'] == i].iloc[0].to_dict()
        # drop unnecessary variables
        for drop in ['battery_name', 'finishtime', 'required', 'response',
                     'response_text', 'worker_id']:
            try:
                item.pop(drop)
            except KeyError:
                continue
        if type(item['options']) != list:
            item['options'] = eval(item['options'])
        # scoring
        values = [int(i['value']) for i in item['options']]
        sorted_values = list(range(1,len(values)+1))
        if values == sorted_values:
            item['scoring'] = 'Forward'
        elif values == sorted_values[::-1]:
            item['scoring'] = 'Reverse'
        else:
            item['scoring'] = 'Misc'
        # convert from numpy.int64 since it's not json serializable
        item['question_num']=int(item['question_num'])
        metadata.append(item)
    return metadata

def print_confusion_matrix(y_true,y_pred,labels=[0,1]):
    cm=confusion_matrix(y_true,y_pred)
    print('Confusion matrix')
    print('\t\tPredicted')
    print('\t\t0\t1')
    print('Actual\t0\t%d\t%d'%(cm[0,0],cm[0,1]))
    print('\t1\t%d\t%d'%(cm[1,0],cm[1,1]))
