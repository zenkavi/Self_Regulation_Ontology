# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy,pandas
import os
import json

from metadata_utils import write_metadata,metadata_reverse_scale

from utils import get_info

basedir=os.path.join(get_info('base_directory'),'discovery_survey_analyses')

def get_data(basedir=basedir):

    datafile=os.path.join(basedir,'health.csv')

    data=pandas.read_csv(datafile,index_col=0)
    data=data.rename(columns={'worker_id':'worker'})
    return data,basedir



def get_health_items(data):
    health_items={}
    for i,r in data.iterrows():
        if not r.text in health_items.keys():
            if r.text.find('If')==0:
                health_items[r.id+'-'+r.text]=r
            else:
                health_items[r.text]=r
    return health_items


def setup_itemid_dict():
    nominalvars=[]
    id_to_name={}
    id_to_name['k6_survey_3']='Nervous'
    id_to_name['k6_survey_4']='Hopeless'
    id_to_name['k6_survey_5']='RestlessFidgety'
    id_to_name['k6_survey_6']='Depressed'
    id_to_name['k6_survey_7']='EverythingIsEffort'
    id_to_name['k6_survey_8']='Worthless'
    id_to_name['k6_survey_9']='Last30DaysUsual'
    id_to_name['k6_survey_11']='DaysLostLastMonth'
    id_to_name['k6_survey_12']='DaysHalfLastMonth'
    id_to_name['k6_survey_13']='DoctorVisitsLastMonth'
    id_to_name['k6_survey_14']='DaysPhysicalHealthFeelings'
    id_to_name['k6_survey_15']='PsychDiagnoses'
    nominalvars.append('k6_survey_15')
    id_to_name['k6_survey_16']='PsychDiagnosesOther'
    nominalvars.append('k6_survey_16')
    id_to_name['k6_survey_17']='NeurologicalDiagnoses'
    id_to_name['k6_survey_18']='NeurologicalDiagnosesDescribe'
    nominalvars.append('k6_survey_18')
    id_to_name['k6_survey_19']='DiseaseDiagnoses'
    nominalvars.append('k6_survey_19')
    id_to_name['k6_survey_20']='DiseaseDiagnosesOther'
    nominalvars.append('k6_survey_20')
    return id_to_name,nominalvars

def get_metadata(health_items):
    all_itemids=[]
    id_to_name,nominalvars=setup_itemid_dict()
    health_dict={"MeasurementToolMetadata": {"Description": 'Health',
            "TermURL": ''}}
    for i in health_items:
            r=health_items[i]
            if not pandas.isnull(r.options):
                itemoptions=eval(r.options)
            else:
                itemoptions=None
            itemid='_'.join(r['id'].split('_')[:3])
            assert itemid not in health_dict  # check for duplicates
            health_dict[itemid]={}
            health_dict[itemid]['Description']=r.text
            health_dict[itemid]['Levels']={}
            if itemid in nominalvars:
                health_dict[itemid]['Nominal']=True
            levelctr=0
            if itemoptions is not None:
                for i in itemoptions:
                    if not 'value' in i:
                        value=levelctr
                        levelctr+=1
                    else:
                        value=i['value']
                    health_dict[itemid]['Levels'][value]=i['text']
    #rename according to more useful names
    health_dict_renamed={}
    for k in health_dict.keys():
        if not k in id_to_name.keys():
            health_dict_renamed[k]=health_dict[k]
        else:
            health_dict_renamed[id_to_name[k]]=health_dict[k]
    return health_dict_renamed

def add_health_item_labels(data):
    item_ids=[]
    for i,r in data.iterrows():
        item_ids.append('_'.join(r['id'].split('_')[:3]))
    data['item_id']=item_ids
    return data

def fix_item(d,v,metadata):
    """
    clean up responses
    """

    id_to_name,nominalvars=setup_itemid_dict()
    vname=id_to_name[v]
    # variables that need to have scale reversed - from 5-1 to 1-5
    reverse_scale=['DaysPhysicalHealthFeelings','Depressed','EverythingIsEffort',
                'Hopeless','Nervous','RestlessFidgety','Worthless',"Last30DaysUsual"]
    if vname in reverse_scale:
        tmp=numpy.array([float(i) for i in d])
        d.iloc[:]=tmp*-1 + len(tmp)
        print('reversed scale:',v,vname)
        metadata=metadata_reverse_scale(metadata)
    return d,metadata

def save_health_data(data,survey_metadata,
              outdir=basedir + '/surveydata'):
    id_to_name,nominalvars=setup_itemid_dict()
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    unique_items=list(data.item_id.unique())
    surveydata=pandas.DataFrame(index=list(data.worker.unique()))
    for i in unique_items:
        qresult=data.query('item_id=="%s"'%i)
        matchitem=qresult.response
        matchitem.index=qresult['worker']
        matchitem,survey_metadata[id_to_name[i]]=fix_item(matchitem,i,survey_metadata[id_to_name[i]])
        surveydata.ix[matchitem.index,i]=matchitem

    surveydata_renamed=surveydata.rename(columns=id_to_name)
    surveydata_renamed.to_csv(os.path.join(outdir,'health.tsv'),sep='\t')
    for v in nominalvars:
        del surveydata[v]
    surveydata_renamed_ord=surveydata.rename(columns=id_to_name)
    surveydata_renamed_ord.to_csv(os.path.join(outdir,'health_ordinal.tsv'),sep='\t')

    return outdir,surveydata_renamed

if __name__=='__main__':
    id_to_name,nominalvars=setup_itemid_dict()
    data,basedir=get_data()
    health_items=get_health_items(data)
    health_metadata=get_metadata(health_items)
    data=add_health_item_labels(data)
    datadir,surveydata_renamed=save_health_data(data,health_metadata)
    metadatadir=write_metadata(health_metadata,'health.json')
