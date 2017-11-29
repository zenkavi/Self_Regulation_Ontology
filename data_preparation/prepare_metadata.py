"""
create metadata in json format for all measures
"""

from metadata_validator import validate_exp

import os,pickle,sys
import json
from selfregulation.utils.utils import get_info,get_behav_data,get_item_metadata,get_var_category

from measure_dictionaries import measure_longnames,measure_termurls,measure_sobcurls

basedir=get_info('base_directory')
dataset=get_info('dataset')
outdir=os.path.join(basedir,'Data/Derived_Data/%s'%dataset)

# first get variable-level metadata
# based on variable set in meaningful_variables

behavdata=get_behav_data(dataset)
measures={}
for c in list(behavdata.columns):
    c_s=c.split('.')
    m=c_s[0]
    v='.'.join(c_s[1:])
    if not m in measures:
        measures[m]={'dataElements':[]}
    measures[m]['dataElements'].append(v)

metadata={}
# three entries are: class, type, and whether we are looking at beginning
# of string - this lets us find drift differences
# derived are for those that are
task_vars=[('hddm_drift','DDMDriftRate','rate',True),
            ("hddm_non_decision",'DDMNondecisionTime','milliseconds',True),
            ('hddm_thresh','DDMThreshold','other',True),
            ("load",'load','count',False),
            ('hyp_discount_rate','hyperbolicDiscountRate','rate',True),
            ('SSRT','SSRT','milliseconds',True),
            ('span','span','count',False),
            ('percent','other','percentage',False),
            ('SSRT','differenceSSRT','milliseconds',False),
            ('hddm_drift','differenceDDMDriftRate','rate',False),
            ('slowing','differenceSlowing','milliseconds',False)]

for m in measures.keys():
    metadata[m]={'measureType':get_var_category(m),
        'title':measure_longnames[m],
        'URL':{'CognitiveAtlasURL':measure_termurls[m],
                'SOBCURL':measure_sobcurls[m]},
        "expFactoryName":m,
        'dataElements':{}}
    if get_var_category(m)=='survey':
        item_md=get_item_metadata(m)
        metadata[m]['dataElements']['surveyItems']=item_md
    for e in measures[m]['dataElements']:
        metadata[m]['dataElements'][e]={}
        if get_var_category(m)=='survey':
            metadata[m]['dataElements'][e]['variableClass']='surveySummary'
            metadata[m]['dataElements'][e]['variableUnits']='arbitrary'
        else:
            # get variable type for tasks
            for k in task_vars:
                if k[3] is True:
                    print(m,e,k)
                    if e.find(k[0])==0:
                        print("found!")
                        metadata[m]['dataElements'][e]['variableClass']=k[1]
                        metadata[m]['dataElements'][e]['variableUnits']=k[2]
                else:
                    print(m,e,k)
                    if e.find(k[0])>0:
                        print("found!")
                        metadata[m]['dataElements'][e]['variableClass']=k[1]
                        metadata[m]['dataElements'][e]['variableUnits']=k[2]
            # override switch cost hddm definition
            if e.find('switch_cost')>-1:
                metadata[m]['dataElements'][e]['variableClass']='differenceSwitchCost'
                metadata[m]['dataElements'][e]['variableUnits']='milliseconds'

            if not 'variableClass' in metadata[m]['dataElements'][e]:
                print('not found, setting to other')
                metadata[m]['dataElements'][e]['variableClass']='other'
                metadata[m]['dataElements'][e]['variableUnits']='other'

# doublecheck that everythign is there

json.dump(metadata, open('./metadata.json','w'), sort_keys = True, indent = 4,
                  ensure_ascii=False)
