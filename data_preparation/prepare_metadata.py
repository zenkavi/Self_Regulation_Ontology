"""
create metadata in json format for all measures
"""

from metadata_validator import validate_exp

import os,pickle,sys
import json
from selfregulation.utils.utils import get_info,get_behav_data,get_item_metadata,get_var_category

from measure_dictionaries import measure_longnames,measure_termurls

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
task_var_dict={'hddm_drift':('DDMDriftRate','rate',True),
            "hddm_non_decision":('DDMNondecisionTime','milliseconds',True),
            'hddm_thresh':('DDMThreshold','other',True),
            "load":('load','count',True),
            'hyp_discount_rate':('hyperbolicDiscountRate','rate',True),
            'span':('span','count',False),
            'SSRT':('SSRT','milliseconds',True),
            'SSRT':('differenceSSRT','milliseconds',False),
            'hddm_drift':('differenceDDMDriftRate','rate',False),
            'slowing':('differenceSlowing','milliseconds',False),

            }
for m in measures.keys():
    metadata[m]={'measureType':get_var_category(m),
        'title':measure_longnames[m],
        'URL':{'CognitiveAtlasURL':measure_termurls[m]},
        "expFactoryName":m,
        'dataElements':{}}
    if False: # and get_var_category(m)=='survey':
        metadata[m]['dataElements']['surveyItems']=get_item_metadata(m)
    for e in measures[m]['dataElements']:
        metadata[m]['dataElements'][e]={'expFactoryName':e}
        if get_var_category(m)=='survey':
            metadata[m]['dataElements'][e]['variableClass']='surveySummary'
            metadata[m]['dataElements'][e]['variableUnits']='arbitrary'
        else:
            # get variable type for tasks
            for k in task_var_dict.keys():
                if task_var_dict[k][2]:
                    if e.find(k)==0:
                        metadata[m]['dataElements'][e]['variableClass']=task_var_dict[k][0]
                        metadata[m]['dataElements'][e]['variableUnits']=task_var_dict[k][1]
                else:
                    if e.find(k)>-1:
                        metadata[m]['dataElements'][e]['variableClass']=task_var_dict[k][0]
                        metadata[m]['dataElements'][e]['variableUnits']=task_var_dict[k][1]
            # override switch cost hddm definition
            if e.find('switch_cost')>-1:
                metadata[m]['dataElements'][e]['variableClass']='differenceSwitchCost'
                metadata[m]['dataElements'][e]['variableUnits']='milliseconds'

            if not 'variableClass' in metadata[m]['dataElements'][e]:
                metadata[m]['dataElements'][e]['variableClass']='other'
                metadata[m]['dataElements'][e]['variableUnits']='other'

json.dump(metadata, open('./metadata.json','w'), sort_keys = True, indent = 4,
                  ensure_ascii=False)
