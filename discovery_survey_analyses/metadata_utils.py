"""
utilities for fixing metadata
"""
import os,json

def write_metadata(metadata,fname,
    outdir='/Users/poldrack/code/Self_Regulation_Ontology/discovery_survey_analyses/metadata'):
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    with open(os.path.join(outdir,fname), 'w') as outfile:
            json.dump(metadata, outfile, sort_keys = True, indent = 4,
                  ensure_ascii=False)
    return outdir

def metadata_subtract_one(md):
    LevelsOrig=md['Levels'].copy()
    NewLevels={}
    for l in LevelsOrig:
        NewLevels['%d'%int(int(l)*-1+5)]=LevelsOrig[l]
    md['LevelsOrig']=LevelsOrig
    md['Levels']=NewLevels
    return md

def metadata_reverse_scale(md):
    LevelsOrig=md['Levels'].copy()
    NewLevels={}
    for l in LevelsOrig:
        NewLevels['%d'%int(int(l)*-1 + 5)]=LevelsOrig[l]
    md['LevelsOrig']=LevelsOrig
    md['Levels']=NewLevels
    return md

def metadata_replace_zero_with_nan(md):
    LevelsOrig=md['Levels'].copy()
    NewLevels={}
    for l in LevelsOrig:
        if not l.find('0')>-1:
            NewLevels[l]=LevelsOrig[l]
        else:
            NewLevels['n/a']=LevelsOrig[l]
    md['LevelsOrig']=LevelsOrig
    md['Levels']=NewLevels
    return md

def metadata_change_two_to_zero_for_no(md):
    LevelsOrig=md['Levels'].copy()
    NewLevels={}
    for l in LevelsOrig:
        if l.find('2')>-1:
            NewLevels['0']=LevelsOrig[l]
        else:
            NewLevels[l]=LevelsOrig[l]
    md['LevelsOrig']=LevelsOrig
    md['Levels']=NewLevels
    return md
