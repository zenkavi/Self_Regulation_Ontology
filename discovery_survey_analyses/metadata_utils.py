"""
utilities for fixing metadata
"""


def metadata_subtract_one(md):
    LevelsOrig=md['Levels'].copy()
    NewLevels={}
    for l in LevelsOrig:
        NewLevels['%d'%int(int(l)-1)]=LevelsOrig[l]
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
