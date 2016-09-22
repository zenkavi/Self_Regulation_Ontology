# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:14:30 2016

@author: ian
"""
import pandas as pd
lookup = pd.DataFrame.from_csv('../variable_lookup.csv')

var = DV_df.columns
extra_vars = set(lookup['Variable_Name']) - set(var)
missing_vars = set(var) - set(lookup['Variable_Name'])
missing_vars = pd.DataFrame(list(missing_vars), columns = ['Variable_Name'])
updated_lookup = lookup.query('Variable_Name not in %s' % list(extra_vars))
updated_lookup = pd.concat([updated_lookup, missing_vars], ignore_index = True).sort_values(by = 'Variable_Name')

updated_lookup.to_csv('../variable_lookup.csv', index = False)