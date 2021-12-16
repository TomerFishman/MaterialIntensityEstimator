# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:42:14 2020

@author: Tomer Fishman
"""
# %% libraries and set up dimensions
# import numpy as np
from os import chdir
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

chdir('C:\\Users\\Tomer\\Dropbox\\-the research\\2020 10 IIASA\\MI_project\\buildings_db')

dims_structure_import = pd.read_excel("dims_structure.xlsx", sheet_name="dims_structure")

dims_names = list(dims_structure_import.columns)

# TODO remove unused dimensions
dims_names = dims_names[7:]

dims_list = []
dims_len = []

for dim_x in dims_names:
    # calculate the number of entities in the dimension
    dim_lastvalidrow = dims_structure_import[dim_x].last_valid_index() + 1
    dims_list += [list(dims_structure_import[dim_x][2:dim_lastvalidrow])]
    dims_len.append(len(dims_structure_import[dim_x][2:dim_lastvalidrow]))


# %% load the MI database with const. type ML results from Orange

buildings_import = pd.read_excel("buildings_v2-const_type_ML.xlsx", sheet_name="Sheet1")

# create new column const_short where U from 'Construction type' is replaced by 'Random Forest'
buildings_import['const_short'] = buildings_import['Random Forest'].where((buildings_import['Construction type'].str.match('U')), buildings_import['Construction type'])

# clean up buildings_import
buildings_import = buildings_import[['id', 'concrete', 'steel', 'wood', 'brick', 'R5_32', 'use_short', 'const_short']]

# slice and selection examples. See also https://stackoverflow.com/questions/53927460/select-rows-in-pandas-multiindex-dataframe
buildings_import.loc[(buildings_import['R5_32'].str.match('OECD_JPN')), ('id', 'steel', 'R5_32')]
buildings_import.loc[(buildings_import['R5_32'].str.match('OECD_JPN')) & (buildings_import['use_short'].str.match('RM')), ('id', 'steel', 'R5_32', 'use_short')]
buildings_import.query("R5_32 == 'OECD_JPN' and use_short == 'RM'")

# violin plots https://seaborn.pydata.org/generated/seaborn.violinplot.html
sns.violinplot(x="concrete", y="const_short", data=buildings_import, cut=0, linewidth=1).legend_.remove()
sns.violinplot(x="concrete", y="const_short", hue="use_short", data=buildings_import, cut=0, linewidth=1, scale="width")
sns.violinplot(x="const_short", y="concrete", hue="use_short", data=buildings_import, cut=0, linewidth=.5, scale="width", bw=.1, height=2, aspect=1)
sns.catplot(x="const_short", y="concrete", hue="use_short", kind="violin", data=buildings_import, cut=0, linewidth=1, scale="width", bw=.15, height=8, aspect=1.8)
sns.catplot(x="use_short", y="concrete", hue="const_short", kind="violin", data=buildings_import, cut=0, linewidth=1, scale="width", bw=.15, height=8, aspect=1.8)

# bivariate distribution plots https://seaborn.pydata.org/tutorial/distributions.html#visualizing-bivariate-distributions
sns.jointplot(data=buildings_import, x="concrete", y="steel", hue="const_short", linewidth=.1, xlim=(0, 3500), ylim=(0, 500), marginal_ticks=False)
sns.jointplot(data=buildings_import, x="concrete", y="steel", hue="const_short", linewidth=.1, xlim=(0, round(max(buildings_import['concrete']), ndigits=-2)), ylim=(0, 500))
sns.jointplot(data=buildings_import, x="concrete", y="wood", hue="const_short", linewidth=.1, xlim=(0, round(max(buildings_import['concrete']), ndigits=-2)), ylim=(0, 300))
sns.jointplot(data=buildings_import, x="concrete", y="brick", hue="const_short", linewidth=.1, xlim=(0, round(max(buildings_import['concrete']), ndigits=-2)), ylim=(0, round(max(buildings_import['brick']), ndigits=-2)))

# set up the same multiindex as the other dataframes
buildings_import.set_index(dims_names, inplace=True)
# note the previous .loc slice and selection examples don't work now. Use query (or xs from the link)
buildings_import.query("R5_32 == 'OECD_JPN' and use_short == 'RM'")[['id', 'steel']]

# %% create a new dataframe of the counts of unique combinations that exist in the DB
# including unspecifieds
db_combinations = pd.DataFrame(data=None, index=pd.MultiIndex.from_product(dims_list, names=dims_names))

# add columns of counts in the db for each mmterial
db_combinations['concrete_count'] = buildings_import.groupby(dims_names).concrete.count()
db_combinations['steel_count'] = buildings_import.groupby(dims_names).steel.count()
db_combinations['wood_count'] = buildings_import.groupby(dims_names).wood.count()
db_combinations['brick_count'] = buildings_import.groupby(dims_names).brick.count()

# add columns of means and SDs in the db for each mmterial
# db_combinations['concrete_mean'] = buildings_import.groupby(dims_names).concrete.mean()
# db_combinations['concrete_sd'] = buildings_import.groupby(dims_names).concrete.std()
# db_combinations['steel_mean'] = buildings_import.groupby(dims_names).steel.mean()
# db_combinations['steel_std'] = buildings_import.groupby(dims_names).steel.std()
# db_combinations['wood_mean'] = buildings_import.groupby(dims_names).wood.mean()
# db_combinations['wood_std'] = buildings_import.groupby(dims_names).wood.std()
# db_combinations['brick_mean'] = buildings_import.groupby(dims_names).brick.mean()
# db_combinations['brick_std'] = buildings_import.groupby(dims_names).brick.std()

# replace NANs with zeros for consistency
db_combinations = db_combinations.fillna(0)

# slice db_combinations that are unspecified in either use or const (should be equivalent to db_combinations.query("use_short == 'UN' or use_short == 'RU') because const_short == 'U' should not exist
db_combinations.query("use_short == 'UN' or use_short == 'RU' or const_short == 'U'")
# slice db_combinations that are NOT unspecified in either use or const
db_combinations.query("use_short != 'UN' and use_short != 'RU' and const_short != 'U'")

# exoort db_combinations
# db_combinations.to_excel("db_combinations.xlsx", sheet_name="sheet1")
# db_combinations.unstack().to_clipboard()

# %% create a dataframe with all practical (i.e. not unspecifieds) combination options to be filled with data

# remove 'unspecified' entities !!make sure to change the list indexes as needed
dims_list_specified = dims_list[:]
dims_list_specified[0] = [x for x in dims_list_specified[0] if 'U' not in x]
dims_list_specified[1] = [x for x in dims_list_specified[1] if 'U' not in x]

# create multi index dataframe of all possible combinations
mi_estimation = pd.DataFrame(data=None, index=pd.MultiIndex.from_product(dims_list_specified, names=dims_names))
mi_estimation.memory_usage(deep=True).sum()

# add counts from db_combination_counts
mi_estimation['steel_count'] = db_combinations['steel_count']
mi_estimation['wood_count'] = db_combinations['wood_count']
mi_estimation['brick_count'] = db_combinations['brick_count']
mi_estimation['concrete_count'] = db_combinations['concrete_count']

# %% selection algorithm: create a temporary df of selected MIs from buildings_import

# TODO further simplifications for SANDBOX: sort by count of appearances in the db
mi_estimation = mi_estimation.sort_values(by="steel_count", ascending=False)

# TODO in sandbox only work with steel so create a version of buildings_import without nan for steel
buildings_import_backup = buildings_import
buildings_import = buildings_import[buildings_import['steel'].notna()]

# most common combination is
c = 0  # (this will become the running index for combinations in mi_df)
combination = mi_estimation.index[c]  # returns a tuple. if a list is needed use list(mi_df.index[0])
# combination = mi_df.index[c:c+11]  # returns a multiindex
# TODO maybe move steel from index to a column when we'll deal with multiple materials

# create a dataframe to keep scores
db_scores = pd.DataFrame(data=0, index=(buildings_import.index), columns=(combination[:-1]))
db_scores[combination[0]][buildings_import['R5.2'].str.match(combination[0])] = 1
db_scores[buildings_import['R5.2'].str.match(combination[0])] = 1
