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
import numpy as np
from scipy import stats

chdir('C:\\Users\\Tomer\\Dropbox\\-the research\\2020 10 IIASA\\MI_project\\git\\MaterialIntensityEstimator')

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

# TODO removed IN 'informal' because there are simply not enough datapoints for meaningful estimations, consider including later
dims_list[0] = dims_list[0][1:]

# %% load the MI database with const. type ML results from Orange

buildings_import = pd.read_excel("buildings_v2-const_type_ML.xlsx", sheet_name="Sheet1")

# create new column const_short where U from 'Construction type' is replaced by 'Random Forest'
buildings_import['const_short'] = buildings_import['Random Forest'].where((buildings_import['Construction type'].str.match('U')), buildings_import['Construction type'])

# clean up buildings_import
buildings_import = buildings_import[['id', 'concrete', 'steel', 'wood', 'brick', 'R5_32', 'use_short', 'const_short']]

# SSP 5 regions
buildings_import['R5'] = buildings_import['R5_32'].str.split('_').str[0]

# slice and selection examples. See also https://stackoverflow.com/questions/53927460/select-rows-in-pandas-multiindex-dataframe
buildings_import.loc[(buildings_import['R5_32'].str.match('OECD_JPN')), ('id', 'steel', 'R5_32')]
buildings_import.loc[(buildings_import['R5_32'].str.match('OECD_JPN')) & (buildings_import['use_short'].str.match('RM')), ('id', 'steel', 'R5_32', 'use_short')]
buildings_import.query("R5_32 == 'OECD_JPN' and use_short == 'RM'")

# %% database plots


# violin plots https://seaborn.pydata.org/generated/seaborn.violinplot.html
sns.violinplot(x="concrete", y="const_short", data=buildings_import, cut=0, linewidth=1).legend_.remove()
sns.violinplot(x="concrete", y="const_short", hue="use_short", data=buildings_import, cut=0, linewidth=1, scale="width")
sns.violinplot(x="const_short", y="concrete", hue="use_short", data=buildings_import, cut=0, linewidth=.5, scale="width", bw=.1, height=2, aspect=1)
sns.catplot(x="const_short", y="concrete", hue="use_short", kind="violin", data=buildings_import, cut=0, linewidth=1, scale="width", bw=.15, height=8, aspect=1.8)
sns.catplot(x="use_short", y="concrete", hue="const_short", kind="violin", data=buildings_import, cut=0, linewidth=1, scale="width", bw=.15, height=8, aspect=1.8)
sns.catplot(x="use_short", y="concrete", row="const_short", kind="violin", data=buildings_import, cut=0, linewidth=1, scale="width", bw=.2, height=3, aspect=6)

sns.catplot(x="use_short", y="concrete", col="const_short", kind="violin", data=buildings_import, cut=0, linewidth=1, scale="width", bw=.2, height=3, aspect=1.2)
sns.catplot(x="use_short", y="steel", col="const_short", kind="violin", data=buildings_import.query('steel < 450'), cut=0, linewidth=1, scale="width", bw=.2, height=3, aspect=1.2)
sns.catplot(x="use_short", y="wood", col="const_short", kind="violin", data=buildings_import.query('wood < 300'), cut=0, linewidth=1, scale="width", bw=.2, height=3, aspect=1.2)
sns.catplot(x="use_short", y="brick", col="const_short", kind="violin", data=buildings_import, cut=0, linewidth=1, scale="width", bw=.2, height=3, aspect=1.2)

sns.catplot(x="const_short", y="concrete", col="use_short", kind="violin", data=buildings_import, cut=0, linewidth=1, scale="width", bw=.2, height=3, aspect=1.2)
sns.catplot(x="const_short", y="steel", col="use_short", kind="violin", data=buildings_import.query('steel < 450'), cut=0, linewidth=1, scale="width", bw=.2, height=3, aspect=1.2)
sns.catplot(x="const_short", y="wood", col="use_short", kind="violin", data=buildings_import.query('wood < 300'), cut=0, linewidth=1, scale="width", bw=.2, height=3, aspect=1.2)
sns.catplot(x="const_short", y="brick", col="use_short", kind="violin", data=buildings_import, cut=0, linewidth=1, scale="width", bw=.2, height=3, aspect=1.2)

sns.catplot(x="use_short", y="concrete", col="const_short", row="R5", kind="violin", data=buildings_import, cut=0, linewidth=1, scale="width", bw=.2, height=3, aspect=1.2)
sns.catplot(x="use_short", y="steel", col="const_short", row="R5", kind="violin", data=buildings_import.query('steel < 450'), cut=0, linewidth=1, scale="width", bw=.2, height=3, aspect=1.2)
sns.catplot(x="const_short", y="steel", col="use_short", row="R5", kind="violin", data=buildings_import.query('steel < 450'), cut=0, linewidth=1, scale="width", bw=.2, height=3, aspect=1.2)

# bivariate distribution plots https://seaborn.pydata.org/tutorial/distributions.html#visualizing-bivariate-distributions
kdebw = .6
scattersize = 120
scatteralpha = .6
sns.jointplot(data=buildings_import, x="concrete", y="steel", hue="const_short", linewidth=0, height=8,
              xlim=(0, round(max(buildings_import['concrete']), ndigits=-2)),
              ylim=(0, round(max(buildings_import['steel']), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))
sns.jointplot(data=buildings_import, x="concrete", y="wood", hue="const_short", linewidth=0, height=8,
              xlim=(0, round(max(buildings_import['concrete']), ndigits=-2)),
              ylim=(0, round(max(buildings_import['wood']), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))
sns.jointplot(data=buildings_import, x="concrete", y="brick", hue="const_short", linewidth=0, height=8,
              xlim=(0, round(max(buildings_import['concrete']), ndigits=-2)),
              ylim=(0, round(max(buildings_import['brick']), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))

sns.jointplot(data=buildings_import, x="steel", y="brick", hue="const_short", linewidth=0, height=8,
              xlim=(0, round(max(buildings_import['steel']), ndigits=-2)),
              ylim=(0, round(max(buildings_import['brick']), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))
sns.jointplot(data=buildings_import, x="steel", y="wood", hue="const_short", linewidth=0, height=8,
              xlim=(0, round(max(buildings_import['steel']), ndigits=-2)),
              ylim=(0, round(max(buildings_import['wood']), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))

sns.jointplot(data=buildings_import, x="wood", y="brick", hue="const_short", linewidth=0, height=8,
              xlim=(0, round(max(buildings_import['wood']), ndigits=-2)),
              ylim=(0, round(max(buildings_import['brick']), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))
# without outlyiers
sns.jointplot(data=buildings_import, x="concrete", y="steel", hue="const_short", linewidth=0, height=8,
              xlim=(0, round(max(buildings_import['concrete']), ndigits=-2)),
              ylim=(0, 500),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))
sns.jointplot(data=buildings_import, x="concrete", y="wood", hue="const_short", linewidth=0, height=8,
              xlim=(0, round(max(buildings_import['concrete']), ndigits=-2)),
              ylim=(0, 300),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))
sns.jointplot(data=buildings_import, x="concrete", y="brick", hue="const_short", linewidth=0, height=8,
              xlim=(0, round(max(buildings_import['concrete']), ndigits=-2)),
              ylim=(0, round(max(buildings_import['brick']), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))

sns.jointplot(data=buildings_import, x="steel", y="brick", hue="const_short", linewidth=0, height=8,
              xlim=(0, 500),
              ylim=(0, round(max(buildings_import['brick']), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))
sns.jointplot(data=buildings_import, x="steel", y="wood", hue="const_short", linewidth=0, height=8,
              xlim=(0, 500),
              ylim=(0, 300),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))

sns.jointplot(data=buildings_import, x="wood", y="brick", hue="const_short", linewidth=0, height=8,
              xlim=(0, 300),
              ylim=(0, round(max(buildings_import['brick']), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))

# use types, without outlyiers
sns.jointplot(data=buildings_import, x="concrete", y="steel", hue="use_short", linewidth=0, height=8,
              xlim=(0, round(max(buildings_import['concrete']), ndigits=-2)),
              ylim=(0, 500),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))
sns.jointplot(data=buildings_import, x="concrete", y="wood", hue="use_short", linewidth=0, height=8,
              xlim=(0, round(max(buildings_import['concrete']), ndigits=-2)),
              ylim=(0, 300),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))
sns.jointplot(data=buildings_import, x="concrete", y="brick", hue="use_short", linewidth=0, height=8,
              xlim=(0, round(max(buildings_import['concrete']), ndigits=-2)),
              ylim=(0, round(max(buildings_import['brick']), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))

sns.jointplot(data=buildings_import, x="steel", y="brick", hue="use_short", linewidth=0, height=8,
              xlim=(0, 500),
              ylim=(0, round(max(buildings_import['brick']), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))
sns.jointplot(data=buildings_import, x="steel", y="wood", hue="use_short", linewidth=0, height=8,
              xlim=(0, 500),
              ylim=(0, 300),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))

sns.jointplot(data=buildings_import, x="wood", y="brick", hue="use_short", linewidth=0, height=8,
              xlim=(0, 300),
              ylim=(0, round(max(buildings_import['brick']), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))

# ssp regions 32 to check that none of the regions govern these results, without outlyiers
sns.jointplot(data=buildings_import, x="concrete", y="steel", hue="R5_32", palette=("Set2"), linewidth=0, height=8,
              xlim=(0, round(max(buildings_import['concrete']), ndigits=-2)),
              ylim=(0, 500),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize / 2)))
sns.jointplot(data=buildings_import, x="concrete", y="wood", hue="R5_32", palette=("Set2"), linewidth=0, height=8,
              xlim=(0, round(max(buildings_import['concrete']), ndigits=-2)),
              ylim=(0, 300),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize / 2)))
sns.jointplot(data=buildings_import, x="concrete", y="brick", hue="R5_32", palette=("Set2"), linewidth=0, height=8,
              xlim=(0, round(max(buildings_import['concrete']), ndigits=-2)),
              ylim=(0, round(max(buildings_import['brick']), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize / 2)))

sns.jointplot(data=buildings_import, x="steel", y="brick", hue="R5_32", palette=("Set2"), linewidth=0, height=8,
              xlim=(0, 500),
              ylim=(0, round(max(buildings_import['brick']), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize / 2)))
sns.jointplot(data=buildings_import, x="steel", y="wood", hue="R5_32", palette=("Set2"), linewidth=0, height=8,
              xlim=(0, 500),
              ylim=(0, 300),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize / 2)))

sns.jointplot(data=buildings_import, x="wood", y="brick", hue="R5_32", palette=("Set2"), linewidth=0, height=8,
              xlim=(0, 300),
              ylim=(0, round(max(buildings_import['brick']), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize / 2)))

# ssp 5 regions to check that none of the regions govern these results, without outlyiers
sns.jointplot(data=buildings_import, x="concrete", y="steel", hue="R5", linewidth=0, height=8,
              xlim=(0, round(max(buildings_import['concrete']), ndigits=-2)),
              ylim=(0, 500),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize / 2)))
sns.jointplot(data=buildings_import, x="concrete", y="wood", hue="R5", linewidth=0, height=8,
              xlim=(0, round(max(buildings_import['concrete']), ndigits=-2)),
              ylim=(0, 300),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize / 2)))
sns.jointplot(data=buildings_import, x="concrete", y="brick", hue="R5", linewidth=0, height=8,
              xlim=(0, round(max(buildings_import['concrete']), ndigits=-2)),
              ylim=(0, round(max(buildings_import['brick']), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize / 2)))

sns.jointplot(data=buildings_import, x="steel", y="brick", hue="R5", linewidth=0, height=8,
              xlim=(0, 500),
              ylim=(0, round(max(buildings_import['brick']), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize / 2)))
sns.jointplot(data=buildings_import, x="steel", y="wood", hue="R5", linewidth=0, height=8,
              xlim=(0, 500),
              ylim=(0, 300),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize / 2)))

sns.jointplot(data=buildings_import, x="wood", y="brick", hue="R5", linewidth=0, height=8,
              xlim=(0, 300),
              ylim=(0, round(max(buildings_import['brick']), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize / 2)))


# %% Kolmogorov-Smirnov Tests

group_a = buildings_import.loc[(buildings_import['use_short'].str.match('RM')) & (buildings_import['const_short'].str.match('C')) & (buildings_import['R5_32'].str.match('OECD_JPN')), ('steel')]
group_b = buildings_import.loc[(buildings_import['use_short'].str.match('RS')) & (buildings_import['const_short'].str.match('T')) & (buildings_import['R5_32'].str.match('OECD_JPN')), ('steel')]

group_a = buildings_import.loc[(buildings_import['use_short'].str.match('RU')) & (buildings_import['const_short'].str.match('C')) & (buildings_import['R5_32'].str.match('LAM_LAM-M')), ('steel')]
group_b = buildings_import.loc[(buildings_import['use_short'].str.match('RM')) & (buildings_import['const_short'].str.match('C')) & (buildings_import['R5_32'].str.match('LAM_LAM-M')), ('steel')]

stats.ks_2samp(group_a, group_b)  # if p < 0.05 we reject the null hypothesis. Hence the two sample datasets do not come from the same distribution.
stats.kruskal(group_a, group_b)  # if p < 0.05 we reject the null hypothesis. Hence the two sample datasets have different medians.
stats.epps_singleton_2samp(group_a, group_b, t=(0.4, 0.8))
stats.anderson_ksamp([group_a, group_b])  # works only when n>=2. If p < 0.05 we reject the null hypothesis. Hence the two sets do not come from the same distribution.

stats.ks_2samp(buildings_import.query("R5_32 == 'OECD_JPN' and use_short == 'RM'")['steel'], buildings_import.query("R5_32 == 'OECD_JPN' and use_short == 'RS'")['steel'])

# # approach that goes column by column
# db_combination_index = pd.MultiIndex.from_product(dims_list, names=dims_names)
# i = 0
# current_column = pd.DataFrame(data=None, index=db_combination_index[i:], columns=db_combination_index[i:i + 1])
# group_a = buildings_import.loc[(buildings_import['use_short'].str.match(current_column.index[:][0][0])) & (buildings_import['const_short'].str.match(current_column.index[:][0][1]) & (buildings_import['R5_32'].str.match(current_column.index[:][0][2]))), ('steel')]

# current_column.iloc[:] = stats.ks_2samp(group_a, buildings_import.loc[(buildings_import['use_short'].str.match(current_column.reset_index().iloc[:,0])) & (buildings_import['const_short'].str.match(current_column.reset_index().iloc[:,1]) & (buildings_import['R5_32'].str.match(current_column.reset_index().iloc[:,2]))), ('steel')])

# approach that moves cell by cell with 2 for loops - takes 20+ minutes

db_combination_index = pd.MultiIndex.from_product(dims_list, names=dims_names)
pairwise = pd.DataFrame(data=None, index=db_combination_index, columns=db_combination_index)
pairwise_ks_p = pairwise.copy()
pairwise_ks_s = pairwise.copy()
pairwise_kw_p = pairwise.copy()
pairwise_kw_s = pairwise.copy()
pairwise_ands_p = pairwise.copy()

# i = 0
# j = 0
for i in range(0, len(db_combination_index)):
    for j in range(i + 1, len(db_combination_index)):
        indexname = pairwise.iloc[[i, j]].index
        group_a = buildings_import.loc[(buildings_import['use_short'].str.match(indexname[0][0])) & (buildings_import['const_short'].str.match(indexname[0][1]) & (buildings_import['R5_32'].str.match(indexname[0][2]))), ('steel')]
        group_b = buildings_import.loc[(buildings_import['use_short'].str.match(indexname[1][0])) & (buildings_import['const_short'].str.match(indexname[1][1]) & (buildings_import['R5_32'].str.match(indexname[1][2]))), ('steel')]
#        if not(group_a.empty | group_b.empty):            if not(group_a.empty | group_b.empty):
        if (len(group_a) > 1 | len(group_b) > 1):
            ks_result = stats.ks_2samp(group_a, group_b)
            kp_result = stats.kruskal(group_a, group_b)
            ands_result = stats.anderson_ksamp([group_a, group_b])
            pairwise_ks_p.iloc[j, i] = ks_result[1]
            pairwise_kw_p.iloc[j, i] = kp_result[1]
            pairwise_ands_p.iloc[j, i] = ands_result[-1]
        else:
            pairwise_ks_p.iloc[j, i] = 2
            pairwise_kw_p.iloc[j, i] = 2
            pairwise_ands_p.iloc[j, i] = 2


pairwise_ks_p_clean = pairwise_ks_p.replace(2, np.NAN)
pairwise_ks_p_clean.dropna(how="all", inplace=True)
pairwise_ks_p_clean.dropna(axis='columns', how="all", inplace=True)
pairwise_ks_p_clean.to_excel("ks_p.xlsx", merge_cells=False)
pairwise_ks_p_clean_long = pairwise_ks_p_clean.stack([0, 1, 2])
sns.heatmap(pairwise_ks_p_clean, cmap="RdYlBu_r", center=0.05, xticklabels=1, yticklabels=1, robust=True)

pairwise_kw_p_clean = pairwise_kw_p.replace(2, np.NAN)
pairwise_kw_p_clean.dropna(how="all", inplace=True)
pairwise_kw_p_clean.dropna(axis='columns', how="all", inplace=True)
pairwise_kw_p_clean.to_excel("kw_p.xlsx", merge_cells=False)


# long form, seems to take much longer
# pairwise2 = pd.DataFrame(data=None, index=db_combination_index, columns=db_combination_index)
# pairwiselong = pairwise2.stack([0, 1, 2], dropna=False)
# for i in range(0, len(pairwiselong)):
#     group_a = buildings_import.loc[(buildings_import['use_short'].str.match(pairwiselong.index[i][0])) & (buildings_import['const_short'].str.match(pairwiselong.index[i][1]) & (buildings_import['R5_32'].str.match(pairwiselong.index[i][2]))), ('steel')]
#     group_b = buildings_import.loc[(buildings_import['use_short'].str.match(pairwiselong.index[i][3])) & (buildings_import['const_short'].str.match(pairwiselong.index[i][4]) & (buildings_import['R5_32'].str.match(pairwiselong.index[i][5]))), ('steel')]
#     if not(group_a.empty | group_b.empty):
#         ks_result = stats.ks_2samp(group_a, group_b)
#         pairwiselong.iloc[i] = ks_result[1]
#     else:
#         pairwiselong.iloc[i] = 2

# pairwiselong = pairwise2.stack([0, 1, 2], dropna=False)
# pairwiselong.index.names = ['use_short_a', 'const_short_a', 'R5_32_a', 'use_short_b', 'const_short_b', 'R5_32_b']
# pairwiselong = pairwiselong.reset_index()
# pairwiselong.rename(columns={0: 'ks'}, inplace=True)
# pairwiselong['ks'] = stats.ks_2samp(
#     buildings_import.loc[(buildings_import['use_short'].str.match(pairwiselong['use_short_a'])) & (buildings_import['const_short'].str.match(pairwiselong['const_short_a']) & (buildings_import['R5_32'].str.match(pairwiselong['R5_32_a']))), ('steel')].
#     buildings_import.loc[(buildings_import['use_short'].str.match(pairwiselong['use_short_b'])) & (buildings_import['const_short'].str.match(pairwiselong['const_short_b']) & (buildings_import['R5_32'].str.match(pairwiselong['R5_32_b']))), ('steel')])[1]

# %% final setups of the database data

# set up the same multiindex as the other dataframes
buildings_import.set_index(dims_names, inplace=True)
# note the previous .loc slice and selection examples don't work now. Use query (or xs from the link)
buildings_import.query("R5_32 == 'OECD_JPN' and use_short == 'RM'")[['id', 'steel']]

# TODO for sandbox: export to excel for toy model
buildings_import.reset_index().to_excel("buildings_import.xlsx", sheet_name="sheet1")

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

# TODO for simple tracking in the sandbox: sort by count of appearances in the db
mi_estimation = mi_estimation.sort_values(by="steel_count", ascending=False)

# TODO in sandbox only work with steel so create a version of buildings_import without nan for steel
buildings_import_backup = buildings_import
buildings_import = buildings_import[buildings_import['steel'].notna()]

c = 0  # running index for the current combination in mi_df
current_combi = mi_estimation.index[c]  # the current combination. Returns a tuple. if a list is needed use list(mi_df.index[0])
# current_combi = mi_df.index[c:c+11]  # returns a multiindex
# TODO maybe move steel from index to a column when we'll deal with multiple materials

# create a helper dataframe to keep scores
db_scores = pd.DataFrame(data=0, index=(buildings_import.index), columns=(current_combi[:-1]))
db_scores[current_combi[0]][buildings_import['R5.2'].str.match(current_combi[0])] = 1
db_scores[buildings_import['R5.2'].str.match(current_combi[0])] = 1
