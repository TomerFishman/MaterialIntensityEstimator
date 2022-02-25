# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:42:14 2020

@author: Tomer Fishman
"""
# %% libraries and load dimensions
# import numpy as np
from os import chdir
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

chdir('C:\\Users\\Tomer\\Dropbox\\-the research\\2020 10 IIASA\\MI_project\\git\\MaterialIntensityEstimator')

dims_structure_import = pd.read_excel("data_input_and_ml_processing\\dims_structure.xlsx", sheet_name="dims_structure")

dims_names = list(dims_structure_import.columns)

# HINT remove unused dimensions
dims_names = dims_names[7:]

dims_list = []
# dims_len = []

for dim_x in dims_names:
    # calculate the number of entities in the dimension
    dim_lastvalidrow = dims_structure_import[dim_x].last_valid_index() + 1
    dims_list += [list(dims_structure_import[dim_x][2:dim_lastvalidrow])]
    # dims_len.append(len(dims_structure_import[dim_x][2:dim_lastvalidrow]))

# HINT removed IN 'informal' because there are simply not enough datapoints for meaningful estimations, consider including later
dims_list[0] = dims_list[0][1:]

# %% load the MI database with const. type ML results from Orange

buildings_import = pd.read_excel("data_input_and_ml_processing\\buildings_v2-const_type_ML.xlsx", sheet_name="Sheet1")

# create new column const_short where U from 'Construction type' is replaced by 'Random Forest'
buildings_import['const_short'] = buildings_import['Random Forest'].where((buildings_import['Construction type'].str.match('U')), buildings_import['Construction type'])

# clean up buildings_import
buildings_import = buildings_import[['id', 'concrete', 'steel', 'wood', 'brick', 'R5_32', 'use_short', 'const_short']]

# optional stuff and usage examples
# # SSP 5 regions
# buildings_import['R5'] = buildings_import['R5_32'].str.split('_').str[0]

# # slice and selection examples. See also https://stackoverflow.com/questions/53927460/select-rows-in-pandas-multiindex-dataframe
# buildings_import.loc[(buildings_import['R5_32'].str.match('OECD_JPN')), ('id', 'steel', 'R5_32')]
# buildings_import.loc[(buildings_import['R5_32'].str.match('OECD_JPN')) & (buildings_import['use_short'].str.match('RM')), ('id', 'steel', 'R5_32', 'use_short')]
# buildings_import.query("R5_32 == 'OECD_JPN' and use_short == 'RM'")

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
pairwise_ks_p_clean.to_excel("db_analysis\\ks_p.xlsx", merge_cells=False)
pairwise_ks_p_clean_long = pairwise_ks_p_clean.stack([0, 1, 2])
sns.heatmap(pairwise_ks_p_clean, cmap="RdYlBu_r", center=0.05, xticklabels=1, yticklabels=1, robust=True)

pairwise_kw_p_clean = pairwise_kw_p.replace(2, np.NAN)
pairwise_kw_p_clean.dropna(how="all", inplace=True)
pairwise_kw_p_clean.dropna(axis='columns', how="all", inplace=True)
pairwise_kw_p_clean.to_excel("db_analysis\\kw_p.xlsx", merge_cells=False)


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

# optional stuff and usage examples
# # note the previous .loc slice and selection examples don't work now. Use query (or xs from the link)
# buildings_import.query("R5_32 == 'OECD_JPN' and use_short == 'RM'")[['id', 'steel']]

# # for sandbox: export to excel for toy model
# buildings_import.reset_index().to_excel("MI_results\\buildings_db_processed.xlsx", sheet_name="sheet1")

# %% create a new dataframe of the counts of unique combinations that exist in the DB
# including unspecifieds
db_combinations = pd.DataFrame(data=None, index=pd.MultiIndex.from_product(dims_list, names=dims_names))

# add columns of counts in the db for each material
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

# replace NANs with zeros for consistency, or keep only those with values
db_combinations_valid = db_combinations.dropna()
# db_combinations = db_combinations.fillna(0)

# optional stuff and usage examples
# # slice db_combinations that are unspecified in either use or const (should be equivalent to db_combinations.query("use_short == 'UN' or use_short == 'RU') because const_short == 'U' should not exist
# db_combinations.query("use_short == 'UN' or use_short == 'RU' or const_short == 'U'")
# # slice db_combinations that are NOT unspecified in either use or const
# db_combinations.query("use_short != 'UN' and use_short != 'RU' and const_short != 'U'")

# # exoort db_combinations
# db_combinations.to_excel("MI_results\\db_combinations.xlsx", sheet_name="sheet1")
# db_combinations.unstack().to_clipboard()

# %% separate buildings_import to individual dataframes by valid combinations

# sort to make pandas faster and with less warnings
buildings_import = buildings_import.sort_index()

# prefiltered as a list only with valid combinations (i.e. existing in buildings_import): [combination tuple, dataframe, [no. of rows in df, counts of each material], expansion score set to 0]
# # of all materials
# db_combinations_data = []
# [db_combinations_data.append([row[0], buildings_import.loc[row[0]], list(buildings_import.loc[row[0]].count()), 0]) for row in db_combinations_valid.itertuples()]

# of only steel
db_combinations_steel = []
[db_combinations_steel.append([row[0], buildings_import.loc[row[0]], int(db_combinations_valid.loc[row[0], 'steel_count']), 0]) for row in db_combinations_valid.itertuples() if db_combinations_valid.loc[row[0], 'steel_count'] > 0]
# optional stuff and usage examples
# # its size in memory is bigger than buildings_import
# buildings_import.memory_usage(deep=True).sum()
# db_mem = 0
# for i in range(len(db_combinations_data)):
#     db_mem += db_combinations_data[i][1].memory_usage(deep=True).sum()

# # find contents in this list
# "ASIA" in db_combinations_data[0][0][2]
# "ASIA_C" in db_combinations_data[0][0][2]
# "ASIA_CB" in db_combinations_data[0][0][2]  # doesn't exist

# # example of filtering
# # filtered = [db_combinations_data[i] for i in range(len(db_combinations_data)) if "ASIA" in db_combinations_data[i][0][2]]
# filtered = [v for i, v in enumerate(db_combinations_data) if "ASIA" in v[0][2]]
# filtered = [v for i, v in enumerate(db_combinations_data) if ('RM' in v[0][0]) and ('C' in v[0][1]) and ('OECD' in v[0][2])]
# # combine the filtered data into one dataframe
# filtered_combined = pd.concat([v[1] for i, v in enumerate(filtered)])

# # HINT  the timing of this list "filter" is dramatically faster than pandas options: 12 µs ± 280 ns vs filtering with pandas query (3.57 ms ± 251 µs with OECD, 1.78 ms ± 141 µs with exact region) or loc (190 µs ± 17 µs but with an exact region)

# # as a dict - excellent for organization and filtering logic but can't have duplicate keys :-(
# db_combinations_data = {}

# # only with valid combinations (i.e. existing in buildings_import)
# for row in db_combinations_valid.itertuples():
#     db_combinations_data[row[0]] = buildings_import.loc[row[0]]

# # with all possible (and relevant) combinations, much slower of course
# for row in db_combinations.itertuples():
#     if buildings_import.index.isin([row[0]]).any():
#         db_combinations_data[row[0]] = buildings_import.loc[row[0]]
#     else:
#         db_combinations_data[row[0]] = pd.DataFrame(data=None)

# # access a combination
# current_combi = ('RM', 'C', 'OECD_EU15')
# db_combinations_data[current_combi]

# # example of filtering
# filtered = {k: v for (k, v) in db_combinations_data.items() if 'ASIA' in k[2]}
# filtered = {k: v for (k, v) in db_combinations_data.items() if ('RM' in k[0]) and ('C' in k[1]) and ('ASIA' in k[2])}
# # and if the filter doesn't exist at all? return an empty dict
# filtered = {k: v for (k, v) in db_combinations_data.items() if 'BLAH' in k[2]}
# # combine the filtered data into one dataframe
# filtered_combined = pd.concat([v for v in filtered.values()])

# %% create a dataframe with all practical (i.e. not unspecifieds) combination options to be filled with data

# remove 'unspecified' entities !!make sure to change the list indexes as needed
dims_list_specified = dims_list[:]
dims_list_specified[0] = [x for x in dims_list_specified[0] if 'U' not in x]
dims_list_specified[1] = [x for x in dims_list_specified[1] if 'U' not in x]

# create multi index dataframe of all possible combinations. This will be the final output of this algorithm
# mi_estimation = pd.DataFrame(data=None, index=pd.MultiIndex.from_product(dims_list_specified, names=dims_names), columns=['steel_db_count', 'wood_db_count', 'brick_db_count', 'concrete_db_count'])
# # mi_estimation.memory_usage(deep=True).sum()

# # add counts from db_combination_counts
# mi_estimation['concrete_db_count'] = db_combinations['concrete_count']
# mi_estimation['steel_db_count'] = db_combinations['steel_count']
# mi_estimation['wood_db_count'] = db_combinations['wood_count']
# mi_estimation['brick_db_count'] = db_combinations['brick_count']

# dict for storing the current selection MIs with their IDs for backup and reference
mi_estimation_steel_data = {}

# only for steel
mi_estimation_steel_stats = pd.DataFrame(data=None, index=pd.MultiIndex.from_product(dims_list_specified, names=dims_names),
                                         columns=['R5', 'db_count', 'db_avg', 'db_sd', 'db_5', 'db_25', 'db_50', 'db_75', 'db_95',
                                                  'expand_count', 'expand_avg', 'expand_sd', 'expand_5', 'expand_25', 'expand_50', 'expand_75', 'expand_95', 'expand_rounds'])  # , 'p1', 'p5', 'p10', 'p20', 'p25', 'p30', 'p40', 'p50', 'p60', 'p70', 'p75', 'p80', 'p90', 'p95', 'p99'
mi_estimation_steel_stats = mi_estimation_steel_stats.reset_index()
mi_estimation_steel_stats['R5'] = mi_estimation_steel_stats['R5_32'].str.split('_').str[0]  # SSP 5 regions
mi_estimation_steel_stats = mi_estimation_steel_stats.set_index(['use_short', 'const_short', 'R5_32'])
mi_estimation_steel_stats['db_count'] = db_combinations['steel_count']
mi_estimation_steel_stats['db_avg'] = buildings_import.groupby(dims_names).steel.mean()
mi_estimation_steel_stats['db_sd'] = buildings_import.groupby(dims_names).steel.std()
mi_estimation_steel_stats['db_5'] = buildings_import.groupby(dims_names).steel.quantile(q=0.05)  # TODO decide which interpolation is best i.e. what does excel do
mi_estimation_steel_stats['db_25'] = buildings_import.groupby(dims_names).steel.quantile(q=0.25)  # ), interpolation='nearest')
mi_estimation_steel_stats['db_50'] = buildings_import.groupby(dims_names).steel.quantile(q=0.5)
mi_estimation_steel_stats['db_75'] = buildings_import.groupby(dims_names).steel.quantile(q=0.75)
mi_estimation_steel_stats['db_95'] = buildings_import.groupby(dims_names).steel.quantile(q=0.95)

# %% selection algorithm

stop_count = 10


def expand_selection(selection, count, condition):
    newselection = [list(v) for v in db_combinations_steel if eval(condition)]
    if newselection:  # pythonic way to check if newselection is not empty
        selection += newselection
        count = 0
        for item in selection:
            item[-1] += 1
            count += item[2] * item[-1]
    return selection, count


# HINT cosmetic: for simple tracking in the sandbox: sort by count of appearances in the db
mi_estimation_steel_stats = mi_estimation_steel_stats.sort_values(by="db_count", ascending=False)

for current_index in mi_estimation_steel_stats.itertuples():  # running index for the current combination in mi_estimation
    current_combi = current_index[0]  # the current combination, a tuple. if a list is needed use list(mi_estimation.index[0])

    current_selection = []
    current_count = 0

    # 1.1 add perfect matches
    if current_count < stop_count:
        current_condition = 'current_combi == v[0]'
        current_selection, current_count = expand_selection(current_selection, current_count, current_condition)
    # 1.2 add similar use types
    if current_count < stop_count:
        if current_combi[0] == 'NR':
            current_condition = "('UN' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2] == v[0][2])"  # TODO this reselects the perfect combination! see RS T OECD_USA
            current_selection, current_count = expand_selection(current_selection, current_count, current_condition)
        else:  # i.e. if current_combi[0][0] == 'R':
            current_condition = "('RU' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2] == v[0][2])"  # TODO consider whether to first add UN (currently in the IF below) and only then RU?
            current_selection, current_count = expand_selection(current_selection, current_count, current_condition)
            if current_count < stop_count:  # TODO this adds UN. consider whether to add the opposite R type e.g. if we're at RS then add RM and vice versa
                current_condition = "('UN' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2] == v[0][2])"
                current_selection, current_count = expand_selection(current_selection, current_count, current_condition)

    # 2.1 repeat for bigger 5-level region, not including the current 32-level region
    if current_count < stop_count:
        current_condition = "(current_combi[0] == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2][:3] in v[0][2]) and (current_combi[2] != v[0][2])"
        current_selection, current_count = expand_selection(current_selection, current_count, current_condition)
    # 2.2 add similar use types
    if current_count < stop_count:
        if current_combi[0] == 'NR':
            current_condition = "('UN' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2][:3] in v[0][2]) and (current_combi[2] != v[0][2])"
            current_selection, current_count = expand_selection(current_selection, current_count, current_condition)
        else:  # make sure to keep it conformed to 1.2 TODO decisions!
            current_condition = "('RU' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2][:3] in v[0][2]) and (current_combi[2] != v[0][2])"
            current_selection, current_count = expand_selection(current_selection, current_count, current_condition)
            if current_count < stop_count:
                current_condition = "('UN' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2][:3] in v[0][2]) and (current_combi[2] != v[0][2])"
                current_selection, current_count = expand_selection(current_selection, current_count, current_condition)

    # 3.1 repeat for all regions
    # TODO consider if stop_count or if stop_count-x to not expand to the entire world if we're already close to stop_count
    if current_count < stop_count:
        current_condition = "(current_combi[0] == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2][:3] not in v[0][2])"
        current_selection, current_count = expand_selection(current_selection, current_count, current_condition)
    # 3.2 add similar use types
    if current_count < stop_count:
        if current_combi[0] == 'NR':
            current_condition = "('UN' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2][:3] not in v[0][2])"
            current_selection, current_count = expand_selection(current_selection, current_count, current_condition)
        else:  # make sure to keep it conformed to 1.2 TODO decisions!
            current_condition = "('RU' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2][:3] not in v[0][2])"
            current_selection, current_count = expand_selection(current_selection, current_count, current_condition)
            if current_count < stop_count:
                current_condition = "('UN' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2][:3] not in v[0][2])"
                current_selection, current_count = expand_selection(current_selection, current_count, current_condition)

    # When done: concatenate current_selection into one dataframe, including repetition of selections from previous expansion rounds i.e. v[3] in the second for loop
    try:  # TODO temporary solution for empty combinations
        current_selection_combined = pd.concat([v[1] for v in current_selection for i in range(v[3])], copy=True).loc[:, ['id', 'steel']].dropna()
        # fill results into mi_estimation_stats 'expanded_count', 'avg', 'sd', 'p5', 'p25', 'p50', 'p75', 'p95', 'expansion_rounds'
        mi_estimation_steel_stats.loc[current_combi, 'expand_count'] = current_count
        mi_estimation_steel_stats.loc[current_combi, 'expand_avg'] = current_selection_combined['steel'].mean()  # can also use iloc for generalization, a bit slower: current_selection_combined.iloc[:, 1].mean()
        mi_estimation_steel_stats.loc[current_combi, 'expand_sd'] = current_selection_combined['steel'].std()
        mi_estimation_steel_stats.loc[current_combi, 'expand_5'] = np.quantile(current_selection_combined['steel'], q=0.05)  # faster than pandas's current_selection_combined['steel'].quantile(q=0.05)
        mi_estimation_steel_stats.loc[current_combi, 'expand_25'] = np.quantile(current_selection_combined['steel'], q=0.25)
        mi_estimation_steel_stats.loc[current_combi, 'expand_50'] = np.quantile(current_selection_combined['steel'], q=0.50)
        mi_estimation_steel_stats.loc[current_combi, 'expand_75'] = np.quantile(current_selection_combined['steel'], q=0.75)
        mi_estimation_steel_stats.loc[current_combi, 'expand_95'] = np.quantile(current_selection_combined['steel'], q=0.95)
        mi_estimation_steel_stats.loc[current_combi, 'expand_rounds'] = current_selection[0][3]
    except ValueError:
        current_selection_combined = pd.DataFrame()

    # save current_selection_combined for backup and reference
    mi_estimation_steel_data[current_combi] = current_selection_combined.copy()

# HINT cosmetic: resort by index
mi_estimation_steel_stats.sort_index(inplace=True)


# %% analysis

# TODO list:
# make it output a pdf with all violin plots
# for the paper we'll select some examples of how the algorithm behaves with lots of perfect matches, the cse of 10-15, and the case of 0-1 datapoints
# we should create a user interface that shows violin plots and percentiles before users download the data
# TODO 24/2:
# compare results with previous studies: marinova, heeren, etc.
# consider adding glass aluminum copper, etc as in marinova, even though we won't get much between-region variation, it's still an improvement over marinova's simple avg.
# maybe for these extra materials we don't need the regional differentiation, so we'd use a simplified algorithm on 5 ssp or just global but still differentiate use and const.

mi_estimation_steel_stats.to_clipboard()

mi_estimation_steel_stats['db_count'] = mi_estimation_steel_stats['db_count'].fillna(0)
sns.relplot(x="expand_count", y="expand_rounds", size="db_count", sizes=(15, 200), hue="use_short", data=mi_estimation_steel_stats)
sns.relplot(x="expand_count", y="expand_rounds", size="db_count", sizes=(15, 200), hue="const_short", data=mi_estimation_steel_stats)
sns.relplot(x="expand_count", y="expand_rounds", size="db_count", sizes=(15, 200), hue="R5", data=mi_estimation_steel_stats)
sns.relplot(x="expand_count", y="db_count", hue="expand_rounds", data=mi_estimation_steel_stats)
sns.relplot(x="expand_count", y="db_count", hue="R5", data=mi_estimation_steel_stats)
sns.relplot(x="db_count", y="expand_count", hue="R5", size="expand_rounds", sizes=(10, 300), alpha=0.5, data=mi_estimation_steel_stats)
sns.relplot(x="expand_rounds", y="db_count", hue="R5", size="expand_count", sizes=(1, 400), alpha=0.3, data=mi_estimation_steel_stats)
sns.swarmplot(x="expand_rounds", y="expand_count", hue="R5", data=mi_estimation_steel_stats)


# %% before-after comparisons

comparisons = {}
for row in mi_estimation_steel_stats.itertuples():
    if buildings_import.index.isin([row[0]]).any():
        comparisons[row[0]] = buildings_import.loc[row[0], ['id', 'steel']]
    else:
        comparisons[row[0]] = pd.DataFrame(data=None)
    comparisons[row[0]] = comparisons[row[0]].assign(MIs="before")
    comparisons[row[0]] = pd.concat([comparisons[row[0]], mi_estimation_steel_data[row[0]]])
    comparisons[row[0]]['MIs'] = comparisons[row[0]]['MIs'].fillna("after")
    comparisons[row[0]].dropna(inplace=True)
    comparisons[row[0]] = comparisons[row[0]].assign(combination=str(row[0]))
    comparisons[row[0]] = comparisons[row[0]].assign(use=row[0][0])
    comparisons[row[0]] = comparisons[row[0]].assign(construction=row[0][1])
    comparisons[row[0]] = comparisons[row[0]].assign(region=row[0][2])


# viol = sns.violinplot(y="combination", x="steel", data=comparisons[('RS', 'C', 'OECD_EU15')], hue="MIs", hue_order=("after", "before"), split=True, inner="stick", scale="count", bw=0.1, linewidth=1)
# viol.set_xlim(left=0, right=500)

# swar = sns.swarmplot(y="combination", x="steel", data=comparisons[('RS', 'C', 'OECD_EU15')], hue="MIs", hue_order=("before", "after"))
# stri = sns.stripplot(y="combination", x="steel", data=comparisons[('RS', 'C', 'OECD_EU15')], hue="MIs", hue_order=("before", "after"))

# def compareviolin_portrait(u, c, r, axx, axy):
#     yaxiscut = 200
#     violcombi = (u, c, r)
#     viol = sns.violinplot(x="combination", y="steel", data=comparisons[violcombi], hue="MIs", hue_order=("before", "after"), split=True,
#                           inner="quartile", scale="count", cut=0, bw=.1, linewidth=1, ax=axs[axx, axy])
#     viol.set_title(violcombi)
#     viol.set_ylim(bottom=0, top=yaxiscut)
#     viol.set_xlabel("")
#     viol.set_xticks([])
#     viol.legend([], [], frameon=False)
#     viol.annotate('After (n = ' + str(mi_estimation_steel_stats.loc[violcombi, "expand_count"]) + ')\n'
#                   'p5 = ' + str(round(mi_estimation_steel_stats.loc[violcombi, "expand_5"], 2)) + '\n'
#                   'p25 = ' + str(round(mi_estimation_steel_stats.loc[violcombi, "expand_25"], 2)) + '\n'
#                   'median = ' + str(round(mi_estimation_steel_stats.loc[violcombi, "expand_50"], 2)) + '\n'
#                   'p75 = ' + str(round(mi_estimation_steel_stats.loc[violcombi, "expand_75"], 2)) + '\n'
#                   'p95 = ' + str(round(mi_estimation_steel_stats.loc[violcombi, "expand_95"], 2)) + '\n'
#                   'outliers > ' + str(yaxiscut) + ' = ' + str(comparisons[violcombi].loc[(comparisons[violcombi]['steel'] > yaxiscut)].count()[1]) + '\n',
#                   # xy=(0.49, comparisons[violcombi]['steel'].max()), va='top', ha='right', fontsize='small', color='C1')
#                   xy=(0.49, 0), va='bottom', ha='right', fontsize='small', color='C1')
#     if mi_estimation_steel_stats.loc[violcombi, "db_count"] > 0:
#         viol.annotate('Before (n = ' + str(round(mi_estimation_steel_stats.loc[violcombi, "db_count"])) + ')\n'
#                       'p5 = ' + str(round(mi_estimation_steel_stats.loc[violcombi, "db_25"], 2)) + '\n'
#                       'p25 = ' + str(round(mi_estimation_steel_stats.loc[violcombi, "db_25"], 2)) + '\n'
#                       'median = ' + str(round(mi_estimation_steel_stats.loc[violcombi, "db_50"], 2)) + '\n'
#                       'q75 = ' + str(round(mi_estimation_steel_stats.loc[violcombi, "db_75"], 2)) + '\n'
#                       'q95 = ' + str(round(mi_estimation_steel_stats.loc[violcombi, "db_95"], 2)) + '\n',
#                       # 'outliers > ' + str(yaxiscut) + ' = ' + str(comparisons[violcombi].loc[(comparisons[violcombi]['steel'] > yaxiscut)].count()[1]) + '\n',
#                       # xy=(-0.49, comparisons[violcombi]['steel'].max()), va='top', fontsize='small', color='C0')
#                       xy=(-0.49, 0), va='bottom', fontsize='small', color='C0')
#     else:
#         viol.annotate('Before (n = 0)', va='bottom', fontsize='small', color='C0',
#                       # xy=(-0.49, comparisons[violcombi]['steel'].max()))
#                       xy=(-0.49, 0))
#     return viol


# # portrait mode
# with PdfPages('stop_at_' + str(stop_count) + '.pdf') as pdf:
#     for region in dims_list_specified[2]:
#         fig, axs = plt.subplots(4, 3, figsize=(20, 30))
#         compareviolin_portrait('NR', 'C', region, 0, 0)
#         compareviolin_portrait('RS', 'C', region, 0, 1)
#         compareviolin_portrait('RM', 'C', region, 0, 2)
#         compareviolin_portrait('NR', 'M', region, 1, 0)
#         compareviolin_portrait('RS', 'M', region, 1, 1)
#         compareviolin_portrait('RM', 'M', region, 1, 2)
#         compareviolin_portrait('NR', 'S', region, 2, 0)
#         compareviolin_portrait('RS', 'S', region, 2, 1)
#         compareviolin_portrait('RM', 'S', region, 2, 2)
#         compareviolin_portrait('NR', 'T', region, 3, 0)
#         compareviolin_portrait('RS', 'T', region, 3, 1)
#         compareviolin_portrait('RM', 'T', region, 3, 2)
#         pdf.savefig(fig)


def compareviolin_landscape(u, c, r, axx, axy):
    xaxiscut = 200
    violcombi = (u, c, r)
    viol = sns.violinplot(y="combination", x="steel", data=comparisons[violcombi], hue="MIs", hue_order=("after", "before"), split=True,
                          inner="quartile", scale="count", cut=0, bw=.1, linewidth=1, ax=axs[axx, axy])
    viol.set_title(violcombi)
    viol.set_xlim(left=0, right=xaxiscut)
    viol.set_ylabel("")
    viol.set_yticks([])
    viol.legend([], [], frameon=False)
    viol.annotate('After (n = ' + str(mi_estimation_steel_stats.loc[violcombi, "expand_count"]) + ')\n'
                  'p5 = ' + str(round(mi_estimation_steel_stats.loc[violcombi, "expand_5"], 2)) + '\n'
                  'p25 = ' + str(round(mi_estimation_steel_stats.loc[violcombi, "expand_25"], 2)) + '\n'
                  'median = ' + str(round(mi_estimation_steel_stats.loc[violcombi, "expand_50"], 2)) + '\n'
                  'p75 = ' + str(round(mi_estimation_steel_stats.loc[violcombi, "expand_75"], 2)) + '\n'
                  'p95 = ' + str(round(mi_estimation_steel_stats.loc[violcombi, "expand_95"], 2)) + '\n'
                  'outliers > ' + str(xaxiscut) + ' = ' + str(comparisons[('NR', 'C', 'ASIA_TWN')].query('(steel > @xaxiscut) and (MIs == "after")').count()[1]),
                  xy=(xaxiscut - 1, -0.49), va='top', ha='right', color='C0')
    if mi_estimation_steel_stats.loc[violcombi, "db_count"] > 0:
        viol.annotate('Before (n = ' + str(round(mi_estimation_steel_stats.loc[violcombi, "db_count"])) + ')\n'
                      'p5 = ' + str(round(mi_estimation_steel_stats.loc[violcombi, "db_25"], 2)) + '\n'
                      'p25 = ' + str(round(mi_estimation_steel_stats.loc[violcombi, "db_25"], 2)) + '\n'
                      'median = ' + str(round(mi_estimation_steel_stats.loc[violcombi, "db_50"], 2)) + '\n'
                      'q75 = ' + str(round(mi_estimation_steel_stats.loc[violcombi, "db_75"], 2)) + '\n'
                      'q95 = ' + str(round(mi_estimation_steel_stats.loc[violcombi, "db_95"], 2)) + '\n'
                      'outliers > ' + str(xaxiscut) + ' = ' + str(comparisons[('NR', 'C', 'ASIA_TWN')].query('(steel > @xaxiscut) and (MIs == "before")').count()[1]),
                      # xy=(-0.49, comparisons[violcombi]['steel'].max()), va='top', fontsize='small', color='C0')
                      xy=(xaxiscut - 1, 0.49), va='bottom', ha='right', color='C1')
    else:
        viol.annotate('Before (n = 0)', va='bottom', ha='right', color='C1',
                      # xy=(-0.49, comparisons[violcombi]['steel'].max()))
                      xy=(xaxiscut - 1, 0.49))
    return viol


# landscape mode. Less informative because it's hard to compare similar const. types down the columns
with PdfPages('MI_results\\stop_at_' + str(stop_count) + '_l.pdf') as pdf:
    for region in dims_list_specified[2]:
        fig, axs = plt.subplots(3, 4, figsize=(30, 20))
        compareviolin_landscape('NR', 'C', region, 0, 0)
        compareviolin_landscape('NR', 'M', region, 0, 1)
        compareviolin_landscape('NR', 'S', region, 0, 2)
        compareviolin_landscape('NR', 'T', region, 0, 3)
        compareviolin_landscape('RM', 'C', region, 1, 0)
        compareviolin_landscape('RM', 'M', region, 1, 1)
        compareviolin_landscape('RM', 'S', region, 1, 2)
        compareviolin_landscape('RM', 'T', region, 1, 3)
        compareviolin_landscape('RS', 'C', region, 2, 0)
        compareviolin_landscape('RS', 'M', region, 2, 1)
        compareviolin_landscape('RS', 'S', region, 2, 2)
        compareviolin_landscape('RS', 'T', region, 2, 3)
        pdf.savefig(fig)

mi_estimation_steel_stats.reset_index().to_excel('MI_results\\stop_at_' + str(stop_count) + '.xlsx', sheet_name=('stop_at_' + str(stop_count)))
