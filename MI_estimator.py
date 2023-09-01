# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:42:14 2020

@author: Tomer Fishman

The buildings material intensity estimator
"""
# %% load libraries and load dimensions

from os import chdir
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import date

export_to_excel = False

today = date.today().strftime("%Y%m%d")

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

# HINT removed IN 'informal' because there are simply not enough datapoints for meaningful estimations, consider including later
dims_list[0] = dims_list[0][1:]

materials = ['concrete', 'brick', 'wood', 'steel', 'glass', 'plastics', 'aluminum', 'copper']

# %% load the MI database with structure type ML results from Orange

buildings_import_full = pd.read_excel("data_input_and_ml_processing\\buildings_v2-structure_type_ML.xlsx", sheet_name="Sheet1")

# update column 'structure' where U from the original 'structure' is replaced by 'Random Forest'
buildings_import_full['structure'] = buildings_import_full['Random Forest'].where((buildings_import_full['structure'].str.match('U')), buildings_import_full['structure'])

# combine all plastics types
buildings_import_full['plastics'] = buildings_import_full[['plastics', 'PVC', 'polystyrene']].sum(axis=1, min_count=1)

# clean up buildings_import
buildings_import = buildings_import_full[['id'] + materials + dims_names]

# %% EDA: pairwise database plots (optional, run to create figure 1)

# buildings_import.reset_index(inplace=True)

# pairwise plots of materials
lim = -10
# figure 1a
s = sns.pairplot(data=buildings_import_full.query('~(`structure` == "U")'), corner=True,
                 hue="structure", hue_order=['C', 'M', 'T', 'S'], palette="tab10",
                 x_vars=materials[0:4], y_vars=materials[0:4],
                 plot_kws=(dict(alpha=0.3, s=20, linewidth=0)),
                 # diag_kind="hist", diag_kws=dict(element="step", linewidth=0.5))
                 diag_kws=dict(alpha=0.3, bw_adjust=1.5, linewidth=0, common_norm=False),
                 height=2, aspect=1)
# sns.move_legend(s, loc='upper center', ncol=2, bbox_to_anchor=(.45, .95))
s.map_lower(sns.kdeplot, alpha=1, levels=2, bw_adjust=2)
s.axes[3, 0].set_xlim(left=lim, right=4000)
s.axes[3, 1].set_xlim(left=lim, right=3000)
s.axes[3, 2].set_xlim(left=lim, right=300)
s.axes[3, 3].set_xlim(left=lim, right=450)
s.axes[1, 0].set_ylim(bottom=lim, top=3000)
s.axes[2, 0].set_ylim(bottom=lim, top=300)
s.axes[3, 0].set_ylim(bottom=lim, top=450)
s._legend.remove()
# s.axes[1, 0].legend(loc='upper left', bbox_to_anchor=(1, 1.5))

# figure 1b
f = sns.pairplot(data=buildings_import_full.query('~(function == "UN") & ~(function == "IN")'), corner=True,
                 hue="function", hue_order=['NR', 'RS', 'RM', 'RU'], palette="Set1_r",
                 x_vars=materials[0:4], y_vars=materials[0:4],
                 plot_kws=(dict(alpha=0.5, s=20, linewidth=0)),
                 # diag_kind="hist", diag_kws=dict(element="step", linewidth=0.5))
                 diag_kws=dict(alpha=0.3, bw_adjust=1.5, linewidth=0, common_norm=False),
                 height=2, aspect=1)
# sns.move_legend(f, 'upper center', bbox_to_anchor=(.55, .65))
f.map_lower(sns.kdeplot, alpha=1, levels=2, bw_adjust=2)
f.axes[3, 0].set_xlim(left=lim, right=4000)
f.axes[3, 1].set_xlim(left=lim, right=3000)
f.axes[3, 2].set_xlim(left=lim, right=300)
f.axes[3, 3].set_xlim(left=lim, right=450)
f.axes[1, 0].set_ylim(bottom=lim, top=3000)
f.axes[2, 0].set_ylim(bottom=lim, top=300)
f.axes[3, 0].set_ylim(bottom=lim, top=450)
f._legend.remove()


stats.ks_2samp(buildings_import_full.query('(`structure` == "T")')['concrete'], buildings_import_full.query('(`structure` == "S")')['concrete'])  # if p < 0.05 we reject the null hypothesis. Hence the two sample datasets do not come from the same distribution.
stats.kruskal(buildings_import_full.query('(`structure` == "T")')['concrete'], buildings_import_full.query('(`structure` == "S")')['concrete'])  # if p < 0.05 we reject the null hypothesis. Hence the two sample datasets have different medians.
stats.anderson_ksamp(buildings_import_full.query('(`structure` == "T")')['concrete'], buildings_import_full.query('(`structure` == "S")')['concrete'])  # works only when n>=2. If p < 0.05 we reject the null hypothesis. Hence the two sets do not come from the same distribution.

# %% EDA: violin and more pairwise database plots (optional, run only for extra visualizations)
# SSP 5 regions
buildings_import['R5'] = buildings_import['R5_32'].str.split('_').str[0]

# violin plots https://seaborn.pydata.org/generated/seaborn.violinplot.html
sns.violinplot(x="concrete", y="structure", data=buildings_import, cut=0, linewidth=1)
sns.violinplot(x="concrete", y="structure", hue="function", data=buildings_import, cut=0, linewidth=1, scale="width")
sns.violinplot(x="structure", y="concrete", hue="function", data=buildings_import, cut=0, linewidth=.5, scale="width", bw=.1, height=2, aspect=1)
sns.catplot(x="structure", y="concrete", hue="function", kind="violin", data=buildings_import, cut=0, linewidth=1, scale="width", bw=.15, height=8, aspect=1.8)
sns.catplot(x="function", y="concrete", hue="structure", kind="violin", data=buildings_import, cut=0, linewidth=1, scale="width", bw=.15, height=8, aspect=1.8)
sns.catplot(x="function", y="concrete", row="structure", kind="violin", data=buildings_import, cut=0, linewidth=1, scale="width", bw=.2, height=3, aspect=6)

sns.catplot(x="function", y="concrete", col="structure", kind="violin", data=buildings_import, cut=0, linewidth=1, scale="width", bw=.2, height=3, aspect=1.2)
sns.catplot(x="function", y="steel", col="structure", kind="violin", data=buildings_import.query('steel < 450'), cut=0, linewidth=1, scale="width", bw=.2, height=3, aspect=1.2)
sns.catplot(x="function", y="wood", col="structure", kind="violin", data=buildings_import.query('wood < 300'), cut=0, linewidth=1, scale="width", bw=.2, height=3, aspect=1.2)
sns.catplot(x="function", y="brick", col="structure", kind="violin", data=buildings_import, cut=0, linewidth=1, scale="width", bw=.2, height=3, aspect=1.2)

sns.catplot(x="structure", y="concrete", col="function", kind="violin", data=buildings_import, cut=0, linewidth=1, scale="width", bw=.2, height=3, aspect=1.2)
sns.catplot(x="structure", y="steel", col="function", kind="violin", data=buildings_import.query('steel < 450'), cut=0, linewidth=1, scale="width", bw=.2, height=3, aspect=1.2)
sns.catplot(x="structure", y="wood", col="function", kind="violin", data=buildings_import.query('wood < 300'), cut=0, linewidth=1, scale="width", bw=.2, height=3, aspect=1.2)
sns.catplot(x="structure", y="brick", col="function", kind="violin", data=buildings_import, cut=0, linewidth=1, scale="width", bw=.2, height=3, aspect=1.2)

sns.catplot(x="function", y="concrete", col="structure", row="R5", kind="violin", data=buildings_import, cut=0, linewidth=1, scale="width", bw=.2, height=3, aspect=1.2)
sns.catplot(x="function", y="steel", col="structure", row="R5", kind="violin", data=buildings_import.query('steel < 450'), cut=0, linewidth=1, scale="width", bw=.2, height=3, aspect=1.2)
sns.catplot(x="structure", y="glass", col="function", row="R5", kind="violin", data=buildings_import.query('steel < 450'), cut=0, linewidth=1, scale="width", bw=.2, height=3, aspect=1.2)

sns.catplot(x="structure", y="glass", col="function", row="R5", kind="strip", data=buildings_import, linewidth=1, height=3, aspect=1.2)
sns.catplot(x="structure", y="aluminum", col="function", row="R5", kind="strip", data=buildings_import, linewidth=1, height=3, aspect=1.2)
sns.catplot(x="structure", y="copper", col="function", row="R5", kind="strip", data=buildings_import, linewidth=1, height=3, aspect=1.2)

# pairwise plots of materials
sns.pairplot(data=buildings_import.query('~(wood > 300) & ~(steel > 500)'), corner=False,
             hue="structure", hue_order=['C', 'M', 'T', 'S'],
             x_vars=materials[0:4], y_vars=materials[0:4],
             kind="kde", plot_kws=(dict(alpha=.3, levels=2, bw_adjust=1, fill=True)),
             diag_kind="hist", diag_kws=dict(element="step"))

sns.pairplot(data=buildings_import.query('~(wood > 300) & ~(steel > 500)'), corner=True,
             hue="structure", hue_order=['C', 'M', 'T', 'S'],
             x_vars=materials[0:4], y_vars=materials[0:4],
             kind="kde", plot_kws=(dict(alpha=1, levels=2, bw_adjust=1, fill=False)),
             diag_kind="hist", diag_kws=dict(element="step"))

sns.pairplot(data=buildings_import.query('~(wood > 300) & ~(steel > 500)'), corner=True,
             hue="function", palette="Set1",
             x_vars=materials[0:4], y_vars=materials[0:4],
             kind="kde", plot_kws=(dict(alpha=1, levels=2, bw_adjust=1, fill=False)),
             diag_kind="hist", diag_kws=dict(element="step"))

# bivariate distribution plots https://seaborn.pydata.org/tutorial/distributions.html#visualizing-bivariate-distributions
kdebw = .6
scattersize = 120
scatteralpha = .6
sns.jointplot(data=buildings_import, x="concrete", y="steel", hue="structure", linewidth=0, height=8,
              xlim=(0, round(buildings_import['concrete'].max(), ndigits=-2)),
              ylim=(0, round(buildings_import['steel'].max(), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))
sns.jointplot(data=buildings_import, x="concrete", y="wood", hue="structure", linewidth=0, height=8,
              xlim=(0, round(buildings_import['concrete'].max(), ndigits=-2)),
              ylim=(0, round(buildings_import['wood'].max(), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))
sns.jointplot(data=buildings_import, x="concrete", y="brick", hue="structure", linewidth=0, height=8,
              xlim=(0, round(buildings_import['concrete'].max(), ndigits=-2)),
              ylim=(0, round(buildings_import['brick'].max(), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))

sns.jointplot(data=buildings_import, x="steel", y="brick", hue="structure", linewidth=0, height=8,
              xlim=(0, round(buildings_import['steel'].max(), ndigits=-2)),
              ylim=(0, round(buildings_import['brick'].max(), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))
sns.jointplot(data=buildings_import, x="steel", y="wood", hue="structure", linewidth=0, height=8,
              xlim=(0, round(buildings_import['steel'].max(), ndigits=-2)),
              ylim=(0, round(buildings_import['wood'].max(), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))

sns.jointplot(data=buildings_import, x="wood", y="brick", hue="structure", linewidth=0, height=8,
              xlim=(0, round(buildings_import['wood'].max(), ndigits=-2)),
              ylim=(0, round(buildings_import['brick'].max(), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))
# without outlyiers
sns.jointplot(data=buildings_import, x="concrete", y="steel", hue="structure", linewidth=0, height=8,
              xlim=(0, round(buildings_import['concrete'].max(), ndigits=-2)),
              ylim=(0, 500),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))
sns.jointplot(data=buildings_import, x="concrete", y="wood", hue="structure", linewidth=0, height=8,
              xlim=(0, round(buildings_import['concrete'].max(), ndigits=-2)),
              ylim=(0, 300),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))
sns.jointplot(data=buildings_import, x="concrete", y="brick", hue="structure", linewidth=0, height=8,
              xlim=(0, round(buildings_import['concrete'].max(), ndigits=-2)),
              ylim=(0, round(buildings_import['brick'].max(), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))

sns.jointplot(data=buildings_import, x="steel", y="brick", hue="structure", linewidth=0, height=8,
              xlim=(0, 500),
              ylim=(0, round(buildings_import['brick'].max(), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))
sns.jointplot(data=buildings_import, x="steel", y="wood", hue="structure", linewidth=0, height=8,
              xlim=(0, 500),
              ylim=(0, 300),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))

sns.jointplot(data=buildings_import, x="wood", y="brick", hue="structure", linewidth=0, height=8,
              xlim=(0, 300),
              ylim=(0, round(buildings_import['brick'].max(), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))

# function types, without outlyiers
sns.jointplot(data=buildings_import, x="concrete", y="steel", hue="function", linewidth=0, height=8,
              xlim=(0, round(buildings_import['concrete'].max(), ndigits=-2)),
              ylim=(0, 500),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))
sns.jointplot(data=buildings_import, x="concrete", y="wood", hue="function", linewidth=0, height=8,
              xlim=(0, round(buildings_import['concrete'].max(), ndigits=-2)),
              ylim=(0, 300),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))
sns.jointplot(data=buildings_import, x="concrete", y="brick", hue="function", linewidth=0, height=8,
              xlim=(0, round(buildings_import['concrete'].max(), ndigits=-2)),
              ylim=(0, round(buildings_import['brick'].max(), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))

sns.jointplot(data=buildings_import, x="steel", y="brick", hue="function", linewidth=0, height=8,
              xlim=(0, 500),
              ylim=(0, round(buildings_import['brick'].max(), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))
sns.jointplot(data=buildings_import, x="steel", y="wood", hue="function", linewidth=0, height=8,
              xlim=(0, 500),
              ylim=(0, 300),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))

sns.jointplot(data=buildings_import, x="wood", y="brick", hue="function", linewidth=0, height=8,
              xlim=(0, 300),
              ylim=(0, round(buildings_import['brick'].max(), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize)))

# ssp regions 32 to check that none of the regions govern these results, without outlyiers
sns.jointplot(data=buildings_import, x="concrete", y="steel", hue="R5_32", palette=("Set2"), linewidth=0, height=8,
              xlim=(0, round(buildings_import['concrete'].max(), ndigits=-2)),
              ylim=(0, 500),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize / 2)))
sns.jointplot(data=buildings_import, x="concrete", y="wood", hue="R5_32", palette=("Set2"), linewidth=0, height=8,
              xlim=(0, round(buildings_import['concrete'].max(), ndigits=-2)),
              ylim=(0, 300),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize / 2)))
sns.jointplot(data=buildings_import, x="concrete", y="brick", hue="R5_32", palette=("Set2"), linewidth=0, height=8,
              xlim=(0, round(buildings_import['concrete'].max(), ndigits=-2)),
              ylim=(0, round(buildings_import['brick'].max(), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize / 2)))

sns.jointplot(data=buildings_import, x="steel", y="brick", hue="R5_32", palette=("Set2"), linewidth=0, height=8,
              xlim=(0, 500),
              ylim=(0, round(buildings_import['brick'].max(), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize / 2)))
sns.jointplot(data=buildings_import, x="steel", y="wood", hue="R5_32", palette=("Set2"), linewidth=0, height=8,
              xlim=(0, 500),
              ylim=(0, 300),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize / 2)))

sns.jointplot(data=buildings_import, x="wood", y="brick", hue="R5_32", palette=("Set2"), linewidth=0, height=8,
              xlim=(0, 300),
              ylim=(0, round(buildings_import['brick'].max(), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize / 2)))

# ssp 5 regions to check that none of the regions govern these results, without outlyiers
sns.jointplot(data=buildings_import, x="concrete", y="steel", hue="R5", linewidth=0, height=8,
              xlim=(0, round(buildings_import['concrete'].max(), ndigits=-2)),
              ylim=(0, 500),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize / 2)))
sns.jointplot(data=buildings_import, x="concrete", y="wood", hue="R5", linewidth=0, height=8,
              xlim=(0, round(buildings_import['concrete'].max(), ndigits=-2)),
              ylim=(0, 300),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize / 2)))
sns.jointplot(data=buildings_import, x="concrete", y="brick", hue="R5", linewidth=0, height=8,
              xlim=(0, round(buildings_import['concrete'].max(), ndigits=-2)),
              ylim=(0, round(buildings_import['brick'].max(), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize / 2)))

sns.jointplot(data=buildings_import, x="steel", y="brick", hue="R5", linewidth=0, height=8,
              xlim=(0, 500),
              ylim=(0, round(buildings_import['brick'].max(), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize / 2)))
sns.jointplot(data=buildings_import, x="steel", y="wood", hue="R5", linewidth=0, height=8,
              xlim=(0, 500),
              ylim=(0, 300),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize / 2)))

sns.jointplot(data=buildings_import, x="wood", y="brick", hue="R5", linewidth=0, height=8,
              xlim=(0, 300),
              ylim=(0, round(buildings_import['brick'].max(), ndigits=-2)),
              marginal_kws=dict(bw_adjust=kdebw, cut=0), joint_kws=(dict(alpha=scatteralpha, s=scattersize / 2)))

# %% EDA: minor materials: visualize if there's a clear difference in categories (optional, run only for extra visualizations)
sns.boxplot(data=buildings_import.reset_index(), x="copper", y="structure")
sns.boxplot(data=buildings_import.reset_index(), x="copper", y="function")
sns.boxplot(data=buildings_import.reset_index(), x="copper", y="R5_32")

sns.boxplot(data=buildings_import.reset_index(), x="aluminum", y="structure")
sns.boxplot(data=buildings_import.reset_index(), x="aluminum", y="function")
sns.boxplot(data=buildings_import.reset_index(), x="aluminum", y="R5_32")

sns.boxplot(data=buildings_import.reset_index(), x="plastics", y="structure")
sns.boxplot(data=buildings_import.reset_index(), x="plastics", y="function")
sns.boxplot(data=buildings_import.reset_index(), x="plastics", y="R5_32")

# %% final setups of the database data

# HINT remove IN 'informal' because there are simply not enough datapoints for meaningful estimations, consider including later
buildings_import = buildings_import[buildings_import.function != 'IN']
# set up the same multiindex as the other dataframes
buildings_import.set_index(dims_names, inplace=True)
# sort to make pandas faster and with less warnings
buildings_import.sort_index(inplace=True)

# export buildings_import to Excel file
if export_to_excel:
    buildings_import.to_excel("db_analysis\\Buildings_import_" + today + ".xlsx", merge_cells=False)


# %% create a new dataframe of the counts of unique combinations that exist in the DB
# including unspecifieds

db_combinations_stats = [pd.DataFrame(data=None, index=pd.MultiIndex.from_product(dims_list, names=dims_names)),
                         buildings_import[materials].groupby(dims_names).count(),
                         buildings_import[materials].groupby(dims_names).mean(),
                         buildings_import[materials].groupby(dims_names).std()[materials],
                         buildings_import[materials].groupby(dims_names).quantile(q=0.05),
                         buildings_import[materials].groupby(dims_names).quantile(q=0.25),
                         buildings_import[materials].groupby(dims_names).quantile(q=0.50),
                         buildings_import[materials].groupby(dims_names).quantile(q=0.75),
                         buildings_import[materials].groupby(dims_names).quantile(q=0.95)]

db_combinations_stats = pd.concat(db_combinations_stats, axis=1, keys=['', 'count', 'avg', 'sd', 'p5', 'p25', 'p50', 'p75', 'p95'])

db_combinations_stats[('count', 'concrete')]
db_combinations_stats.loc[('NR', 'C', 'ASIA_CHN'), ('count', 'concrete')]
db_combinations_stats.loc[('NR', 'C', 'ASIA_CHN'), :]
db_combinations_stats.loc[:, db_combinations_stats.columns.isin(['concrete'], level=1)]

# dataframe that keeps only rows with values
db_combinations_stats_valid = db_combinations_stats.dropna(how='all')

# export db_combinations_stats
if export_to_excel:
    db_combinations_stats_valid.to_excel("MI_results\\db_combinations_stats_" + today + ".xlsx", merge_cells=False)
    # db_combinations_stats.unstack().to_clipboard()

# %% EDA: Nonparametric statistical tests (optional)

if export_to_excel:
    pairwise_writer = pd.ExcelWriter("db_analysis\\nonparametric_tests_" + today + ".xlsx", engine='openpyxl')

for current_material in materials:
    pairwise_ks_p = pd.DataFrame(data=None, index=db_combinations_stats_valid.index, columns=db_combinations_stats_valid.index)
    pairwise_kw_p = pairwise_ks_p.copy()
    pairwise_ands_p = pairwise_ks_p.copy()
    # pairwise_ks_s = pairwise_ks_p.copy()
    # pairwise_kw_s = pairwise_ks_p.copy()

    pairwise_col_count = 1
    for pairwise_ind_combi in pairwise_ks_p.index:
        pairwise_ind_data = buildings_import.loc[pairwise_ind_combi, current_material]
        for pairwise_col_combi in pairwise_ks_p.columns[pairwise_col_count:]:
            pairwise_col_data = buildings_import.loc[pairwise_col_combi, current_material]
            if (len(pairwise_ind_data) > 2 and len(pairwise_col_data) > 2):
                pairwise_ks_p.loc[pairwise_ind_combi, pairwise_col_combi] = stats.ks_2samp(pairwise_ind_data, pairwise_col_data)[1]  # if p < 0.05 we reject the null hypothesis. Hence the two sample datasets do not come from the same distribution.
                pairwise_kw_p.loc[pairwise_ind_combi, pairwise_col_combi] = stats.kruskal(pairwise_ind_data, pairwise_col_data)[1]  # if p < 0.05 we reject the null hypothesis. Hence the two sample datasets have different medians.
                # pairwise_ands_p.loc[pairwise_ind_combi, pairwise_col_combi] = stats.anderson_ksamp([pairwise_ind_data, pairwise_col_data])[-1]  # works only when n>=2. If p < 0.05 we reject the null hypothesis. Hence the two sets do not come from the same distribution.
        pairwise_col_count = pairwise_col_count + 1

    pairwise_ks_p.dropna(how="all", inplace=True)
    pairwise_ks_p.dropna(axis='columns', how="all", inplace=True)

    pairwise_kw_p.dropna(how="all", inplace=True)
    pairwise_kw_p.dropna(axis='columns', how="all", inplace=True)

    # pairwise_ands_p.dropna(how="all", inplace=True)
    # pairwise_ands_p.dropna(axis='columns', how="all", inplace=True)

    pairwise_ks_p.T.to_excel(pairwise_writer, sheet_name=current_material + "_ks", merge_cells=False)
    pairwise_kw_p.T.to_excel(pairwise_writer, sheet_name=current_material + "_kw", merge_cells=False)
    # pairwise_ands_p.T.to_excel(pairwise_writer, sheet_name=current_material + "_ands", merge_cells=False)
    # # export pairwise matrices as individual Excel files
    # pairwise_ks_p.to_excel("db_analysis\\" + current_material + "_pairwise_ks_p.xlsx", merge_cells=False)
    # pairwise_kw_p.to_excel("db_analysis\\" + current_material + "_pairwise_kw_p.xlsx", merge_cells=False)
    # pairwise_ands_p.to_excel("db_analysis\\" + current_material + "_pairwise_ands_p.xlsx", merge_cells=False)

    # summary p results
    pairwise_long = pd.concat([pairwise_ks_p.stack([0, 1, 2]), pairwise_kw_p.stack([0, 1, 2]), pairwise_ands_p.stack([0, 1, 2])], axis=1)
    pairwise_long.index.names = ['function_1', 'structure_1', 'R5_32_1', 'function_2', 'structure_2', 'R5_32_2']
    pairwise_long.columns = ['Kolmogorov_Smirnov_test', 'Kruskal_Wallis_H_test', 'k_sample_Anderson_Darling_test']
    pairwise_long.sort_index(inplace=True)
    pairwise_long['tests_over_0.05_ie_same_dists'] = 0
    pairwise_long.loc[pairwise_long['Kolmogorov_Smirnov_test'] > 0.05, 'tests_over_0.05_ie_same_dists'] += 1
    pairwise_long.loc[pairwise_long['Kruskal_Wallis_H_test'] > 0.05, 'tests_over_0.05_ie_same_dists'] += 1
    # pairwise_long.loc[pairwise_long['k_sample_Anderson_Darling_test'] > 0.05, 'tests_over_0.05_ie_same_dists'] += 1
    # pairwise_long.to_excel(pairwise_writer, sheet_name=current_material + "_summary", merge_cells=False)
    # pairwise_long.to_excel("db_analysis\\" + current_material + "_nonparametric_tests.xlsx", merge_cells=False)  # export as individual Excel file

if export_to_excel:
    pairwise_writer.close()

# visualize cumulative distributions (what the Kolmogorov-Smirnov test does)
# pairwise:
sns.ecdfplot(x="brick", hue='function',
             data=pd.concat([buildings_import.loc[('NR', 'M', 'OECD_EU15')],
                             buildings_import.loc[('RM', 'T', 'OECD_EU15')]]).reset_index())
sns.histplot(x="brick", hue='function', element="step", stat="percent", common_norm=False,
             data=pd.concat([buildings_import.loc[('NR', 'T', 'ASIA_CHN')],
                             buildings_import.loc[('UN', 'T', 'ASIA_CHN')]]).reset_index())
# multiple categories:
sns.ecdfplot(data=buildings_import.loc[('RS', 'C')].reset_index(), x="concrete", hue='R5_32')
sns.ecdfplot(data=buildings_import.reset_index().query("structure == 'C' and R5_32 == 'ASIA_CHN'"), x="concrete", hue='function')
# compare:
sns.ecdfplot(data=buildings_import.reset_index().query("structure == 'C' and R5_32 == 'OECD_EU15'"), x="wood", hue='function')
sns.ecdfplot(data=buildings_import.reset_index().query("structure == 'S' and R5_32 == 'OECD_EU15'"), x="wood", hue='function')
sns.ecdfplot(data=buildings_import.reset_index().query("structure == 'T' and R5_32 == 'OECD_EU15'"), x="wood", hue='function')
sns.ecdfplot(data=buildings_import.reset_index().query("structure == 'W' and R5_32 == 'OECD_EU15'"), x="wood", hue='function')
# with:
sns.ecdfplot(data=buildings_import.reset_index().query("function == 'RS' and R5_32 == 'OECD_EU15'"), x="wood", hue='structure')
sns.ecdfplot(data=buildings_import.reset_index().query("function == 'RM' and R5_32 == 'OECD_EU15'"), x="wood", hue='structure')
sns.ecdfplot(data=buildings_import.reset_index().query("function == 'RU' and R5_32 == 'OECD_EU15'"), x="wood", hue='structure')
sns.ecdfplot(data=buildings_import.reset_index().query("function == 'NR' and R5_32 == 'OECD_EU15'"), x="wood", hue='structure')
sns.ecdfplot(data=buildings_import.reset_index().query("function == 'UN' and R5_32 == 'OECD_EU15'"), x="wood", hue='structure')

# %% separate buildings_import to individual dataframes by valid combinations

# for each material in this dict, prefiltered as a list only with valid combinations (i.e. existing in buildings_import): [combination tuple, dataframe, [no. of rows in df, counts of each material], increment score set to 0]
# it's a list and not a dict in case the iterative incremental pooling algorithm needs to duplicate list items. A dict can't have duplicate keys.
db_combinations_data = {}
for current_material in materials:
    db_combinations_data[current_material] = []
    [db_combinations_data[current_material].append([row[0], buildings_import.loc[row[0]], int(db_combinations_stats_valid.loc[row[0], ('count', current_material)]), 0]) for row in db_combinations_stats_valid.itertuples() if db_combinations_stats_valid.loc[row[0], ('count', current_material)] > 0]


# %% create a dataframe with all practical (i.e. not unspecifieds) combination options to be filled with data

# remove 'unspecified' entities !!make sure to change the list indexes as needed
dims_list_specified = dims_list[:]
dims_list_specified[0] = [x for x in dims_list_specified[0] if 'U' not in x]
dims_list_specified[1] = [x for x in dims_list_specified[1] if 'U' not in x]


# dict for storing the current pool MIs with their IDs for backup and reference
mi_estimation_data = {}
mi_estimation_stats = {}
for current_material in materials:
    mi_estimation_data[current_material] = {}
    mi_estimation_stats[current_material] = pd.DataFrame(data=None, index=pd.MultiIndex.from_product(dims_list_specified, names=dims_names),
                                                         columns=['R5', 'raw_HF_db_count', 'increment_iterations', 'incremented_count',
                                                                  'p_0', 'p_5', 'p_25', 'p_50', 'p_75', 'p_95', 'p_100'])
    mi_estimation_stats[current_material] = mi_estimation_stats[current_material].reset_index()
    mi_estimation_stats[current_material]['R5'] = mi_estimation_stats[current_material]['R5_32'].str.split('_').str[0]  # SSP 5 regions
    mi_estimation_stats[current_material] = mi_estimation_stats[current_material].set_index(['function', 'structure', 'R5_32'])
    mi_estimation_stats[current_material]['raw_HF_db_count'] = db_combinations_stats[('count', current_material)]
    mi_estimation_stats[current_material]['raw_HF_db_count'].fillna(0, inplace=True)
    # mi_estimation_stats[current_material]['db_avg'] = db_combinations_stats[('avg', current_material)]
    # mi_estimation_stats[current_material]['db_sd'] = db_combinations_stats[('sd', current_material)]
    # mi_estimation_stats[current_material]['db_5'] = db_combinations_stats[('p5', current_material)]
    # mi_estimation_stats[current_material]['db_25'] = db_combinations_stats[('p25', current_material)]
    # mi_estimation_stats[current_material]['db_50'] = db_combinations_stats[('p50', current_material)]
    # mi_estimation_stats[current_material]['db_75'] = db_combinations_stats[('p75', current_material)]
    # mi_estimation_stats[current_material]['db_95'] = db_combinations_stats[('p95', current_material)]

# %% calculate scores for materials' coverage
# to decide whether to use the incremental pooling algorithm or global statistics


def gini(x):  # https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python
    x = np.asarray(x)
    sorted_x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(sorted_x, dtype=float)
    return round((n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n, 3)


db_material_scores = pd.DataFrame(index=materials)
db_material_scores['count'] = db_combinations_stats_valid['count'].sum()
db_material_scores['combi_coverage'] = db_combinations_stats_valid['avg'].count() / db_combinations_stats_valid['count'].count()
db_material_scores['gini'] = [gini(db_combinations_stats_valid['count'][current_material]) for current_material in db_material_scores.index]
db_material_scores['score'] = 0
db_material_scores.loc[db_material_scores['count'] > len(buildings_import) * 0.15, "score"] += 1  # covers at least 15% of the db
db_material_scores.loc[db_material_scores['combi_coverage'] > 0.5, "score"] += 1  # covers at least 50% of the existing combinations in the db
db_material_scores.loc[db_material_scores['gini'] < 0.8, "score"] += 1  # datapoints don't all come from very few combinations

# %% pooling algorithm

stop_count = 30
export_to_excel = True


def expand_pool(selection, count, condition, material):
    newselection = [list(v) for v in db_combinations_data[material] if eval(condition)]
    if newselection:  # pythonic way to check if newselection is not empty
        selection += newselection
        count = 0
        for item in selection:
            item[3] += 1  # counter for how many iterations this selection was in an increment
            count += item[2] * item[3]  # count how many datapoints are in selection
    return selection, count


if export_to_excel:
    mi_estimation_writer = pd.ExcelWriter('MI_results\\MI_ranges_' + today + ".xlsx", engine='openpyxl')

# pooling algorithm for materials with coverage score >= 2
for current_material in db_material_scores.query('score >= 2').index:
    mi_estimation_stats[current_material] = mi_estimation_stats[current_material].sort_values(by="raw_HF_db_count", ascending=False)

    for current_combi in mi_estimation_stats[current_material].index:  # running index for the current combination in mi_estimation
        print(current_material + str(current_combi))
        current_pool = []
        current_count = 0

        # 1.1 add perfect matches
        current_condition = 'current_combi == v[0]'
        current_pool, current_count = expand_pool(current_pool, current_count, current_condition, current_material)

        # 1.2 add similar structure types
        if current_count < stop_count:
            if current_combi[0] == 'NR':
                current_condition = "('UN' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2] == v[0][2])"
                current_pool, current_count = expand_pool(current_pool, current_count, current_condition, current_material)
            else:  # i.e. if current_combi[0][0] == 'R':
                current_condition = "('RU' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2] == v[0][2])"  # TODO consider whether to first add UN (currently in the IF below) and only then RU?
                current_pool, current_count = expand_pool(current_pool, current_count, current_condition, current_material)
                if current_count < stop_count:  # TODO this adds UN. consider whether to add the opposite R type e.g. if we're at RS then add RM and vice versa
                    current_condition = "('UN' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2] == v[0][2])"
                    current_pool, current_count = expand_pool(current_pool, current_count, current_condition, current_material)

        # 2.1 repeat for bigger 5-level region, not including the current 32-level region
        if current_count < stop_count:
            current_condition = "(current_combi[0] == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2][:3] in v[0][2]) and (current_combi[2] != v[0][2])"
            current_pool, current_count = expand_pool(current_pool, current_count, current_condition, current_material)
        # 2.2 add similar function types
        if current_count < stop_count:
            if current_combi[0] == 'NR':
                current_condition = "('UN' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2][:3] in v[0][2]) and (current_combi[2] != v[0][2])"
                current_pool, current_count = expand_pool(current_pool, current_count, current_condition, current_material)
            else:  # make sure to keep it conformed to 1.2 TODO decisions!
                current_condition = "('RU' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2][:3] in v[0][2]) and (current_combi[2] != v[0][2])"
                current_pool, current_count = expand_pool(current_pool, current_count, current_condition, current_material)
                if current_count < stop_count:
                    current_condition = "('UN' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2][:3] in v[0][2]) and (current_combi[2] != v[0][2])"
                    current_pool, current_count = expand_pool(current_pool, current_count, current_condition, current_material)

        # 3.1 repeat for all regions
        # TODO consider if stop_count or if stop_count-x to not expand to the entire world if we're already close to stop_count
        if current_count < stop_count:
            current_condition = "(current_combi[0] == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2][:3] not in v[0][2])"
            current_pool, current_count = expand_pool(current_pool, current_count, current_condition, current_material)
        # 3.2 add similar function types
        if current_count < stop_count:
            if current_combi[0] == 'NR':
                current_condition = "('UN' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2][:3] not in v[0][2])"
                current_pool, current_count = expand_pool(current_pool, current_count, current_condition, current_material)
            else:  # make sure to keep it conformed to 1.2 TODO decisions!
                current_condition = "('RU' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2][:3] not in v[0][2])"
                current_pool, current_count = expand_pool(current_pool, current_count, current_condition, current_material)
                if current_count < stop_count:
                    current_condition = "('UN' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2][:3] not in v[0][2])"
                    current_pool, current_count = expand_pool(current_pool, current_count, current_condition, current_material)

        # When done: concatenate current_pool into one dataframe, including repetition of selections from previous increment iterations i.e. v[3] in the second for loop
        current_pool_combined = pd.concat([v[1] for v in current_pool for i in range(v[3])], copy=True).loc[:, ['id', current_material]].dropna()
        current_pool_combined['increment_iterations'] = current_pool_combined.groupby('id').cumcount()
        current_pool_combined['increment_iterations'] = current_pool_combined['increment_iterations'].max() - current_pool_combined['increment_iterations']
        if current_combi not in current_pool_combined.index:
            current_pool_combined['increment_iterations'] += 1
        # fill results into mi_estimation_stats 'incremented_count', 'avg', 'sd', 'p5', 'p25', 'p50', 'p75', 'p95', 'increment_iterations'
        mi_estimation_stats[current_material].loc[current_combi, 'incremented_count'] = current_count
        mi_estimation_stats[current_material].loc[current_combi, 'p_0'] = np.quantile(current_pool_combined[current_material], q=0.00)
        mi_estimation_stats[current_material].loc[current_combi, 'p_5'] = np.quantile(current_pool_combined[current_material], q=0.05)  # faster than pandas's current_pool_combined['steel'].quantile(q=0.05)
        mi_estimation_stats[current_material].loc[current_combi, 'p_25'] = np.quantile(current_pool_combined[current_material], q=0.25)
        mi_estimation_stats[current_material].loc[current_combi, 'p_50'] = np.quantile(current_pool_combined[current_material], q=0.50)
        mi_estimation_stats[current_material].loc[current_combi, 'p_75'] = np.quantile(current_pool_combined[current_material], q=0.75)
        mi_estimation_stats[current_material].loc[current_combi, 'p_95'] = np.quantile(current_pool_combined[current_material], q=0.95)
        mi_estimation_stats[current_material].loc[current_combi, 'p_100'] = np.quantile(current_pool_combined[current_material], q=1.00)
        mi_estimation_stats[current_material].loc[current_combi, 'increment_iterations'] = current_pool_combined['increment_iterations'].max()

        # save current_pool_combined for backup and reference
        mi_estimation_data[current_material][current_combi] = current_pool_combined.copy()

    # re-sort by index
    mi_estimation_stats[current_material].sort_index(inplace=True)

    # export as sheet
    if export_to_excel:
        mi_estimation_stats[current_material].reset_index().to_excel(mi_estimation_writer, sheet_name=current_material)

# use global statistics for materials with coverage score < 2
print('global statistics for materials with score < 2')
for current_material in db_material_scores.query('score < 2').index:  # bulk edit all combinations with the global statistics, to avoid cycling through all combinations unnecessarily
    current_pool_combined = buildings_import[['id', current_material]].copy().dropna(how='any')
    current_pool_combined['increment_iterations'] = 1
    current_pool_combined['increment_iterations'] = current_pool_combined['increment_iterations'].astype('int64')
    mi_estimation_stats[current_material]['increment_iterations'] = 1
    mi_estimation_stats[current_material]['incremented_count'] = current_pool_combined[current_material].count()
    mi_estimation_stats[current_material]['p_0'] = np.quantile(current_pool_combined[current_material], q=0.00)
    mi_estimation_stats[current_material]['p_5'] = np.quantile(current_pool_combined[current_material], q=0.05)  # faster than pandas's current_pool_combined['steel'].quantile(q=0.05)
    mi_estimation_stats[current_material]['p_25'] = np.quantile(current_pool_combined[current_material], q=0.25)
    mi_estimation_stats[current_material]['p_50'] = np.quantile(current_pool_combined[current_material], q=0.50)
    mi_estimation_stats[current_material]['p_75'] = np.quantile(current_pool_combined[current_material], q=0.75)
    mi_estimation_stats[current_material]['p_95'] = np.quantile(current_pool_combined[current_material], q=0.95)
    mi_estimation_stats[current_material]['p_100'] = np.quantile(current_pool_combined[current_material], q=1.00)
    for current_combi in mi_estimation_stats[current_material].index:
        mi_estimation_data[current_material][current_combi] = current_pool_combined.copy()
        print(current_material + str(current_combi))
    # add perfect combinations for the few that have them, to give them some weight, and rerun the statistics
    for current_combi in mi_estimation_stats[current_material].query('raw_HF_db_count >= 1').index:
        mi_estimation_data[current_material][current_combi] = pd.concat([current_pool_combined, buildings_import.loc[current_combi, ['id', current_material]].copy().dropna(how='any')])
        mi_estimation_data[current_material][current_combi]['increment_iterations'].fillna(0, inplace=True)
        mi_estimation_data[current_material][current_combi]['increment_iterations'] = mi_estimation_data[current_material][current_combi]['increment_iterations'].astype('int64')
        # mi_estimation_stats[current_material].loc[current_combi, 'increment_iterations'] += 1
        mi_estimation_stats[current_material].loc[current_combi, 'incremented_count'] = mi_estimation_data[current_material][current_combi][current_material].count()
        mi_estimation_stats[current_material].loc[current_combi, 'p_0'] = np.quantile(mi_estimation_data[current_material][current_combi][current_material], q=0.00)
        mi_estimation_stats[current_material].loc[current_combi, 'p_5'] = np.quantile(mi_estimation_data[current_material][current_combi][current_material], q=0.05)  # faster than pandas's current_pool_combined['steel'].quantile(q=0.05)
        mi_estimation_stats[current_material].loc[current_combi, 'p_25'] = np.quantile(mi_estimation_data[current_material][current_combi][current_material], q=0.25)
        mi_estimation_stats[current_material].loc[current_combi, 'p_50'] = np.quantile(mi_estimation_data[current_material][current_combi][current_material], q=0.50)
        mi_estimation_stats[current_material].loc[current_combi, 'p_75'] = np.quantile(mi_estimation_data[current_material][current_combi][current_material], q=0.75)
        mi_estimation_stats[current_material].loc[current_combi, 'p_95'] = np.quantile(mi_estimation_data[current_material][current_combi][current_material], q=0.95)
        mi_estimation_stats[current_material].loc[current_combi, 'p_100'] = np.quantile(mi_estimation_data[current_material][current_combi][current_material], q=1.00)

    # export as sheet
    if export_to_excel:
        mi_estimation_stats[current_material].reset_index().to_excel(mi_estimation_writer, sheet_name=current_material)

if export_to_excel:
    mi_estimation_writer.close()

    print('exporting full data to Excel')
    mi_data_writer = pd.ExcelWriter('MI_results\\MI_data_' + today + ".xlsx", engine='openpyxl')
    for current_material in materials:
        pd.concat([mi_estimation_data[current_material][current_combi].reset_index()
                  for current_combi in mi_estimation_data[current_material].keys()],
                  axis=0,
                  keys=list(mi_estimation_data[current_material].keys())).sort_index().to_excel(mi_data_writer, sheet_name=current_material)
    mi_data_writer.close()
    print('done')

# %% analysis: summary boxplots (figure 3 for all 32 regions, and variants)

# merge results data for boxplots, start with a list of dataframes and then concatenate it to one dataframe
# design structure should be: combination dict: function, structure, r32, material, value
# then we can use seaborn for boxplots etc. of various layouts
dims_list_specified_named = dict()
dims_list_specified_named[dims_names[0]] = ['Nonresidential', 'Multifamily', 'Single family']
dims_list_specified_named[dims_names[1]] = ['Reinforced concrete', 'Masonry structure', 'Timber structure', 'Steel structure']
dims_list_specified_named[dims_names[2]] = ["China", "Indonesia", "India", "Other Asia (FCP)", "Other Asia (Low)", "Other Asia (Med)", "Pakistan", "Taiwan",
                                            "Brazil", "LATAM (low)", "LATAM (Med)", "Mexico",
                                            "MEA (High)", "MEA (Med)", "North Africa", "South Africa", "Subsaharan (Low)", "Subsaharan (Med)",
                                            "Australia & NZ", "Canada", "SE Europe", "EFTA", "EU12 (High)", "EU12 (Med)", "EU15", "Japan", "S. Korea", "Turkey", "USA",
                                            "Central Asia", "East Europe (FCP)", "Russia"]

analysis_comparison_data = {}
for current_material in materials:
    for key, value in mi_estimation_data[current_material].items():
        analysis_comparison_data[key + (current_material,)] = mi_estimation_data[current_material][key]
        analysis_comparison_data[key + (current_material,)] = analysis_comparison_data[key + (current_material,)].reset_index()
        analysis_comparison_data[key + (current_material,)] = analysis_comparison_data[key + (current_material,)].drop(['id'] + dims_names, axis=1)
        analysis_comparison_data[key + (current_material,)]['combination'] = str(key)
analysis_comparison_data = pd.concat(analysis_comparison_data)
analysis_comparison_data['value'] = analysis_comparison_data.sum(axis=1, numeric_only=True)
analysis_comparison_data.index.rename(dims_names + ['material', 'id'], inplace=True)
analysis_comparison_data.reset_index(inplace=True)
analysis_comparison_data.drop(materials + ['id'], axis=1, inplace=True)
analysis_comparison_data['R5'] = analysis_comparison_data['R5_32'].str.split('_').str[0]

# examples
# sns.boxplot(data=analysis_comparison_data.loc[analysis_comparison_data['combination'] == "('RS', 'C', 'OECD_EU15')"], x='material', y='value')
# sns.catplot(data=analysis_comparison_data.loc[analysis_comparison_data['R5_32'] == "OECD_EU15"], x='material', y='value', row='function', col='structure', kind="violin")

# region = "OECD_EU15"
# boxes = sns.catplot(kind="box",
#                     data=analysis_comparison_data.loc[analysis_comparison_data['R5_32'] == region],
#                     hue='material', y='value',
#                     row='structure', row_order=dims_list_specified[1],
#                     x='function', order=dims_list_specified[0],
#                     linewidth=0.8, showfliers=False,
#                     aspect=3, sharey=True, legend_out=False)
# boxes.set_titles(region + ", {row_name}")

# boxes = sns.catplot(kind="boxen",
#                     data=analysis_comparison_data.loc[analysis_comparison_data['R5_32'] == region],
#                     hue='material', y='value',
#                     row='structure', row_order=dims_list_specified[1],
#                     x='function', order=dims_list_specified[0],
#                     linewidth=0.8, showfliers=False,
#                     aspect=3, sharey=True, legend_out=False, k_depth="proportion", outlier_prop=0.05)
# boxes.set_titles(region + ", {row_name}")

# sns.catplot(kind="box",
#             data=analysis_comparison_data.loc[analysis_comparison_data['R5_32'] == "OECD_EU15"],
#             x='material', y='value',
#             row='function', row_order=dims_list_specified[0],
#             col='structure', col_order=dims_list_specified[1],
#             linewidth=0.8, showfliers=False,
#             )

# boxen by region as page, function type as rows, materials as columns, structure type in plot
with PdfPages('postestimation\\boxplots_regions_' + str(stop_count) + '_' + today + '.pdf') as pdf:
    for region in list(enumerate(dims_list_specified[2])):
        boxes = sns.catplot(kind="boxen", k_depth=3, scale="linear", saturation=0.75,
                            data=analysis_comparison_data.loc[analysis_comparison_data['R5_32'] == region[1]],
                            y='structure', order=['C', 'M', 'T', 'S'], palette="deep",
                            x='value', sharex=False,
                            col='material', col_order=materials,
                            row='function', row_order=dims_list_specified[0],
                            linewidth=0.0, showfliers=False,
                            aspect=0.8, legend_out=False)
        boxes.fig.subplots_adjust(top=0.95, left=0.1, bottom=0.1)
        boxes.fig.suptitle(dims_list_specified_named['R5_32'][region[0]], fontsize=20)
        boxes.set_titles("")
        boxes.despine(left=True)
        boxes.set_yticklabels(dims_list_specified_named['structure'], size=15)
        for boxax in boxes.axes_dict.values():
            boxax.tick_params(left=False)
            for line in boxax.get_lines():
                line.set(color='white', linewidth=1.5, alpha=1)  # https://ourpython.com/python/how-to-change-the-colour-a-boxplot-line-in-seaborn
        for boxax in range(3):
            boxes.axes[boxax][0].set_ylabel(dims_list_specified_named['function'][boxax], size=20)
            boxes.axes[boxax][0].set(xlim=(0, 1700))
            boxes.axes[boxax][1].set(xlim=(0, 1700))
            boxes.axes[boxax][2].set(xlim=(0, 400))
            boxes.axes[boxax][3].set(xlim=(0, 400))
            boxes.axes[boxax][4].set(xlim=(0, 20))
            boxes.axes[boxax][5].set(xlim=(0, 20))
            boxes.axes[boxax][6].set(xlim=(0, 10))
            boxes.axes[boxax][7].set(xlim=(0, 10))
        for boxax in range(8):
            boxes.axes[2][boxax].set_xlabel(materials[boxax] + "\n $kg/m^2$ ", size=20)
        pdf.savefig(boxes.fig)
        plt.close()


# by material as page, structure type as rows, function type as columns, regions in plot, r5 as colors
with PdfPages('postestimation\\boxplots_materials_' + str(stop_count) + '_' + today + '.pdf') as pdf:
    for current_material in materials:
        boxes = sns.catplot(kind="boxen", k_depth=3, scale="linear", saturation=0.75,
                            data=analysis_comparison_data.loc[analysis_comparison_data['material'] == current_material],
                            x='R5_32', y='value', order=dims_list_specified[2],
                            palette="plasma", dodge=False, hue='R5', hue_order=['ASIA', 'MAF', 'REF', 'LAM', 'OECD'],
                            row='structure', row_order=dims_list_specified[1],
                            col='function', col_order=dims_list_specified[0],
                            linewidth=0, showfliers=False,
                            aspect=3, legend_out=False)
        boxes.fig.subplots_adjust(left=0.05, top=0.95, bottom=0.15)
        boxes.set_xticklabels(dims_list_specified_named['R5_32'], size=20, rotation="vertical")
        boxes.fig.suptitle(current_material, fontsize=20)
        boxes.set_titles("")
        boxes.despine(left=True)
        for boxax in boxes.axes_dict.values():
            boxax.tick_params(left=False, bottom=False)
            for line in boxax.get_lines():
                line.set(color='white', linewidth=1.5, alpha=1)  # https://ourpython.com/python/how-to-change-the-colour-a-boxplot-line-in-seaborn
        for boxax in range(4):
            boxes.axes[boxax][0].set_ylabel(dims_list_specified_named['structure'][boxax] + "\n $kg/m^2$ ", size=20)
        for boxax in range(3):
            boxes.axes[3][boxax].set_xlabel(dims_list_specified_named['function'][boxax], size=20)
        pdf.savefig(boxes.fig)
        plt.close()


# # by region as page, function type as rows, structure type as columns, materials in plot
# with PdfPages('postestimation\\boxplots_' + str(stop_count) + '_' + today + '.pdf') as pdf:
#     structure_types = ['Reinforced concrete', 'Masonry structure', 'Steel structure', 'Timber structure']
#     function_types = ['Nonresidential', 'Multifamily', 'Single family']
#     for region in dims_list_specified[2]:
#         boxes = sns.catplot(kind="box",
#                             data=analysis_comparison_data.loc[analysis_comparison_data['R5_32'] == region],
#                             x='value', y='material', palette="bright",
#                             col='structure', col_order=dims_list_specified[1],
#                             row='function', row_order=dims_list_specified[0],
#                             linewidth=1, showfliers=False,
#                             aspect=1.5, legend_out=False)
#         boxes.set(xlim=(0, 1750))
#         boxes.fig.subplots_adjust(top=0.95, left=0.1)
#         boxes.fig.suptitle(region, fontsize=20)
#         boxes.set_titles("")
#         # boxes.set_titles("{col_name}, {row_name}", size=20)
#         # boxes.set_axis_labels("$kg/m^2$", "", size=20)
#         boxes.despine(left=True)
#         boxes.set_yticklabels(materials, size=20)
#         for boxax in boxes.axes_dict.values():
#             boxax.tick_params(left=False)
#         for boxax in range(3):
#             boxes.axes[boxax][0].set_ylabel(function_types[boxax], size=20)
#         for boxax in range(4):
#             boxes.axes[2][boxax].set_xlabel("$kg/m^2$ \n" + structure_types[boxax], size=20)
#         pdf.savefig(boxes.fig)
#         plt.close()

# # by region as page, structure type as rows, functiontype as columns, materials in plot
# with PdfPages('postestimation\\boxplots_' + str(stop_count) + '_' + today + '.pdf') as pdf:
#     structure_types = ['Reinforced concrete', 'Masonry structure', 'Steel structure', 'Timber structure']
#     function_types = ['Nonresidential', 'Multifamily', 'Single family']
#     for region in dims_list_specified[2]:
#         boxes = sns.catplot(kind="box",
#                             data=analysis_comparison_data.loc[analysis_comparison_data['R5_32'] == region],
#                             x='value', y='material', palette="bright",
#                             row='structure', row_order=dims_list_specified[1],
#                             col='function', col_order=dims_list_specified[0],
#                             linewidth=1, showfliers=False,
#                             aspect=1.5, legend_out=False)
#         boxes.set(xlim=(0, 1750))
#         boxes.fig.subplots_adjust(top=0.95, left=0.1, bottom=0.1)
#         boxes.fig.suptitle(region, fontsize=20)
#         boxes.set_titles("")
#         # boxes.set_titles("{col_name}, {row_name}", size=20)
#         # boxes.set_axis_labels("$kg/m^2$", "", size=20)
#         boxes.despine(left=True)
#         boxes.set_yticklabels(materials, size=20)
#         for boxax in boxes.axes_dict.values():
#             boxax.tick_params(left=False)
#         for boxax in range(4):
#             boxes.axes[boxax][0].set_ylabel(structure_types[boxax], size=20)
#         for boxax in range(3):
#             boxes.axes[2][boxax].set_xlabel("$kg/m^2$ \n" + function_types[boxax], size=20)
#         pdf.savefig(boxes.fig)
#         plt.close()

# # boxplot by region as page, function type as rows, materials as columns, structure type in plot
# with PdfPages('postestimation\\boxplots_inv_' + str(stop_count) + '_' + today + '.pdf') as pdf:
#     structure_types = ['Reinforced concrete', 'Masonry structure', 'Timber structure', 'Steel structure']
#     function_types = ['Nonresidential', 'Multifamily', 'Single family']
#     for region in dims_list_specified[2]:
#         boxes = sns.catplot(kind="box",
#                             data=analysis_comparison_data.loc[analysis_comparison_data['R5_32'] == region],
#                             y='structure', order=['C', 'M', 'T', 'S'], palette="deep",
#                             x='value', sharex=False,
#                             col='material', col_order=materials,
#                             row='function', row_order=dims_list_specified[0],
#                             linewidth=0.5, showfliers=False,
#                             medianprops=dict(color="white", linewidth=2, alpha=0.9), whis=(5, 95),
#                             showmeans=True, meanprops=dict(marker='o', markerfacecolor="white", markeredgecolor='0.6'),
#                             aspect=0.8, legend_out=False)
#         boxes.fig.subplots_adjust(top=0.95, left=0.1, bottom=0.1)
#         boxes.fig.suptitle(region, fontsize=20)
#         boxes.set_titles("")
#         boxes.despine(left=True)
#         boxes.set_yticklabels(structure_types, size=15)
#         for boxax in boxes.axes_dict.values():
#             boxax.tick_params(left=False)
#         for boxax in range(3):
#             boxes.axes[boxax][0].set_ylabel(function_types[boxax], size=20)
#             boxes.axes[boxax][0].set(xlim=(0, 1700))
#             boxes.axes[boxax][1].set(xlim=(0, 1700))
#             boxes.axes[boxax][2].set(xlim=(0, 400))
#             boxes.axes[boxax][3].set(xlim=(0, 400))
#             boxes.axes[boxax][4].set(xlim=(0, 20))
#             boxes.axes[boxax][5].set(xlim=(0, 20))
#             boxes.axes[boxax][6].set(xlim=(0, 10))
#             boxes.axes[boxax][7].set(xlim=(0, 10))
#         for boxax in range(8):
#             boxes.axes[2][boxax].set_xlabel(materials[boxax] + "\n $kg/m^2$ ", size=20)
#         pdf.savefig(boxes.fig)
#         plt.close()


# # by region as page, materials as rows, functiontype as columns, structure type in plot
# with PdfPages('postestimation\\boxplots_inv_' + str(stop_count) + '_' + today + '.pdf') as pdf:
#     structure_types = ['Reinforced concrete', 'Masonry structure', 'Steel structure', 'Timber structure']
#     function_types = ['Nonresidential', 'Multifamily', 'Single family']
#     for region in dims_list_specified[2]:
#         boxes = sns.catplot(kind="box", sharey=False,
#                             data=analysis_comparison_data.loc[analysis_comparison_data['R5_32'] == region],
#                             x='structure', order=['C', 'M', 'T', 'S'], palette="bright", y='value',
#                             row='material', row_order=materials,
#                             col='function', col_order=dims_list_specified[0],
#                             linewidth=1, showfliers=False,
#                             aspect=2, legend_out=False)
#         boxes.fig.subplots_adjust(top=0.95, left=0.1)
#         boxes.fig.suptitle(region, fontsize=20)
#         boxes.set_titles("")
#         boxes.despine(left=True)
#         boxes.set_xticklabels(['C', 'M', 'T', 'S'], size=20)
#         for boxax in boxes.axes_dict.values():
#             boxax.tick_params(left=False)
#         for boxax in range(7):
#             boxes.axes[boxax][0].set_ylabel(materials[boxax] + "\n $kg/m^2$ ", size=20)
#         for boxax in range(3):
#             boxes.axes[6][boxax].set_xlabel(function_types[boxax], size=20)
#             boxes.axes[0][boxax].set(ylim=(0, 2000))
#             boxes.axes[1][boxax].set(ylim=(0, 2000))
#             boxes.axes[2][boxax].set(ylim=(0, 500))
#             boxes.axes[3][boxax].set(ylim=(0, 500))
#             boxes.axes[4][boxax].set(ylim=(0, 20))
#             boxes.axes[5][boxax].set(ylim=(0, 20))
#             boxes.axes[6][boxax].set(ylim=(0, 10))
#         pdf.savefig(boxes.fig)
#         plt.close()

# %% analysis: individual combination increment of distributions by increment iteration, histograms (figure 2)
# setup
material = "concrete"
combi = ("RS", "S", "OECD_AUNZ")
outliercut = buildings_import.max()[material]
if material == 'steel':
    outliercut = 200
elif material == 'wood':
    outliercut = 350
binsize = outliercut / 17
sns.set_palette("magma", 7)
increment_data = mi_estimation_data[material][combi].copy().reset_index()
increment_max = increment_data['increment_iterations'].max()
increment_data['increment_iterations'] = increment_data['increment_iterations'].astype("string")

# export individual increment histogram plot
increment_hist_indv = sns.histplot(data=increment_data, x=material,
                                   hue="increment_iterations", hue_order=["6", "5", "4", "3", "2", "1", "0"],
                                   alpha=1, linewidth=0,
                                   stat="count", element="step",
                                   binwidth=binsize, multiple="stack")
increment_hist_indv.set(xlim=(0, outliercut))
# plt.savefig('postestimation\\' + material + '_' + str(combi) + '.svg')

# export individual increment boxplot
increment_box_indv = sns.boxplot(data=increment_data, x=material,
                                 linewidth=0.8, showfliers=False,
                                 color="C" + str(6 - increment_max), width=0.1)
increment_box_indv.set(xlim=(0, outliercut))
# sns.despine(left=True)
# plt.savefig('postestimation\\' + material + '_' + str(combi) + '_box.svg')

# %% analysis: increment of distributions by increment iteration, histograms (alternative version of figure 2)

# # examples
# current_combi = ('RS', 'T', 'ASIA_TWN')
# current_combi = ('RM', 'T', 'OECD_EU15')

# increment, increment_axes = plt.subplots(3, gridspec_kw={"height_ratios": (.1, .1, .8)})
# increment_data = mi_estimation_data[current_material][current_combi].copy().reset_index()
# increment_data['increment_iterations'] = increment_data['increment_iterations'].astype("string")
# sns.set_palette("magma", 7)
# sns.boxplot(data=increment_data, x=current_material,
#             linewidth=0.8, showfliers=False,
#             color="C" + increment_data['increment_iterations'].max(),
#             ax=increment_axes[0])
# sns.boxplot(data=increment_data.query('increment_iterations == "0"'), x=current_material,
#             linewidth=0.8, showfliers=False,
#             color="C1",
#             ax=increment_axes[1])
# sns.histplot(data=increment_data, x=current_material,
#              hue="increment_iterations", hue_order=["0", "1", "2", "3", "4", "5", "6"],
#              alpha=1, linewidth=0,
#              stat="count", element="step",
#              binwidth=20,
#              ax=increment_axes[2])
# increment_axes[0].set(yticks=[], xticks=[], xlim=(0, 700),
#                    xlabel='after, n=' + str(increment_data.count()[0]),
#                    title=str(current_combi)[1:-1])
# increment_axes[1].set(yticks=[], xticks=[], xlim=(0, 700),
#                    xlabel='before, n=' + str(increment_data.query('increment_iterations == "0"').count()[0]))
# increment_axes[2].set(xlim=(0, 700), xlabel='kg/m2 \n')
# sns.despine(ax=increment_axes[0], left=True, bottom=True)
# sns.despine(ax=increment_axes[1], left=True, bottom=True)
# sns.despine(ax=increment_axes[2])


def compareincrement_hist(u, c, r, material, axx_before, axx_after, axx_hist, axy):
    combi = (u, c, r)
    outliercut = buildings_import.max()[material]
    if material == 'steel':
        outliercut = 200
    elif material == 'wood':
        outliercut = 350
    binsize = outliercut / 17

    increment_data = mi_estimation_data[material][combi].copy().reset_index()
    increment_data['increment_iterations_str'] = increment_data['increment_iterations'].astype("string")

    sns.set_palette("magma", 7)

    sns.boxplot(data=increment_data, x=material,
                linewidth=0.8, showfliers=False,
                color="C" + str(6 - increment_data['increment_iterations'].max()),
                ax=increment_axes[axx_after, axy])
    sns.boxplot(data=increment_data.query('increment_iterations_str == "0"'), x=material,
                linewidth=0.8, showfliers=False,
                color=".6",
                ax=increment_axes[axx_before, axy])
    sns.histplot(data=increment_data, x=material,
                 hue="increment_iterations_str", hue_order=["6", "5", "4", "3", "2", "1", "0"],
                 alpha=1, linewidth=0,
                 stat="count", element="step",
                 binwidth=binsize,
                 multiple="stack",
                 ax=increment_axes[axx_hist, axy])

    increment_axes[axx_after, axy].set(yticks=[], xticks=[], xlim=(0, outliercut),
                                       xlabel='after, n=' + str(increment_data.count()[0]))
    increment_axes[axx_before, axy].set(yticks=[], xticks=[], xlim=(0, outliercut),
                                        xlabel='before, n=' + str(increment_data.query('increment_iterations_str == "0"').count()[0]),
                                        title="\n" + str(combi)[1:-1])
    increment_axes[axx_hist, axy].set(xlim=(0, outliercut), xlabel='kg/m2 \n')

    sns.despine(ax=increment_axes[axx_after, axy], left=True, bottom=True)
    sns.despine(ax=increment_axes[axx_before, axy], left=True, bottom=True)
    sns.despine(ax=increment_axes[axx_hist, axy])

    if not(axx_hist == 2 and axy == 3):
        increment_axes[axx_hist, axy].legend([], [], frameon=False)

    return None


for current_material in materials:
    filename = current_material + '_expand_until_at_least_' + str(stop_count) + '_' + today
    with PdfPages('postestimation\\' + filename + '.pdf') as pdf:
        for region in dims_list_specified[2]:
            increment, increment_axes = plt.subplots(9, 4, gridspec_kw={"height_ratios": (.1, .1, .8, .1, .1, .8, .1, .1, .8)}, figsize=(30, 20))
            compareincrement_hist("NR", "C", region, current_material, 0, 1, 2, 0)
            compareincrement_hist("NR", "M", region, current_material, 0, 1, 2, 1)
            compareincrement_hist("NR", "S", region, current_material, 0, 1, 2, 2)
            compareincrement_hist("NR", "T", region, current_material, 0, 1, 2, 3)
            compareincrement_hist("RM", "C", region, current_material, 3, 4, 5, 0)
            compareincrement_hist("RM", "M", region, current_material, 3, 4, 5, 1)
            compareincrement_hist("RM", "S", region, current_material, 3, 4, 5, 2)
            compareincrement_hist("RM", "T", region, current_material, 3, 4, 5, 3)
            compareincrement_hist("RS", "C", region, current_material, 6, 7, 8, 0)
            compareincrement_hist("RS", "M", region, current_material, 6, 7, 8, 1)
            compareincrement_hist("RS", "S", region, current_material, 6, 7, 8, 2)
            compareincrement_hist("RS", "T", region, current_material, 6, 7, 8, 3)
            increment.set_constrained_layout(True)
            pdf.savefig(increment)
            plt.close()
