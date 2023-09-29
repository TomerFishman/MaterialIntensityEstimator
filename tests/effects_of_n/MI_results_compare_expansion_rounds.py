# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 10:14:47 2022

@author: tomer
"""

from os import chdir
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

chdir('C:\\Users\\Tomer\\Dropbox\\-the research\\2020 10 IIASA\\MI_project\\git\\MaterialIntensityEstimator\\tests\\effects_of_n')

materials = ['concrete', 'steel', 'brick', 'wood', 'glass']  # , 'aluminum', 'copper']

stop_list = list(np.arange(10, 71, step=5))
# load MI result files
stop_results = {}
for stop in stop_list:
    stop_results["n_" + str(stop)] = pd.read_excel("MI_ranges_n" + str(stop) + "_20230927.xlsx", sheet_name=materials, index_col=[1, 2, 3])

db_combinations_stats = pd.read_excel("db_combinations_stats_20230927.xlsx", index_col=[0, 1, 2])

# concatenate results for comparisons
stop_compare = {}
current_material = 'concrete'
for current_material in materials:
    stop_compare[current_material] = {}
    stop_compare[current_material]["n"] = pd.concat([db_combinations_stats['count.' + current_material]] + [value[current_material]['incremented_count'] for value in stop_results.values()], keys=[0] + stop_list)
    stop_compare[current_material]["median"] = pd.concat([db_combinations_stats['p50.' + current_material]] + [value[current_material]['p_50'] for value in stop_results.values()], keys=[0] + stop_list)
    stop_compare[current_material] = pd.concat([stop_compare[current_material]['n'], stop_compare[current_material]['median']], axis=1).reset_index()
    stop_compare[current_material].rename(columns={0: 'n', 1: 'median', 'level_0': 'n_limit'}, inplace=True)
    stop_compare[current_material]['combi'] = stop_compare[current_material]['function'].astype("string") + '_' + stop_compare[current_material]['structure'].astype("string") + '_' + stop_compare[current_material]['R5_32'].astype("string")
    stop_compare[current_material]['function_structure'] = stop_compare[current_material]['function'].astype("string") + '_' + stop_compare[current_material]['structure'].astype("string")
    stop_compare[current_material]['R5'] = stop_compare[current_material]['R5_32'].str.split('_').str[0]

# %% pivots to show progress as n limits grow

stop_compare_medians = {}
stop_compare_ns = {}
for current_material in materials:
    stop_compare_medians[current_material] = stop_compare[current_material].pivot(index='n_limit', columns='combi', values='median')
    stop_compare_ns[current_material] = stop_compare[current_material].pivot(index='n_limit', columns='combi', values='n')

stop_compare_medians_concat = pd.concat([stop_compare_medians[current_material] for current_material in materials], axis=1, keys=materials)

stop_compare_medians_concat_compared = pd.DataFrame(index=stop_compare_medians_concat.index, columns=stop_compare_medians_concat.columns)
for stop in [0] + stop_list:
    stop_compare_medians_concat_compared.loc[stop] = (stop_compare_medians_concat.loc[stop] - stop_compare_medians_concat.loc[30]) / stop_compare_medians_concat.loc[30]
stop_compare_medians_concat_compared_mean = stop_compare_medians_concat_compared.mean(axis=1) * 100


stop_compare_medians_concat_diff = stop_compare_medians_concat.diff().dropna(axis=1, how='all')
stop_compare_medians_concat_diff.loc[0] = 0
stop_compare_medians_concat_diff1 = stop_compare_medians_concat_diff.where(stop_compare_medians_concat_diff == 0, 1)
stop_compare_medians_concat_diff1_cumsum = stop_compare_medians_concat_diff1.cumsum()

stop_compare_medians_concat_diff1_cumsum_compared = pd.DataFrame(index=stop_compare_medians_concat_diff1_cumsum.index, columns=stop_compare_medians_concat_diff1_cumsum.columns)
for stop in [0] + stop_list:
    stop_compare_medians_concat_diff1_cumsum_compared.loc[stop] = stop_compare_medians_concat_diff1_cumsum.loc[30] == stop_compare_medians_concat_diff1_cumsum.loc[stop]

stop_compare_medians_concat_diff1_cumsum_compared_count = stop_compare_medians_concat_diff1_cumsum_compared.sum(axis=1)
stop_compare_medians_concat_diff1_cumsum_compared_count.plot(kind='line')

stop_compare_medians_concat_diff1_cumsum_compared_percent = stop_compare_medians_concat_diff1_cumsum_compared_count / stop_compare_medians_concat_diff1_cumsum_compared_count[30]
stop_compare_medians_concat_diff1_cumsum_compared_percent.iloc[1:].plot(kind='line')
# %% example plots
nplot = sns.lineplot(data=stop_compare[current_material], x="n_limit", y="n", hue="structure", size="combi", sizes=(.25, .30), palette="bright", errorbar=None)
handles, labels = nplot.get_legend_handles_labels()
nplot.legend(handles=handles[0:labels.index('combi')], labels=labels[0:labels.index('combi')])

medplot = sns.lineplot(data=stop_compare[current_material], x="n_limit", y="median", hue="structure", size="combi", sizes=(.25, .30), palette="bright", errorbar=None)
handles, labels = medplot.get_legend_handles_labels()
medplot.legend(handles=handles[0:labels.index('combi')], labels=labels[0:labels.index('combi')])
medplot.set(ylim=(0, 1500))

nplot = sns.lineplot(data=stop_compare[current_material], x="n_limit", y="n", hue="structure", size="combi", sizes=(.25, .30), legend=False, palette="bright", errorbar=None)
medplot = sns.lineplot(data=stop_compare[current_material], x="n_limit", y="median", hue="combi", errorbar=None, legend=False, palette="magma")

# plot with achieved n and medians on the axes
sns.lineplot(data=stop_compare['steel'], x="n", y="median", hue="structure", size="combi", sizes=(.25, .30), legend=False, palette="bright", errorbar=None, markers=True)
sns.lineplot(data=stop_compare['glass'], x="n", y="median", hue="function", size="combi", sizes=(.25, .30), legend=False, palette="bright", errorbar=None)
sns.relplot(data=stop_compare['steel'].reset_index(), x="n", y="median", kind="line", hue="R5_32", size="combi", sizes=(.25, .30), legend=False, palette="bright", col="structure", row="function", markers=True)


# plots for all materials
with PdfPages('postestimation_analysis\\stops_compare.pdf') as pdf:
    for hue_dim in ["R5", "use_short", "const_short"]:
        compare, compare_axs = plt.subplots(2, 5, figsize=(40, 20))
        sns.lineplot(data=stop_compare[materials[0]], x="n_limit", y="n", hue=hue_dim, size="combi", sizes=(.25, .30), palette="bright", errorbar=None, ax=compare_axs[0, 0])
        sns.lineplot(data=stop_compare[materials[0]], x="n_limit", y="median", hue=hue_dim, size="combi", sizes=(.25, .30), legend=False, palette="bright", errorbar=None, ax=compare_axs[1, 0])
        sns.lineplot(data=stop_compare[materials[1]], x="n_limit", y="n", hue=hue_dim, size="combi", sizes=(.25, .30), legend=False, palette="bright", errorbar=None, ax=compare_axs[0, 1])
        sns.lineplot(data=stop_compare[materials[1]], x="n_limit", y="median", hue=hue_dim, size="combi", sizes=(.25, .30), legend=False, palette="bright", errorbar=None, ax=compare_axs[1, 1])
        sns.lineplot(data=stop_compare[materials[2]], x="n_limit", y="n", hue=hue_dim, size="combi", sizes=(.25, .30), legend=False, palette="bright", errorbar=None, ax=compare_axs[0, 2])
        sns.lineplot(data=stop_compare[materials[2]], x="n_limit", y="median", hue=hue_dim, size="combi", sizes=(.25, .30), legend=False, palette="bright", errorbar=None, ax=compare_axs[1, 2])
        sns.lineplot(data=stop_compare[materials[3]], x="n_limit", y="n", hue=hue_dim, size="combi", sizes=(.25, .30), legend=False, palette="bright", errorbar=None, ax=compare_axs[0, 3])
        sns.lineplot(data=stop_compare[materials[3]], x="n_limit", y="median", hue=hue_dim, size="combi", sizes=(.25, .30), legend=False, palette="bright", errorbar=None, ax=compare_axs[1, 3])
        sns.lineplot(data=stop_compare[materials[4]], x="n_limit", y="n", hue=hue_dim, size="combi", sizes=(.25, .30), legend=False, palette="bright", errorbar=None, ax=compare_axs[0, 4])
        sns.lineplot(data=stop_compare[materials[4]], x="n_limit", y="median", hue=hue_dim, size="combi", sizes=(.25, .30), legend=False, palette="bright", errorbar=None, ax=compare_axs[1, 4])
        compare_axs[0, 0].set_title(materials[0])
        compare_axs[0, 1].set_title(materials[1])
        compare_axs[0, 2].set_title(materials[2])
        compare_axs[0, 3].set_title(materials[3])
        compare_axs[0, 4].set_title(materials[4])
        compare_axs[1, 1].set_ylim(0, 250)
        compare_axs[1, 3].set_ylim(0, 200)
        handles, labels = compare_axs[0, 0].get_legend_handles_labels()
        compare_axs[0, 0].legend(handles=handles[0:labels.index('combi')], labels=labels[0:labels.index('combi')])
        pdf.savefig(compare)
