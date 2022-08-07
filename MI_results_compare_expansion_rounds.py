# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 10:14:47 2022

@author: tomer
"""

from os import chdir
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

chdir('C:\\Users\\Tomer\\Dropbox\\-the research\\2020 10 IIASA\\MI_project\\git\\MaterialIntensityEstimator')

materials = ['concrete', 'steel', 'brick', 'wood', 'glass']  # , 'aluminum', 'copper']

# load MI result files
stops_results = {}
stops_results["stop_at_15"] = pd.read_excel("MI_results\\stop_at_15_20220405.xlsx", sheet_name=materials, index_col=[1, 2, 3])
stops_results["stop_at_20"] = pd.read_excel("MI_results\\stop_at_20_20220405.xlsx", sheet_name=materials, index_col=[1, 2, 3])
stops_results["stop_at_30"] = pd.read_excel("MI_results\\stop_at_30_20220405.xlsx", sheet_name=materials, index_col=[1, 2, 3])
stops_results["stop_at_45"] = pd.read_excel("MI_results\\stop_at_45_20220405.xlsx", sheet_name=materials, index_col=[1, 2, 3])
stops_results["stop_at_60"] = pd.read_excel("MI_results\\stop_at_60_20220405.xlsx", sheet_name=materials, index_col=[1, 2, 3])
stops_results["stop_at_75"] = pd.read_excel("MI_results\\stop_at_75_20220405.xlsx", sheet_name=materials, index_col=[1, 2, 3])
stops_results["stop_at_90"] = pd.read_excel("MI_results\\stop_at_90_20220405.xlsx", sheet_name=materials, index_col=[1, 2, 3])

# concatenate results for comparisons
stops_compare = {}
current_material = 'concrete'
for current_material in materials:
    stops_compare[current_material] = {}
    stops_compare[current_material]["n"] = pd.concat([stops_results["stop_at_15"][current_material]['db_count']] + [value[current_material]['expand_count'] for value in stops_results.values()], keys=['0', '15', '20', '30', '45', '60', '100'])
    stops_compare[current_material]["median"] = pd.concat([stops_results["stop_at_15"][current_material]['db_50']] + [value[current_material]['expand_50'] for value in stops_results.values()], keys=['0', '15', '20', '30', '45', '60', '100'])
    stops_compare[current_material] = pd.concat([stops_compare[current_material]['n'], stops_compare[current_material]['median']], axis=1).reset_index()
    stops_compare[current_material].rename(columns={0: 'n', 1: 'median', 'level_0': 'n_limit'}, inplace=True)
    stops_compare[current_material]['combi'] = stops_compare[current_material]['use_short'].astype("string") + '_' + stops_compare[current_material]['const_short'].astype("string") + '_' + stops_compare[current_material]['R5_32'].astype("string")
    stops_compare[current_material]['use_const'] = stops_compare[current_material]['use_short'].astype("string") + '_' + stops_compare[current_material]['const_short'].astype("string")
    stops_compare[current_material]['R5'] = stops_compare[current_material]['R5_32'].str.split('_').str[0]

# example plots
nplot = sns.lineplot(data=stops_compare[current_material], x="n_limit", y="n", hue="R5", size="combi", sizes=(.25, .30), palette="bright", ci=None)
handles, labels = nplot.get_legend_handles_labels()
nplot.legend(handles=handles[0:labels.index('combi')], labels=labels[0:labels.index('combi')])

nplot = sns.lineplot(data=stops_compare[current_material], x="n_limit", y="n", hue="const_short", size="combi", sizes=(.25, .30), legend=False, palette="bright", ci=None)
medplot = sns.lineplot(data=stops_compare[current_material], x="n_limit", y="median", hue="combi", ci=None, legend=False, palette="magma")

# plot with achieved n and medians on the axes
sns.lineplot(data=stops_compare['steel'], x="n", y="median", hue="const_short", size="combi", sizes=(.25, .30), legend=False, palette="bright", ci=None, markers=True)
sns.lineplot(data=stops_compare['glass'], x="n", y="median", hue="use_short", size="combi", sizes=(.25, .30), legend=False, palette="bright", ci=None)
sns.relplot(data=stops_compare['steel'].reset_index(), x="n", y="median", kind="line", hue="R5_32", size="combi", sizes=(.25, .30), legend=False, palette="bright", col="const_short", row="use_short", markers=True)


# plots for all materials
with PdfPages('postestimation_analysis\\stops_compare.pdf') as pdf:
    for hue_dim in ["R5", "use_short", "const_short"]:
        compare, compare_axs = plt.subplots(2, 5, figsize=(40, 20))
        sns.lineplot(data=stops_compare[materials[0]], x="n_limit", y="n", hue=hue_dim, size="combi", sizes=(.25, .30), palette="bright", ci=None, ax=compare_axs[0, 0])
        sns.lineplot(data=stops_compare[materials[0]], x="n_limit", y="median", hue=hue_dim, size="combi", sizes=(.25, .30), legend=False, palette="bright", ci=None, ax=compare_axs[1, 0])
        sns.lineplot(data=stops_compare[materials[1]], x="n_limit", y="n", hue=hue_dim, size="combi", sizes=(.25, .30), legend=False, palette="bright", ci=None, ax=compare_axs[0, 1])
        sns.lineplot(data=stops_compare[materials[1]], x="n_limit", y="median", hue=hue_dim, size="combi", sizes=(.25, .30), legend=False, palette="bright", ci=None, ax=compare_axs[1, 1])
        sns.lineplot(data=stops_compare[materials[2]], x="n_limit", y="n", hue=hue_dim, size="combi", sizes=(.25, .30), legend=False, palette="bright", ci=None, ax=compare_axs[0, 2])
        sns.lineplot(data=stops_compare[materials[2]], x="n_limit", y="median", hue=hue_dim, size="combi", sizes=(.25, .30), legend=False, palette="bright", ci=None, ax=compare_axs[1, 2])
        sns.lineplot(data=stops_compare[materials[3]], x="n_limit", y="n", hue=hue_dim, size="combi", sizes=(.25, .30), legend=False, palette="bright", ci=None, ax=compare_axs[0, 3])
        sns.lineplot(data=stops_compare[materials[3]], x="n_limit", y="median", hue=hue_dim, size="combi", sizes=(.25, .30), legend=False, palette="bright", ci=None, ax=compare_axs[1, 3])
        sns.lineplot(data=stops_compare[materials[4]], x="n_limit", y="n", hue=hue_dim, size="combi", sizes=(.25, .30), legend=False, palette="bright", ci=None, ax=compare_axs[0, 4])
        sns.lineplot(data=stops_compare[materials[4]], x="n_limit", y="median", hue=hue_dim, size="combi", sizes=(.25, .30), legend=False, palette="bright", ci=None, ax=compare_axs[1, 4])
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
