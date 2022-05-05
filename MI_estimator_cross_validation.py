# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 13:42:14 2020

@author: Tomer Fishman

copied from the main file 5/5/2022:
    removed cells:
        EDA: database plots
        EDA: Nonparametric statistical tests
        analysis: growth of distributions by expansion round, histograms
        analysis: growth of distributions by expansion round, kde
        analysis: before-after comparisons with violin plots
        further analysis

"""
# %% libraries and load dimensions

from os import chdir
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import date
from sklearn.model_selection import train_test_split

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

materials = ['concrete', 'steel', 'wood', 'brick']
materials += ['glass', 'aluminum', 'copper']

# %% load the MI database with const. type ML results from Orange

buildings_import = pd.read_excel("data_input_and_ml_processing\\buildings_v2-const_type_ML.xlsx", sheet_name="Sheet1")

# create new column const_short where U from 'Construction type' is replaced by 'Random Forest'
buildings_import['const_short'] = buildings_import['Random Forest'].where((buildings_import['Construction type'].str.match('U')), buildings_import['Construction type'])

# clean up buildings_import
buildings_import = buildings_import[['id'] + materials + dims_names]


# %% final setups of the database data

# HINT remove IN 'informal' because there are simply not enough datapoints for meaningful estimations, consider including later
buildings_import = buildings_import[buildings_import.use_short != 'IN']
# set up the same multiindex as the other dataframes
buildings_import.set_index(dims_names, inplace=True)
# sort to make pandas faster and with less warnings
buildings_import.sort_index(inplace=True)

buildings_import_full = buildings_import.copy()

# %% cross validation: partition buildings_import to a test set and training set

cross_round = 4

buildings_import, test_data = train_test_split(buildings_import_full, test_size=0.1, random_state=cross_round)  # 10% for test


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
# TODO decide which quantile interpolation is best i.e. what does excel do?

db_combinations_stats = pd.concat(db_combinations_stats, axis=1, keys=['', 'count', 'avg', 'sd', 'p5', 'p25', 'p50', 'p75', 'p95'])

db_combinations_stats[('count', 'concrete')]
db_combinations_stats.loc[('NR', 'C', 'ASIA_CHN'), ('count', 'concrete')]
db_combinations_stats.loc[('NR', 'C', 'ASIA_CHN'), :]
db_combinations_stats.loc[:, db_combinations_stats.columns.isin(['concrete'], level=1)]

# replace NANs with zeros for consistency, or keep only those with values
db_combinations_stats_valid = db_combinations_stats.dropna(how='all')
# db_combinations_stats = db_combinations_stats.fillna(0)

# # exoort db_combinations_stats
# db_combinations_stats_valid.to_excel("MI_results\\db_combinations_stats.xlsx", merge_cells=False)
# db_combinations_stats.unstack().to_clipboard()


# %% separate buildings_import to individual dataframes by valid combinations

# for each material in this dict, prefiltered as a list only with valid combinations (i.e. existing in buildings_import): [combination tuple, dataframe, [no. of rows in df, counts of each material], expansion score set to 0]
# it's a list and not a dict in case the selection algorithm needs to duplicate list items. A dict can't have duplicate keys.
db_combinations_data = {}
for current_material in materials:
    db_combinations_data[current_material] = []
    [db_combinations_data[current_material].append([row[0], buildings_import.loc[row[0]], int(db_combinations_stats_valid.loc[row[0], ('count', current_material)]), 0]) for row in db_combinations_stats_valid.itertuples() if db_combinations_stats_valid.loc[row[0], ('count', current_material)] > 0]


# %% create a dataframe with all practical (i.e. not unspecifieds) combination options to be filled with data

# remove 'unspecified' entities !!make sure to change the list indexes as needed
dims_list_specified = dims_list[:]
dims_list_specified[0] = [x for x in dims_list_specified[0] if 'U' not in x]
dims_list_specified[1] = [x for x in dims_list_specified[1] if 'U' not in x]


# dict for storing the current selection MIs with their IDs for backup and reference
mi_estimation_data = {}
mi_estimation_stats = {}
for current_material in materials:
    mi_estimation_data[current_material] = {}
    mi_estimation_stats[current_material] = pd.DataFrame(data=None, index=pd.MultiIndex.from_product(dims_list_specified, names=dims_names),
                                                         columns=['R5', 'db_count', 'db_avg', 'db_sd', 'db_5', 'db_25', 'db_50', 'db_75', 'db_95',
                                                                  'expand_count', 'expand_avg', 'expand_sd', 'expand_5', 'expand_25', 'expand_50', 'expand_75', 'expand_95', 'expand_rounds'])  # , 'p1', 'p5', 'p10', 'p20', 'p25', 'p30', 'p40', 'p50', 'p60', 'p70', 'p75', 'p80', 'p90', 'p95', 'p99'
    mi_estimation_stats[current_material] = mi_estimation_stats[current_material].reset_index()
    mi_estimation_stats[current_material]['R5'] = mi_estimation_stats[current_material]['R5_32'].str.split('_').str[0]  # SSP 5 regions
    mi_estimation_stats[current_material] = mi_estimation_stats[current_material].set_index(['use_short', 'const_short', 'R5_32'])
    mi_estimation_stats[current_material]['db_count'] = db_combinations_stats[('count', current_material)]
    mi_estimation_stats[current_material]['db_avg'] = db_combinations_stats[('avg', current_material)]
    mi_estimation_stats[current_material]['db_sd'] = db_combinations_stats[('sd', current_material)]
    mi_estimation_stats[current_material]['db_5'] = db_combinations_stats[('p5', current_material)]
    mi_estimation_stats[current_material]['db_25'] = db_combinations_stats[('p25', current_material)]
    mi_estimation_stats[current_material]['db_50'] = db_combinations_stats[('p50', current_material)]
    mi_estimation_stats[current_material]['db_75'] = db_combinations_stats[('p75', current_material)]
    mi_estimation_stats[current_material]['db_95'] = db_combinations_stats[('p95', current_material)]

# %% calculate scores for materials' coverage
# to decide whether to use the selection algorithm or global statistics


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
db_material_scores.loc[db_material_scores['count'] > len(buildings_import) * 0.15, "score"] += 1  # covers at least 15% of the db (at least 120 datapoints)
db_material_scores.loc[db_material_scores['combi_coverage'] > 0.5, "score"] += 1  # covers at least 50% of the existing combinations in the db
db_material_scores.loc[db_material_scores['gini'] < 0.8, "score"] += 1  # datapoints don't all come from very few combinations

# %% selection algorithm

stop_count = 30


def expand_selection(selection, count, condition, material):
    newselection = [list(v) for v in db_combinations_data[material] if eval(condition)]
    if newselection:  # pythonic way to check if newselection is not empty
        selection += newselection
        count = 0
        for item in selection:
            item[3] += 1  # counter for how many rounds this selection was in an expansion
            count += item[2] * item[3]  # count how many datapoints are in selection
    return selection, count


mi_estimation_writer = pd.ExcelWriter('MI_cross_validation\\round' + str(cross_round) + '_' + today + ".xlsx", engine='openpyxl')

# expand selection algorithm for materials with score >= 2
for current_material in db_material_scores.query('score >= 2').index:
    mi_estimation_stats[current_material] = mi_estimation_stats[current_material].sort_values(by="db_count", ascending=False)

    for current_combi in mi_estimation_stats[current_material].index:  # running index for the current combination in mi_estimation

        current_selection = []
        current_count = 0

        # 1.1 add perfect matches
        if current_count < stop_count:
            current_condition = 'current_combi == v[0]'
            current_selection, current_count = expand_selection(current_selection, current_count, current_condition, current_material)
        # 1.2 add similar use types
        if current_count < stop_count:
            if current_combi[0] == 'NR':
                current_condition = "('UN' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2] == v[0][2])"
                current_selection, current_count = expand_selection(current_selection, current_count, current_condition, current_material)
            else:  # i.e. if current_combi[0][0] == 'R':
                current_condition = "('RU' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2] == v[0][2])"  # TODO consider whether to first add UN (currently in the IF below) and only then RU?
                current_selection, current_count = expand_selection(current_selection, current_count, current_condition, current_material)
                if current_count < stop_count:  # TODO this adds UN. consider whether to add the opposite R type e.g. if we're at RS then add RM and vice versa
                    current_condition = "('UN' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2] == v[0][2])"
                    current_selection, current_count = expand_selection(current_selection, current_count, current_condition, current_material)

        # 2.1 repeat for bigger 5-level region, not including the current 32-level region
        if current_count < stop_count:
            current_condition = "(current_combi[0] == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2][:3] in v[0][2]) and (current_combi[2] != v[0][2])"
            current_selection, current_count = expand_selection(current_selection, current_count, current_condition, current_material)
        # 2.2 add similar use types
        if current_count < stop_count:
            if current_combi[0] == 'NR':
                current_condition = "('UN' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2][:3] in v[0][2]) and (current_combi[2] != v[0][2])"
                current_selection, current_count = expand_selection(current_selection, current_count, current_condition, current_material)
            else:  # make sure to keep it conformed to 1.2 TODO decisions!
                current_condition = "('RU' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2][:3] in v[0][2]) and (current_combi[2] != v[0][2])"
                current_selection, current_count = expand_selection(current_selection, current_count, current_condition, current_material)
                if current_count < stop_count:
                    current_condition = "('UN' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2][:3] in v[0][2]) and (current_combi[2] != v[0][2])"
                    current_selection, current_count = expand_selection(current_selection, current_count, current_condition, current_material)

        # 3.1 repeat for all regions
        # TODO consider if stop_count or if stop_count-x to not expand to the entire world if we're already close to stop_count
        if current_count < stop_count:
            current_condition = "(current_combi[0] == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2][:3] not in v[0][2])"
            current_selection, current_count = expand_selection(current_selection, current_count, current_condition, current_material)
        # 3.2 add similar use types
        if current_count < stop_count:
            if current_combi[0] == 'NR':
                current_condition = "('UN' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2][:3] not in v[0][2])"
                current_selection, current_count = expand_selection(current_selection, current_count, current_condition, current_material)
            else:  # make sure to keep it conformed to 1.2 TODO decisions!
                current_condition = "('RU' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2][:3] not in v[0][2])"
                current_selection, current_count = expand_selection(current_selection, current_count, current_condition, current_material)
                if current_count < stop_count:
                    current_condition = "('UN' == v[0][0]) and (current_combi[1] == v[0][1]) and (current_combi[2][:3] not in v[0][2])"
                    current_selection, current_count = expand_selection(current_selection, current_count, current_condition, current_material)

        # When done: concatenate current_selection into one dataframe, including repetition of selections from previous expansion rounds i.e. v[3] in the second for loop
        # try:  # TODO temporary solution for empty combinations
        current_selection_combined = pd.concat([v[1] for v in current_selection for i in range(v[3])], copy=True).loc[:, ['id', current_material]].dropna()
        current_selection_combined['expansion_round'] = current_selection_combined.groupby('id').cumcount()
        current_selection_combined['expansion_round'] = current_selection_combined['expansion_round'].max() - current_selection_combined['expansion_round']
        if current_combi not in current_selection_combined.index:
            current_selection_combined['expansion_round'] += 1
        # fill results into mi_estimation_stats 'expanded_count', 'avg', 'sd', 'p5', 'p25', 'p50', 'p75', 'p95', 'expansion_rounds'
        mi_estimation_stats[current_material].loc[current_combi, 'expand_count'] = current_count
        mi_estimation_stats[current_material].loc[current_combi, 'expand_avg'] = current_selection_combined[current_material].mean()
        mi_estimation_stats[current_material].loc[current_combi, 'expand_sd'] = current_selection_combined[current_material].std()
        mi_estimation_stats[current_material].loc[current_combi, 'expand_5'] = np.quantile(current_selection_combined[current_material], q=0.05)  # faster than pandas's current_selection_combined['steel'].quantile(q=0.05)
        mi_estimation_stats[current_material].loc[current_combi, 'expand_25'] = np.quantile(current_selection_combined[current_material], q=0.25)
        mi_estimation_stats[current_material].loc[current_combi, 'expand_50'] = np.quantile(current_selection_combined[current_material], q=0.50)
        mi_estimation_stats[current_material].loc[current_combi, 'expand_75'] = np.quantile(current_selection_combined[current_material], q=0.75)
        mi_estimation_stats[current_material].loc[current_combi, 'expand_95'] = np.quantile(current_selection_combined[current_material], q=0.95)
        mi_estimation_stats[current_material].loc[current_combi, 'expand_rounds'] = current_selection[0][3]
        # except ValueError:
        #     current_selection_combined = pd.DataFrame(columns=['id', current_material, 'expansion_round'])

        # save current_selection_combined for backup and reference
        mi_estimation_data[current_material][current_combi] = current_selection_combined.copy()

    # HINT cosmetic: resort by index
    mi_estimation_stats[current_material].sort_index(inplace=True)

    # # export as individual Excel file
    # filename = current_material + '_stop_at_' + str(stop_count) + '_' + today
    # mi_estimation_stats[current_material].reset_index().to_excel('MI_results\\' + filename + '.xlsx', sheet_name=(current_material))

    # export as sheet
    mi_estimation_stats[current_material].reset_index().to_excel(mi_estimation_writer, sheet_name=current_material)

# use global statistics for materials with score < 2
for current_material in db_material_scores.query('score < 2').index:  # bulk edit all combinations with the global statistics, to avoid cycling through all combinations unnecessarily
    current_selection_combined = buildings_import[['id', current_material]].copy().dropna(how='any')
    current_selection_combined['expansion_round'] = 1
    current_selection_combined['expansion_round']
    mi_estimation_stats[current_material]['expand_count'] = current_selection_combined[current_material].count()
    mi_estimation_stats[current_material]['expand_avg'] = current_selection_combined[current_material].mean()
    mi_estimation_stats[current_material]['expand_sd'] = current_selection_combined[current_material].std()
    mi_estimation_stats[current_material]['expand_5'] = np.quantile(current_selection_combined[current_material], q=0.05)  # faster than pandas's current_selection_combined['steel'].quantile(q=0.05)
    mi_estimation_stats[current_material]['expand_25'] = np.quantile(current_selection_combined[current_material], q=0.25)
    mi_estimation_stats[current_material]['expand_50'] = np.quantile(current_selection_combined[current_material], q=0.50)
    mi_estimation_stats[current_material]['expand_75'] = np.quantile(current_selection_combined[current_material], q=0.75)
    mi_estimation_stats[current_material]['expand_95'] = np.quantile(current_selection_combined[current_material], q=0.95)
    mi_estimation_stats[current_material]['expand_rounds'] = 1
    for current_combi in mi_estimation_stats[current_material].index:
        mi_estimation_data[current_material][current_combi] = current_selection_combined.copy()
    for current_combi in mi_estimation_stats[current_material].query('db_count >= 1').index:  # add perfect combinations for the few that have them
        mi_estimation_data[current_material][current_combi] = pd.concat([current_selection_combined, buildings_import.loc[current_combi, ['id', current_material]].copy().dropna(how='any')])
        mi_estimation_data[current_material][current_combi]['expansion_round'].fillna(0, inplace=True)
        mi_estimation_data[current_material][current_combi]['expansion_round'] = mi_estimation_data[current_material][current_combi]['expansion_round'].astype("int")
        mi_estimation_stats[current_material].loc[current_combi, 'expand_count'] = mi_estimation_data[current_material][current_combi][current_material].count()
        mi_estimation_stats[current_material].loc[current_combi, 'expand_avg'] = mi_estimation_data[current_material][current_combi][current_material].mean()
        mi_estimation_stats[current_material].loc[current_combi, 'expand_sd'] = mi_estimation_data[current_material][current_combi][current_material].std()
        mi_estimation_stats[current_material].loc[current_combi, 'expand_5'] = np.quantile(mi_estimation_data[current_material][current_combi][current_material], q=0.05)  # faster than pandas's current_selection_combined['steel'].quantile(q=0.05)
        mi_estimation_stats[current_material].loc[current_combi, 'expand_25'] = np.quantile(mi_estimation_data[current_material][current_combi][current_material], q=0.25)
        mi_estimation_stats[current_material].loc[current_combi, 'expand_50'] = np.quantile(mi_estimation_data[current_material][current_combi][current_material], q=0.50)
        mi_estimation_stats[current_material].loc[current_combi, 'expand_75'] = np.quantile(mi_estimation_data[current_material][current_combi][current_material], q=0.75)
        mi_estimation_stats[current_material].loc[current_combi, 'expand_95'] = np.quantile(mi_estimation_data[current_material][current_combi][current_material], q=0.95)

    # # export as individual Excel file
    # filename = current_material + '_stop_at_' + str(stop_count) + '_' + today
    # mi_estimation_stats[current_material].reset_index().to_excel('MI_results\\' + filename + '.xlsx', sheet_name=(current_material))

    # export as sheet
    # mi_estimation_stats[current_material].reset_index().to_excel(mi_estimation_writer, sheet_name=current_material)

mi_estimation_writer.save()
mi_estimation_writer.close()

# %% visualize cross validations

test_data.sort_index(inplace=True)

analysis_comparison_data = {}
for current_material in materials:
    for key, value in mi_estimation_data[current_material].items():
        analysis_comparison_data[key + (current_material,)] = mi_estimation_data[current_material][key]
        analysis_comparison_data[key + (current_material,)] = analysis_comparison_data[key + (current_material,)].reset_index()
        # try:
        analysis_comparison_data[key + (current_material,)] = analysis_comparison_data[key + (current_material,)].drop(['id'] + dims_names, axis=1)
        # finally:
        analysis_comparison_data[key + (current_material,)]['combination'] = str(key)
analysis_comparison_data = pd.concat(analysis_comparison_data)
analysis_comparison_data['value'] = analysis_comparison_data.sum(axis=1, numeric_only=True)
analysis_comparison_data.index.rename(dims_names + ['material', 'id'], inplace=True)
analysis_comparison_data.reset_index(inplace=True)
analysis_comparison_data.drop(materials + ['id'], axis=1, inplace=True)
analysis_comparison_data['R5'] = analysis_comparison_data['R5_32'].str.split('_').str[0]

with PdfPages('MI_cross_validation\\round' + str(cross_round) + '_' + today + '.pdf') as pdf:
    for current_combi in test_data.query("(use_short != 'RU') and (use_short != 'UN')").index.unique():
        cross_compare = sns.boxplot(data=analysis_comparison_data.loc[analysis_comparison_data['combination'] == str(current_combi)], x='material', y='value', showfliers=False)
        cross_compare = sns.stripplot(data=test_data.loc[current_combi][materials], palette="dark")
        cross_compare.set_title(current_combi)
        pdf.savefig(cross_compare.figure)
        plt.close()
