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

addded cross validation code

revised with up-to-date code from the main file 5/9/2023

"""
# %% load libraries and load dimensions/features

from os import chdir
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import date
from sklearn.model_selection import train_test_split

# change this number to define a new cross validation round (e.g. round 5) and then re-run the entire code
cross_round = 19
# what % (from 0 to 1) of the MIs will be left out as a test set
test = 0.1

export_to_excel = False  # True will export to spreadsheets the intermediate and final results

today = date.today().strftime("%Y%m%d")  # for versioning purposes

chdir('C:\\Users\\Tomer\\Dropbox\\-the research\\2020 10 IIASA\\MI_project\\git\\MaterialIntensityEstimator\\tests\\cross_validation')  # set working directory

dims_structure_import = pd.read_excel("..\\..\\data_input_and_ml_processing\\dims_structure.xlsx", sheet_name="dims_structure")  # load the names of dimensions (labels) and their optional entities

dims_names = list(dims_structure_import.columns)  # create a list of the names of dimensions

dims_names = dims_names[7:]  # HINT remove unused dimensions to keep things clean

# create list of lists of the entities of each dimension
dims_list = []
for dim_x in dims_names:
    dim_lastvalidrow = dims_structure_import[dim_x].last_valid_index() + 1  # get the number of entities in the dimension
    dims_list += [list(dims_structure_import[dim_x][2:dim_lastvalidrow])]  # create a list of the entities in the dimension

dims_list[0] = dims_list[0][1:]  # remove IN 'informal' because there are simply not enough datapoints for meaningful estimations, consider including in future versions

materials = ['concrete', 'brick', 'wood', 'steel', 'glass', 'plastics', 'aluminum', 'copper']  # list of the materials

# %% load the MI database with structure type ML results from Orange

# load the Heeren & Fishman data after the Orange ML runs
buildings_import_full = pd.read_excel("..\\..\\data_input_and_ml_processing\\buildings_v2-structure_type_ML.xlsx", sheet_name="Sheet1")

# update column 'structure' where U from the original 'structure' is replaced by 'Random Forest'
buildings_import_full['structure'] = buildings_import_full['Random Forest'].where((buildings_import_full['structure'].str.match('U')), buildings_import_full['structure'])

# combine all plastics types
buildings_import_full['plastics'] = buildings_import_full[['plastics', 'PVC', 'polystyrene']].sum(axis=1, min_count=1)

# clean up buildings_import
buildings_import = buildings_import_full[['id'] + materials + dims_names]

# %% final setups of the database data

# remove IN 'informal' because there are simply not enough datapoints for meaningful estimations, consider including in future versions
buildings_import = buildings_import[buildings_import.function != 'IN']
# set the same function-structure-region multiindex as the other dataframes
buildings_import.set_index(dims_names, inplace=True)
# sort to make pandas faster and with less warnings
buildings_import.sort_index(inplace=True)

buildings_import_all = buildings_import.copy()

# %% cross validation: partition buildings_import to a test set and training set

buildings_import, test_data = train_test_split(buildings_import_all, test_size=test, random_state=cross_round)


# %% create a new dataframe of the counts of unique combinations that exist in the DB and their statistics

# including unspecifieds
# list of datafranes, one for each material...
db_combinations_stats = [pd.DataFrame(data=None, index=pd.MultiIndex.from_product(dims_list, names=dims_names)),
                         buildings_import[materials].groupby(dims_names).count(),
                         buildings_import[materials].groupby(dims_names).mean(),
                         buildings_import[materials].groupby(dims_names).std()[materials],
                         buildings_import[materials].groupby(dims_names).quantile(q=0.05),
                         buildings_import[materials].groupby(dims_names).quantile(q=0.25),
                         buildings_import[materials].groupby(dims_names).quantile(q=0.50),
                         buildings_import[materials].groupby(dims_names).quantile(q=0.75),
                         buildings_import[materials].groupby(dims_names).quantile(q=0.95)]
# ... and concatenate all materials to one df
db_combinations_stats = pd.concat(db_combinations_stats, axis=1, keys=['', 'count', 'avg', 'sd', 'p5', 'p25', 'p50', 'p75', 'p95'])

# examples of slicing and querying the db_combinations_stats dataframe
db_combinations_stats[('count', 'concrete')]
db_combinations_stats.loc[('NR', 'C', 'ASIA_CHN'), ('count', 'concrete')]
db_combinations_stats.loc[('NR', 'C', 'ASIA_CHN'), :]
db_combinations_stats.loc[:, db_combinations_stats.columns.isin(['concrete'], level=1)]

# cleaned-up dataframe that keeps only rows with values
db_combinations_stats_valid = db_combinations_stats.dropna(how='all')

# %% separate buildings_import to individual dataframes by valid combinations

# for each material in this dict, prefiltered as a list only with valid combinations (i.e. existing in buildings_import): [combination tuple, dataframe, [no. of rows in df, counts of each material], increment score set to 0]
# the items in the dict are a list and not a dict in case the iterative incremental pooling algorithm needs to duplicate list items. A dict can't have duplicate keys.
db_combinations_data = {}
for current_material in materials:
    db_combinations_data[current_material] = []
    [db_combinations_data[current_material].append([row[0], buildings_import.loc[row[0]], int(db_combinations_stats_valid.loc[row[0], ('count', current_material)]), 0]) for row in db_combinations_stats_valid.itertuples() if db_combinations_stats_valid.loc[row[0], ('count', current_material)] > 0]


# %% create a dataframe with all specified (i.e. not unspecifieds) combination options to be filled with data

# remove 'unspecified' entities !!make sure to change the list indexes as needed
dims_list_specified = dims_list[:]
dims_list_specified[0] = [x for x in dims_list_specified[0] if 'U' not in x]
dims_list_specified[1] = [x for x in dims_list_specified[1] if 'U' not in x]


# create empty dicts for storing the current pool MIs with their IDs and raw data for backup and reference
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
    # # optional: copy over the statistics of the raw MI data, for before-after comparisons
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

stop_count = 30  # minimal MI datapoints before stopping the expansion

export_to_excel = False  # opportunity to change your mind and export only the final results


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
    mi_estimation_writer = pd.ExcelWriter('cross_validation_round_' + str(cross_round) + '_' + today + ".xlsx", engine='openpyxl')

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

    # print('exporting full data to Excel')
    # mi_data_writer = pd.ExcelWriter('MI_results\\MI_data_' + today + ".xlsx", engine='openpyxl')
    # for current_material in materials:
    #     pd.concat([mi_estimation_data[current_material][current_combi].reset_index()
    #               for current_combi in mi_estimation_data[current_material].keys()],
    #               axis=0,
    #               keys=list(mi_estimation_data[current_material].keys())).sort_index().to_excel(mi_data_writer, sheet_name=current_material)
    # mi_data_writer.close()
    # print('done')

# %% cross validation

# how many test-set datapoints are within the 25%-75% percentile range, and how many are within 5%-95%?
test_data.sort_index(inplace=True)

# material = 'concrete'
# current_combi = ('NR', 'C', 'ASIA_CHN')

in_range = {}
in_range_count = {}
for material in materials:
    in_range[material] = pd.concat([test_data.query("(function != 'RU') and (function != 'UN')")[['id', material]], pd.DataFrame(columns=['in_25_75', 'in_05_95'])])
    for current_combi in in_range[material].index.unique():
        in_range[material].loc[current_combi, 'in_25_75'] = in_range[material].loc[current_combi, material].between(mi_estimation_stats[material].loc[current_combi, 'p_25'], mi_estimation_stats[material].loc[current_combi, 'p_75'])
        in_range[material].loc[current_combi, 'in_05_95'] = in_range[material].loc[current_combi, material].between(mi_estimation_stats[material].loc[current_combi, 'p_5'], mi_estimation_stats[material].loc[current_combi, 'p_95'])
    in_range_groupby = in_range[material].groupby(['function', 'structure', 'R5_32'])
    in_range_count[material] = pd.DataFrame()  # index=test_data.query("(function != 'RU') and (function != 'UN')").index.unique())
    in_range_count[material]['test_set_count'] = in_range_groupby[material].count()
    in_range_count[material] = in_range_count[material].loc[(in_range_count[material] != 0).any(axis=1)]
    in_range_count[material]['in_25_75_count'] = in_range_groupby['in_25_75'].sum().astype(int)
    in_range_count[material]['in_05_95_count'] = in_range_groupby['in_05_95'].sum().astype(int)
    in_range_count[material]['in_25_75_percent'] = in_range_count[material]['in_25_75_count'] / in_range_count[material]['test_set_count']
    in_range_count[material]['in_05_95_percent'] = in_range_count[material]['in_05_95_count'] / in_range_count[material]['test_set_count']

# in_range_summary = pd.DataFrame(index=materials, columns=['in_25_75_percent', 'in_05_95_percent'])
# for material in materials:
#     in_range_summary.loc[material] = in_range_count[material].loc[:, ['in_25_75_percent', 'in_05_95_percent']].mean()

in_range_count_concat = pd.concat(in_range_count)
in_range_summary = pd.DataFrame()
in_range_summary[cross_round] = in_range_count_concat.loc[:, ['in_25_75_percent', 'in_05_95_percent']].mean()
in_range_summary['average'] = in_range_summary.mean(axis=1)

in_range_summary.to_excel('cross_validation_test MIs_percent_in_range_' + today + '.xlsx')

# %% visualize cross validation for this run

test_data.sort_index(inplace=True)

# organize the results for visualization
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

# visualization of the test set overlaying the MI results box plots
with PdfPages('cross_validation_round_' + str(cross_round) + '_' + today + '.pdf') as pdf:
    for current_combi in test_data.query("(function != 'RU') and (function != 'UN')").index.unique():
        cross_compare = sns.boxplot(data=analysis_comparison_data.loc[analysis_comparison_data['combination'] == str(current_combi)], x='material', y='value', showfliers=False)
        cross_compare = sns.stripplot(data=test_data.loc[current_combi][materials], palette="dark")
        cross_compare.set_title(current_combi)
        pdf.savefig(cross_compare.figure)
        plt.close()

# %% estimate % changes from the current cross_round run compared to the full dataset run

# # load the full dataset run results
# full_results_stats = {}
# for material in materials:
#     full_results_stats[material] = pd.read_excel("..\\..\\MI_results\\MI_ranges_20230905.xlsx", sheet_name=material)
#     full_results_stats[material].set_index(dims_names, inplace=True)
#     full_results_stats[material].drop(columns=['Unnamed: 0'], inplace=True)

# # % changes dataframe with eaxct same structure as mi_estimation_stats and full_results_stats
# percent_change = full_results_stats.copy()
# for material in materials:
#     for p in ['p_0', 'p_5', 'p_25', 'p_50', 'p_75', 'p_95', 'p_100']:
#         percent_change[material][p] = (full_results_stats[material][p] - mi_estimation_stats[material][p]) / full_results_stats[material][p]

# # summaries
# percent_change_summary = pd.DataFrame(columns=percent_change[material].columns)
# for material in materials:
#     percent_change_summary.loc[material] = percent_change[material].mean()


# %% estimate % changes from the all cross_round runs compared to the full dataset run

# load the full dataset run results
full_results_stats = {}
for material in materials:
    full_results_stats[material] = pd.read_excel("..\\..\\MI_results\\MI_ranges_20230905.xlsx", sheet_name=material)
    full_results_stats[material].set_index(dims_names, inplace=True)
    full_results_stats[material].drop(columns=['Unnamed: 0'], inplace=True)

# load the various cross validation runs results
cross_validation_stats = {}
for material in materials:
    cross_validation_stats[material] = {}
    for run in range(cross_round + 1):
        cross_validation_stats[material][run] = pd.read_excel("cross_validation_round_" + str(run) + "_20230906.xlsx", sheet_name=material)
        cross_validation_stats[material][run].set_index(dims_names, inplace=True)
        cross_validation_stats[material][run].drop(columns=['Unnamed: 0'], inplace=True)

# % changes dataframe with eaxct same structure as mi_estimation_stats and full_results_stats
percent_change_all = {}
for material in materials:
    percent_change_all[material] = {}
    for run in range(cross_round + 1):
        percent_change_all[material][run] = pd.DataFrame(columns=cross_validation_stats[material][run].columns, index=cross_validation_stats[material][run].index)
        for p in ['p_0', 'p_5', 'p_25', 'p_50', 'p_75', 'p_95', 'p_100']:
            percent_change_all[material][run][p] = (cross_validation_stats[material][run][p] - full_results_stats[material][p]) / full_results_stats[material][p]
        percent_change_all[material][run].replace([np.inf, -np.inf], np.nan, inplace=True)


# summaries for each run
percent_change_all_summary = {}
for run in range(cross_round + 1):
    percent_change_all_summary[run] = pd.DataFrame(columns=percent_change_all[material][run].columns)
    for material in materials:
        percent_change_all_summary[run].loc[material] = percent_change_all[material][run].mean()
    percent_change_all_summary[run].loc[:, ['raw_HF_db_count', 'increment_iterations', 'incremented_count']] = np.nan

# average of the average % changes across runs
# first concat
percent_change_all_summary['all_runs'] = pd.concat(percent_change_all_summary)
percent_change_all_summary['mean_across_runs'] = pd.DataFrame(columns=percent_change_all[material][run].columns, index=materials)
for material in materials:
    percent_change_all_summary['mean_across_runs'].loc[material] = percent_change_all_summary['all_runs'].loc[(slice(None), material), :].mean()

# visualize the % changes of all runs
all_summary = sns.relplot(kind="line",
                          data=percent_change_all_summary['all_runs'].reorder_levels([1, 0]).transpose().loc[:, (materials[0:5], slice(None))],
                          dashes=False, height=6, aspect=1.5,
                          errorbar="sd", estimator='mean')
#                         errorbar=("pi", 100), estimator='median')
all_summary.set_xticklabels(['', '', '', '', '0th', '5th', '25th', 'median', '75th', '95th', '100th'])
all_summary.set(ylim=(-0.5, 0.4), xlim=(4, 10))
all_summary.set_axis_labels(x_var='MI range percentiles', y_var='Averaged relative change from the full dataset')
sns.move_legend(all_summary, "upper right", bbox_to_anchor=(.9, .95), frameon=False)
all_summary.savefig('relative_change.svg')

# sns.lineplot(data=percent_change_all_summary['all_runs'].reset_index().set_index('level_1').transpose(), dashes=False)
