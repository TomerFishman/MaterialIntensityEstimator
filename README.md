# RASMI: Regional Assessment of buildings’ Material Intensities

A framework to estimate ranges of buildings' material intensities (kg/m^2^)

Please refer to the Data Descriptor article for full details.

## What's RASMI?

The Regional Assessment of buildings’ Material Intensities (RASMI) is a dataset and accompanying method that provides comprehensive and consistent representative MI value ranges. Value ranges embody the inherent variability that exists in buildings. 

RASMI consists of 3072 MI ranges for:
- `[material]` 8 materials (concrete, steel, bricks, wood, glass, copper, aluminum, and plastics) 
- `[structure type]` 4 structural construction types (reinforced concrete structure, masonry structure, timber structure, and steel frame structure) 
- `[function type]` 3 functional use types (Residential single-family, residential multifamily, and non-residential) 
- `[region]` 32 global regions compatible with global IAM applications like the Shared Socioeconomic Pathways (SSP). 

Each datapoint is a range of values that represent one of the unique combinations of these dimensions. This yields 8 x 4 x 3 x 32 = 3072 MI ranges.

For instance, the material intensity of *concrete* `[material]` in *steel frame structures* `[structure type]` used for *multifamily housing* `[function type]` in *Japan* `[region]`.

## Motivation

The construction materials used in buildings have significant environmental impacts and implications for global material flows and emissions. Material Intensity (MI) is a metric that measures the mass of construction materials per unit of floor area in a building, and is used to model buildings’ materials and assess their resource use and environmental performance. However, the availability and quality of MI data is inconsistent, incomparable, and limited, especially for regions in the Global South. 
The dataset is reproducible, traceable, and updatable, using synthetic data when required. It can be used for estimating historical and future material flows and emissions, assessing demolition waste and at-risk stocks, and evaluating urban mining potentials.

## The data
is in MI_results\
- MI_ranges_(date).xlsx is the dataset of the estimated MI ranges. **This is probably the file you're looking for.** 
- MI_data_(date).xlsx is the raw pools of MI used to create the MI ranges. This is mostly for reproducability.

### Versioning
Versions are marked by the (date) in the filename.

## Descriptions of main files in the repository
<details>
<summary>Expand to view</summary>

| Folder | File | Decription |
|-|-|-|
|(root) |MI_estimator.py |Python 3 code to create the MI ranges |
|MI_results |See above | |
|data_input_and_ml_processing\ |buildings_v2.xlsx |data from the Heeren & Fishman DB |
| |buildings_v2-structure_type_ML.ows |classification of structure types (Orange suite file) |
| |buildings_v2-structure_type_ML.xlsx |output of the classification of structure types |
|postestimation\ |various files |outputs of the postestimation code in MI_estimator.py |
|tests\ | various folders and files |outputs of the tests of cross validation and effects of the pool size on the MI results |

Refer to the Data Descriptor article for details.
</details>

## How to cite
Please cite both the Data Descriptor and the specific data version used:

**Data Descriptor:** Tomer Fishman, Alessio Mastrucci, Yoav Peled, Shoshanna Saxe, Bas van Ruijven (2023) Global ranges of building material intensities differentiated by region, structure, and function: the RASMI dataset *In review*

**Data version:** refer to Zenodo

## Contact
Tomer Fishman t.fishman@cml.leidenuniv.nl
