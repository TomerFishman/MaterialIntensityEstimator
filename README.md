# RASMI: Regional Assessment of buildings’ Material Intensities

A framework to estimate ranges of buildings' material intensities (kg/m<sup>2</sup>)

RASMI aims to answer the question ***which MI data are appropriate for my country/region of interest?***

Please refer to the Data Descriptor article for full details, see "how to cite" below.

![sample MI ranges box-letter plots](/postestimation/sample_ranges.png)

## :house: What's RASMI?

The Regional Assessment of buildings’ Material Intensities (RASMI) is a dataset and accompanying method that provides comprehensive and consistent representative MI value ranges. Value ranges embody the inherent variability that exists in buildings. 

RASMI consists of 3072 MI ranges for:
- `[material]` 8 materials (concrete, steel, bricks, wood, glass, copper, aluminum, and plastics) 
- `[structure type]` 4 structural construction types (reinforced concrete structure, masonry structure, timber structure, and steel frame structure) 
- `[function type]` 3 functional use types (Residential single-family, residential multifamily, and non-residential) 
- `[region]` 32 global regions compatible with global IAM applications like the Shared Socioeconomic Pathways (SSP). 

Each datapoint is a range of values that represent one of the unique combinations of these dimensions. This yields 8 x 4 x 3 x 32 = 3072 MI ranges.

### Simple example
The table below shows how to use RASMI to estimate the expected range of total mass of `[materials]` stocked in a 120m<sup>2</sup> *single-family house* `[function type]` with a *reinforced concrete structure* `[structure type]` in *Brazil* `[region]`:

![mi_table](https://github.com/user-attachments/assets/7b6ed3e7-98c6-4df5-8323-ada1d0806619)
(This uses the 20230905 version of the data).

## :hospital: Motivation

The construction materials used in buildings have significant environmental impacts and implications for global material flows and emissions. Material Intensity (MI) is a metric that measures the mass of construction materials per unit of floor area in a building, and is used to model buildings’ materials and assess their resource use and environmental performance. However, the availability and quality of MI data is inconsistent, incomparable, and limited, especially for regions in the Global South. 
The dataset is reproducible, traceable, and updatable, using synthetic data when required. It can be used for estimating historical and future material flows and emissions, assessing demolition waste and at-risk stocks, and evaluating urban mining potentials.

## :factory: The data
is in MI_results\
- `MI_ranges_(date).xlsx` is the dataset of the estimated MI ranges. ***This is probably the file you're looking for.***
- `MI_data_(date).xlsx` is the raw pools of MI used to create the MI ranges. This is mostly for reproducability.

### Versioning
Versions are marked by the (date) in the filename.

## :office: Descriptions of main files in the repository
<details>
<summary>Expand to view</summary>

| Folder | File | Decription |
|-|-|-|
|(root) |MI_estimator.py |Python 3 code to create the MI ranges |
|MI_results |See above | |
|data_input_and_ml_processing\ |buildings_v2.xlsx |data from the Heeren & Fishman DB |
| |buildings_v2-structure_type_ML.ows |classification of structure types (Orange suite file) |
| |buildings_v2-structure_type_ML.xlsx |output of the classification of structure types |
| |dims_structure.xlsx |structure and label options for the various dimensions of the data|
|postestimation\ |various files |outputs of the postestimation code in MI_estimator.py |
|tests\ | various folders and files |outputs of the tests of cross validation and effects of the pool size on the MI results |

Refer to the Data Descriptor article for details, see "how to cite" below.
</details>

## :speech_balloon: How to cite
Please cite both the Data Descriptor journal article and the specific data version used:

**Data Descriptor article:** Tomer Fishman, Alessio Mastrucci, Yoav Peled, Shoshanna Saxe, Bas van Ruijven. ***RASMI: Global Ranges of Building Material Intensities Differentiated by Region, Structure, and Function***. Scientific Data 2024, 11 (1), 418. https://doi.org/10.1038/s41597-024-03190-7.

**Data version:** preferably use the DOI of the Zeonodo release, e.g. [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10124952.svg)](https://doi.org/10.5281/zenodo.10124952) Refer to the release number (on the right)

Cite all versions? You can cite all versions by using the DOI 10.5281/zenodo.10124951. This DOI represents all versions, and will always resolve to the latest one.

## :e-mail: Contact
Tomer Fishman t.fishman@cml.leidenuniv.nl

## :memo: Acknowledgements

This work was conducted with support by the IIASA-Israel program, and by the Israel Science Foundation project RUSTY (grant no. 2706/19). Funding was also provided by the Horizon Europe research and innovation programme under grant agreement no. 101056868 (CIRCOMOD) for TF and grant agreement No 101056810 (CircEUlar) for AM. Opinions are those of the authors only and do not necessarily reflect those of the European Union. Neither the European Union nor the granting authority can be held responsible for this. BvR and AM have been supported by the Energy Demand changes Induced by Technological and Social innovations (EDITS) project, which is an initiative coordinated by the Research Institute of Innovative Technology for the Earth (RITE) and the International Institute for Applied Systems Analysis (IIASA), and funded by the Ministry of Economy, Trade, and Industry (METI), Japan. SS was supported by the Canada Research Chair in Sustainable Infrastructure, Grant Number: 232970.

