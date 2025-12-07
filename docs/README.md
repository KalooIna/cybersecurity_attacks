# Cybersecurity Attacks ML Analizer

## Team Composition

Eugenio La Cava: Project Coordinator - DS

Otmane Qorchi: Adjunct Project Cordinator - DS

Janagam Vasantha: - ??

Elly Smagghe: - ??

Kaloina Rakotobe: - DS

Sanchana Krishna Kumar: - ??

Siham Eldjouher: - ??

## Description

## Repo Organization

```bash
cybersecurity_attacks
├── data
│   ├── admin1CodesASCII.txt
│   ├── alternateNamesV2.txt
│   ├── cybersecurity_attacks.csv
│   ├── dataframe_with_location.parquet
│   ├── df_location_data.parquet
│   ├── geo_data.parquet
│   ├── IN.txt
│   ├── india_cities.parquet
│   ├── iso-languagecodes.txt
│   ├── missing_data.parquet
│   └── readme.txt
├── db
│   ├── GeoLite2-City_20251202
│   │   ├── COPYRIGHT.txt
│   │   ├── GeoLite2-City.mmdb
│   │   ├── LICENSE.txt
│   │   └── README.txt
│   └── india-251205.osm.pbf
├── docs
│   ├── ML Python Labs Group Work Distribution.docx
│   └── README.md
├── generate_ascii_dir_repr.ps1
├── LICENSE
├── pipeline.py
├── pixi.lock
├── pixi.toml
└── src
    ├── __init__.py
    ├── download_files.py
    ├── EDA.py
    └── EDA1.py
```

## Composition

### EDA

Exploratory Data Analysis class for cybersecurity attack location data.

This Section performs comprehensive EDA on cybersecurity attack data, including
geocoding Indian cities and states, processing geographic data from GeoNames,
and enriching attack data with precise latitude/longitude coordinates.

The class handles data loading, preprocessing, city-state matching with multiple
strategies (exact match, city-only, alternate names, historical names), and
exports processed geographic data.

Attributes:
    data_dir (str): Directory path for storing datasets.
    required_files (list): List of file paths that must be present for EDA.
    max_city_population (int): Threshold for filtering out non-city entities.
    india_df (pd.DataFrame): DataFrame containing Indian cities data from GeoNames.
    admin_df (pd.DataFrame): DataFrame containing Indian administrative codes.
    cybersecurity_df (pd.DataFrame): DataFrame containing cybersecurity attack data.

## Results

So far the EDA gathers the latitude and longitude data for the Geolocation column for the 93.5% of the rows cybersecurity dataset.

## Utilities

The powershell script "generate_ascii_dir_repr.ps1" generates the ASCII representation of the directory tree  with options to hide the "." preceded directories and files

## Package management

The environment is managed through Pixi (https://pixi.sh/) a multi-language package manager that utilizes uv and anaconda repos under the hood of a Rust-based package manager.

### Available app commands

```bash
# This command runs the entire pipeline
run-pipeline = "python pipeline.py"

#This command runs the generator for the ascii representation of the directory tree excluding directories and files preceded by '.'
gen-dir-repr = "pwsh -Command './generate_ascii_dir_repr.ps1 ./ -Exclude .*, __pycache__ -Depth 3'"
```
